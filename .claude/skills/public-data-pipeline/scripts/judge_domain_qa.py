# -*- coding: utf-8 -*-
"""Kimi 裁判脚本（蒸馏闭环第二/三步）：QA 过筛 与 PT vs SFT 对比评分。

三种模式：

1) judge（默认）：读 generate_domain_qa.py 的 manifest，用 Kimi 对每条 QA 三维评分
   （grounding/terminology/value 各 1~5），三档分流 pass/review/drop，产出：
   - domain_env_qa_judged.jsonl  逐条评分明细
   - domain_env_qa_sft.jsonl     过筛后的 SFT 训练集（pass + --promote 改判）
   - domain_env_qa_review.md     裁判报告（兼人工抽检文档）
   冒烟: python .claude/skills/public-data-pipeline/scripts/judge_domain_qa.py --limit 3
   全量: python .claude/skills/public-data-pipeline/scripts/judge_domain_qa.py
   改判: python .claude/skills/public-data-pipeline/scripts/judge_domain_qa.py --promote "12,27"（复核合格条目强制入 sft 集）

2) init-compare：从 manifest 的 eval 条目生成对比骨架（question/reference 已填，
   answer_pt/answer_sft 留空，人工用 llamafactory-cli chat 各答 10 题后粘入）：
   python .claude/skills/public-data-pipeline/scripts/judge_domain_qa.py --mode init-compare

3) compare：对填好的骨架逐题评分（两答案各三维分 + prefer），出对比报告：
   python .claude/skills/public-data-pipeline/scripts/judge_domain_qa.py --mode compare

需项目根 .env 配置 MOONSHOT_API_KEY（.env 已被 gitignore；勿写 .env.local——它被 git 追踪）。
裁判默认 kimi-k3（钥匙来自 platform.moonshot.cn；模型名以 Moonshot 文档为准）。
第三方端点兼容性不一：不用 response_format、宽松解析；thinking disabled 参数不认时自动裸调。
"""

import argparse
import json
import os
import re
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
DATA_DIR = os.path.join(REPO, "data")
ENV_FILE = os.path.join(REPO, ".env")

JUDGE_SYSTEM = "你是环境保护领域的技术评审，负责严格评审用于模型蒸馏的问答对质量。"

JUDGE_TEMPLATE = """【原文片段】
{block}

【问题】{question}
【答案】{answer}
【模型自报价引】「{quote}」
【程序校验】{grounding_note}

请按三个维度各打 1~5 分（1 差 5 优）：
- grounding(原文依据)：答案的每个关键论断能否在原文找到支持、引用是否真实可靠
- terminology(术语准确)：条款号/数值/单位/专业术语是否正确、有无编造
- value(问题价值)：问题是否具体、有专业价值、非背景复述、非一望而知

只输出一个 json 对象，不要 markdown 代码块，不要任何多余文字：
{{"grounding": 4, "terminology": 5, "value": 3, "issues": "发现的问题(没有则空串)", "verdict_comment": "一句话总评"}}"""

COMPARE_SYSTEM = "你是环境保护领域的技术评审，负责对比评审两个模型对同一问题的回答。"

COMPARE_TEMPLATE = """【原文片段】
{block}

【问题】{question}
【参考答案】（另一大模型所写，仅供参照）{reference}

【A 模型答案】{answer_a}
【B 模型答案】{answer_b}

请按三个维度各给两个答案打 1~5 分（1 差 5 优）：
- terminology(术语准确)：条款号/数值/单位/专业术语是否正确、有无编造
- faithfulness(忠实原文)：是否忠于原文、有无编造或臆测
- completeness(完整回答)：是否完整、直接地回答了问题

只输出一个 json 对象，不要 markdown 代码块，不要任何多余文字：
{{"A": {{"terminology": 4, "faithfulness": 3, "completeness": 4}}, "B": {{"terminology": 5, "faithfulness": 4, "completeness": 4}}, "prefer": "A|B|tie 三选一", "comment": "一句话理由"}}"""


def load_env(path):
    """极简 .env 解析：读 KEY=VALUE，忽略空行与 # 注释。"""
    env = {}
    if not os.path.exists(path):
        return env
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def parse_json_block(raw):
    """宽松解析模型输出：剥围栏、截首{尾}、json.loads。失败返回 None。"""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"\A```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\Z", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def clamp_score(v):
    """分数兜底：非数字或越界时返回 None。"""
    try:
        score = int(v)
    except (TypeError, ValueError):
        return None
    return score if 1 <= score <= 5 else None


def call_llm(client, model, system, user_content, retries):
    """调一次评审 API，返回回复文本或 None。

    兼容性逐级降级（不同端点/模型的参数差异）：
    - 优先 temperature=0(评分求稳定) + thinking disabled：GLM-4.5+ 等推理模型不关思考时
      max_tokens 会被 reasoning_content 烧光、content 返回空串；
    - 模型只允许 temperature=1（如 kimi-k3 传 temperature=0 直接 400）→ 省略 temperature；
    - 端点不认 thinking 参数 → 裸调。
    一轮参数组合全失败才计入一次重试（指数退避）。max_tokens=2048 给思考留余量。
    """
    base = dict(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_content}],
        max_tokens=2048,
    )
    variants = [
        dict(base, temperature=0, extra_body={"thinking": {"type": "disabled"}}),
        dict(base, extra_body={"thinking": {"type": "disabled"}}),
        dict(base, temperature=0),
        dict(base),
    ]
    for attempt in range(retries + 1):
        last_note = "未知错误"
        for kwargs in variants:
            try:
                resp = client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content
                if content and content.strip():
                    return content
                last_note = "返回空 content(疑似思考烧尽 tokens)"
            except Exception as exc:
                last_note = str(exc)
        if attempt >= retries:
            print(f"[skip] API 调用失败(重试 {retries} 次后): {last_note[:120]}")
            return None
        wait = 2 ** (attempt + 1)
        print(f"[retry] 第 {attempt + 1} 次失败({last_note[:80]})，{wait}s 后重试")
        time.sleep(wait)
    return None


def classify(scores, threshold):
    """三档分流：硬伤剔除 / 过筛 / 边缘进人工复核。"""
    g, t, v = scores["grounding"], scores["terminology"], scores["value"]
    mean = (g + t + v) / 3
    if mean < 3.0 or g <= 2:
        return "drop"
    if mean >= threshold and min(g, t, v) >= 3:
        return "pass"
    return "review"


def mode_judge(client, args):
    """裁判模式：QA 三维评分 + 三档分流 + 报告。"""
    records = []
    for line in open(args.manifest, encoding="utf-8"):
        line = line.strip()
        if line:
            records.append(json.loads(line))
    targets = [r for r in records if r.get("status") == "ok"]
    if args.split != "all":
        targets = [r for r in targets if r.get("split") == args.split]
    if args.limit:
        targets = targets[: args.limit]
    print(f"待评审 QA {len(targets)} 条(阈值 {args.threshold})")

    promote = {int(x) for x in re.split(r"[,\s]+", args.promote.strip()) if x} if args.promote else set()

    judged = []
    for n, rec in enumerate(targets, 1):
        note = "引用已确认存在于原文（程序校验）"  # 进裁判的都是机械校验通过的 ok 条目
        user_content = JUDGE_TEMPLATE.format(
            block=rec["block"], question=rec["question"], answer=rec["answer"],
            quote=rec.get("quote", ""), grounding_note=note,
        )
        raw = call_llm(client, args.model, JUDGE_SYSTEM, user_content, args.retries)
        obj = parse_json_block(raw or "") if raw else None
        scores = {"grounding": 0, "terminology": 0, "value": 0}
        if obj:
            for k in scores:
                scores[k] = clamp_score(obj.get(k)) or 0  # 非法分按 0 计(进 drop/review)
        verdict = classify(scores, args.threshold)
        if rec.get("split") == "train" and rec.get("block_idx") in promote:
            verdict = "pass"  # 人工复核改判，强制入 sft 集
        judged.append({
            "block_idx": rec["block_idx"], "split": rec["split"],
            "question": rec["question"], "answer": rec["answer"], "quote": rec.get("quote", ""),
            **scores, "issues": (obj or {}).get("issues", ""), "verdict_comment": (obj or {}).get("verdict_comment", ""),
            "verdict": verdict, "parse_ok": obj is not None,
        })
        print(f"[{n}/{len(targets)}] idx={rec['block_idx']}({rec['split']}) "
              f"g/t/v={scores['grounding']}/{scores['terminology']}/{scores['value']} → {verdict}")
        time.sleep(args.sleep)

    # 逐条评分明细
    with open(args.out, "w", encoding="utf-8") as f:
        for row in judged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 过筛后的 sft 训练集（仅 train split 的 pass）
    sft_rows = [{"instruction": r["question"], "input": "", "output": r["answer"]}
                for r in judged if r["split"] == "train" and r["verdict"] == "pass"]
    with open(args.sft_out, "w", encoding="utf-8") as f:
        for row in sft_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 报告（人工抽检文档）
    n_pass = sum(1 for r in judged if r["verdict"] == "pass")
    n_review = sum(1 for r in judged if r["verdict"] == "review")
    n_drop = sum(1 for r in judged if r["verdict"] == "drop")
    def dim_mean(k):
        vals = [r[k] for r in judged if r["parse_ok"]]
        return round(sum(vals) / len(vals), 2) if vals else 0.0
    lines = [
        f"# domain_env QA 裁判报告（{args.model}，{time.strftime('%Y-%m-%d %H:%M')}）", "",
        f"统计: pass {n_pass} / review {n_review} / drop {n_drop}"
        f"（其中 --promote 改判 {len(promote & {r['block_idx'] for r in judged})} 条）", "",
        f"维度均分(仅解析成功条目): grounding {dim_mean('grounding')} / "
        f"terminology {dim_mean('terminology')} / value {dim_mean('value')}", "",
        f"SFT 训练集: {args.sft_out}（{len(sft_rows)} 条）", "",
        "## 需人工复核（务必看；合格的用 --promote \"idx,...\" 改判后重跑）", "",
        "| block_idx | g/t/v | 均分 | issues |", "|---|---|---|---|",
    ]
    for r in judged:
        if r["verdict"] == "review":
            mean = round((r["grounding"] + r["terminology"] + r["value"]) / 3, 1)
            lines.append(f"| {r['block_idx']} | {r['grounding']}/{r['terminology']}/{r['value']} | {mean} | {str(r['issues'])[:60]} |")
    lines += ["", "## 已剔除（含原因）", ""]
    for r in judged:
        if r["verdict"] == "drop":
            lines.append(f"- idx {r['block_idx']}: g/t/v={r['grounding']}/{r['terminology']}/{r['value']} {r['issues']}")
    lines += ["", "## 全量明细", ""]
    for r in judged:
        mean = round((r["grounding"] + r["terminology"] + r["value"]) / 3, 1)
        lines += [
            f"### [idx:{r['block_idx']}] {r['grounding']}/{r['terminology']}/{r['value']} 均 {mean} · {r['verdict']}",
            f"**问题**：{r['question']}", f"**答案**：{r['answer']}", f"**引用**：「{r['quote']}」",
            f"裁判意见：{r['verdict_comment']}", "",
        ]
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"pass {n_pass} / review {n_review} / drop {n_drop}；sft 集 {len(sft_rows)} 条")
    print("报告:", args.report)


def mode_init_compare(args):
    """对比模式第一步：从 manifest eval 条目生成对比骨架。"""
    records = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    evals = [r for r in records if r.get("status") == "ok" and r.get("split") == "eval"]
    with open(args.compare, "w", encoding="utf-8") as f:
        for r in evals:
            row = {"question": r["question"], "reference": r["answer"], "block": r["block"],
                   "answer_pt": "", "answer_sft": ""}  # 两个空位：人工 chat 后粘入
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"对比骨架已生成: {args.compare}（{len(evals)} 题，请填 answer_pt / answer_sft）")


def mode_compare(client, args):
    """对比模式第二步：逐题给 PT/SFT 两答案评分 + 偏好，出报告。"""
    rows = [json.loads(l) for l in open(args.compare, encoding="utf-8") if l.strip()]
    dims = ("terminology", "faithfulness", "completeness")
    results = []
    for n, row in enumerate(rows, 1):
        if not row.get("answer_pt", "").strip() or not row.get("answer_sft", "").strip():
            print(f"[skip] 第 {n} 题答案未填全，跳过")
            continue
        # 奇偶行交换 A/B 位置，降低裁判的位置偏差；评分后映射回 pt/sft
        swap = n % 2 == 0
        answer_a, answer_b = (row["answer_sft"], row["answer_pt"]) if swap else (row["answer_pt"], row["answer_sft"])
        user_content = COMPARE_TEMPLATE.format(
            block=row["block"], question=row["question"], reference=row["reference"],
            answer_a=answer_a, answer_b=answer_b,
        )
        raw = call_llm(client, args.model, COMPARE_SYSTEM, user_content, args.retries)
        obj = parse_json_block(raw or "") if raw else None
        pt_scores = {k: 0 for k in dims}
        sft_scores = {k: 0 for k in dims}
        prefer = "tie"
        if obj:
            src_a, src_b = obj.get("A", {}), obj.get("B", {})
            for k in dims:
                a, b = clamp_score(src_a.get(k)) or 0, clamp_score(src_b.get(k)) or 0
                (sft_scores if swap else pt_scores)[k] = a
                (pt_scores if swap else sft_scores)[k] = b
            raw_prefer = str(obj.get("prefer", "tie")).strip().lower()
            prefer = {"a": "sft" if swap else "pt", "b": "pt" if swap else "sft"}.get(raw_prefer, "tie")
        results.append({"idx": n, "question": row["question"], "answer_pt": row["answer_pt"],
                        "answer_sft": row["answer_sft"], "pt": pt_scores, "sft": sft_scores,
                        "prefer": prefer, "comment": (obj or {}).get("comment", "")})
        print(f"[{n}/{len(rows)}] pt均{sum(pt_scores.values()) / 3:.1f} sft均{sum(sft_scores.values()) / 3:.1f} → {prefer}")
        time.sleep(args.sleep)

    def model_mean(key):
        if not results:
            return 0.0
        return round(sum(sum(r[key].values()) for r in results) / (3 * len(results)), 2)

    n_pt = sum(1 for r in results if r["prefer"] == "pt")
    n_sft = sum(1 for r in results if r["prefer"] == "sft")
    n_tie = sum(1 for r in results if r["prefer"] == "tie")
    lines = [
        f"# PT vs SFT 留出题对比报告（裁判 {args.model}，{time.strftime('%Y-%m-%d %H:%M')}）", "",
        f"题数: {len(results)}；SFT 胜 {n_sft} / PT 胜 {n_pt} / 平 {n_tie}", "",
        f"PT  总均分: {model_mean('pt')}（术语/忠实/完整均值见下）",
        f"SFT 总均分: {model_mean('sft')}", "",
        "| # | PT 均 | SFT 均 | prefer |", "|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['idx']} | {sum(r['pt'].values()) / 3:.1f} | {sum(r['sft'].values()) / 3:.1f} | {r['prefer']} |")
    lines += ["", "## 逐题明细", ""]
    for r in results:
        lines += [
            f"### [题 {r['idx']}] prefer={r['prefer']}",
            f"**问题**：{r['question']}", f"**PT 答**：{r['answer_pt']}", f"**SFT 答**：{r['answer_sft']}",
            f"裁判：{r['comment']}", "",
        ]
    with open(args.compare_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"对比报告: {args.compare_report}（SFT {n_sft} 胜 / PT {n_pt} 胜 / 平 {n_tie}）")


def main():
    ap = argparse.ArgumentParser(description="GLM 裁判：QA 过筛三档分流 + PT/SFT 留出题对比评分")
    ap.add_argument("--mode", choices=["judge", "init-compare", "compare"], default="judge",
                    help="judge=QA 过筛(默认)；init-compare=生成对比骨架；compare=对比评分")
    ap.add_argument("--manifest", default=os.path.join(DATA_DIR, "domain_env_qa_manifest.jsonl"),
                    help="generate_domain_qa.py 的 manifest(事实源，内嵌原文)")
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "domain_env_qa_judged.jsonl"), help="逐条评分明细输出")
    ap.add_argument("--sft-out", default=os.path.join(DATA_DIR, "domain_env_qa_sft.jsonl"), help="过筛后 SFT 训练集输出")
    ap.add_argument("--report", default=os.path.join(DATA_DIR, "domain_env_qa_review.md"), help="裁判报告(人工抽检文档)")
    ap.add_argument("--compare", default=os.path.join(DATA_DIR, "domain_env_qa_compare.jsonl"),
                    help="对比骨架文件(init-compare 写 / compare 读)")
    ap.add_argument("--compare-report", default=os.path.join(DATA_DIR, "domain_env_qa_compare_report.md"), help="对比报告输出")
    ap.add_argument("--split", choices=["train", "eval", "all"], default="all", help="judge 模式评哪个 split(默认 all)")
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 条(冒烟用；0=全部)")
    ap.add_argument("--threshold", type=float, default=4.0, help="pass 的均分线(默认 4.0)")
    ap.add_argument("--promote", default="", help='人工复核改判: --promote "12,27" 强制指定 block_idx 入 sft 集')
    ap.add_argument("--model", default="kimi-k3", help="裁判模型(默认 kimi-k3；模型名以 Moonshot 文档为准)")
    ap.add_argument("--base-url", default="https://api.moonshot.cn/v1", help="API base url")
    ap.add_argument("--api-key-env", default="MOONSHOT_API_KEY", help="从 .env 读哪个键名")
    ap.add_argument("--sleep", type=float, default=1.0, help="相邻 API 调用间隔秒(默认 1.0)")
    ap.add_argument("--retries", type=int, default=3, help="单条 API 失败重试次数")
    args = ap.parse_args()

    if args.mode == "init-compare":  # 纯本地，不需要钥匙
        mode_init_compare(args)
        return

    api_key = load_env(ENV_FILE).get(args.api_key_env, "")
    if not api_key:
        print(f"[exit] .env({ENV_FILE}) 里没有 {args.api_key_env}，请先配置后再跑")
        return
    try:
        from openai import OpenAI
    except ImportError:
        print('[exit] 缺少 openai 包：pip install "openai>=1.5.0"')
        return
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    if args.mode == "judge":
        mode_judge(client, args)
    else:
        mode_compare(client, args)


if __name__ == "__main__":
    main()
