# -*- coding: utf-8 -*-
"""从过筛 QA 生成 DPO/RM 偏好对（蒸馏闭环第四步：定向劣化造 rejected）。

读 judge_domain_qa.py 产出的过筛 SFT 集（alpaca 格式 {"instruction","input","output"}，
output 即已过 Kimi 三维评分的 chosen），对每条调 DeepSeek 按指定缺陷把答案劣化成
"看似合理但有具体错误" 的 rejected，落盘 alpaca ranking 格式：
  {"instruction": ..., "input": "", "chosen": ..., "rejected": ...}
供 stage: dpo 直接训练，或 stage: rm 先训奖励模型（PPO 前置）。

缺陷类型按行序确定性轮转（路由交给代码，不交给模型）：
  数值篡改 → 主体张冠李戴 → 关键截断 → 模糊化

manifest 是唯一事实源（内嵌 QA 全文与 rejected 全文），输出 jsonl 每次由 manifest
全量重写——天然幂等，中断重跑不重复扣费。

用法（conda env llama-factory，需 PYTHONUTF8=1）：

1) 冒烟（前 2 条；产物即正式产物的前缀，不作废）：
   python .claude/skills/public-data-pipeline/scripts/generate_domain_pref.py --limit 2
2) 正式（全量；已成功的 QA 按 sha1 自动跳过）：
   python .claude/skills/public-data-pipeline/scripts/generate_domain_pref.py

需项目根 .env 配置 DEEPSEEK_API_KEY（.env 已被 gitignore；勿写 .env.local——它被 git 追踪）。
需 openai>=1.5.0。约 1 元/百条（answer+rejected 各数百 token）。
"""

import argparse
import hashlib
import json
import os
import re
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
DATA_DIR = os.path.join(REPO, "data")
ENV_FILE = os.path.join(REPO, ".env")

# 缺陷类型：名称 → 给模型的改写指令。默认轮转只取原 4 类（顺序不变，保持 v1 行为可复现）；
# 无中生有 仅在 --defects 显式点名时参与轮转（2026-08-24 DPO A/B 败局分析：
# 模型在没把握处"编造更具体的数值/主体"且更自信——需要 rejected=编造细节、chosen=忠于原文的反向信号）。
DEFECTS = [
    ("数值篡改", "把答案中的关键数值（浓度限值/罚款倍数/百分比/年限等）改成错误但数量级合理的值；其余内容保持原样。"),
    ("主体张冠李戴", "把答案中的责任主体、监管部门或适用对象调换成同领域里另一个看似合理的主体；其余内容保持原样。"),
    ("关键截断", "保留答案的前半段和整体语气，但删去最关键的裁定结论或数值细节，让答案方向对却无法落地执行。"),
    ("模糊化", "把答案里所有具体数值和明确条款替换成『有关部门依法处理』『按规定执行』式的含混表述，长度相近。"),
    ("无中生有", "在保持答案原有内容的基础上，额外编造原文中不存在的具体细节（如新增罚款数额、审批部门、时限、条款号），使答案看起来更详尽更自信；原有内容保持原样。"),
]

SYSTEM_PROMPT = (
    "你是对抗样本工程师，负责为偏好优化(DPO)构造高质量负样本："
    "把正确答案改写成有指定缺陷、但表面上专业流畅的版本。"
)

USER_TEMPLATE = """【问题】
{question}

【正确答案】
{answer}

======
【任务】
基于上面的正确答案写一个劣化版：{defect_desc}
硬性要求：
1. 只植入这一种缺陷，其余表述尽量沿用原文（包括「」引用的风格），不得改问题；
2. 劣化版必须仍然流畅、自信、格式与原答案一致——禁止出现"抱歉/我不确定"等露怯痕迹；
3. 长度与原答案相差不超过三分之一。

【输出格式】
只输出一个 json 对象，不要 markdown 代码块，不要任何多余文字：
{{"rejected": "劣化后的完整答案"}}"""

SAME_AS_CHOSEN_HINT = (
    "\n\n【重试提示】你上一次给出的 rejected 与原答案几乎相同、为空或解析失败，"
    "请严格按缺陷要求大刀阔斧地改写（数值篡改须改动具体数字；其余缺陷须重写相应表述）"
    "后重新输出完整的 json 对象。"
)


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


def load_qa(src):
    """读过筛 QA jsonl（alpaca 行），返回 (instruction, output) 列表。"""
    rows = []
    for line in open(src, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if isinstance(rec.get("instruction"), str) and isinstance(rec.get("output"), str) and rec["output"].strip():
            rows.append((rec["instruction"], rec["output"]))
    return rows


def norm_text(s):
    """去全部空白后比对（中文语料空白无语义）。"""
    return re.sub(r"\s+", "", s)


def parse_rejected_json(raw):
    """宽松解析模型输出：剥代码块围栏、截取首{尾}、校验 rejected 字段。失败返回 None。"""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"\A```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\Z", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except Exception:
        return None
    if not isinstance(obj.get("rejected"), str) or not obj["rejected"].strip():
        return None
    return obj["rejected"]


def call_llm(client, model, user_content, retries):
    """调一次 API，带指数退避重试。返回 (回复文本, usage) 或 (None, None)。"""
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.7,
                max_tokens=1024,
                response_format={"type": "json_object"},  # DeepSeek JSON mode(要求 prompt 含 "json"，模板已满足)
            )
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            if resp.usage:
                usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
            return resp.choices[0].message.content, usage
        except Exception as exc:
            if attempt >= retries:
                print(f"[skip] API 调用失败(重试 {retries} 次后): {exc}")
                return None, None
            wait = 2 ** (attempt + 1)  # 2/4/8 秒
            print(f"[retry] 第 {attempt + 1} 次失败({exc})，{wait}s 后重试")
            time.sleep(wait)
    return None, None


def load_manifest(path):
    """载入已有 manifest，返回 (记录列表, 已成功 QA sha1 集合)。"""
    records, done = [], set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
            if rec.get("status") == "ok":
                done.add(rec["qa_sha1"])
    return records, done


def valid_rejected(rejected, chosen, defect_name=None):
    """机械校验劣化是否落地：数字集合变了(最小对比对) 或 改写幅度够大。
    注意：只确认"确有具体错误"，不强制错误类型与缺陷指令一致——模型常偷懒改数字，
    但数字错误本身即合格负样本。defect 字段记录的是"出题指令"而非实测错误类型。"""
    if not rejected or len(rejected) < 30:
        return False
    if norm_text(rejected) == norm_text(chosen):
        return False  # 原样照抄=没劣化
    if re.findall(r"\d+(?:\.\d+)?", chosen) != re.findall(r"\d+(?:\.\d+)?", rejected):
        return True  # 数值变了(含最小对比对)，机械可证的错误
    import difflib
    ratio = difflib.SequenceMatcher(None, norm_text(chosen), norm_text(rejected)).ratio()
    return ratio <= 0.97  # 无数字时，改写幅度须挡住"原样复述/仅换说法"


def rebuild_outputs(records, out_train, out_stats, totals):
    """从 manifest 全量重写输出（manifest 是唯一事实源）。"""
    rows = []
    for rec in records:
        if rec.get("status") != "ok":
            continue
        rows.append({"instruction": rec["question"], "input": "",
                     "chosen": rec["chosen"], "rejected": rec["rejected"]})
    with open(out_train, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_ok = sum(1 for r in records if r.get("status") == "ok")
    n_api = sum(1 for r in records if r.get("status") == "api_fail")
    n_bad = len(records) - n_ok - n_api
    by_defect = {}
    for r in records:
        if r.get("status") == "ok":
            by_defect[r["defect"]] = by_defect.get(r["defect"], 0) + 1
    stats = (
        f"manifest 记录数: {len(records)}  (ok {n_ok} / api_fail {n_api} / 劣化不合格 {n_bad})\n"
        f"缺陷分布: {json.dumps(by_defect, ensure_ascii=False)}\n"
        f"偏好对总数: {len(rows)}\n"
        f"本次调用消耗: prompt_tokens {totals['prompt_tokens']:,} + completion_tokens {totals['completion_tokens']:,}\n"
    )
    with open(out_stats, "w", encoding="utf-8") as f:
        f.write(stats)
    print(stats)


def main():
    ap = argparse.ArgumentParser(description="从过筛 QA 定向劣化生成 DPO/RM 偏好对")
    ap.add_argument("--qa", default=os.path.join(DATA_DIR, "domain_env_qa_sft.jsonl"),
                    help='过筛 QA jsonl(alpaca 行；默认 data/domain_env_qa_sft.jsonl)')
    ap.add_argument("--out-prefix", default="domain_env_pref", help="输出文件名前缀(默认 domain_env_pref)")
    ap.add_argument("--data-dir", default=DATA_DIR, help=f"输出目录(默认 {DATA_DIR})")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条(冒烟用，0=全量)")
    ap.add_argument("--model", default="deepseek-chat", help="劣化模型(默认 deepseek-chat)")
    ap.add_argument("--base-url", default="https://api.deepseek.com", help="API base url")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY", help="从 .env 读哪个键名")
    ap.add_argument("--sleep", type=float, default=1.0, help="相邻 API 调用间隔秒(默认 1.0)")
    ap.add_argument("--retries", type=int, default=3, help="单条 API 失败重试次数(指数退避 2/4/8s)")
    ap.add_argument("--defects", default="数值篡改,主体张冠李戴,关键截断,模糊化",
                    help="逗号分隔的缺陷轮转清单(默认原 4 类全轮转=历史行为；定向批次可只选子集或点名无中生有)")
    ap.add_argument("--force", action="store_true", help="丢弃已有 manifest 重新生成")
    args = ap.parse_args()

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

    out_train = os.path.join(args.data_dir, f"{args.out_prefix}.jsonl")
    out_stats = os.path.join(args.data_dir, f"{args.out_prefix}_stats.txt")
    out_manifest = os.path.join(args.data_dir, f"{args.out_prefix}_manifest.jsonl")

    qa_rows = load_qa(args.qa)
    if args.limit > 0:
        qa_rows = qa_rows[: args.limit]
    wanted = [d.strip() for d in args.defects.split(",") if d.strip()]
    defect_pool = [d for d in DEFECTS if d[0] in wanted]
    missing = [name for name in wanted if name not in {d[0] for d in DEFECTS}]
    if missing:
        print(f"[exit] --defects 含未知缺陷名: {missing}；可用: {[d[0] for d in DEFECTS]}")
        return
    if not defect_pool:
        print("[exit] --defects 过滤后为空")
        return
    print(f"读入 QA {len(qa_rows)} 条(来自 {args.qa})；缺陷轮转: {[d[0] for d in defect_pool]}")

    records, done = ([], set()) if args.force else load_manifest(out_manifest)
    if args.force and os.path.exists(out_manifest):
        os.remove(out_manifest)
        print("[force] 已丢弃旧 manifest")
    totals = {"prompt_tokens": 0, "completion_tokens": 0}
    manifest_f = open(out_manifest, "a", encoding="utf-8")

    for n, (question, answer) in enumerate(qa_rows, 1):
        digest = hashlib.sha1(question.encode("utf-8")).hexdigest()
        if digest in done:
            print(f"[{n}/{len(qa_rows)}] 已生成，跳过")
            continue

        defect_name, defect_desc = defect_pool[(n - 1) % len(defect_pool)]  # 行序轮转，确定性
        user_content = USER_TEMPLATE.format(question=question, answer=answer, defect_desc=defect_desc)
        raw, usage = call_llm(client, args.model, user_content, args.retries)
        time.sleep(args.sleep)
        rejected, status = None, "api_fail"

        def _parse_and_check(text):
            rej = parse_rejected_json(text or "")
            return rej if (rej is not None and valid_rejected(rej, answer, defect_name)) else None

        if raw is not None:
            rejected = _parse_and_check(raw)
            if rejected is None:
                # 解析失败或劣化未落地：追加提示重试一次
                raw2, usage2 = call_llm(client, args.model, user_content + SAME_AS_CHOSEN_HINT, args.retries)
                time.sleep(args.sleep)
                if usage2:
                    usage["prompt_tokens"] += usage2["prompt_tokens"]
                    usage["completion_tokens"] += usage2["completion_tokens"]
                rejected = _parse_and_check(raw2)
                raw = raw2 or raw
            status = "ok" if rejected is not None else "corrupt_fail"

        totals["prompt_tokens"] += usage.get("prompt_tokens", 0)
        totals["completion_tokens"] += usage.get("completion_tokens", 0)
        rec = {"qa_sha1": digest, "defect": defect_name, "status": status,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "usage": usage,
               "question": question, "chosen": answer}
        if status == "ok":
            rec["rejected"] = rejected
            print(f"[{n}/{len(qa_rows)}] ok({defect_name})")
        else:
            rec["raw"] = (raw or "")[:800]  # 留证据便于诊断
            print(f"[skip] {status}(第 {n} 条, {defect_name})")
        manifest_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        manifest_f.flush()

    manifest_f.close()
    print("生成完毕，重建输出…")
    final_records, _ = load_manifest(out_manifest)
    rebuild_outputs(final_records, out_train, out_stats, totals)
    print("偏好对集:", out_train)


if __name__ == "__main__":
    main()
