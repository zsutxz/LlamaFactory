# -*- coding: utf-8 -*-
"""用 DeepSeek 从领域语料生成蒸馏 QA 对（蒸馏闭环第一步）。

从 data/domain_env.jsonl（{"text": ...} 每行的 PT 语料块）等距抽块，调 DeepSeek
生成锚定原文的问答对，机械校验 quote 确实存在于原文后，写入 alpaca 格式训练集，
供 stage: sft 在 PT adapter 上续训。manifest 是唯一事实源（内嵌原文与 QA 全文），
输出 jsonl 每次由 manifest 全量重写——天然幂等，中断重跑不重复扣费。

用法（conda env llama-factory，需 PYTHONUTF8=1）：

1) 冒烟（3 块；固定 60 级梯子使冒烟块必然是正式块的子集，产物不作废）：
   python .claude/skills/public-data-pipeline/scripts/generate_domain_qa.py --num 3 --eval-num 0
2) 正式（50 训练 + 10 留出评测；已生成的块按 sha1 自动跳过）：
   python .claude/skills/public-data-pipeline/scripts/generate_domain_qa.py

需项目根 .env 配置 DEEPSEEK_API_KEY（.env 已被 gitignore；勿写 .env.local——它被 git 追踪）。
需 openai>=1.5.0。60 次调用约 72K in + 17K out（deepseek-chat 不足 0.5 元）。
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

LADDER_SIZE = 60  # 固定 60 级采样梯子：冒烟(3) 是正式(50+10) 的严格前缀

SYSTEM_PROMPT = (
    "你是环境保护领域的法规与标准专家，负责为模型蒸馏出题："
    "基于给定的语料片段出一道高质量的中文问答对。"
)

USER_TEMPLATE = """【语料片段】
{block}

【出题要求】
1. 问题必须只能依靠上面的语料片段回答，禁止引入外部知识或假设；问题自身不得出现"根据片段/根据上文/材料中提到"等字样（样本要独立成题）。
2. 问题要具体、有专业价值，优先考：条款适用情形、污染物限值与适用范围、责任主体、罚则幅度、术语定义、时间节点；禁止"这段话讲了什么"式概述题。
3. 若片段含数值标准（浓度限值/排放标准/处罚倍数等），优先考"某污染物的限值或适用范围"。
4. 答案 100~250 字：先直接作答，再用「」引用原文关键句佐证，不得整段照抄。

【输出格式】
只输出一个 json 对象，不要 markdown 代码块，不要任何多余文字：
{{"question": "...", "answer": "...", "quote": "答案所引用的原文原句"}}"""

GROUNDING_RETRY_HINT = (
    "\n\n【重试提示】你上一次给出的 quote 在原文中找不到（比对已忽略空白差异），"
    "请严格从原文逐字摘取 quote 后重新输出完整的 json 对象。"
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


def load_blocks(src):
    """读语料 jsonl，返回 text 块列表。"""
    blocks = []
    for line in open(src, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        blocks.append(rec["text"])
    return blocks


def build_ladder(candidates, size):
    """等距取样：candidates 的行序已被上游 seed(42) 洗牌，等距 = 随机 + 均匀覆盖 + 确定。"""
    if len(candidates) <= size:
        return list(candidates)
    return [candidates[round(k * (len(candidates) - 1) / (size - 1))] for k in range(size)]


def norm_text(s):
    """去全部空白后比对（中文语料空白无语义，模型复述时常有微差）。"""
    return re.sub(r"\s+", "", s)


def parse_qa_json(raw):
    """宽松解析模型输出：剥代码块围栏、截取首{尾}、校验三字段。失败返回 None。"""
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
    if not all(isinstance(obj.get(k), str) and obj.get(k).strip() for k in ("question", "answer", "quote")):
        return None
    if not 50 <= len(obj["answer"]) <= 500:
        return None
    return obj


def call_llm(client, model, user_content, retries, sleep_s):
    """调一次 API，带指数退避重试。返回 (回复文本, usage) 或 (None, None)。"""
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
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
    """载入已有 manifest，返回 (记录列表, 已成功块 sha1 集合)。"""
    records, done = [], set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
            if rec.get("status") == "ok":
                done.add(rec["block_sha1"])
    return records, done


def rebuild_outputs(records, out_train, out_eval, out_stats, totals):
    """从 manifest 全量重写三件套（manifest 是唯一事实源）。"""
    train_rows, eval_rows = [], []
    for rec in records:
        if rec.get("status") != "ok":
            continue
        row = {"instruction": rec["question"], "input": "", "output": rec["answer"]}
        (train_rows if rec.get("split") == "train" else eval_rows).append(row)

    def write(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write(out_train, train_rows)
    write(out_eval, eval_rows)

    n_ok = sum(1 for r in records if r.get("status") == "ok")
    n_fail = sum(1 for r in records if r.get("status") == "grounding_fail")
    n_api = len(records) - n_ok - n_fail
    stats = (
        f"manifest 记录数: {len(records)}  (ok {n_ok} / grounding_fail {n_fail} / api_fail {n_api})\n"
        f"训练 QA 数: {len(train_rows)}  留出评测 QA 数: {len(eval_rows)}\n"
        f"本次调用消耗: prompt_tokens {totals['prompt_tokens']:,} + completion_tokens {totals['completion_tokens']:,}\n"
    )
    with open(out_stats, "w", encoding="utf-8") as f:
        f.write(stats)
    print(stats)


def main():
    ap = argparse.ArgumentParser(description="用 DeepSeek 从领域语料生成蒸馏 QA 对(锚定原文+机械校验)")
    ap.add_argument("--src", default=os.path.join(DATA_DIR, "domain_env.jsonl"),
                    help='源语料 jsonl(每行 {"text": ...}；默认 data/domain_env.jsonl)')
    ap.add_argument("--out-prefix", default="domain_env_qa", help="输出文件名前缀(默认 domain_env_qa)")
    ap.add_argument("--data-dir", default=DATA_DIR, help=f"输出目录(默认 {DATA_DIR})")
    ap.add_argument("--num", type=int, default=50, help="训练 QA 对数(默认 50)")
    ap.add_argument("--eval-num", type=int, default=10, help="留出评测 QA 对数，与训练块完全不相交(默认 10)")
    ap.add_argument("--model", default="deepseek-chat", help="生成模型(默认 deepseek-chat)")
    ap.add_argument("--base-url", default="https://api.deepseek.com", help="API base url")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY", help="从 .env 读哪个键名")
    ap.add_argument("--sleep", type=float, default=1.0, help="相邻 API 调用间隔秒(默认 1.0)")
    ap.add_argument("--retries", type=int, default=3, help="单块 API 失败重试次数(指数退避 2/4/8s)")
    ap.add_argument("--min-block-chars", type=int, default=400, help="过短块不入选(默认 400 字符)")
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

    out_dir = args.data_dir
    out_train = os.path.join(out_dir, f"{args.out_prefix}.jsonl")
    out_eval = os.path.join(out_dir, f"{args.out_prefix}_eval.jsonl")
    out_stats = os.path.join(out_dir, f"{args.out_prefix}_stats.txt")
    out_manifest = os.path.join(out_dir, f"{args.out_prefix}_manifest.jsonl")

    blocks = load_blocks(args.src)
    candidates = [i for i, b in enumerate(blocks) if len(b) >= args.min_block_chars]
    ladder_size = max(LADDER_SIZE, args.num + args.eval_num)
    ladder = build_ladder(candidates, ladder_size)
    train_idx = ladder[: args.num]
    eval_idx = ladder[len(ladder) - args.eval_num :] if args.eval_num > 0 else []
    print(f"语料 {len(blocks)} 块，合格候选 {len(candidates)} 块，梯子 {len(ladder)} 级")
    print(f"目标: train {len(train_idx)} 块 + eval {len(eval_idx)} 块")

    records, done = ([], set()) if args.force else load_manifest(out_manifest)
    if args.force and os.path.exists(out_manifest):
        os.remove(out_manifest)
        print("[force] 已丢弃旧 manifest")
    totals = {"prompt_tokens": 0, "completion_tokens": 0}
    manifest_f = open(out_manifest, "a", encoding="utf-8")

    targets = [(i, "train") for i in train_idx] + [(i, "eval") for i in eval_idx]
    for n, (idx, split) in enumerate(targets, 1):
        block = blocks[idx]
        digest = hashlib.sha1(block.encode("utf-8")).hexdigest()
        if digest in done:
            print(f"[{n}/{len(targets)}] block_idx={idx} 已生成，跳过")
            continue

        user_content = USER_TEMPLATE.format(block=block)
        raw, usage = call_llm(client, args.model, user_content, args.retries, args.sleep)
        time.sleep(args.sleep)
        if raw is None:
            manifest_f.write(json.dumps({"block_idx": idx, "block_sha1": digest, "split": split,
                                         "status": "api_fail", "ts": time.strftime("%Y-%m-%dT%H:%M:%S")},
                                        ensure_ascii=False) + "\n")
            manifest_f.flush()
            continue

        qa = parse_qa_json(raw)
        grounded = qa is not None and norm_text(qa["quote"]) in norm_text(block)
        if qa is not None and not grounded:
            # 引用不在原文：把失败原因追加进 prompt 重试一次
            raw2, usage2 = call_llm(client, args.model, user_content + GROUNDING_RETRY_HINT, args.retries, args.sleep)
            time.sleep(args.sleep)
            if usage2:
                usage["prompt_tokens"] += usage2["prompt_tokens"]
                usage["completion_tokens"] += usage2["completion_tokens"]
            qa2 = parse_qa_json(raw2 or "")
            if qa2 is not None and norm_text(qa2["quote"]) in norm_text(block):
                qa, grounded = qa2, True

        totals["prompt_tokens"] += usage.get("prompt_tokens", 0)
        totals["completion_tokens"] += usage.get("completion_tokens", 0)
        rec = {"block_idx": idx, "block_sha1": digest, "split": split,
               "status": "ok" if (qa and grounded) else "grounding_fail",
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "usage": usage, "block": block}
        if qa and grounded:
            rec.update({"question": qa["question"], "answer": qa["answer"], "quote": qa["quote"]})
            print(f"[{n}/{len(targets)}] block_idx={idx}({split}) ok")
        else:
            print(f"[skip] 引用不在原文或解析失败 block_idx={idx}({split})")
        manifest_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        manifest_f.flush()

    manifest_f.close()
    print("生成完毕，重建输出三件套…")
    final_records, _ = load_manifest(out_manifest)
    rebuild_outputs(final_records, out_train, out_eval, out_stats, totals)
    print("训练集:", out_train)
    print("留出评测集:", out_eval)


if __name__ == "__main__":
    main()
