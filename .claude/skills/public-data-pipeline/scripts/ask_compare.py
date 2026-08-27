# -*- coding: utf-8 -*-
"""自动问答回填：把对比骨架里的 10 题逐条问本地 LLaMA-Factory api 服务，答案写回 jsonl。

配合留出题对比评测（judge_domain_qa.py --mode compare）使用：
1) 先起 api 服务（例：llamafactory-cli api <infer yaml> adapter_name_or_path=saves/.../pt）
2) python .claude/skills/public-data-pipeline/scripts/ask_compare.py --field answer_pt            # 回填 PT 答案
3) 换 pt_then_sft adapter 重启服务，再 --field answer_sft           # 回填 SFT 答案
think 链 SFT vs DPO（骨架先 judge_domain_qa.py --mode init-compare --pair sft-dpo 生成）：
  换 --compare data/domain_env_think_qa_compare.jsonl，--field answer_sft / answer_dpo 各跑一轮。
只回填目标字段为空的行，已填的跳过（幂等，中断重跑不重复问）。
"""

import argparse
import json
import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
DATA_DIR = os.path.join(REPO, "data")


def main():
    ap = argparse.ArgumentParser(description="把对比骨架问题逐条发给本地 api 服务并回填答案")
    ap.add_argument("--compare", default=os.path.join(DATA_DIR, "domain_env_qa_compare.jsonl"), help="对比骨架文件")
    ap.add_argument("--field", choices=["answer_pt", "answer_sft", "answer_dpo", "answer_a", "answer_b"], required=True, help="本轮回填哪个字段")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="本地 api 服务地址")
    ap.add_argument("--api-key", default="0", help="API_KEY(本地服务未设鉴权时随便填)")
    ap.add_argument("--temperature", type=float, default=0.3, help="生成温度(两个 adapter 保持一致才公平)")
    ap.add_argument("--max-tokens", type=int, default=512, help="单题生成上限")
    args = ap.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        print('[exit] 缺少 openai 包：pip install "openai>=1.5.0"')
        return
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # 服务端模型名取 /v1/models 第一个（默认 = model_name_or_path）
    model = client.models.list().data[0].id
    print(f"服务模型: {model}")

    rows = [json.loads(l) for l in open(args.compare, encoding="utf-8") if l.strip()]
    asked = 0
    for n, row in enumerate(rows, 1):
        if row.get(args.field, "").strip():
            print(f"[{n}/{len(rows)}] 已填，跳过")
            continue
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": row["question"]}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            row[args.field] = (resp.choices[0].message.content or "").strip()
            print(f"[{n}/{len(rows)}] {row['question'][:30]}… -> {len(row[args.field])} 字")
        except Exception as exc:
            print(f"[skip] 第 {n} 题调用失败: {exc}")
            continue  # 该行保持空，compare 模式会跳过未填全的行
        asked += 1

    with open(args.compare, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    filled = sum(1 for r in rows if r.get(args.field, "").strip())
    print(f"回填 {asked} 题；{args.field} 现有 {filled}/{len(rows)} 题有答案")


if __name__ == "__main__":
    main()
