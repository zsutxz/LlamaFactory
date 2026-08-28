# -*- coding: utf-8 -*-
"""SFT+DPO 评测编排：106 题三轮多数决 A/B（扩量 SFT 基线 vs 其 DPO 后继）。

回答 §13.7 开放问题：知识更到位的基线（净 +11）上，DPO 能否拿到 v4（净 -2）拿不到的增益。
与 run_eval_sft_ab.py 同流程（服务只起 2 次，同 adapter 连续回填三轮骨架副本）：
  1) 复制 domain_env_qa_compare.jsonl -> _dpo_r1/_r2/_r3 三份
  2) 起 api 服务(SFT adapter=pt_think_then_sft) -> ask_compare 依次回填三份的 answer_a -> 停服务
  3) 起 api 服务(dpo adapter=pt_think_sft_then_dpo) -> 依次回填 answer_b -> 停服务
  4) kimi --mode compare --pair sft-dpo 逐轮出报告（_dpo_report_r{r}.md）
  5) 三轮 prefer 多数决汇总 -> _dpo_report_majority.md

用法：python train_test/run_eval_sft_dpo_ab.py [--rounds 3]
前置：SFT+DPO 训练已完成；.env 有 MOONSHOT_API_KEY；GPU 空闲。
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request

REPO = r"E:\AI\LLaMA-Factory"
DATA = os.path.join(REPO, "train_test", "data")
LF_EXE = r"C:\Users\skype\.conda\envs\llama-factory\Scripts\llamafactory-cli.exe"
INFER_YAML = os.path.join(REPO, "train_test", "examples", "inference", "qwen3_5_9b_think_domain_chat.yaml")
ADAPTER_A = os.path.join(REPO, "train_test", "saves", "Qwen3.5-9B-domain-env", "lora", "pt_think_then_sft")
ADAPTER_B = os.path.join(REPO, "train_test", "saves", "Qwen3.5-9B-domain-env", "lora", "pt_think_sft_then_dpo")
SKEL = os.path.join(DATA, "domain_env_qa_compare.jsonl")
ASK = os.path.join(REPO, ".claude", "skills", "public-data-pipeline", "scripts", "ask_compare.py")
JUDGE = os.path.join(REPO, ".claude", "skills", "public-data-pipeline", "scripts", "judge_domain_qa.py")
BASE_URL = "http://127.0.0.1:8000/v1"
ENV = {**os.environ, "PYTHONUTF8": "1", "LF_ALLOW_TORCH29_CONV3D": "1",
       "HF_HOME": os.path.join(REPO, "hf_cache"), "HF_HUB_OFFLINE": "1"}


def wait_ready(timeout=420):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(BASE_URL + "/models", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def port_free(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(BASE_URL + "/models", timeout=3):
                time.sleep(5)  # 还能连上，继续等退出
        except Exception:
            return True
    return False


def serve(adapter, tag):
    log = open(os.path.join(REPO, "train_test", "logs", f"api_{tag}.log"), "w", encoding="utf-8")
    proc = subprocess.Popen([LF_EXE, "api", INFER_YAML, f"adapter_name_or_path={adapter}"],
                            cwd=REPO, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    print(f"[serve] {tag} adapter={os.path.basename(adapter)} pid={proc.pid}，等待就绪…", flush=True)
    if not wait_ready():
        raise RuntimeError(f"api 服务 {tag} 420s 未就绪，看 train_test/logs/api_{tag}.log")
    print(f"[serve] {tag} 就绪", flush=True)
    return proc


def stop(proc, tag):
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    if port_free():
        print(f"[stop] {tag} 已停", flush=True)
    else:
        raise RuntimeError(f"{tag} 端口 8000 未释放")


def run(py, *extra):
    r = subprocess.run([sys.executable, py, *extra], cwd=REPO, env=ENV)
    if r.returncode != 0:
        raise RuntimeError(f"{py} 退出码 {r.returncode}")


def majority(rounds):
    """rounds: [{idx: prefer}, ...] -> 每 idx 多数决；三方各不同记 tie。"""
    out = {}
    for idx in rounds[0]:
        votes = [r.get(idx, "tie") for r in rounds]
        for v in ("sft", "sft_dpo", "tie"):
            if votes.count(v) >= 2:
                out[idx] = v
                break
        else:
            out[idx] = "tie"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    copies = [os.path.join(DATA, f"domain_env_qa_compare_dpo_r{r}.jsonl") for r in range(1, args.rounds + 1)]
    for c in copies:
        shutil.copyfile(SKEL, c)

    proc = serve(ADAPTER_A, "sft")
    try:
        for c in copies:
            run(ASK, "--compare", c, "--field", "answer_a")
    finally:
        stop(proc, "sft")

    proc = serve(ADAPTER_B, "sft_dpo")
    try:
        for c in copies:
            run(ASK, "--compare", c, "--field", "answer_b")
    finally:
        stop(proc, "sft_dpo")

    parsed = []
    for r, c in enumerate(copies, 1):
        report = os.path.join(DATA, f"domain_env_qa_compare_dpo_report_r{r}.md")
        run(JUDGE, "--mode", "compare", "--pair", "sft-dpo", "--compare", c,
            "--compare-report", report, "--sleep", "0.3")
        prefs = {}
        for line in open(report, encoding="utf-8"):
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) == 4 and parts[0].isdigit():
                prefs[int(parts[0])] = parts[3]
        parsed.append(prefs)
        print(f"[round {r}] 解析 {len(prefs)} 题 prefer", flush=True)

    mo = majority(parsed)
    n_b = sum(1 for v in mo.values() if v == "sft_dpo")
    n_a = sum(1 for v in mo.values() if v == "sft")
    n_t = sum(1 for v in mo.values() if v == "tie")
    lines = [
        "# SFT vs SFT+DPO 三轮多数决（106 题骨架）", "",
        f"多数决：SFT+DPO 胜 {n_b} / SFT 胜 {n_a} / 平 {n_t}（净 {n_b - n_a:+d}）", "",
        f"口径：106 题骨架 = 旧 36 剔 5 污染 + 75 新留出过筛；对照 = §13.7 开放问题（v4 基线上 DPO 净 -2）。",
        f"轮次报告：domain_env_qa_compare_dpo_report_r1~r{args.rounds}.md", "",
        "| # | r1 | r2 | r3 | 多数决 |", "|---|---|---|---|---|",
    ]
    for idx in sorted(mo):
        cells = [parsed[r].get(idx, "-") for r in range(args.rounds)] if args.rounds == 3 else ["-"] * 3
        lines.append(f"| {idx} | {cells[0]} | {cells[1]} | {cells[2]} | {mo[idx]} |")
    with open(os.path.join(DATA, "domain_env_qa_compare_dpo_report_majority.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[done] 多数决 SFT+DPO {n_b} / SFT {n_a} / 平 {n_t}（净 {n_b - n_a:+d}）-> dpo_report_majority.md", flush=True)


if __name__ == "__main__":
    main()
