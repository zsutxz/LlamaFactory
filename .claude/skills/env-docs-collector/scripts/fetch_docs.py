#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量下载环保资料到 data_raw/<category>/，并写幂等清单 _manifest.jsonl。

输入清单(JSON 数组)，每项:
  {"category": "laws|standards|papers", "title": "...", "url": "...", "kind": "pdf|xml|txt|md"}

下载后端: curl 优先(Windows 自动 --ssl-no-revoke 规避 schannel 吊销检查) → urllib 兜底。
幂等: 已成功(ok/skip)或永久失败(invalid)的 URL 跳过；网络 fail 下次重试。
校验: PDF 须以 %PDF 开头；xml/txt/md 拒收疑似 HTML 页面(避免把 403 错误页存成假文件)。

用法:
  python fetch_docs.py [--data-raw ./data_raw] [--list ./data_raw/_fetch_list.json]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

DEFAULT_DATA_RAW = os.path.join(os.getcwd(), "data_raw")
ALLOWED_CATEGORY = {"laws", "standards", "papers"}
ALLOWED_KIND = {"pdf", "xml", "txt", "md"}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
REQUEST_DELAY = 1.0  # 秒，限速
# 成功或永久失败 → 跳过；其余(fail=网络) → 重试
SKIP_STATUS = {"ok", "skip", "invalid"}


def safe_name(title, kind):
    """标题 → 安全文件名。"""
    name = re.sub(r"[\\/:*?\"<>|]+", "_", title or "").strip().strip(".")
    name = re.sub(r"\s+", "_", name) or "unnamed"
    return f"{name}.{kind}"


def load_skip_hashes(mani_path):
    """读已有 manifest，返回应跳过的 url_hash 集合(成功 ok/skip 或永久失败 invalid)。"""
    skip = set()
    if os.path.exists(mani_path):
        with open(mani_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("status") in SKIP_STATUS:
                        skip.add(rec.get("url_hash"))
                except json.JSONDecodeError:
                    pass
    return skip


def fetch_bytes(url):
    """curl 优先 → urllib 兜底，返回字节数。两者皆失败抛异常。"""
    if shutil.which("curl"):
        try:
            r = subprocess.run(
                ["curl", "--ssl-no-revoke", "-sS", "-L",
                 "-A", USER_AGENT, "--max-time", "60", url],
                capture_output=True, timeout=90,
            )
            if r.returncode == 0 and r.stdout:
                return r.stdout
        except Exception:
            pass  # 回退 urllib
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:  # 自动跟随重定向
        return resp.read()


def validate(data, kind):
    """校验内容真实性，失败抛 ValueError。"""
    if len(data) < 100:
        raise ValueError("内容过短(疑似错误页)")
    if kind == "pdf" and not data.startswith(b"%PDF"):
        raise ValueError("非 PDF(疑似 403/错误页)")
    if kind != "pdf" and re.match(rb"\s*<(?:!doctype\s+html|html[\s>])", data, re.I):
        raise ValueError(f"疑似 HTML 页面(非 {kind})")


def main():
    ap = argparse.ArgumentParser(description="批量下载环保资料到 data_raw")
    ap.add_argument("--data-raw", default=DEFAULT_DATA_RAW, help="data_raw 根目录")
    ap.add_argument("--list", default=None, help="下载清单 JSON (默认 data_raw/_fetch_list.json)")
    args = ap.parse_args()

    list_path = args.list or os.path.join(args.data_raw, "_fetch_list.json")
    mani_path = os.path.join(args.data_raw, "_manifest.jsonl")
    if not os.path.exists(list_path):
        sys.exit(f"[错误] 清单不存在: {list_path}\n请先按 SKILL.md 流程生成 _fetch_list.json。")

    with open(list_path, encoding="utf-8") as f:
        items = json.load(f)

    os.makedirs(args.data_raw, exist_ok=True)
    skip_hashes = load_skip_hashes(mani_path)
    processed = set()          # 本次运行内去重
    new_success = set()        # 本次新成功，防止同轮重复下载
    counts = {"ok": 0, "skip": 0, "fail": 0}

    with open(mani_path, "a", encoding="utf-8") as mani:
        for it in items:
            cat = it.get("category")
            title = it.get("title", "")
            url = it.get("url", "")
            kind = it.get("kind", "pdf")
            url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
            record = {"category": cat, "title": title, "url": url, "url_hash": url_hash}

            # 1) 本轮重复 / 历史已成功或永久失败 → 跳过(不写行)
            if url_hash in processed or url_hash in skip_hashes or url_hash in new_success:
                counts["skip"] += 1
                continue
            processed.add(url_hash)

            # 2) 非法 category/kind → 永久失败 invalid(下次自动跳过)
            if cat not in ALLOWED_CATEGORY:
                reason = f"非法 category: {cat}"
            elif kind not in ALLOWED_KIND:
                reason = f"非法 kind: {kind}(仅允许 pdf/xml/txt/md)"
            else:
                reason = None
            if reason:
                record.update(status="invalid", error=reason)
                mani.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["fail"] += 1
                print(f"[invalid] {title or url}: {reason}")
                continue

            # 3) 文件已存在且非空 → 登记为 skip
            dest_dir = os.path.join(args.data_raw, cat)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, safe_name(title, kind))
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                record.update(status="skip",
                              local_path=os.path.relpath(dest, args.data_raw),
                              bytes=os.path.getsize(dest))
                mani.write(json.dumps(record, ensure_ascii=False) + "\n")
                new_success.add(url_hash)
                counts["skip"] += 1
                print(f"[skip] {cat}/{os.path.basename(dest)}  已存在")
                continue

            # 4) 下载 + 校验
            try:
                data = fetch_bytes(url)
                validate(data, kind)
                with open(dest, "wb") as f:
                    f.write(data)
                record.update(status="ok",
                              local_path=os.path.relpath(dest, args.data_raw),
                              bytes=len(data))
                mani.write(json.dumps(record, ensure_ascii=False) + "\n")
                new_success.add(url_hash)
                counts["ok"] += 1
                print(f"[ok]   {cat}/{os.path.basename(dest)}  {len(data):,} bytes")
            except Exception as exc:
                record.update(status="fail", error=f"{type(exc).__name__}: {exc}")
                mani.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["fail"] += 1
                print(f"[fail] {title or url}: {exc}")
            time.sleep(REQUEST_DELAY)

    print(f"\n完成: 成功 {counts['ok']}  跳过 {counts['skip']}  失败 {counts['fail']}")
    print(f"清单: {mani_path}")


if __name__ == "__main__":
    main()
