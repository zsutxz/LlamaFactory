# -*- coding: utf-8 -*-
"""Build a continued-pretraining corpus from local professional docs & papers.

Sources:
  - E:\\AI\\Book\\*.pdf                        (专业文档/论文手册)
  - E:\\AI\\teach-fish-to-swim\\**            (英文论文全文: md/html)

Output (LLaMA-Factory PT 数据集):
  - data/domain_papers.jsonl       训练集
  - data/domain_papers_eval.jsonl  验证集(每 10 块留 1 块)

用法(用带 PyMuPDF 的解释器运行,在仓库根目录下):
  python scripts/data/build_domain_corpus.py
"""

import glob
import hashlib
import html
import json
import os
import random
import re

SRC_PDF_DIRS = [
    r"E:\AI\Book",  # 专业手册 + 论文合集
    r"E:\AI\5-Day-AI-Agents-Intensive-Course-with-Google-2025",  # 专业课程文档
]
SRC_PAPER_ROOT = r"E:\AI\teach-fish-to-swim"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
OUT_TRAIN = os.path.join(OUT_DIR, "domain_papers.jsonl")
OUT_EVAL = os.path.join(OUT_DIR, "domain_papers_eval.jsonl")
OUT_STATS = os.path.join(OUT_DIR, "domain_papers_stats.txt")

CHUNK_CHARS = 1800  # 每块目标字符数(约 1000~1500 token)
MIN_CHUNK_CHARS = 150
EVAL_EVERY = 10  # 每 10 块留 1 块做验证


def pdf_text(path):
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf
    doc = pymupdf.open(path)
    try:
        return "\n\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def html_text(path):
    raw = open(path, encoding="utf-8", errors="ignore").read()
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "svg", "button", "form"]):
            tag.decompose()
        return soup.get_text("\n")
    except Exception:
        raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
        raw = re.sub(r"(?s)<[^>]+>", " ", raw)
        return html.unescape(raw)


def clean(text):
    """去重、去噪、规整空白，返回干净的纯文本。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # 图片语法
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # 链接只留文字
    text = re.sub(r"```.*?```", " ", text, flags=re.S)  # 代码块(论文正文不适用)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)  # 标题符号
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)  # 列表符号
    text = re.sub(r"[ \t\u00a0]+", " ", text)

    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) < 2:
            continue
        if re.fullmatch(r"[\W_]{1,40}", line):  # 纯符号行(分页符/装饰线)
            continue
        lines.append(line)
    return "\n".join(lines)


def chunk_text(text, max_chars=CHUNK_CHARS):
    """按段落边界贪心打包成不超过 max_chars 的块。"""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, current = [], []
    cur_len = 0
    for para in paragraphs:
        if current and cur_len + len(para) > max_chars:
            chunks.append("\n".join(current))
            current, cur_len = [], 0
        current.append(para)
        cur_len += len(para) + 1
    if current:
        chunks.append("\n".join(current))
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def collect_documents():
    docs = []  # (来源, 文本)

    # 1) 专业文档/论文 PDF
    for src_dir in SRC_PDF_DIRS:
        for path in sorted(glob.glob(os.path.join(src_dir, "*.pdf"))):
            try:
                docs.append((os.path.basename(path), pdf_text(path)))
            except Exception as exc:
                print(f"[skip] PDF 解析失败 {path}: {exc}")

    # 2) teach-fish-to-swim 下的论文(md 优先, 否则 html)
    md_files = glob.glob(os.path.join(SRC_PAPER_ROOT, "**", "raw-content.md"), recursive=True)
    seen_md_dirs = {os.path.dirname(p) for p in md_files}
    for path in sorted(md_files):
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
            docs.append((os.path.relpath(path, SRC_PAPER_ROOT), text))
        except Exception as exc:
            print(f"[skip] md 读取失败 {path}: {exc}")

    for path in sorted(glob.glob(os.path.join(SRC_PAPER_ROOT, "**", "index.html"), recursive=True)):
        if os.path.dirname(path) in seen_md_dirs:  # 已有 md 就不再吃 html
            continue
        try:
            docs.append((os.path.relpath(path, SRC_PAPER_ROOT), html_text(path)))
        except Exception as exc:
            print(f"[skip] html 解析失败 {path}: {exc}")

    return docs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    docs = collect_documents()
    print(f"收集到 {len(docs)} 份文档")

    all_chunks = []
    for src, text in docs:
        text = clean(text)
        if len(text) < MIN_CHUNK_CHARS:
            print(f"[skip] 清洗后过短 {src}: {len(text)} chars")
            continue
        for chunk in chunk_text(text):
            all_chunks.append((src, chunk))
    print(f"切块后共 {len(all_chunks)} 块(去重前)")

    # 去重: 以块内容哈希去重
    seen, unique = set(), []
    for src, chunk in all_chunks:
        digest = hashlib.sha1(chunk.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        unique.append((src, chunk))
    print(f"去重后共 {len(unique)} 块")

    # 洗牌(固定种子)并划分训练/验证
    random.seed(42)
    random.shuffle(unique)
    eval_chunks, train_chunks = [], []
    for i, item in enumerate(unique):
        (eval_chunks if i % EVAL_EVERY == 0 else train_chunks).append(item)

    def write(out_path, items):
        with open(out_path, "w", encoding="utf-8") as f:
            for _, chunk in items:
                f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")

    write(OUT_TRAIN, train_chunks)
    write(OUT_EVAL, eval_chunks)

    total_chars = sum(len(c) for _, c in unique)
    stats = (
        f"来源文档数: {len(docs)}\n"
        f"切块数(去重前): {len(all_chunks)}\n"
        f"去重后块数: {len(unique)}\n"
        f"训练块数: {len(train_chunks)}  验证块数: {len(eval_chunks)}\n"
        f"总字符数: {total_chars:,}\n"
        f"平均每块字符: {total_chars // max(len(unique), 1)}\n"
    )
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        f.write(stats)
    print(stats)
    print("训练集:", OUT_TRAIN)
    print("验证集:", OUT_EVAL)


if __name__ == "__main__":
    main()
