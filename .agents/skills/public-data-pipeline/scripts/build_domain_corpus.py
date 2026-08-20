# -*- coding: utf-8 -*-
"""Build a continued-pretraining corpus from local professional docs & papers.

两种用法：

1) 默认模式(原 AI-agents 领域，硬编码源，无参即等价旧行为)：
   python .claude/skills/public-data-pipeline/scripts/build_domain_corpus.py
   源: E:\\AI\\Book\\*.pdf, E:\\AI\\5-Day-AI-Agents-Intensive-Course-with-Google-2025\\*.pdf,
       E:\\AI\\teach-fish-to-swim\\** 的 raw-content.md / index.html
   出: data/domain_papers.jsonl / _eval.jsonl / _stats.txt

2) 目录扫描模式(供 public-data-pipeline skill 采集的 data_raw 用)：
   python .claude/skills/public-data-pipeline/scripts/build_domain_corpus.py --src data_raw --out-prefix domain_env
   源: 递归扫描 --src 下的 *.md / *.html / *.pdf (跳过 _ 前缀元数据文件)
   出: data/<prefix>.jsonl / _eval.jsonl / _stats.txt

需 PyMuPDF(解析 PDF) + BeautifulSoup(解析 HTML)；缺 PDF 库时自动跳过 PDF。
"""

import argparse
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
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), "data")
OUT_TRAIN = os.path.join(OUT_DIR, "domain_papers.jsonl")
OUT_EVAL = os.path.join(OUT_DIR, "domain_papers_eval.jsonl")
OUT_STATS = os.path.join(OUT_DIR, "domain_papers_stats.txt")

CHUNK_CHARS = 1800  # 每块目标字符数(约 1000~1500 token)
MIN_CHUNK_CHARS = 150
EVAL_EVERY = 10  # 每 10 块留 1 块做验证
TEXT_EXT = {".pdf", ".html", ".htm", ".md", ".markdown", ".txt", ".xml"}


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


def md_text(path):
    text = open(path, encoding="utf-8", errors="ignore").read()
    text = re.sub(r"\A---\n.*?\n---\n", "", text, count=1, flags=re.S)  # 去 YAML frontmatter
    return text


def read_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return pdf_text(path)
    if ext in (".html", ".htm"):
        return html_text(path)
    if ext in (".md", ".markdown", ".txt", ".xml"):
        return md_text(path)
    return None


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
    """默认模式：从硬编码的专业文档/论文目录收集(原 AI-agents 领域)。"""
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


def collect_from_tree(root):
    """目录扫描模式：递归收集 root 下的 md/html/pdf（跳过 _ 前缀元数据文件）。"""
    docs = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.startswith("_"):  # _catalog.md / _fetch_list.json / _manifest.jsonl
                continue
            if os.path.splitext(fn)[1].lower() not in TEXT_EXT:
                continue
            path = os.path.join(dirpath, fn)
            try:
                text = read_any(path)
                if text and text.strip():
                    docs.append((os.path.relpath(path, root), text))
                else:
                    print(f"[skip] 内容为空 {path}")
            except Exception as exc:
                print(f"[skip] 读取失败 {path}: {exc}")
    return docs


def main():
    ap = argparse.ArgumentParser(description="构建领域继续预训练语料(切块+去重+划分训练/验证)")
    ap.add_argument("--src", default=None,
                    help="源目录(递归扫描 md/html/pdf)；不给则用默认硬编码源(原 AI-agents 领域)")
    ap.add_argument("--out-prefix", default="domain_papers", help="输出文件名前缀(默认 domain_papers)")
    ap.add_argument("--data-dir", default=OUT_DIR, help=f"输出目录(默认 {OUT_DIR})")
    args = ap.parse_args()

    out_dir = args.data_dir
    out_train = os.path.join(out_dir, f"{args.out_prefix}.jsonl")
    out_eval = os.path.join(out_dir, f"{args.out_prefix}_eval.jsonl")
    out_stats = os.path.join(out_dir, f"{args.out_prefix}_stats.txt")

    os.makedirs(out_dir, exist_ok=True)
    docs = collect_from_tree(args.src) if args.src else collect_documents()
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

    write(out_train, train_chunks)
    write(out_eval, eval_chunks)

    total_chars = sum(len(c) for _, c in unique)
    stats = (
        f"来源文档数: {len(docs)}\n"
        f"切块数(去重前): {len(all_chunks)}\n"
        f"去重后块数: {len(unique)}\n"
        f"训练块数: {len(train_chunks)}  验证块数: {len(eval_chunks)}\n"
        f"总字符数: {total_chars:,}\n"
        f"平均每块字符: {total_chars // max(len(unique), 1)}\n"
    )
    with open(out_stats, "w", encoding="utf-8") as f:
        f.write(stats)
    print(stats)
    print("训练集:", out_train)
    print("验证集:", out_eval)


if __name__ == "__main__":
    main()
