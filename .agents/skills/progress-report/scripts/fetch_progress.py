#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从多个需登录的 Web 系统抓取项目/任务进度，汇总成一份进度报表。

每个数据源 = config/sources.json 里的一项:
  登录(GET 登录页取隐藏域 → POST 账号密码) → 取数据页 → 解析含 key_header 的 <table>
  → 按 columns 映射成任务行 → 归一化(progress/deadline)。

凭据: 从 .env 读环境变量(名由 credentials.*_env 指定)，绝不入库、绝不打印。
后端: Python 标准库 only(urllib + http.cookiejar + html.parser)，零新依赖。
失败处理: 单源失败不影响其它源，计入 _run.json 并在 report.md 单列(显性失败)。

能力边界: 只能处理标准 HTML 表单登录；JS/SSO/验证码登录会失败(报表标注，不造假数据)。

用法:
  python fetch_progress.py [--config config/sources.json] [--out progress_report] [--env .env] [--insecure]
"""
import argparse
import datetime
import html
import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
import http.cookiejar

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
DEFAULT_CONFIG = os.path.join(os.getcwd(), "config", "sources.json")
DEFAULT_OUT = os.path.join(os.getcwd(), "progress_report")
DEFAULT_ENV = os.path.join(os.getcwd(), ".env")
DEFAULT_PROGRESS_RE = r"(\d+)\s*%"
TASK_FIELDS = ("title", "owner", "status", "progress_raw", "deadline")
DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M", "%Y.%m.%d")


# ----------------------------- .env 加载 ---------------------------------
def load_env(path):
    """极简 .env 解析：KEY=VALUE 写入 os.environ(已存在的不覆盖)。"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k:
                os.environ.setdefault(k, v)


# ----------------------------- HTML 解析 ---------------------------------
class _FormParser(HTMLParser):
    """收集页面里的 <form>：action + 其下 <input> 的 name/value(仅 hidden/text 类)。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.forms = []
        self._cur = None

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag == "form":
            self._cur = {"action": d.get("action", ""), "method": (d.get("method") or "POST").upper(), "fields": {}}
        elif tag == "input" and self._cur is not None:
            name = d.get("name")
            if not name:
                return
            itype = (d.get("type") or "text").lower()
            # 复选/单选/按钮类不自动发送，避免发送未勾选项与多余 submit
            if itype in ("checkbox", "radio", "submit", "image", "button", "reset"):
                return
            self._cur["fields"][name] = d.get("value", "") or ""

    def handle_endtag(self, tag):
        if tag == "form" and self._cur is not None:
            self.forms.append(self._cur)
            self._cur = None


class _TableParser(HTMLParser):
    """收集页面里的 <table>：每张表 = 行列表，每行 = 单元格文本列表。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = [[]]
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables.append([])
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self.tables:
                self.tables[-1].append(self._row)
            self._row = None


def _decode(resp, raw):
    """按响应头 charset → utf-8 → gbk 顺序解码。"""
    charset = None
    try:
        charset = resp.headers.get_content_charset()
    except Exception:
        pass
    for enc in (charset, "utf-8", "gbk", "gb18030"):
        if not enc:
            continue
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


# ----------------------------- 抓取会话 ----------------------------------
def build_opener(insecure):
    if insecure:
        ctx = ssl._create_unverified_context()
        https = urllib.request.HTTPSHandler(context=ctx)
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()), https
        )
    else:
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )
    opener.addheaders = [("User-Agent", UA)]
    return opener


def http_get(opener, url, referer=None, timeout=30):
    headers = {}
    if referer:
        headers["Referer"] = referer
    req = urllib.request.Request(url, headers=headers)
    try:
        with opener.open(req, timeout=timeout) as resp:
            return _decode(resp, resp.read())
    except urllib.error.HTTPError as e:
        # 跟随到错误页时也尝试读 body（很多登录失败返回 200 登录页，少数返回 4xx）
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(f"HTTP {e.code} {e.reason}") from e


def http_post(opener, url, form, referer=None, timeout=30):
    data = urllib.parse.urlencode(form).encode("utf-8")
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if referer:
        headers["Referer"] = referer
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with opener.open(req, timeout=timeout) as resp:
            return _decode(resp, resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} {e.reason}") from e


# --------------------------- 单源：登录+抓取 -----------------------------
def login_and_fetch(src, opener):
    """登录该源并取回数据页 HTML。失败抛异常(带原因)。不接触明文密码。"""
    login = src.get("login", {})
    login_url = login.get("url")
    user_field = login.get("username_field")
    pass_field = login.get("password_field")
    user_env = src.get("credentials", {}).get("username_env")
    pass_env = src.get("credentials", {}).get("password_env")
    if not (login_url and user_field and pass_field and user_env and pass_env):
        raise ValueError("配置不全: 需 login.url/username_field/password_field 与 credentials.*_env")
    username = os.environ.get(user_env)
    password = os.environ.get(pass_env)
    if not username or not password:
        raise ValueError(f"凭据缺失: 环境变量 {user_env}/{pass_env} 未设置(检查 .env)")

    # 1) GET 登录页，捕获隐藏域(CSRF)
    try:
        page = http_get(opener, login_url)
    except Exception as exc:
        raise RuntimeError(f"登录页拉取失败: {exc}") from exc
    fp = _FormParser()
    fp.feed(page)
    form = None
    for f in fp.forms:
        if user_field in f["fields"] or pass_field in f["fields"]:
            form = f
            break
    fields = dict(form["fields"]) if form else {}
    fields[user_field] = username
    fields[pass_field] = password

    # action 用表单 action，缺省回退到 login_url；相对 URL 拼 base_url
    action = (form["action"] if form else "") or login_url
    action = urllib.parse.urljoin(src.get("base_url", ""), action)
    if not action.strip():
        action = login_url

    # 2) POST 登录
    try:
        http_post(opener, action, fields, referer=login_url)
    except Exception as exc:
        raise RuntimeError(f"登录提交失败: {exc}") from exc

    # 3) 取数据页
    data_url = src.get("fetch", {}).get("url")
    if not data_url:
        raise ValueError("配置不全: 需 fetch.url")
    try:
        data_html = http_get(opener, data_url, referer=login_url)
    except Exception as exc:
        raise RuntimeError(f"数据页拉取失败: {exc}") from exc

    # 4) 登录成功判定(显性失败核心)
    marker = login.get("success_marker")
    if marker:
        if marker not in data_html:
            raise RuntimeError(
                f"登录疑似失败: 数据页未命中 success_marker '{marker}'(可能账号密码错/被踢回登录页)"
            )
    else:
        # 兜底启发式：数据页又出现密码框 → 多半被弹回登录页
        if pass_field in data_html and 'type="password"' in data_html:
            raise RuntimeError("登录疑似失败: 数据页仍含密码输入框(未配 success_marker，启发式判定)")
    return data_html


# --------------------------- 单源：解析表格 ------------------------------
def extract_tasks(src, data_html):
    """从数据页找含 key_header 的 <table>，按 columns 映射成归一化任务列表。"""
    extract = src.get("extract", {})
    key_header = extract.get("key_header")
    columns = extract.get("columns", {}) or {}
    if not key_header or not columns:
        raise ValueError("配置不全: 需 extract.key_header 与 extract.columns")

    tp = _TableParser()
    tp.feed(data_html)
    target = None
    for tbl in tp.tables:
        if not tbl:
            continue
        header = tbl[0]
        if any(key_header in (h or "") for h in header):
            target = tbl
            break
    if not target:
        raise RuntimeError(f"未找到表头含 '{key_header}' 的表格(页面结构变? 或登录后被重定向?)")

    header = [html.unescape(h or "") for h in target[0]]
    # 列名 → 列下标(支持子串匹配)
    col_idx = {}
    for col_name, field in columns.items():
        for i, h in enumerate(header):
            if col_name in h:
                col_idx[field] = i
                break
    if "title" not in col_idx:
        raise RuntimeError(f"columns 映射后找不到 title 列(检查 columns 与实际表头: {header})")

    prog_re = re.compile(src.get("progress_regex") or DEFAULT_PROGRESS_RE)
    done_status = set(src.get("done_status", []) or [])
    tasks = []
    name = src.get("name", "source")
    for row in target[1:]:
        if not row or all(not (c or "").strip() for c in row):
            continue
        cells = [html.unescape(c or "") for c in row]
        task = {"source": name}
        for field, idx in col_idx.items():
            if idx < len(cells):
                task[field] = cells[idx].strip()
        if "title" not in task or not task.get("title"):
            continue
        # progress 归一化
        task["progress"] = None
        raw_p = task.get("progress_raw", "")
        m = prog_re.search(raw_p or "")
        if m:
            try:
                task["progress"] = max(0, min(100, int(m.group(1))))
            except ValueError:
                pass
        task["done"] = bool(done_status) and task.get("status", "") in done_status
        tasks.append(task)
    return tasks


# --------------------------- 渲染报表 ------------------------------------
def _md_escape(s):
    return (str(s).replace("|", "\\|").replace("\n", " ").replace("\r", " ")).strip()


def _parse_date(s):
    s = (s or "").strip()
    if not s:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.datetime.strptime(s[:19], fmt).date()
        except ValueError:
            continue
    return None


def _rate_by_progress(tasks):
    ps = [t["progress"] for t in tasks if t.get("progress") is not None]
    return round(sum(ps) / len(ps)) if ps else None


def _rate_by_status(tasks):
    if not tasks:
        return None
    done = sum(1 for t in tasks if t.get("done"))
    return round(done / len(tasks) * 100)


def render_report(results, out_dir):
    today = datetime.date.today()
    all_tasks = []
    src_stats = []
    for r in results:
        if r["status"] == "ok":
            all_tasks.extend(r["tasks"])
            src_stats.append((r["name"], True, len(r["tasks"]), None))
        else:
            src_stats.append((r["name"], False, 0, r["error"]))
    ok_src = sum(1 for _, ok, *_ in src_stats if ok)
    fail_src = len(src_stats) - ok_src

    lines = []
    lines.append("# 项目进度报表")
    lines.append("")
    lines.append(
        f"> 生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  "
        f"| 数据源: {ok_src} 成功 / {fail_src} 失败  | 任务总数: {len(all_tasks)}"
    )
    lines.append("")

    # 概览
    rp = _rate_by_progress(all_tasks)
    rs = _rate_by_status(all_tasks)
    overdue = [
        t for t in all_tasks
        if not t.get("done") and _parse_date(t.get("deadline")) and _parse_date(t.get("deadline")) < today
    ]
    lines.append("## 概览")
    lines.append(f"- 整体完成率(进度均值): {rp}%" if rp is not None else "- 整体完成率(进度均值): N/A")
    done_n = sum(1 for t in all_tasks if t.get("done"))
    lines.append(f"- 已完成(状态): {done_n}/{len(all_tasks)}" + (f" ({rs}%)" if rs is not None else ""))
    lines.append(f"- 逾期未完成: {len(overdue)}")
    lines.append("")

    # 按状态分布
    status_count = {}
    for t in all_tasks:
        s = t.get("status") or "(空)"
        status_count[s] = status_count.get(s, 0) + 1
    if status_count:
        lines.append("## 按状态分布")
        lines.append("| 状态 | 数量 |")
        lines.append("|---|---|")
        for s, c in sorted(status_count.items(), key=lambda x: -x[1]):
            lines.append(f"| {_md_escape(s)} | {c} |")
        lines.append("")

    # 按负责人
    owner_stat = {}
    for t in all_tasks:
        o = t.get("owner") or "(未指派)"
        owner_stat.setdefault(o, []).append(t)
    if owner_stat:
        lines.append("## 按负责人")
        lines.append("| 负责人 | 任务数 | 完成率(进度均值) |")
        lines.append("|---|---|---|")
        for o in sorted(owner_stat.keys()):
            ts = owner_stat[o]
            op = _rate_by_progress(ts)
            lines.append(f"| {_md_escape(o)} | {len(ts)} | {op}%" if op is not None else f"| {_md_escape(o)} | {len(ts)} | N/A |")
        lines.append("")

    # 各源明细
    lines.append("## 各数据源明细")
    for r in results:
        lines.append("")
        if r["status"] != "ok":
            lines.append(f"### {r['name']}  ❌ 抓取失败")
            lines.append(f"- 原因: {r['error']}")
            continue
        ts = r["tasks"]
        rp_s = _rate_by_progress(ts)
        lines.append(f"### {r['name']}  ({len(ts)} 条)")
        lines.append(f"- 完成率(进度均值): {rp_s}%" if rp_s is not None else "- 完成率(进度均值): N/A")
        lines.append("")
        lines.append("| 任务 | 负责人 | 状态 | 进度 | 截止 |")
        lines.append("|---|---|---|---|---|")
        for t in ts:
            prog = f"{t['progress']}%" if t.get("progress") is not None else "-"
            lines.append(
                f"| {_md_escape(t.get('title',''))} | {_md_escape(t.get('owner',''))} | "
                f"{_md_escape(t.get('status',''))} | {prog} | {_md_escape(t.get('deadline',''))} |"
            )
    lines.append("")

    # 逾期
    if overdue:
        lines.append("## 逾期未完成任务")
        lines.append("| 任务 | 负责人 | 源 | 截止 |")
        lines.append("|---|---|---|---|")
        for t in overdue:
            lines.append(
                f"| {_md_escape(t.get('title',''))} | {_md_escape(t.get('owner',''))} | "
                f"{_md_escape(t.get('source',''))} | {_md_escape(t.get('deadline',''))} |"
            )
        lines.append("")

    # 失败汇总(再强调一次，显性失败)
    fails = [r for r in results if r["status"] != "ok"]
    if fails:
        lines.append("## ⚠️ 抓取失败清单")
        for r in fails:
            lines.append(f"- **{r['name']}**: {r['error']}")
        lines.append("")
        lines.append("> 修复建议: 核对账号密码/字段名/登录页 URL；JS·SSO·验证码登录本脚本不支持(见 SKILL.md 能力边界)。")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report_path


# ------------------------------- main ------------------------------------
def main():
    ap = argparse.ArgumentParser(description="从多个登录 Web 系统抓取任务进度，汇总进度报表")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="数据源配置 JSON")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出目录")
    ap.add_argument("--env", default=DEFAULT_ENV, help=".env 路径(存账号密码)")
    ap.add_argument("--insecure", action="store_true", help="跳过 HTTPS 证书校验(内网自签证书时用)")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        sys.exit(
            f"[错误] 配置不存在: {args.config}\n"
            f"请先: cp .claude/skills/progress-report/config/sources.example.json config/sources.json 再改。"
        )

    load_env(args.env)
    with open(args.config, encoding="utf-8") as f:
        sources = json.load(f)
    if not isinstance(sources, list) or not sources:
        sys.exit("[错误] config 必须是非空 JSON 数组。")

    os.makedirs(args.out, exist_ok=True)
    raw_dir = os.path.join(args.out, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    results = []
    opener = build_opener(args.insecure)
    for src in sources:
        name = src.get("name", "source")
        print(f"→ 抓取: {name}")
        try:
            data_html = login_and_fetch(src, opener)
            tasks = extract_tasks(src, data_html)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            results.append({"name": name, "status": "fail", "error": err, "tasks": []})
            print(f"  [fail] {name}: {err}")
            continue
        # raw 落盘(调试用)
        safe = re.sub(r"[\\/:*?\"<>|]+", "_", name).strip().strip(".") or "source"
        with open(os.path.join(raw_dir, f"{safe}.json"), "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
        results.append({"name": name, "status": "ok", "error": None, "tasks": tasks})
        print(f"  [ok]   {name}: {len(tasks)} 条任务")

    report_path = render_report(results, args.out)

    run = {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": os.path.relpath(args.config, os.getcwd()),
        "sources": [
            {"name": r["name"], "status": r["status"], "task_count": len(r["tasks"]), "error": r["error"]}
            for r in results
        ],
        "total_tasks": sum(len(r["tasks"]) for r in results if r["status"] == "ok"),
    }
    with open(os.path.join(args.out, "_run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n完成: 成功 {ok}/{len(results)} 源, 任务 {run['total_tasks']} 条")
    print(f"报表: {report_path}")


if __name__ == "__main__":
    main()
