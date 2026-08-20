---
name: progress-report
description: 从多个需要登录的 Web 系统抓取项目/任务进度，汇总成一份进度报表。账号密码从 .env 读取(不入库)，抓取由确定性 Python 脚本(stdlib only)完成。当用户要求"汇总各系统任务进度""生成进度报表""拉一下项目进度"时使用。
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
metadata:
  version: "1.0.0"
  category: "reporting"
  tags: ["进度报表", "任务进度", "项目进度", "多数据源", "登录抓取"]
examples:
  - "汇总各系统任务进度生成进度报表"
  - "拉一下项目进度出一份报表"
  - "把禅道/Redmine 的任务进度汇总成报表"
---

# 项目进度报表 (progress-report)

## 用途
从**多个需要账号密码登录的 Web 系统**（如 禅道 / Redmine / Jira Server / 自研 OA）
抓取「项目/任务进度」，归一化为统一任务结构，渲染成一份汇总报表。

> 与 [[public-data-pipeline]] 的分工：那个采集**公开语料**进 `data_raw/` 并清洗入库；
> 这个登录**内部/受保护系统**拉**任务进度**进 `progress_report/`。互不复用。

## 能力边界（先看清，避免抓不动）
本 skill 的抓取后端是 **Python 标准库 `urllib`**（与 `fetch_docs.py` 同款，零新依赖），**只能处理标准 HTML 表单登录**：
- ✅ 支持：`POST username+password` + 页面里的隐藏域(CSRF)→ 拿 Cookie/Session → 抓 HTML 表格。
- ❌ 不支持：JS 渲染登录、SSO/OAuth 跳转(钉钉/飞书/企业微信单点)、图形验证码、滑块。
- 遇到 ❌ 场景：脚本会**显性失败**并在报表里标出该源抓取失败。需要的话后续给该源
  加一个 Playwright 浏览器适配器（`type: browser`，脚本已预留适配器入口）。

## 凭据安全（强制）
- 账号密码**只**放在仓库根的 **`.env`**（已被 `.gitignore` 覆盖）。
- **不要**写进 `sources.json`、`sources.example.json`、代码、或任何会提交的文件。
- 脚本只 `print` 源名和 `[ok]/[fail]`，**绝不**打印账号密码。
- `sources.json` 默认也 gitignore（可能含内网 URL）；committed 的只是脱敏的 `sources.example.json`。

## 输出布局
```
progress_report/           # 输出目录(已 gitignore)
  report.md                # 最终进度报表
  raw/<源名>.json          # 每源归一化后的任务数组(调试用)
  _run.json                # 本次运行元数据(各源 ok/fail、任务数、时间戳)
config/
  sources.example.json     # 配置模板(committed, 无密钥)
  sources.json             # 真实配置(gitignored, 你按模板填)
.env                       # 账号密码(gitignored)
```

## 配置：config/sources.json
一个 JSON 数组，每个元素 = 一个数据源。先复制模板再改：
```bash
cp .Codex/skills/progress-report/config/sources.example.json config/sources.json
```
字段说明：
| 字段 | 必填 | 说明 |
|------|------|------|
| `name` | ✅ | 源显示名（也是 raw 文件名，须文件名安全） |
| `base_url` | ✅ | 站点根，用于拼相对链接 |
| `login.url` | ✅ | 登录页/登录接口 URL |
| `login.method` | 默认 POST | 登录提交方法 |
| `login.username_field` / `password_field` | ✅ | 登录表单里用户名/密码 `<input>` 的 `name` |
| `login.success_marker` | 推荐 | 登录成功后才出现的页面文字（如 `退出`/`logout`/用户昵称）。用于判定登录是否成功 |
| `credentials.username_env` / `password_env` | ✅ | 取账号/密码的**环境变量名**（值在 `.env` 里） |
| `fetch.url` | ✅ | 登录后要抓取的数据页 URL |
| `extract.key_header` | ✅ | 任务表头里能唯一标识任务表的列名（如 `任务名称`/`标题`） |
| `extract.columns` | ✅ | 表头列名 → 任务字段的映射（`title`/`owner`/`status`/`progress_raw`/`deadline`） |
| `done_status` | 可选 | 视为"已完成"的状态值数组，用于算完成率 |
| `progress_regex` | 可选 | 从 `progress_raw` 单元格提取整数百分度的正则，默认 `(\d+)\s*%` |

### .env 写法（环境变量名要和 `credentials.*_env` 对齐）
```
PROGRESS_ZENTAO_USER=zhangsan
PROGRESS_ZENTAO_PASS=your-password-here
```

## 工作流（Codex 按序执行）
1. **配源**：若 `config/sources.json` 不存在 → 复制模板，和用户确认每个源：
   登录页 URL、用户名/密码字段 `name`（让用户在浏览器按 F12 看）、数据页 URL、
   表头列名 → 任务字段的映射、`success_marker`。改完 `Write` 落盘 `config/sources.json`。
2. **填凭据**：让用户把账号密码写进根目录 `.env`（环境变量名与 `credentials.*_env` 一致）。
   **Codex 不接触明文密码**；只核对变量名是否齐全。
3. **抓取+出报表**：在仓库根目录运行
   ```bash
   python .Codex/skills/progress-report/scripts/fetch_progress.py \
     [--config config/sources.json] [--out progress_report] [--env .env]
   ```
   脚本：加载 `.env` → 逐源登录抓取 → 解析表格 → 归一化 → 写 `report.md` + `raw/*.json` + `_run.json`。
4. **核对**：`Read _run.json` 看各源 ok/fail；`Read report.md` 复核报表。
5. **报告**：向用户汇报——总任务数、整体完成率、按负责人/状态分布、逾期任务、**抓取失败的源及原因**
   （登录失败 / 缺凭据 / 表格解析不到 key_header 等），并给修复建议。

## 脚本
`scripts/fetch_progress.py` — stdlib only（`urllib`+`http.cookiejar`+`html.parser`），零新依赖。
- **登录**：GET 登录页自动捕获所有 `<input type="hidden">`(含 CSRF) → 按字段名填账号密码 → POST → 跟随 Cookie。
- **成功判定**：取到数据页后，若配了 `success_marker` 则必须命中，否则判登录失败（显性失败，不写假数据）。
- **解析**：找出表头含 `key_header` 的那张 `<table>`，按 `columns` 映射成任务行。
- **失败处理**：单个源失败不影响其它源；失败源计入 `_run.json` 并在 `report.md` 单列。缺凭据直接跳过该源并标注。
