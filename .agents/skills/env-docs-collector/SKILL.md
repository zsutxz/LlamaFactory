---
name: env-docs-collector
description: 从互联网公开来源采集"环境保护"领域资料(法律法规/行业标准/论文)，下载到项目的 data_raw 目录，作为 LLaMA-Factory 领域继续预训练(PT)的原始语料。当用户要求搜索/下载/采集环保法规、环境标准、环境类论文，或为环保领域 PT 准备原始语料时使用。
allowed-tools:
  - WebSearch
  - WebFetch
  - mcp__web_reader__webReader
  - Bash
  - Read
  - Write
  - Edit
metadata:
  version: "1.0.0"
  category: "data-collection"
  tags: ["环保", "环境保护", "法律法规", "行业标准", "论文", "语料采集", "预训练"]
examples:
  - "搜索环保法律法规和行业标准下载到 data_raw"
  - "采集环境保护领域论文作为 PT 语料"
  - "下载环境保护法、GB 3095 等到 data_raw"
---

# 环保资料采集 (env-docs-collector)

## 用途
从**互联网公开来源**采集中国"环境保护"领域的三类资料，下载到项目 `data_raw/`，供后续用 `scripts/data/build_domain_corpus.py` 思路构建领域继续预训练(PT)语料：
- **法律法规** → `data_raw/laws/`
- **行业标准** → `data_raw/standards/`
- **论文** → `data_raw/papers/`

> 不在本项目内搜索；资料全部来自互联网。

## 输出布局
```
data_raw/
  laws/             # 法律法规全文
  standards/        # 环境标准(GB 强制 / HJ 行业)
  papers/           # 论文(优先开放获取)
  _fetch_list.json  # 本次下载清单(脚本输入, Codex 生成)
  _manifest.jsonl   # 下载记录(脚本输出, 幂等去重)
```

> **落盘格式仅限 `pdf` / `xml` / `txt` / `md` 四种**：网页正文须先转成 markdown 再落盘，禁止下载/保存 `.html` 及其他格式。

## 工作流（Codex 按序执行）
1. **定范围**：与用户确认每类要几份、是否含国际标准(ISO 14000)、语言(中/英)。默认套餐 = 核心法律 ~10 部 + 基础环境质量标准(GB) + 少量 arXiv 开放论文。
2. **找源**：按下方"权威来源门户"用 `WebSearch` 或已知门户 URL 定位具体文档页。
   > ⚠️ WebSearch 仅美国区，中文检索可能稀疏；优先**直接用已知门户 URL 导航**，用 `mcp__web_reader__webReader` 读页面再取下载直链。
3. **取正文**（落盘仅限 pdf/xml/txt/md）：
   - **直链文件**：拿到 `.pdf`/`.xml`/`.txt`/`.md` 直链 → 追加进 `data_raw/_fetch_list.json`(`kind=pdf|xml|txt|md`)，交给脚本下载。
   - **网页正文**：用 `mcp__web_reader__webReader` 抓成 markdown → `Write` 直接落盘 `data_raw/<cat>/<名>.md`；网页 URL **不得**列入清单下载原始 HTML。
4. **批量下载**：运行
   `python .agents/skills/env-docs-collector/scripts/fetch_docs.py`
   读取 `_fetch_list.json`，下载到对应子目录，幂等写 `_manifest.jsonl`（已下载且文件存在则跳过，限速 1s/请求）。
5. **核对**：`Read` `_manifest.jsonl`，统计各类别 ok/skip/fail；失败项标注原因(付费墙/反爬/404)。
6. **报告**：向用户汇报各类别份数、总字节、需人工获取的失败项清单。

## 权威来源门户（核心知识）

### 法律法规 (laws) — 公开可全文下载
| 门户 | URL | 说明 |
|------|-----|------|
| 国家法律法规数据库 | https://flk.npc.gov.cn | 最权威；法律+行政法规全文及 PDF |
| 中国政府网 | https://www.gov.cn | 国务院条例、规范性文件 |
| 生态环境部 | https://www.mee.gov.cn | 部门规章 |

核心法条（到 flk.npc.gov.cn 搜标题取 PDF）：
环境保护法、大气污染防治法、水污染防治法、土壤污染防治法、固体废物污染环境防治法、噪声污染防治法、海洋环境保护法、环境影响评价法、清洁生产促进法、节约能源法；
条例：排污许可管理条例、建设项目环境保护管理条例。

### 行业标准 (standards) — 部分公开
| 门户 | URL | 说明 |
|------|-----|------|
| 国家标准全文公开系统 | http://openstd.samr.gov.cn | **强制性 GB** 可在线阅读/下载 |
| 全国标准信息公共服务平台 | https://std.samr.gov.cn | GB/HJ 检索 |
| 生态环境部-标准 | https://www.mee.gov.cn （"标准"栏目） | HJ 环保行业标准 |

核心：环境空气质量标准 GB 3095、地表水环境质量标准 GB 3838、声环境质量标准 GB 3096、土壤污染风险管控标准 GB 36600、污水综合排放标准 GB 8978。
> ⚠️ 推荐性 HJ 方法标准多在标准销售渠道(付费)。付费内容只记录元数据到清单，标注"需人工获取"，**不绕过付费墙**。

### 论文 (papers) — 优先开放获取
| 门户 | URL | 说明 |
|------|-----|------|
| arXiv | https://arxiv.org | 大气/水质/遥感/碳 等英文预印本，PDF 开放 |
| PubMed | https://pubmed.ncbi.nlm.nih.gov | 环境健康，摘要 + 部分 OA 全文 |
| DOAJ | https://doaj.org | 开放获取期刊 |
| CNKI / 万方 | https://www.cnki.net | 中文环境期刊，多付费墙（仅编目） |

检索词：air pollution / water quality / PM2.5 / carbon emission / 环境监测 / 碳达峰 / 碳中和。

## 合规边界
- 仅采集**公开可访问**内容；付费墙/登录墙内容只记录元数据，不绕过、不盗取。
- 遵守站点频次（脚本默认每请求间隔 1s）。
- 法规文本、强制性国标属公开法律/标准文本，可全文保存。

## _fetch_list.json 格式
```json
[
  {"category": "laws",      "title": "中华人民共和国环境保护法",      "url": "https://flk.npc.gov.cn/.../hbf.pdf", "kind": "pdf"},
  {"category": "standards", "title": "GB 3095-2012 环境空气质量标准", "url": "http://openstd.samr.gov.cn/...",      "kind": "pdf"},
  {"category": "papers",    "title": "Deep learning for air quality",  "url": "https://arxiv.org/pdf/xxxx.pdf",     "kind": "pdf"}
]
```
`kind` 仅允许 `pdf | xml | txt | md`；其他取值(含 `html`)脚本记 invalid 永久跳过。

## 脚本
`scripts/fetch_docs.py` — 读 `data_raw/_fetch_list.json`，按 `category` 下载到 `data_raw/<category>/`，幂等写 `_manifest.jsonl`。
- **下载后端**：curl 优先(Windows 自动带 `--ssl-no-revoke` 规避 schannel 吊销检查) → urllib 兜底。
- **格式限制**：`kind` 仅允许 `pdf/xml/txt/md`，其他取值(含 `html`)记 invalid 永久跳过。
- **校验**：PDF 须以 `%PDF` 开头；xml/txt/md 内容若疑似 HTML 页面则记 fail——防止把 403 错误页存成假文件。
- **幂等语义**：`ok/skip/invalid` 的 URL 下次跳过(不重下、不重写行)；`fail`(网络/校验) 下次**重试**。
用法（在仓库根目录）：
```bash
python .agents/skills/env-docs-collector/scripts/fetch_docs.py \
  [--data-raw ./data_raw] [--list ./data_raw/_fetch_list.json]
```
