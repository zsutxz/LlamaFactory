---
name: public-data-pipeline
description: 从互联网公开来源采集任意领域资料(法规/标准/论文/公开文档)，下载到 data_raw，再清洗切块为 LLaMA-Factory 预训练(PT)语料并注册到 dataset_info.json 的端到端流水线。当用户要求"采集公开数据/清洗数据/构建某领域训练语料/下载资料并入库"，或只做其中一段(只采集、或只清洗已有 data_raw)时使用。
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
  tags: ["公开数据", "语料采集", "数据清洗", "预训练", "领域语料"]
examples:
  - "采集法律领域的公开法规和论文，清洗成训练语料"
  - "把 data_raw 里已有的资料清洗入库"
  - "为医疗领域 PT 准备原始语料并注册数据集"
---

# 公开数据采集清洗流水线 (public-data-pipeline)

## 用途
端到端编排三段现成机制，产出可直接用于 LLaMA-Factory PT 的数据集：

```
① 采集   WebSearch/webReader 找直链 → data_raw/_fetch_list.json
         → python .claude/skills/public-data-pipeline/scripts/fetch_docs.py
         → data_raw/<category>/  (+ _manifest.jsonl 幂等记录)
② 清洗   python scripts/data/build_domain_corpus.py --src data_raw --out-prefix domain_<slug>
         → data/domain_<slug>.jsonl / _eval.jsonl / _stats.txt
③ 注册   data/dataset_info.json 追加两条数据集项
```

> 三段可独立执行：用户只要"清洗"就从②开始（data_raw 已有内容）；只要"采集"就止于①。

## 工作流（Claude 按序执行，每阶段结束向用户汇报一次）

### 阶段 0 · 定范围
与用户确认：**领域**、资料类别(laws/standards/papers 或自定义目录名)、每类份数、语言。
确定输出前缀 `domain_<slug>`（如 环保=domain_env、法律=domain_law），全程复用。

### 阶段 1 · 找源
- **通用三类门户**（任意领域适用，`WebSearch` 定位后用 `mcp__web_reader__webReader` 读文档页取下载直链）：
  | 类别 | 通用门户 |
  |---|---|
  | 法律法规 | 国家法律法规数据库 flk.npc.gov.cn、中国政府网 gov.cn、主管部门官网 |
  | 标准 | 国家标准全文公开系统 openstd.samr.gov.cn、std.samr.gov.cn |
  | 论文 | arXiv、PubMed、DOAJ（优先开放获取） |
- ⚠️ WebSearch 仅美国区，中文检索可能稀疏；优先直接用已知门户 URL 导航。
- **合规红线**：仅采集公开可访问内容；付费墙/登录墙只记录元数据并标注"需人工获取"，不绕过。

**领域门户速查 · 环保（已实测验证过一轮）**
| 类别 | 门户 | 说明 |
|---|---|---|
| laws | https://flk.npc.gov.cn 国家法律法规数据库 | 最权威；法律+行政法规全文及 PDF |
| laws | https://www.gov.cn 中国政府网 | 国务院条例、规范性文件 |
| laws | https://www.mee.gov.cn 生态环境部 | 部门规章 |
| standards | http://openstd.samr.gov.cn 国家标准全文公开系统 | **强制性 GB** 可在线阅读/下载 |
| standards | https://std.samr.gov.cn 全国标准信息公共服务平台 | GB/HJ 检索 |
| papers | https://arxiv.org / https://pubmed.ncbi.nlm.nih.gov / https://doaj.org | 开放获取优先 |

环保核心清单：环境保护法/大气/水/土壤/固废/噪声/海洋环境保护法/环评法 + 排污许可管理条例；GB 3095/3838/3096/36600/8978。
新领域首次采集后，把验证可用的门户补充到本表。

### 阶段 2 · 采集落盘
- 直链文件(`.pdf`/`.xml`/`.txt`/`.md`)追加进 `data_raw/_fetch_list.json`：
  ```json
  [
    {"category": "laws", "title": "中华人民共和国环境保护法", "url": "https://flk.npc.gov.cn/.../hbf.pdf", "kind": "pdf"}
  ]
  ```
  `kind` 仅允许 `pdf | xml | txt | md`，其他取值(含 `html`)脚本记 invalid 永久跳过。
- 网页正文用 webReader 抓成 markdown 后 `Write` 直接落盘 `data_raw/<category>/<名>.md`，URL 不进清单。
- 批量下载（仓库根目录）：
  ```bash
  python .claude/skills/public-data-pipeline/scripts/fetch_docs.py
  ```
- 核对 `_manifest.jsonl`：统计 ok/skip/fail，失败项标注原因。已下载过的 URL 自动跳过，可安全重跑。

### 阶段 3 · 清洗入库
```bash
python scripts/data/build_domain_corpus.py --src data_raw --out-prefix domain_<slug>
```
递归扫描 data_raw 下的 md/html/pdf/txt/xml（`_` 前缀元数据文件自动跳过），完成：去噪规整 → 按段落切块(约1800字符) → 块级哈希去重 → 固定种子划分训练/验证。
产出 `data/domain_<slug>.jsonl`、`data/domain_<slug>_eval.jsonl`、`data/domain_<slug>_stats.txt`。
向用户复述 stats（文档数/去重前后块数/总字符）。

### 阶段 4 · 注册数据集
向 `data/dataset_info.json` 追加（模式与既有 domain_* 项一致）：
```json
"domain_<slug>": {
  "file_name": "domain_<slug>.jsonl",
  "columns": { "prompt": "text" }
},
"domain_<slug>_eval": {
  "file_name": "domain_<slug>_eval.jsonl",
  "columns": { "prompt": "text" }
}
```
之后即可在训练 yaml 的 `dataset:` 里引用 `domain_<slug>`。

## 后续（不在本 skill 范围，仅提示）
PT 语料若要生成 SFT 问答对：`scripts/data/generate_domain_qa.py` + `judge_domain_qa.py`（见蒸馏闭环流程）。

## 脚本
- 采集：`scripts/fetch_docs.py`（本 skill 自带；curl 优先/urllib 兜底、PDF 头校验、限速 1s/请求）
  ```bash
  python .claude/skills/public-data-pipeline/scripts/fetch_docs.py \
    [--data-raw ./data_raw] [--list ./data_raw/_fetch_list.json]
  ```
- 清洗：复用 `scripts/data/build_domain_corpus.py`（需 PyMuPDF 解析 PDF，缺库时自动跳过 PDF）
