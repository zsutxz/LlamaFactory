---
name: public-data-pipeline
description: 从互联网公开来源采集任意领域资料(法规/标准/论文/公开文档)，下载到 data_raw，再清洗切块为 LLaMA-Factory 预训练(PT)语料并注册到 dataset_info.json 的端到端流水线；含蒸馏 QA 生成与定向劣化偏好对(DPO/RM 数据)支线。当用户要求"采集公开数据/清洗数据/构建某领域训练语料/生成偏好对/DPO或RM数据/下载资料并入库"，或只做其中一段(只采集、只清洗、只出题、只造偏好对)时使用。
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
端到端编排两条数据线，产出可直接用于 LLaMA-Factory 训练的数据集：

```
A线·PT 语料（采集→清洗→注册）
① 采集   WebSearch/webReader 找直链 → data_raw/_fetch_list.json
         → python .claude/skills/public-data-pipeline/scripts/fetch_docs.py
         → data_raw/<category>/  (+ _manifest.jsonl 幂等记录)
② 清洗   python .claude/skills/public-data-pipeline/scripts/build_domain_corpus.py \
             --src data_raw --out-prefix domain_<slug>
         → data/domain_<slug>.jsonl / _eval.jsonl / _stats.txt
③ 注册   data/dataset_info.json 追加两条数据集项(prompt=text)

B线·偏好对（出题→裁判→劣化→注册，供 DPO/RM）
⑤ 出题   generate_domain_qa.py 用 DeepSeek 从 A 线语料块生成锚定原文的 QA 对
⑥ 裁判   judge_domain_qa.py 用 Kimi 三维评分过筛，pass 项成 <prefix>_qa_sft.jsonl
⑦ 劣化   generate_domain_pref.py 对过筛答案定向造 rejected → <prefix>_pref.jsonl
   注册   dataset_info.json 加 ranking:true 一项(chosen/rejected 列)
```

> 各阶段可独立执行：只要"清洗"就从②开始；只要"偏好对"且 QA 已有就从⑥开始。

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
python .claude/skills/public-data-pipeline/scripts/build_domain_corpus.py --src data_raw --out-prefix domain_<slug>
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

## B 线 · 蒸馏 QA 与偏好对（DPO/RM 数据）

DPO/RM 的核心都是**偏好对** (prompt + chosen + rejected)；PPO 额外只要一池 prompt（eval QA 即可）。
chosen 来自过筛蒸馏 QA，rejected 由 DeepSeek 定向劣化（数值篡改/主体张冠李戴/关键截断/模糊化，行序轮转）。

### 阶段 5 · 出题（DeepSeek）
```bash
python .claude/skills/public-data-pipeline/scripts/generate_domain_qa.py --num 3 --eval-num 0   # 冒烟
python .claude/skills/public-data-pipeline/scripts/generate_domain_qa.py                        # 正式
```
从 `data/domain_<slug>.jsonl` 语料块生成锚定原文的 QA（quote 机械校验），manifest 幂等。
需 `.env` 配 `DEEPSEEK_API_KEY`。**注意：该脚本的出题 system prompt 目前写死环保领域，换领域需先改 SYSTEM_PROMPT。**

### 阶段 6 · 裁判过筛（Kimi）
```bash
python .claude/skills/public-data-pipeline/scripts/judge_domain_qa.py --limit 3   # 冒烟
python .claude/skills/public-data-pipeline/scripts/judge_domain_qa.py             # 全量
```
三维评分(grounding/terminology/value)三档分流，pass 项写入 `data/<prefix>_qa_sft.jsonl`（SFT 集兼 chosen 池）。
需 `.env` 配 `MOONSHOT_API_KEY`；kimi-k3 只认 temperature=1。

### 阶段 7 · 偏好对生成（DeepSeek 定向劣化）
```bash
python .claude/skills/public-data-pipeline/scripts/generate_domain_pref.py --limit 2   # 冒烟
python .claude/skills/public-data-pipeline/scripts/generate_domain_pref.py             # 正式
```
读 `<prefix>_qa_sft.jsonl`，对每条答案按轮转缺陷劣化成 rejected，机械校验与 chosen 有实质差异，
产出 `data/<prefix>_pref.jsonl`（alpaca ranking 格式）。manifest 幂等、中断重跑不重复扣费。

**注册（与 A 线不同，须显式指定 chosen/rejected 列）：**
```json
"domain_<slug>_pref": {
  "file_name": "domain_<slug>_pref.jsonl",
  "ranking": true,
  "columns": {"prompt": "instruction", "query": "input", "chosen": "chosen", "rejected": "rejected"}
}
```

**训练侧用法**：`stage: dpo` 直接吃该数据集；PPO 则先用同一份 `stage: rm` 训 RM，再用 prompt 池跑 `stage: ppo`。
PPO 训练中 policy 实时生成、RM 实时打分（9B QLoRA 单卡挂不动，先 DPO/KTO）。

## 脚本（全部自带于本 skill 的 scripts/）
| 脚本 | 用途 | 依赖 |
|---|---|---|
| `fetch_docs.py` | 清单批量下载，curl 优先/urllib 兜底、PDF 头校验、限速 1s/请求 | 无 |
| `build_domain_corpus.py` | 清洗/切块/去重/划分，目录扫描模式 | PyMuPDF(可选) |
| `generate_domain_qa.py` | DeepSeek 出题（锚定原文+quote 校验） | openai, .env:DEEPSEEK_API_KEY |
| `judge_domain_qa.py` | Kimi 裁判过筛 + PT/SFT 对比评测 | openai, .env:MOONSHOT_API_KEY |
| `ask_compare.py` | 本地 api 服务自动作答回填对比骨架 | 本地 LLaMA-Factory api |
| `generate_domain_pref.py` | 定向劣化生成偏好对(DPO/RM) | openai, .env:DEEPSEEK_API_KEY |

API 密钥一律读项目根 `.env`（已 gitignore；勿写 .env.local——它被 git 追踪）。跑脚本需 `PYTHONUTF8=1`（conda env llama-factory）。
