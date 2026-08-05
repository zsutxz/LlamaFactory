# 实操记录 1：Qwen3-4B 领域继续预训练（学习 LLM 预训练）

> 日期：2026-08-03 ~ 2026-08-04
> 目标：学习 LLM 预训练，把本机「专业文档 + 论文」语料灌进 Qwen3-4B，跑通「语料构建 → 继续预训练(PT) → 效果评估」全流程。
> 前置知识文档：[learning.md](./learning.md)（LLaMA-Factory 全阶段实操指南）、[frame.md](./frame.md)（框架原理）。

---

## 脚本与工具约定

> 原则：**只有「数据处理 / 清洗」才自写脚本，统一放 `scripts/data/`；训练、评估、推理等其他流程一律用 LLaMA-Factory 原生能力。**

| 流程 | 用什么 | 说明 |
|------|--------|------|
| 数据清洗 | `scripts/data/build_domain_corpus.py` | 本工作流唯一自写脚本（PDF/md/html → jsonl） |
| 训练（PT/SFT） | `llamafactory-cli train` | 见第 4、6 节 |
| PPL 评估 | `scripts/stat_utils/cal_ppl.py`（LF 自带） | `stage=pt`，按注册数据集名加载，口径与训练同源 |
| 续写 / 推理 | `llamafactory-cli chat` | 交互式喂前缀看续写 |

> 以下所有 `train` / `chat` / `cal_ppl.py` 命令都需先设三项环境变量：`PYTHONUTF8=1`、`HF_HOME=E:\AI\LLaMA-Factory\hf_cache`、`HF_HUB_OFFLINE=1`（理由见第 5 节坑 1）。各命令块为可独立拷贝运行，故重复列出。

---

## 1. 本次结论（TL;DR）

- 用 **LoRA 继续预训练** 把 18 份领域文档/论文（约 65 万字符）注入 Qwen3-4B，训练 45 步（5 epochs）约 35 分钟，16GB 显存刚好跑满。
- 验证集困惑度 **PPL 18.49 → 12.56（↓ 32%，均为训练内评、可复现）**；续写能稳定接续领域术语与文风（定性观察）。
- 踩到并解决两个本机必踩的坑：
  1. 默认 HF 缓存目录 `~/.cache/huggingface/datasets` 无法写入 → `datasets` 卡在文件锁上 15 分钟+；
  2. Windows 下 `preprocessing_num_workers` 设 0 或 >1 都会出问题 → 统一设 1。
- 重要认知：PT 注入的是「知识 + 语言风格」，不是「问答能力」；要能回答领域问题还需继续做 SFT。

---

## 2. 环境盘点（动手前先摸清家底）

| 项 | 值 | 说明 |
|----|----|------|
| GPU | RTX 5060 Ti 16GB（WDDM） | bf16 下 Qwen3-4B + LoRA 占 15.7GB，几乎顶满 |
| 基座模型 | `model/Qwen3-4B`（本地完整权重） | 另有 Qwen3-0.6B/1.7B/8B |
| Python 环境 | conda `llama-factory`（Python 3.11.14） | torch 2.9.1+cu130 / transformers 4.57.1 / peft 0.18.1 / datasets 3.6.0 |
| 历史进度 | `saves/` 中已有 SFT 实测记录；PT 此前只有 5 步冒烟 | 本次把 PT 从「未实测」补成完整记录 |
| 语料来源 | `E:\AI\Book`、`E:\AI\5-Day-AI-Agents-Intensive-Course-with-Google-2025`、`E:\AI\teach-fish-to-swim` | 专业手册 + 论文合集 + 课程文档 |

---

## 3. 语料构建

### 3.1 来源与选择

| 来源 | 内容 | 形态 |
|------|------|------|
| `E:\AI\Book` | AI演义 36 篇论文（中文）、完整提示词工程指南（中文）、Paper2Agent（英文） | 3 个 PDF |
| `E:\AI\5-Day-AI-Agents-Intensive-Course-with-Google-2025` | Google AI Agent 课程 6 份讲义 | 6 个 PDF（文本密集，非扫描件） |
| `E:\AI\teach-fish-to-swim` | 多篇论文全文/精读笔记（NVFP4 预训练、递归语言模型等） | md / html |

### 3.2 构建脚本 `scripts/data/build_domain_corpus.py`

处理流水线：

```
PDF(pymupdf) / md / html(bs4)  → 清洗 → 段落切块(~1800字符) → 哈希去重 → 固定种子洗牌 → 每10块留1块做验证
```

清洗规则（论文/文档场景够用）：

- 去掉图片语法、链接只留文字、代码块、标题/列表符号；
- 合并连续空白；丢弃 <150 字符的碎片块和纯符号行（分页符、装饰线）；
- 按段落边界贪心打包，单块不超过 1800 字符。

产出统计（`data/domain_papers_stats.txt`）：

| 指标 | 数值 |
|------|------|
| 文档数 | 18 |
| 切块数（去重前/后） | 374 / 374 |
| 训练块 / 验证块 | 336 / 38 |
| 总字符数 | 647,655（约 14.7 万 token 训练） |

### 3.3 注册数据集

在 `data/dataset_info.json` 追加两个条目（PT 只需 `columns.prompt = "text"`）：

```json
"domain_papers": {
  "file_name": "domain_papers.jsonl",
  "columns": { "prompt": "text" }
},
"domain_papers_eval": {
  "file_name": "domain_papers_eval.jsonl",
  "columns": { "prompt": "text" }
}
```

---

## 4. 训练配置

配置文件：`examples/train_lora/qwen3_4b_domain_pretrain.yaml`

关键参数与理由：

| 分块 | 参数 | 值 | 为什么 |
|------|------|----|--------|
| model | `model_name_or_path` | `model/Qwen3-4B` | 本地路径，不依赖联网 |
| method | `stage: pt` | 无监督语言建模 | PT 不套 chat 模板，纯文本直接喂 |
| method | `finetuning_type: lora`，rank 8，`lora_target: all` | 16GB 显存跑 4B 的常规选择 | 只训 1650 万参数（0.41%） |
| dataset | `cutoff_len: 2048` | 单窗口 token 上限 | PT 自动 packing，多块短文本拼满窗口 |
| dataset | `preprocessing_num_workers: 1` | **Windows 必设 1** | 0 报错、>1 卡死（见第 5 节） |
| train | batch 1 × 累积 8 | 有效 batch = 8 个窗口 | 336 块 ≈ 72 窗口 ≈ 9 步/epoch |
| train | `learning_rate: 1e-4`，cosine，warmup 0.1 | PT 比 SFT（5e-5）高一档 | 知识注入需要更大步长 |
| train | `num_train_epochs: 5.0` | 小语料多过几遍 | 3~5 是常用区间 |
| eval | `eval_dataset: domain_papers_eval`，`eval_steps: 10` | 训练中定期看验证 loss | 用 PPL 决定停训 |

---

## 5. 排坑实录（本次最有价值的部分）

### 坑 1：`datasets` 加载卡死 15 分钟+（HF 缓存目录写不进）

现象：`llamafactory-cli train` 打印完 `Loading dataset domain_papers.jsonl...` 后无任何输出，15 分钟超时被杀。

排查过程：

1. 用 faulthandler 抓调用栈 → 卡在 `filelock._api.acquire`（`datasets/builder.py:390`）；
2. 单独测 `FileLock`：仓库内路径 0 秒拿到，`C:\Users\skype\.cache\huggingface\datasets\probe.lock` 15 秒超时；
3. 实测该目录**连普通文件都创建不了**（`System.IO.File.WriteAllText` 报「访问被拒绝」）；
4. 把 `cache_dir` 指到仓库内 → `load_dataset` 0.9 秒完成。

结论与处理：

- 根因：默认 HF 缓存目录被破坏（权限/同步问题），`datasets` 在 builder 初始化时等文件锁，无限重试。
- 解决：所有 LF 命令前设置 `$env:HF_HOME = 'E:\AI\LLaMA-Factory\hf_cache'`（仓库 `.gitignore` 已忽略 `hf_cache/`，正好是预留的本地缓存位）。
- 经验：**Windows 上任何「看似无进展」的 datasets 操作，先怀疑文件锁/缓存目录可写性**。

### 坑 2：Windows 多进程预处理

现象：`preprocessing_num_workers: 0` 直接报 `ValueError: num_proc must be an integer > 0.`；设 8 时 worker 反复崩溃（spawn 无法重导入入口脚本）并挂起。

解决：统一设 `1`（单进程）。本机语料小（336 块），tokenize 不到 1 秒，多进程毫无收益。

---

## 6. 冒烟测试 → 正式训练

### 6.1 冒烟（3 步，验证管线与显存）

```powershell
$env:PYTHONUTF8='1'; $env:HF_HOME='E:\AI\LLaMA-Factory\hf_cache'; $env:HF_HUB_OFFLINE='1'
& "C:\Users\skype\.conda\envs\llama-factory\Scripts\llamafactory-cli.exe" train `
  examples\train_lora\qwen3_4b_domain_pretrain.yaml max_samples=80 max_steps=3 `
  output_dir=saves\Qwen3-4B-domain\lora\pt_smoke
```

结果：3 步 141 秒，训练损失 2.87 → 2.85，验证 PPL 18.49（即第 7.1 节「训练初期」对照值），管线全通。

### 6.2 正式训练（45 步 / 5 epochs / 35 分钟）

启动（后台 + 日志）：

```powershell
$env:PYTHONUTF8='1'; $env:HF_HOME='E:\AI\LLaMA-Factory\hf_cache'; $env:HF_HUB_OFFLINE='1'
& "C:\Users\skype\.conda\envs\llama-factory\Scripts\llamafactory-cli.exe" train `
  examples\train_lora\qwen3_4b_domain_pretrain.yaml
```

训练过程监控：

| 时刻 | 进度 | 说明 |
|------|------|------|
| 0:46 | 1/45 | ~47 秒/步，GPU 15.7GB / 99% |
| 5:25 | 7/45 | 正常 |
| 16:32 | 21/45 | 过半 |
| 28:15 | 36/45 | checkpoint-10/20/30 已落盘 |
| 35:24 | 45/45 完成 | 最终 adapter 存到 output_dir 根 |

产物（`saves/Qwen3-4B-domain/lora/pt`）：

- `adapter_config.json` + `adapter_model.safetensors`（66MB）——最终 LoRA；
- `checkpoint-30/40/45` ——中间检查点（`save_steps: 10`，`save_total_limit: 3`）；
- `training_loss.png` / `training_eval_loss.png` ——loss 曲线；
- `all_results.json` / `eval_results.json` ——指标。

---

## 7. 效果评估

### 7.1 困惑度（PPL）：训练初期 → 训练后

PPL 以**训练内评**为准（`eval_steps` 在 `domain_papers_eval` 上算，与训练同源、可复现）：

| 指标 | 训练初期（冒烟 step3） | 训练后（45 步） | 变化 |
|------|------------------------|-----------------|------|
| eval loss | ≈2.92 | 2.5304 | ↓ |
| PPL | 18.49 | 12.56 | **↓ 32%** |

> 若需独立的**基座基线** PPL（训练内评只覆盖训练后），用 LF 自带 `scripts/stat_utils/cal_ppl.py`（与训练同源）：

```powershell
& "C:\Users\skype\.conda\envs\llama-factory\python.exe" scripts\stat_utils\cal_ppl.py `
  --model_name_or_path model/Qwen3-4B --stage pt --dataset domain_papers_eval --save_name ppl_base.json
```

### 7.2 续写验证（定性）

用 `llamafactory-cli chat` 加载基座 + adapter，交互式喂前缀看续写：

```powershell
& "C:\Users\skype\.conda\envs\llama-factory\Scripts\llamafactory-cli.exe" chat `
  examples\train_lora\qwen3_4b_domain_pretrain.yaml adapter_name_or_path=saves\Qwen3-4B-domain\lora\pt
```

> 验证方法：取验证集某段开头做前缀，对比「直接基座 chat」（不加 `adapter_name_or_path`）与「加载 adapter 后」的续写。

### 7.3 训练动态观察

- 训练损失 2.87 → 2.49（每步），平均 2.58；
- eval loss 在 step 30 后进入平台期（≈2.53 不再降）→ **5 epochs 对小语料已经够用，验证集 PPL 是停训的好依据**；
- LoRA 权重 16.5M / 40.4 亿参数 = 只动了 0.41% 的参数，训练 35 分钟。

---

## 8. 学习要点（预训练是什么）

1. **预训练 = 下一个 token 预测**：把纯文本切成 token 序列，模型对每个位置预测下一个 token，用交叉熵算 loss、反向传播更新权重。LLaMA-Factory 的 `stage: pt` 就是包装好的这个流程（`DataCollatorForLanguageModeling(mlm=False)`，见 `src/llamafactory/train/pt/workflow.py`）。
2. **从零预训练 vs 继续预训练**：从零训练需要 TB 级数据和大量算力；在消费级显卡上做的是「继续预训练 / 领域适配」——基座已有通用语言能力，用领域语料 + LoRA 微调让它更懂你的领域，成本可接受。
3. **PT 灌的是「知识 + 风格」，不是「问答能力」**：做完 PT 模型能接续领域文本，但直接问它未必好好回答。要「能回答领域问题」，链路是 **PT → SFT** 两步。
4. **LoRA 为什么行**：冻结基座，只训练注入的少量低秩矩阵（本次 0.41% 参数），显存和训练时间都大幅下降，效果对领域适配足够。
5. **PPL 怎么读**：PPL = e^(平均交叉熵)，越低说明模型对这段文本的预测越准。注意口径一致才能对比（是否加 BOS、是否 packing 都会造成小差异）。
6. **小语料别贪 epochs**：验证集 PPL 进入平台期就停，继续训只会过拟合 + 遗忘通用能力。

---

## 9. 复现清单

> 以下命令在同一 PowerShell 会话里顺序执行。三项环境变量对**所有步骤**都必需（理由见第 5 节坑 1），故单列一步。

```powershell
# 1) 构建语料（用带 pymupdf 的解释器，如 base Anaconda；llama-factory 环境无 pymupdf）
python scripts\data\build_domain_corpus.py

# 2) 设环境变量（同会话生效，后续 train / cal_ppl / chat 都依赖）
$env:PYTHONUTF8='1'; $env:HF_HOME='E:\AI\LLaMA-Factory\hf_cache'; $env:HF_HUB_OFFLINE='1'

# 3) 训练
& "C:\Users\skype\.conda\envs\llama-factory\Scripts\llamafactory-cli.exe" train `
  examples\train_lora\qwen3_4b_domain_pretrain.yaml

# 4) 基座 PPL 对照（训练后 PPL 直接看训练内评，见 §7.1）
& "C:\Users\skype\.conda\envs\llama-factory\python.exe" scripts\stat_utils\cal_ppl.py `
  --model_name_or_path model/Qwen3-4B --stage pt --dataset domain_papers_eval --save_name ppl_base.json

# 5) 续写验证（加载 adapter）
& "C:\Users\skype\.conda\envs\llama-factory\Scripts\llamafactory-cli.exe" chat `
  examples\train_lora\qwen3_4b_domain_pretrain.yaml adapter_name_or_path=saves\Qwen3-4B-domain\lora\pt
```

**换语料时改这 5 处**：`build_domain_corpus.py` 的源目录（`SRC_PDF_DIRS` / `SRC_PAPER_ROOT`）→ `data/dataset_info.json` 注册新数据集名 → yaml 的 `dataset` 与 `output_dir` → 步骤 4 的 `--dataset` → 步骤 5 的 `adapter_name_or_path`。

---

## 10. 产物文件清单

| 文件 | 作用 |
|------|------|
| `scripts/data/build_domain_corpus.py` | 语料构建脚本（PDF/md/html → jsonl 训练/验证集 + 统计） |
| `data/domain_papers.jsonl` / `domain_papers_eval.jsonl` | 训练 / 验证数据 |
| `data/domain_papers_stats.txt` | 语料统计（文档数 / 切块数 / 字符数） |
| `data/dataset_info.json`（新增条目） | 数据集注册（追加 `domain_papers` / `domain_papers_eval`） |
| `examples/train_lora/qwen3_4b_domain_pretrain.yaml` | 训练配置（带注释） |
| `saves/Qwen3-4B-domain/lora/pt` | 训练产物（adapter + checkpoint-30/40/45 + 曲线 + 指标） |
