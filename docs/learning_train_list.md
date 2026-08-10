# 领域继续预训练（PT）实操记录：Qwen3-4B → Qwen3-8B QLoRA

> 日期：2026-08-03 ~ 2026-08-08（4B：08-03~04；8B QLoRA：08-07~08）
> 目标：把本机「专业文档 + 论文」语料灌进 Qwen3，跑通「语料构建 → 继续预训练(PT) → 效果评估」全流程；先 4B 起步，再用 4-bit QLoRA 把规格顶到 8B，对比代价与收益。
> 前置知识：[learning.md](./learning.md)（全阶段实操）、[frame.md](./frame.md)（框架原理）。

---

## 脚本与工具约定

> 原则：**只有「数据清洗」才自写脚本，统一放 `scripts/data/`；训练、评估、推理等其他流程一律用 LLaMA-Factory 原生能力。**

| 流程 | 用什么 | 说明 |
|------|--------|------|
| 数据清洗 | `scripts/data/build_domain_corpus.py` | 本工作流唯一自写脚本（PDF/md/html → jsonl） |
| 训练（PT） | `llamafactory-cli train` | 4B bf16 LoRA / 8B 4-bit QLoRA |
| PPL 评估 | 训练内 eval（`eval_steps`） | `cal_ppl.py` 不支持量化、大模型 OOM（§5 坑 6），统一用训练内评 |
| 续写 / 推理 | `llamafactory-cli chat` + **独立 infer yaml** | chat 不收 train yaml（§5 坑 4） |

> 三项环境变量对所有命令必需（理由见 §5 坑 1）：`PYTHONUTF8=1`、`HF_HOME=E:\AI\LLaMA-Factory\hf_cache`、`HF_HUB_OFFLINE=1`。

---

## 1. 结论（TL;DR）

- 同一份领域语料（18 文档 / 65 万字符 / 336 训练块），分别用 **4B bf16 LoRA** 和 **8B 4-bit QLoRA** 做继续预训练，5 epochs / 45 步。
- **4B**：PPL **18.49 → 12.56（↓32%）**，35 分钟，显存 15.7GB（近顶满 16GB）。
- **8B QLoRA**：PPL **15.47 → 11.36（↓27%）**，47 分钟，显存 ~14.3GB（量化后反比 4B 省）。
- 两个核心洞察：
  1. **4-bit 量化是消费级显卡上「把模型规格顶上去」的关键杠杆**——8B QLoRA 比 4B bf16 既更强（PPL 更低）又更省显存，代价是慢约 35%。
  2. **PT 灌的是「知识 + 语言风格」，不是问答能力**——续写对照显示 PT 改善的是领域文本预测分布（PPL 可量化），不显著改变问答行为；要「能回答领域问题」，链路是 **PT → SFT**。
- 踩坑 6 个（§5）：HF 缓存写不进、Windows 多进程、bnb/Blackwell、chat 不收 train yaml、chat 覆盖语法、cal_ppl 不量化。

---

## 2. 环境盘点

| 项 | 值 |
|----|----|
| GPU | RTX 5060 Ti 16GB（Blackwell sm_120，WDDM） |
| Python | conda `llama-factory`（3.11.14）；torch 2.9.1+cu130 / transformers 4.57.1 / peft 0.18.1 / datasets 3.6.0 / bitsandbytes 0.48.2 |
| 基座 | `model/Qwen3-4B`（单文件）/ `model/Qwen3-8B`（5 分片，bf16 ≈15.5GB） |
| 语料来源 | `E:\AI\Book`、`E:\AI\5-Day-AI-Agents-Intensive-Course-with-Google-2025`、`E:\AI\teach-fish-to-swim` |

> **为什么 8B 能跑、4B 反而更挤**：4B bf16 基座 ~8GB，加训练态到 15.7GB（近顶满）；8B bf16 放不进 16GB，必须 4-bit 量化——量化后基座仅 ~5GB，叠加 LoRA/激活/优化器总占用 ~14.3GB，反比 4B 省。

---

## 3. 语料构建（两模型共用）

### 3.1 来源

| 来源 | 内容 | 形态 |
|------|------|------|
| `E:\AI\Book` | AI演义 36 篇（中）、完整提示词工程指南（中）、Paper2Agent（英） | 3 PDF |
| 5-Day-AI-Agents | Google AI Agent 课程 6 份讲义 | 6 PDF（文本密集） |
| teach-fish-to-swim | 多篇论文全文/精读（NVFP4 预训练、递归语言模型等） | md / html |

### 3.2 构建脚本 `scripts/data/build_domain_corpus.py`

处理流水线：

```
PDF(pymupdf) / md / html(bs4)  → 清洗 → 段落切块(~1800字符) → 哈希去重 → 固定种子洗牌 → 每10块留1块做验证
```

清洗规则（论文/文档场景够用）：去图片语法、链接只留文字、代码块、标题/列表符号；合并连续空白；丢弃 <150 字符的碎片块和纯符号行（分页符、装饰线）；按段落边界贪心打包，单块不超过 1800 字符。

产出统计（`data/domain_papers_stats.txt`）：

| 指标 | 数值 |
|------|------|
| 文档数 | 18 |
| 切块数（去重前/后） | 374 / 374 |
| 训练块 / 验证块 | 336 / 38 |
| 总字符数 | 647,655（约 14.7 万训练 token） |

### 3.3 注册数据集

在 `data/dataset_info.json` 追加两条（PT 只需 `columns.prompt = "text"`）：

```json
"domain_papers":       { "file_name": "domain_papers.jsonl",       "columns": { "prompt": "text" } },
"domain_papers_eval":  { "file_name": "domain_papers_eval.jsonl",  "columns": { "prompt": "text" } }
```

> 两模型共用这份语料，差别只在训练配置（§4）。

---

## 4. 训练配置

两份 yaml，**同一 PT 配方**（`stage: pt` / `lora rank8 lora_target: all` / `cutoff_len: 2048` / `preprocessing_num_workers: 1` / batch 1×累积 8 / `lr 1e-4` cosine warmup 0.1 / `5 epochs` / `eval_steps: 10`），差别在模型与量化：

| 分块 | 4B（`qwen3_4b_domain_pretrain.yaml`） | 8B（`qwen3_8b_domain_pretrain.yaml`） |
|------|----------------------------------------|----------------------------------------|
| model | `model/Qwen3-4B` | `model/Qwen3-8B` + **`quantization_bit: 4` / `method: bnb`** |
| 量化 | 无（bf16 放得下） | 必须 4-bit（bf16 15.5GB 放不进 16GB） |
| 可训练参数 | 16.5M / 40.4 亿 = 0.41% | 21.8M / 82 亿 = 0.27% |

> PT 自动 packing：336 块 ≈ 72 个 2048 窗口，有效 batch 8 → ~9 步/epoch × 5 = 45 步。

---

## 5. 排坑实录（最有价值的部分，共 6 个）

### 坑 1：`datasets` 加载卡死 15 分钟+（HF 缓存目录写不进）— 4B 首遇

现象：`llamafactory-cli train` 打印完 `Loading dataset ...` 后无输出，15 分钟超时被杀。
排查：faulthandler 抓调用栈 → 卡在 `filelock._api.acquire`（`datasets/builder.py:390`）；单测 `FileLock`，仓库内路径 0 秒拿到，`C:\Users\skype\.cache\huggingface\datasets\probe.lock` 15 秒超时；该目录**连普通文件都创建不了**（`访问被拒绝`）。
根因：默认 HF 缓存目录被破坏（权限/同步问题），`datasets` 在 builder 初始化时等文件锁、无限重试。
解决：所有 LF 命令前 `$env:HF_HOME = 'E:\AI\LLaMA-Factory\hf_cache'`（`.gitignore` 已忽略 `hf_cache/`）。
经验：**Windows 上任何「看似无进展」的 datasets 操作，先怀疑文件锁/缓存目录可写性**。

### 坑 2：Windows 多进程预处理 — 4B 首遇

现象：`preprocessing_num_workers: 0` 直接报 `ValueError: num_proc must be an integer > 0.`；设 >1 时 worker 反复崩溃（spawn 无法重导入入口脚本）并挂起。
解决：统一设 `1`（单进程）。本机语料小，tokenize 不到 1 秒，多进程毫无收益。

### 坑 3：bitsandbytes 能否在 Blackwell sm_120 上跑（8B 最大风险）— 8B

预检三连（动手前必查，省得训练到一半崩）：

```
torch.cuda.get_device_capability()  -> (12, 0)   # Blackwell sm_120
torch.version.cuda                  -> 13.0
bitsandbytes.__version__            -> 0.48.2    # ≥0.43 即支持 Blackwell
```

结论：组合 OK，QLoRA 全程无 kernel 报错。**经验：50 系卡跑 QLoRA 前，先确认 device cap + bnb 版本，别等训练崩。**

### 坑 4：`llamafactory-cli chat` 不接受训练 yaml — 8B

`chat examples/.../train.yaml` 直接报 `ValueError: Some keys are not used by the HfArgumentParser: ['do_train','learning_rate','save_steps', ...]`。
原因：chat 走 `get_infer_args`，infer 解析器只认模型/推理键，训练专用键一律拒绝。
解决：**单独建一份只含模型键的 infer yaml**（`examples/inference/qwen3_8b_domain_chat.yaml`：`model_name_or_path` + `quantization_bit/method` + `template` + `infer_backend`），不含任何训练键。

### 坑 5：chat 的 key=value 只能「覆盖」、不能「单独成参」— 8B

`chat model_name_or_path=...`（不给 yaml）会报 `Please provide model_name_or_path`——chat 要求**第一个位置参数必须是 yaml**，key=value 只能在 yaml 之上覆盖。正确姿势：`chat infer.yaml adapter_name_or_path=saves/...`。

### 坑 6：`cal_ppl.py` 不支持量化 → 8B 纯基座 PPL 跑不了 — 8B

`scripts/stat_utils/cal_ppl.py` 的 `get_train_args` 用**固定 dict**（无 `quantization_bit`），`load_model` 全精度加载。8B bf16 推理 ≈15.5GB + 激活，在 16GB 卡上 **OOM**。
解决：放弃独立基座 PPL，**用冒烟测试（step3）的 eval PPL 当「训练初期」基线**（与训练同源、可复现）。4B 因放得下，cal_ppl 仍可选。

> 另：训练日志经 `tee` 管道是**块缓冲**，实时看不到 loss——用 `nvidia-smi`（util + 显存）确认在跑，训练后读 `trainer_state.json` 取完整曲线。

---

## 6. 冒烟 → 正式训练

### 6.1 4B（35 分钟）

- **冒烟**（3 步，`max_samples=80 max_steps=3`）：141 秒，loss 2.87→2.85，验证 **PPL 18.49（基线）**，管线全通。
- **正式**（45 步）：~47 秒/步，GPU 15.7GB / 99%，35:24 完成；checkpoint-30/40/45 落盘。

### 6.2 8B QLoRA（47 分钟）

- **冒烟**（3 步）：186 秒（~62 秒/步），loss 2.72→2.71，验证 **PPL 15.47（基线）**，显存 ~14.3GB，无 OOM。
- **正式**（45 步）：99% util 全程，14.3GB，2841s（47.4 分钟），grad_norm 0.15~0.62 **全程稳定无爆炸**（`upcast_layernorm` 警告实测可忽略）；checkpoint-30/40/45 落盘。

---

## 7. 效果评估

### 7.1 困惑度（PPL）：训练初期 → 训练后

训练内评（与训练同源、可复现，「初期」取自冒烟 step3）：

| 指标 | 4B | 8B QLoRA |
|------|----|----------|
| 训练初期 PPL | 18.49 | 15.47（基座更强） |
| 训练后 PPL（45 步） | 12.56 | **11.36（更优）** |
| 降幅 | ↓32% | ↓27%（起点低、空间小） |
| eval loss 平台 | step30 后 ≈2.53 | step30 后 11.56→11.37 |

8B eval PPL 下降曲线（证平台）：step10 **13.16** → 20 **12.15** → 30 **11.56** → 40 **11.37** → 45 **11.36**。

### 7.2 续写对照（定性 + 诚实结论）— 8B

用 `llamafactory-cli chat`（独立 infer yaml，§5 坑 4 的正确姿势），喂**留出验证集**前缀让它续写：

```
prefix: "Multimodal memory" is a crucial concept that describes how an agent handles non-textual
```

- **base**：结构化教科书式回答（「1. Data Integration / 2. Memory Storage / 3. Retrieval and Reasoning」），通用、正确。
- **adapter**：框架明显贴合语料话语——「integrate, store, and retrieve information across multiple modalities」「memory bank」，更靠近训练语料的措辞与概念组织。

**诚实结论**：chat 模板会把前缀当成「提问」，两个模型都倾向于**回答**而非纯续写；adapter 与 base 的差异是**措辞/风格层面的偏移**（adapter 更贴领域语料），而非「突然会答领域题」。**PT 改善的是领域文本的预测分布（PPL 可量化：15.47→11.36），不显著改变问答行为**——要「能回答领域问题」，链路仍是 **PT → SFT**。

### 7.3 训练动态

- 4B：训练损失 step1 **2.84** → step45 **2.49**，均值 2.58；eval loss step30 后进平台。
- 8B：训练损失 step1 **2.669** → step45 **2.399**，均值 2.483；grad_norm warmup 期略升（~0.6）后稳定回落到 ~0.15，无爆炸/NaN。
- 两者 LoRA 都只动 **0.27%~0.41%** 的参数。

---

## 8. 4B vs 8B 对比（核心 takeaway）

| 维度 | 4B（bf16 LoRA） | 8B（4-bit QLoRA） |
|------|-----------------|-------------------|
| 基线 PPL | 18.49 | 15.47（基座更强） |
| 终点 PPL | 12.56 | **11.36（更优）** |
| PPL 降幅 | ↓32% | ↓27%（起点低、空间小） |
| 每步 / 总耗时 | 47s / 35min | 62s / 47min |
| 显存 | 15.7GB（近顶满） | **14.3GB（量化反更省）** |
| 可训练参数 | 16.5M / 0.41% | 21.8M / 0.27% |

**结论**：16GB 卡上，8B QLoRA 比 4B bf16 **既更强（PPL 更低）又更省显存**，代价是训练慢约 35%。量化是「在消费级显卡上把模型规格顶上去」的关键杠杆。

---

## 9. 学习要点（预训练 + 量化）

1. **预训练 = 下一个 token 预测**：纯文本切成 token 序列，每位置预测下一个 token，交叉熵算 loss 反传。LF 的 `stage: pt` 就是包装好的这个流程（`DataCollatorForLanguageModeling(mlm=False)`，见 `src/llamafactory/train/pt/workflow.py`）。
2. **从零 vs 继续预训练**：从零需 TB 级数据 + 大算力；消费级显卡做的是「继续预训练 / 领域适配」——基座已有通用语言能力，用领域语料 + LoRA 让它更懂你的领域。
3. **PT 灌「知识 + 风格」，不是「问答能力」**：续写对照实证（§7.2）——模型更懂领域文本、能接续术语与文风，但直接问未必好好回答。要「能回答」，链路是 **PT → SFT**。
4. **LoRA 为什么行**：冻结基座，只训少量低秩矩阵（本次 0.27%~0.41%），显存和时间大幅下降，效果对领域适配足够。
5. **PPL 怎么读**：PPL = e^(平均交叉熵)，越低 = 对这段文本预测越准。口径一致才能比（是否加 BOS、是否 packing 都会造成小差异）。
6. **小语料别贪 epochs**：验证 PPL 进平台就停（本次 step30 后），继续训只会过拟合 + 遗忘通用能力。
7. **QLoRA = 4-bit 量化基座 + LoRA**：量化基座只占零头显存，训练的只是少量低秩矩阵，让 8B 塞进 16GB——量化是消费级显卡顶规格的杠杆。
8. **新硬件先预检**：Blackwell sm_120 需 bnb≥0.43 + cu130；动手前一句 `get_device_capability()` + `bnb.__version__` 省掉训练崩。
9. **更大模型边际收益递减**：8B 起点 PPL 已更低，同等语料降幅（27%）小于 4B（32%）——语料不变时，越强的基座「被教化」空间越小。

---

## 10. 动手清单（复现 / 自跑）

> PowerShell 流程，逐步勾选；★ 为必查检查点。以 **8B** 为准，跑 4B 只需把 yaml 换成 `qwen3_4b_domain_pretrain.yaml`、去掉量化相关项。

**0. 前置确认**（已就绪，可选验证）
- [ ] 语料 `data/domain_papers.jsonl` / `_eval.jsonl` 在；训练 yaml `examples/train_lora/qwen3_{4,8}b_domain_pretrain.yaml`、续写 yaml `examples/inference/qwen3_8b_domain_chat.yaml` 在；`data/dataset_info.json` 含 `domain_papers` / `domain_papers_eval`
- [ ] 换语料才重跑 `python scripts\data\build_domain_corpus.py`（需带 pymupdf 的环境）

**1. 预检**（2 分钟，§5 坑 1/3）
- [ ] `conda activate llama-factory`
- [ ] `python -c "import torch,bitsandbytes as bnb; print('cap',torch.cuda.get_device_capability(0),'bnb',bnb.__version__)"` → 见 `(12, 0)` + `0.48.2`（4B 可跳过 bnb 检查）
- [ ] `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader` → 显存基本空闲

**2. 环境变量**（每个新会话都要，否则 datasets 卡死）
- [ ] `cd E:\AI\LLaMA-Factory`
- [ ] `$env:PYTHONUTF8='1'; $env:HF_HOME='E:\AI\LLaMA-Factory\hf_cache'; $env:HF_HUB_OFFLINE='1'`

**3. 冒烟 3 步 ★检查点**（~5 分钟）
- [ ] `llamafactory-cli train examples\train_lora\qwen3_8b_domain_pretrain.yaml max_samples=80 max_steps=3 output_dir=saves\Qwen3-8B-domain\lora\pt_smoke`
- [ ] 不报错、不 OOM；结束 PPL ≈ 15（基线）——**任一不对就停，别进第 4 步**

**4. 正式训练**（8B ~47 分钟 / 4B ~35 分钟）
- [ ] `llamafactory-cli train examples\train_lora\qwen3_8b_domain_pretrain.yaml`（`overwrite_output_dir:true` 覆盖旧产物）
- [ ] 另开窗口 `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 10` → 99% util + ~14GB（tee 块缓冲看不到 loss 属正常）

**5. 看结果**（§7）
- [ ] `Get-Content saves\Qwen3-8B-domain\lora\pt\all_results.json` → `eval_perplexity` ≈ 11.4
- [ ] 曲线 `training_eval_loss.png`；每步 `trainer_state.json`；停训信号 = 验证 PPL 进平台

**6. 续写对照**（base vs adapter，§5 坑 4/5）
- [ ] base：`llamafactory-cli chat examples\inference\qwen3_8b_domain_chat.yaml`
- [ ] adapter：`llamafactory-cli chat examples\inference\qwen3_8b_domain_chat.yaml adapter_name_or_path=saves\Qwen3-8B-domain\lora\pt`
- [ ] 喂同一句领域前缀对比续写

> PPL 用训练内评；独立基座 PPL 用 `cal_ppl.py`：4B 可跑（`--model_name_or_path model/Qwen3-4B --stage pt --dataset domain_papers_eval`），8B 会 OOM（§5 坑 6），不建议。

---

## 11. 产物文件清单

| 文件 | 作用 |
|------|------|
| `scripts/data/build_domain_corpus.py` | 语料构建脚本（PDF/md/html → jsonl 训练/验证集 + 统计） |
| `data/domain_papers.jsonl` / `domain_papers_eval.jsonl` | 训练 / 验证数据（两模型共用） |
| `data/domain_papers_stats.txt` | 语料统计（文档数 / 切块数 / 字符数） |
| `data/dataset_info.json`（新增条目） | 数据集注册（`domain_papers` / `domain_papers_eval`） |
| `examples/train_lora/qwen3_4b_domain_pretrain.yaml` | 4B 训练配置（bf16 LoRA） |
| `examples/train_lora/qwen3_8b_domain_pretrain.yaml` | 8B 训练配置（4-bit QLoRA） |
| `examples/inference/qwen3_8b_domain_chat.yaml` | 续写/推理配置（仅模型键，§5 坑 4） |
| `saves/Qwen3-4B-domain/lora/pt` | 4B 训练产物（adapter 66MB + checkpoint-30/40/45 + 曲线 + 指标） |
| `saves/Qwen3-8B-domain/lora/pt` | 8B 训练产物（adapter 87MB + checkpoint-30/40/45 + 曲线 + 指标） |
