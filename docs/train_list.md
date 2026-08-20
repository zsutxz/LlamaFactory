# 领域模型实操记录：PT（4B/8B/9B）→ 蒸馏闭环

> 日期：2026-08-03 ~ 2026-08-08（4B：08-03~04；8B QLoRA：08-07~08）；2026-08-19（蒸馏闭环）
> 目标：把本机「专业文档 + 论文」语料灌进 Qwen3，跑通「语料构建 → 继续预训练(PT) → 效果评估」全流程；先 4B 起步，再用 4-bit QLoRA 把规格顶到 8B，对比代价与收益；后扩展 9B（环保语料）补上 SFT 环，跑通蒸馏闭环（§12）。
> 导航：全流程上手见 [learning.md](./learning.md)；框架原理见 [frame.md](./frame.md)。

---

## 脚本与工具约定

> 原则：**只有「数据处理」（清洗 / 蒸馏造数）才自写脚本，统一放 `.claude/skills/public-data-pipeline/scripts/`；训练、评估、推理等其他流程一律用 LLaMA-Factory 原生能力。**

| 流程 | 用什么 | 说明 |
|------|--------|------|
| 数据清洗 | `.claude/skills/public-data-pipeline/scripts/build_domain_corpus.py` | 语料构建（PDF/md/html → jsonl） |
| 蒸馏造数 | `.claude/skills/public-data-pipeline/scripts/generate_domain_qa.py` | DeepSeek 从语料块生成锚定原文的 QA 对（§12） |
| 裁判过筛 | `.claude/skills/public-data-pipeline/scripts/judge_domain_qa.py` | Kimi 三维评分三档分流 + PT/SFT 对比评分（§12） |
| 留出题评测 | `llamafactory-cli api` + `.claude/skills/public-data-pipeline/scripts/ask_compare.py` | 本地服务自动问答回填，免人工粘贴（§12 坑 7） |
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

### 3.2 构建脚本 `.claude/skills/public-data-pipeline/scripts/build_domain_corpus.py`

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
- [ ] 一次性永久（推荐，只执行一次）：把变量挂到 conda 环境，以后每次 `conda activate llama-factory` 自动带上，训练命令不再需要 `$env:` 前缀：

  ```powershell
  conda env config vars set LF_ALLOW_TORCH29_CONV3D=1 PYTHONUTF8=1 -n llama-factory
  conda deactivate
  conda activate llama-factory
  conda env config vars list    # 确认列出这两个变量
  ```

  - `LF_ALLOW_TORCH29_CONV3D` 仅 Qwen3.5-9B 需要：torch 2.9.x + 视觉模块 Conv3D 触发 `src/llamafactory/model/loader.py` 的防护闸（已知性能回归；纯文本 PT 不执行 Conv3D，放行安全，启动日志出现 bypass WARNING 属预期）
- [ ] 仍需每次手设（`HF_HUB_OFFLINE` 挂死会挡住联网下模型）：`$env:HF_HOME='E:\AI\LLaMA-Factory\hf_cache'; $env:HF_HUB_OFFLINE='1'`

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
| `.claude/skills/public-data-pipeline/scripts/build_domain_corpus.py` | 语料构建脚本（PDF/md/html → jsonl 训练/验证集 + 统计） |
| `data/domain_papers.jsonl` / `domain_papers_eval.jsonl` | 训练 / 验证数据（两模型共用） |
| `data/domain_papers_stats.txt` | 语料统计（文档数 / 切块数 / 字符数） |
| `data/dataset_info.json`（新增条目） | 数据集注册（`domain_papers` / `domain_papers_eval`） |
| `examples/train_lora/qwen3_4b_domain_pretrain.yaml` | 4B 训练配置（bf16 LoRA） |
| `examples/train_lora/qwen3_8b_domain_pretrain.yaml` | 8B 训练配置（4-bit QLoRA） |
| `examples/train_lora/qwen3_5_9b_domain_pretrain.yaml` | 9B 训练配置（4-bit QLoRA，Qwen3.5-9B-Base + 环保语料 domain_env） |
| `examples/inference/qwen3_8b_domain_chat.yaml` | 续写/推理配置（仅模型键，§5 坑 4） |
| `saves/Qwen3-4B-domain/lora/pt` | 4B 训练产物（adapter 66MB + checkpoint-30/40/45 + 曲线 + 指标） |
| `saves/Qwen3-8B-domain/lora/pt` | 8B 训练产物（adapter 87MB + checkpoint-30/40/45 + 曲线 + 指标） |
| `.claude/skills/public-data-pipeline/scripts/generate_domain_qa.py` | 蒸馏造数脚本（DeepSeek 出题 + quote 机械校验 + manifest 幂等，§12） |
| `.claude/skills/public-data-pipeline/scripts/judge_domain_qa.py` | 裁判脚本（Kimi 三维评分三档分流 / PT vs SFT 对比，§12） |
| `.claude/skills/public-data-pipeline/scripts/ask_compare.py` | 留出题自动问答回填（配合本地 api 服务，§12 坑 7） |
| `data/domain_env_qa{,_eval,_sft,_manifest,_judged,_compare}.jsonl` | 蒸馏数据全家桶（生成 → 过筛 → 对比；均不入库） |
| `data/domain_env_qa_review.md` / `_compare_report.md` | 裁判报告（兼人工抽检文档）/ PT vs SFT 对比报告 |
| `examples/train_lora/qwen3_5_9b_domain_pt_then_sft.yaml` | 9B PT→SFT 续训配置（蒸馏 QA，§12） |
| `examples/inference/qwen3_5_9b_domain_chat.yaml` | 9B 推理配置（仅模型键；key=value 切 pt / pt_then_sft adapter） |
| `saves/Qwen3.5-9B-domain-env/lora/pt_then_sft` | 9B PT→SFT 训练产物（§12） |

---

## 12. 蒸馏闭环（模式 D）：DeepSeek 出题 → Kimi 裁判 → PT→SFT 续训

> 背景：§1 已证 PT 只灌知识与文风、不会答领域问题，链路是 PT → SFT。本节用「通用大模型造数据 + 裁判过筛」
> 补上 SFT 这一环：DeepSeek 从环保语料出题（答案锚定原文），Kimi（kimi-k3）三维评分过筛，人工抽检后在 9B 的
> PT adapter 上续训。两家分工同时避开「自评偏差」（出题的不当裁判）。

### 12.1 管线

```
domain_env.jsonl 等距抽 60 块（固定梯子，冒烟=正式前缀）
  → DeepSeek 逐块生成 QA（question/answer/quote 三字段 JSON）
  → 机械校验：quote 去空白后必须存在于原文（失败带提示重试一次，仍败记 grounding_fail）
  → manifest 落盘（唯一事实源，sha1 幂等，中断重跑不重复扣费）
  → Kimi 三维评分（grounding/terminology/value 各 1~5）
  → 三档分流：pass（≥4.0 且三维各 ≥3）/ review（人工复核）/ drop（均分 <3 或 grounding ≤2）
  → 人工抽检报告，review 条目合格用 --promote 改判
  → domain_env_qa_sft.jsonl 注册为 alpaca 数据集
  → qwen3_5_9b_domain_pt_then_sft.yaml 在 PT adapter 上 SFT（~18 步）
  → 留出 10 题（训练从未见过）：pt / pt_then_sft 两 adapter 各答一遍 → Kimi 对比评分出报告
```

### 12.2 命令（conda env llama-factory；脚本另需 PYTHONUTF8=1）

```bash
# ① 蒸馏造数（先冒烟 3 块人工看质量、调 prompt，再正式全量；60 次调用约 0.5 元 / 10~20 分钟）
python .claude/skills/public-data-pipeline/scripts/generate_domain_qa.py --num 3 --eval-num 0   # 冒烟（是正式集的严格前缀）
python .claude/skills/public-data-pipeline/scripts/generate_domain_qa.py                          # 正式 50 训练 + 10 留出

# ② 裁判过筛（先 --limit 3 看分数是否非全同，再全量）
python .claude/skills/public-data-pipeline/scripts/judge_domain_qa.py --limit 3
python .claude/skills/public-data-pipeline/scripts/judge_domain_qa.py
python .claude/skills/public-data-pipeline/scripts/judge_domain_qa.py --promote "12,27"          # 人工复核改判（sft 集自动重写）

# ③ SFT（训练/推理命令都带四项环境变量，9B 必须 LF_ALLOW_TORCH29_CONV3D=1）
LF_ALLOW_TORCH29_CONV3D=1 PYTHONUTF8=1 HF_HOME=E:\AI\LLaMA-Factory\hf_cache HF_HUB_OFFLINE=1 \
  llamafactory-cli train examples/train_lora/qwen3_5_9b_domain_pt_then_sft.yaml max_steps=3   # 冒烟
# 正式：去掉 max_steps=3（约 18 步，几分钟）

# ④ 留出 10 题对比（全自动：本地 api 服务自动问答，免人工 chat 粘贴）
python .claude/skills/public-data-pipeline/scripts/judge_domain_qa.py --mode init-compare        # 生成对比骨架
# 起 PT 服务 → ask_compare 回填 answer_pt → 停服务；换 pt_then_sft adapter 重启 → 回填 answer_sft
LF_ALLOW_TORCH29_CONV3D=1 PYTHONUTF8=1 HF_HOME=... HF_HUB_OFFLINE=1 API_HOST=127.0.0.1 API_PORT=8000 \
  llamafactory-cli api examples/inference/qwen3_5_9b_domain_chat.yaml \
  adapter_name_or_path=saves/Qwen3.5-9B-domain-env/lora/pt
PYTHONUTF8=1 python .claude/skills/public-data-pipeline/scripts/ask_compare.py --field answer_pt
PYTHONUTF8=1 python .claude/skills/public-data-pipeline/scripts/ask_compare.py --field answer_sft   # 换 pt_then_sft adapter 重启服务后
python .claude/skills/public-data-pipeline/scripts/judge_domain_qa.py --mode compare             # Kimi 对比评分 → _compare_report.md
```

API 钥匙放项目根 `.env`（已被 gitignore）：`DEEPSEEK_API_KEY`（出题）/ `MOONSHOT_API_KEY`（裁判）。

### 12.3 机制要点

- **采样梯子**：语料构建时已 seed(42) 洗牌，文件序即随机序；等距取 60 级 = 随机 + 全库均匀覆盖 + 确定，
  冒烟 3 块必然 ⊂ 正式 60 块，manifest 按 block_sha1 幂等跳过，冒烟产物不作废、不重复扣费。
- **manifest 即事实源**：每行内嵌原文全文与 QA 三字段，输出 jsonl 每次由 manifest 全量重写——
  避免「写一半崩溃导致 jsonl 与 manifest 失步」的去重难题。
- **三档而非一刀切**：边缘条目（review）不静默丢弃，进报告人工复核，`--promote` 改判重跑即入 sft 集。
- **对比评分防位置偏差**：compare 模式奇偶行交换 A/B 呈现顺序，评分后映射回 pt/sft。
- **SFT 机制**：同 8B 版（`adapter_name_or_path` resume 同一 adapter；量化基座单 adapter；
  rank/alpha/target 必须与 PT 一致；lr 5e-5 比 PT 降一档）。qwen3_5 是 ReasoningTemplate，
  SFT 空 think 块计入 loss，推理用 `enable_thinking: false` 对齐。

### 12.4 诚实定位与预期

- 50 条蒸馏 QA 是**管线验证批**而非效果批：~18 步 SFT 改变的是**回答格式、直接作答+引原文的习惯、术语使用**，
  知识增量主要来自 PT——评测预期设为「SFT 在格式/术语/忠实度上占优」，别设「正确率大涨」（会翻车误判）。
- 裁判同源局限：参考答案出自 DeepSeek、裁判是 Kimi 已避开自评，但裁判可能偏好某种行文（`--model` 可换）。

### 12.5 实测结果（2026-08-19，管线验证批）

| 环节 | 结果 |
|------|------|
| 蒸馏生成 | 60 块 → 55 ok（92%，重试一轮补 4 块；5 块引用顽固失败）|
| Kimi 裁判 | pass 52 / review 3 / drop 0；g 4.98 / t 5.00 / value 3.36（value 是真区分维度）|
| 人工复核 | promote idx=202（框架题），431（页脚备案号）剔除；重跑后 sft 集 42 条 |
| SFT | 18 步 / 283s，loss 1.91 → 0.84；显存同 PT（~14.3GB 量级）|
| 留出 10 题对比 | **PT 6 胜 / SFT 3 胜 / 平 1，双方绝对分均低（多数均分 1~3）** |

诚实结论：**闭环管线全部跑通，但 42 条验证批没有提升留出题的答题质量**——两个 adapter 都在
硬事实（日期/数值）上大量臆测，PT 靠"更长更啰嗦"偶然更接近原文；SFT 学到的是回答**格式**
（直答、简短，字数 14~390 vs PT 103~1077），但 42 条教不会知识。这与"管线验证批 vs 效果批"
的预判一致：下一步是放量蒸馏（300~500 条、答案加长到 150~300 字）再训，而不是回头怀疑管线。

### 12.6 新增坑

1. **DeepSeek JSON mode** 要求 prompt 里出现 "json" 字样，否则 400——出题模板已内置。
2. **第三方端点 response_format 兼容性不一**：裁判侧不用 JSON mode，统一宽松解析（剥围栏+截首{尾}）。
3. **quote 比对必须空白归一化**：语料 clean() 规整过空白，模型复述时常有微差，裸 substring 大量误杀。
4. **`.env.local` 被 git 追踪**，写密钥必泄漏；钥匙只进 `.env`（gitignore:123 已覆盖）。
5. **裁判模型踩坑实录**：智谱 glm-4.7/5.2 需账户余额(429 错误码 1113)、glm-4.5-flash 免费但是推理模型
   （max_tokens 会被 reasoning_content 烧光导致 content 为空，须 thinking disabled）；后裁判改 Kimi
   （`kimi-k3` + `https://api.moonshot.cn/v1`）。模型名/base-url 都留了 CLI 参数（`--model`/`--base-url`）。
6. **kimi-k3 只允许 temperature=1**：传 temperature=0 直接 400；且同为推理模型（思考会吃 tokens）。
   裁判脚本已改为"参数组合逐级降级"（temperature=0+thinking关 → 省略temperature → 裸调），max_tokens 提到 2048。
   temperature=1 意味着**裁判打分有随机性**：--promote 重跑全量时边缘条目（value 3↔4）会抖动，属预期。
7. **留出题自动评测可免人工**：`llamafactory-cli api <infer yaml> adapter_name_or_path=...` 起本地
   OpenAI 兼容服务（默认模型名 gpt-3.5-turbo），`.claude/skills/public-data-pipeline/scripts/ask_compare.py` 逐题问答回填，免 20 次手动粘贴。
