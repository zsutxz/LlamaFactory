# LLaMA-Factory 学习指南（实操上手）

> 本文档记录在本机上手使用 LLaMA-Factory 的实操流程。
> 框架原理（目录结构 / 核心模块 / 配置 / 概念 / 代码位置）见 [frame.md](./frame.md)。

---

## 实操速查：启动环境与 WebUI（本机）

> 本节记录在本机（Windows + conda）实际跑通的流程，作为快速上手入口。

### 0.1 启动 conda 环境

环境名是 `llama-factory`（不是 `base`）。在 Git Bash 中激活：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llama-factory
```

激活后核对关键依赖（**`datasets` 必须在 3.0.0~3.6.0，否则 modelscope 会崩**）：

| 组件 | 本机版本 |
|-----|---------|
| Python | 3.11.14 |
| llamafactory | 0.9.6.dev0 |
| datasets | 3.6.0 ✅ |
| torch | 2.9.1+cu130，CUDA 可用 ✅ |

```bash
python --version
python -c "import datasets, torch; print(datasets.__version__, torch.__version__, torch.cuda.is_available())"
```

### 0.2 启动 WebUI（推荐新手）

```bash
llamafactory-cli webui
```

启动后在浏览器打开：**http://localhost:7860**

> **本机实测提示**
> - 用 `conda run -n llama-factory llamafactory-cli ...` 首次冷启动会很慢（cli 要重导入 torch/transformers/gradio，实测 `--help` 都能卡 2 分钟以上）。
> - 更快的方式是直接调用环境内的可执行文件，绕过 `conda run`：
>   ```bash
>   "C:/Users/skype/.conda/envs/llama-factory/Scripts/llamafactory-cli.exe" webui
>   ```
>   实测约 5s 即可监听 7860 端口。
> - WebUI 是常驻服务，启动后保持运行；`Ctrl+C` 停止。
> - Claude Code 中每条命令是独立子进程，`conda activate` 不会跨命令保留——后续训练/推理命令需用 `conda run -n llama-factory ...` 或直接用上面的 `.exe`。

### 0.3 加载训练好的 LoRA 进行聊天

配置文件 `examples/inference/qwen3_lora_chat.yaml`（加载 `train_2026-07-27-10-11-28` 这个 adapter）：

```yaml
model_name_or_path: model/Qwen3-4B
adapter_name_or_path: saves/Qwen3-4B-Thinking/lora/train_2026-07-27-10-11-28
template: qwen3
enable_thinking: true          # 与训练一致，开启思维链（关闭改 qwen3_nothink 并删此行）
infer_backend: huggingface
trust_remote_code: true
```

启动（交互式，需在自己的终端跑；Claude Code 非交互，跑不了多轮 REPL）：

```bash
# Git Bash（推荐）
conda activate llama-factory
PYTHONUTF8=1 llamafactory-cli chat examples/inference/qwen3_lora_chat.yaml
```

```powershell
# PowerShell
conda activate llama-factory
$env:PYTHONUTF8=1
llamafactory-cli chat examples/inference\qwen3_lora_chat.yaml
```

> ⚠️ **中文输入必须用 UTF-8 终端（`PYTHONUTF8=1`）**，否则 tokenizer 会崩。本机实测踩坑：

| 现象 | 误判方向 | 真正根因 |
|-----|---------|---------|
| `TypeError: TextEncodeInput...` | fast tokenizer async 竞态 | ❌ |
| `UnicodeEncodeError '\udcaa' surrogates` | slow tokenizer bug | ❌ |
| 实际 | — | ✅ 非 UTF-8 stdin 把中文损坏成孤立代理项，tokenizers 拒绝 |

换 fast/slow tokenizer、调 `enable_thinking` 都没用——`PYTHONUTF8=1` 才是正解。

> **关于「继续训练」**：原训练已完成（6554/6554 步、1.0 epoch），不存在"训练到一半"。若想**接着练**（增量）：用 `adapter_name_or_path` 加载已有 adapter + **新 output_dir**（否则会触发 auto-resume 到已完成的 checkpoint 立即结束）；若训练**真被中断**：把 output_dir 指向那个未完成目录即可自动 resume。逻辑见 `src/llamafactory/hparams/parser.py:484-500`。

### 0.4 生成式评估（在数据集上看模型生成什么）

配置 `examples/extras/nlg_eval/qwen3_4b_thinking_predict.yaml`（`do_predict: true` + `predict_with_generate: true`；adapter 已指向新训练 `saves/Qwen3-4B-Thinking/lora/sft`，`max_samples: 200`）。

```powershell
llamafactory-cli train examples/extras/nlg_eval/qwen3_4b_thinking_predict.yaml
```

产出在 `saves/Qwen3-4B-Thinking/lora/predict/`：

- `generated_predictions.jsonl` — 每行 `label` vs `predict`，**人眼对比最直观**，比 BLEU 分更有参考价值
- `predict_results.json` — BLEU/ROUGE 指标（中文场景分值普遍偏低，别被低分吓到）

> **本机实测提示（0.9.6 版本坑）**
> - `llamafactory-cli eval` **已废弃**（`launcher.py:147` 直接 raise `NotImplementedError`）。所有评估/预测统一用 `train` 命令 + 配置里 `do_predict: true`，走 `sft/workflow.py:152` 的预测分支，效果等同旧 `eval`。配置文件本身不用改，只换子命令。
> - 命令行覆盖 YAML 配置用 **OmegaConf 语法 `key=value`**，不是 `--key value`（`parser.py:91` 用 `OmegaConf.from_cli`）。例：`... cfg.yaml max_samples=500`。用 `--key value` 会报 `ValueError: Some keys are not used by the HfArgumentParser`。
> - 最稳的做法：直接改配置文件值，再跑**无参数**命令（上面的命令就是把 adapter 路径写进文件第 8 行后直接跑）。

### 0.5 合并发布（把 LoRA 烧进基座，导出完整模型）

评估满意后，把 LoRA adapter 合并进基座，导出一个**自包含的完整模型**（可脱离基座单独部署）。配置 `examples/merge_lora/qwen3_4b_thinking_export.yaml`：

```powershell
llamafactory-cli export examples/merge_lora/qwen3_4b_thinking_export.yaml
```

导出到 `saves/Qwen3-4B-Thinking/merged/`（Qwen3-4B bf16 约 8GB，按 `export_size: 5` 分成 2 个 safetensors 分片）。

> **执行要点**
> - `export` 子命令在 0.9.6 **仍可用**（不像 `eval` 被废弃）。它读 `export_dir` / `export_size` / `export_device` / `export_legacy_format` 等参数。
> - `export_device: cpu` 慢但稳（不吃 GPU 显存）；`auto` 用 GPU 快但占显存。本机用 `cpu` 即可。
> - `export_legacy_format: false` 导出 safetensors（推荐）；`true` 导出老的 `pytorch_model.bin`。
> - ⚠️ **`adapter_name_or_path` 要指向含 `adapter_config.json` 的目录**——即训练 output_dir 的**根目录**，**不要**指向 `checkpoint-XXX` 子目录（子目录会被 `save_total_limit` 清理；本机 `checkpoint-6554` 就已不存在，曾导致 export 报 `Can't find 'adapter_config.json'`）。本配置现指向 `saves/Qwen3-4B-Thinking/lora/sft`（与 0.4 评估同一个新训练）。要合并**旧的完整训练**就改成 `saves/Qwen3-4B-Thinking/lora/train_2026-07-27-10-11-28`，或命令行覆盖：`... qwen3_4b_thinking_export.yaml adapter_name_or_path=saves/Qwen3-4B-Thinking/lora/train_2026-07-27-10-11-28`（用 `key=value`，非 `--key`）。
> - ⚠️ **合并时基座不能是量化模型 / 不能设 `quantization_bit`**（合并要求全精度基座）。
> - 合并耗内存 ≈ 模型体积（约 8~16GB 内存），耗时几分钟到十几分钟。

### 0.6 用合并后的模型做推理（验证 merged 能独立跑）

合并的意义就是 `merged/` 是个**自包含的完整模型**——不再需要基座、也不需要 adapter。配置 `examples/inference/qwen3_merged_chat.yaml`（与 0.3 的区别：`model_name_or_path` 直接指向 `merged/`，**没有** `adapter_name_or_path`）：

```yaml
model_name_or_path: saves/Qwen3-4B-Thinking/merged
template: qwen3
enable_thinking: true
infer_backend: huggingface
trust_remote_code: true
```

```powershell
$env:PYTHONUTF8=1
llamafactory-cli chat examples/inference/qwen3_merged_chat.yaml
```

> **与 0.3（LoRA chat）的区别**
> - 0.3 加载「基座 + adapter」（运行时合并）；0.6 直接加载**已合并的完整模型**。两者推理结果应一致，但 0.6 部署更简单——只搬 `merged/` 一个目录即可，不用带基座。
> - 中文输入同样必须 `PYTHONUTF8=1`（见 0.3 的坑）。
> - `merged/` 可直接用 **vLLM / Ollama**（目录里已生成 `Modelfile`）/ transformers 加载部署，不必再走 LLaMA-Factory。

---

## 一、快速开始

### 1.1 安装

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

### 1.2 准备数据

在 `data/dataset_info.json` 中添加：

```json
{
  "my_data": {
    "file_name": "my_data.json",
    "columns": {
      "prompt": "instruction",
      "response": "output"
    }
  }
}
```

数据文件 `data/my_data.json`:

```json
[
  {"instruction": "你好", "output": "你好！有什么可以帮助你的？"},
  {"instruction": "介绍一下Python", "output": "Python是一种高级编程语言..."}
]
```

### 1.3 配置训练

复制示例配置并修改：

```bash
cp examples/train_lora/llama3_lora_sft.yaml my_config.yaml
```

### 1.4 开始训练

```bash
llamafactory-cli train my_config.yaml
```

### 1.5 启动WebUI（推荐新手）

```bash
llamafactory-cli webui
```

然后在浏览器打开 http://localhost:7860

---

## 二、学习路径建议

### 第一阶段：熟悉基本操作
1. 安装项目
2. 使用WebUI完成一次LoRA微调
3. 理解配置文件各参数含义

### 第二阶段：理解核心流程
1. 阅读 `launcher.py` 理解命令路由
2. 阅读 `data/loader.py` 理解数据加载
3. 阅读 `model/loader.py` 理解模型加载

### 第三阶段：深入定制
1. 学习如何添加新模型（注册到 `constants.py`）
2. 学习如何添加新模板（修改 `template.py`）
3. 学习如何自定义数据处理器

### 第四阶段：高级应用
1. 多模态模型微调
2. 分布式训练
3. 模型评估和对比

---

## 三、常见问题

### Q1: 如何选择训练方法？
- **显存充足**：Full或LoRA
- **显存有限**：QLoRA（4-bit或8-bit）
- **快速实验**：LoRA + 小rank

### Q2: 如何添加新模型？
参考 `src/llamafactory/extras/constants.py:153-167` 的 `register_model_group()` 函数

### Q3: 训练不收敛怎么办？
- 检查学习率（通常1e-4到5e-5）
- 检查数据质量
- 尝试增大warmup_ratio
- 检查loss曲线（启用plot_loss）

### Q4: 如何导出模型？
```bash
llamafactory-cli export export_config.yaml
```

---

## 四、资源链接

- **官方文档**: https://github.com/hiyouga/LLaMA-Factory
- **配置示例**: `examples/` 目录
- **示例数据**: `data/` 目录
- **框架原理**: [frame.md](./frame.md)

---

## 五、微调参数详解

> 以 `examples/train_lora/qwen3_4b_thinking_lora_sft.yaml`（你实际跑的 Qwen3-4B + LoRA + Thinking 配置）为蓝本，按配置文件的分块逐段解说。参数定义源码在 `src/llamafactory/hparams/`（`model_args.py` / `finetuning_args.py` / `data_args.py` / `training_args.py`）。

### 5.1 模型相关（### model）

| 参数 | 本配置值 | 含义 |
|-----|---------|------|
| `model_name_or_path` | `model/Qwen3-4B` | 基座模型。HuggingFace hub id 或本地目录 |
| `trust_remote_code` | `true` | 信任远程代码（自定义/新架构模型需要） |
| `template` | `qwen3` | 提示模板，决定 prompt 怎么拼。**必须与基座匹配**；Qwen3 想关思维链改 `qwen3_nothink` |
| `enable_thinking` | `true` | 开思维链（推理模型）。**训练与推理必须一致**，否则效果会坏 |

> 💡 `template` + `enable_thinking` 是 LLaMA-Factory 特有的「ReasoningTemplate」机制：模板控制特殊 token（如 `<think>`）的注入。换基座时这两个一定要跟着改。

### 5.2 训练方法（### method）

| 参数 | 本配置值 | 含义 |
|-----|---------|------|
| `stage` | `sft` | 训练阶段：`pt`(预训练) / `sft`(监督微调) / `rm`(奖励模型) / `ppo` / `dpo` / `kto` 等 |
| `do_train` | `true` | 跑训练（评估/预测则用 `do_predict: true`） |
| `finetuning_type` | `lora` | 微调方式：`full`(全参) / `freeze`(冻结大部分) / `lora`(LoRA 及其变体) |
| `lora_rank` | `8` | LoRA 秩 r，即低秩矩阵的「内在维度」。越大→表达能力越强、可训参数越多、显存越多。常用 8/16/32/64 |
| `lora_alpha` | `16` | LoRA 缩放因子，**默认 = rank × 2**。实际缩放倍率 = `alpha / rank` = 16/8 = 2。调大 alpha ≈ 放大学习率的效果 |
| `lora_dropout` | `0` | LoRA 层的 dropout，防过拟合。小数据集可设 `0.05`~`0.1` |
| `lora_target` | `all` | LoRA 挂载到哪些模块。`all` = 所有线性层（q/k/v/o/gate/up/down）。最省心；显存吃紧可只挂 q/v |

### 5.3 数据（### dataset）

| 参数 | 本配置值 | 含义 |
|-----|---------|------|
| `dataset` | `alpaca_zh_demo,...` | 训练集名称（须先在 `data/dataset_info.json` 注册），逗号分隔可混合多个数据集 |
| `dataset_dir` | `data` | 数据集所在目录 |
| `cutoff_len` | `2048` | 单条样本 token 截断长度（默认 2048）。越长→显存越多、越慢；思维链较长时需要放宽 |
| `max_samples` | `100000` | 每个数据集最多取多少条（**调试专用**）。smoke test 改 `2000~5000` 先跑通流程 |
| `preprocessing_num_workers` | `16` | 分词预处理并行进程数 |
| `packing` | `false` | 序列打包：`true` 把多条短样本拼进一个 `cutoff_len` 窗口，提吞吐但会让样本边界混叠。**SFT 一般 false；pt 阶段自动 true** |

> 💡 `cutoff_len` 也是模型的 `model_max_length`（见 `parser.py:518`）——它同时决定显存上限和模型能处理的最大上下文。

### 5.4 输出与存盘（### output）

| 参数 | 本配置值 | 含义 |
|-----|---------|------|
| `output_dir` | `saves/.../sft` | 输出目录。**CLI 直接写入此目录（不加时间戳）**；带 `train_时间戳` 子目录是 WebUI 的行为。重训前务必换 `output_dir`，否则会覆盖上次产物 |
| `logging_steps` | `5` | 每 5 步打印一次训练 loss |
| `save_steps` | `100` | 每 100 步存一个 checkpoint |
| `save_total_limit` | `3` | 只保留最近 3 个 checkpoint，防止磁盘堆满（你原训练没设这个，存了 65 个） |
| `plot_loss` | `true` | 训练结束自动画 loss 曲线（输出 `training_loss.png`） |
| `report_to` | `none` | 不上报到 wandb / tensorboard 等外部平台 |

### 5.5 训练超参（### train）

| 参数 | 本配置值 | 含义 |
|-----|---------|------|
| `per_device_train_batch_size` | `2` | 每卡每步的样本数。直接受显存限制 |
| `gradient_accumulation_steps` | `8` | 梯度累积步数。**有效 batch = 2 × 8 = 16**（显存不够时用累积换大 batch） |
| `learning_rate` | `5.0e-05` | 学习率。4B + LoRA 建议 `5e-5 ~ 1e-4`（你比默认 `1e-4` 减半，更稳） |
| `num_train_epochs` | `1.0` | 训练轮数。SFT 通常 1~3 epoch，过多易过拟合 |
| `lr_scheduler_type` | `cosine` | 学习率调度：`cosine`(余弦退火) / `linear` / `constant` 等 |
| `warmup_steps` | `0` | 学习率预热步数（从 0 线性升到 lr）。数据少时可设几十~几百步 |
| `max_grad_norm` | `1.0` | 梯度裁剪阈值，防止梯度爆炸 |
| `bf16` | `true` | bf16 混合精度。Ampere 及以上 GPU 用 `bf16`；旧卡（如 20 系）改 `fp16: true` |
| `flash_attn` | `auto` | 注意力实现：`auto` / `disabled` / `sdpa` / `fa2` / `fa3`。`auto` 自动选当前环境最快的 |
| `optim` | `adamw_torch` | 优化器 |
| `seed` | `42` | 随机种子，保证可复现 |
| `include_num_input_tokens_seen` | `true` | 日志里累计「已见 token 数」，方便看真实吞吐 |
| `resume_from_checkpoint` | （注释） | **断点续训**：取消注释并指向某 checkpoint 路径即可从断点接着跑。续训时数据规模须与原训练一致 |

### 5.6 评估（### eval，本配置默认注释掉）

| 参数 | 示例值 | 含义 |
|-----|---------|------|
| `val_size` | `0.1` | 从训练集切 10% 做验证集（0~1 之间） |
| `eval_strategy` | `steps` | 评估节奏：`steps` / `epoch` / `no` |
| `eval_steps` | `500` | 每 500 步评估一次（配合 `eval_strategy: steps`） |

> ⚠️ 开验证集会**减少训练数据**并增加评估开销。小数据集/快速实验可不开；正式训练建议开，配合 `plot_loss` 一起看是否过拟合。

### 5.7 按需求调参速查

| 目标 | 调什么 |
|-----|--------|
| 显存不够（OOM） | 降 `per_device_train_batch_size` → 降 `cutoff_len` → 降 `lora_rank` → 换 QLoRA（`quantization_bit: 4`） |
| 想训得更好（欠拟合） | 适当加 `num_train_epochs` / 调大 `learning_rate` / 加 `lora_rank` |
| 防过拟合（loss 后期回升） | 加 `lora_dropout` / 减 `num_train_epochs` / 开验证集早停 |
| 想跑快点验证流程 | `max_samples: 2000` + `num_train_epochs: 0.1` |
| 训练中途断了 | 设 `resume_from_checkpoint` 指向最近的 checkpoint |

---

*本文档持续更新中...*
