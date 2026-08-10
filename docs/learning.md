# LLaMA-Factory 学习指南（实操上手）

> 本文档记录在本机（Windows + conda）上手使用 LLaMA-Factory 的**完整实操流程**。
> 框架原理（目录结构 / 核心模块 / 配置 / 概念 / 代码位置）见 [frame.md](./frame.md)。

---

## 完整流程总览

```
安装 → 预训练(PT) → 监督微调(SFT) → 偏好对齐(DPO) → 推理 → 蒸馏 → 评估 → 导出
```

| # | 阶段 | LF stage / 命令 | 数据形态 | 实测 |
|---|------|----------------|---------|------|
| 0 | 安装与环境 | `pip install -e .` | — | ✅ |
| 1 | 预训练 PT | `stage: pt` | 纯文本 | ✅ 已实测 |
| 2 | 监督微调 SFT | `stage: sft` | 指令-回答对 | ✅ 已实测 |
| 3 | 偏好对齐 DPO | `stage: dpo` | chosen/rejected 对 | ⚠️ 未实测 |
| 4 | 推理 | `llamafactory-cli chat` | — | ✅ 已实测 |
| 5 | 蒸馏 | **LF 无独立阶段** = 强模型生成数据→SFT | teacher 生成 | ⚠️ 未实测 |
| 6 | 评估 | `train` + `do_predict:true` | 同训练集 | ✅ 已实测 |
| 7 | 导出 | `llamafactory-cli export` | — | ✅ 已实测 |

> ⚠️ 标注「未实测」的章节基于框架配置 + 原理编写，命令和数据格式可信，但没有本机跑通的真机记录；动手前请自行 smoke test。

---

## 0. 安装与环境

### 0.1 安装 LLaMA-Factory

**前置**：Python ≥ 3.11（`pyproject.toml` 硬性要求）。

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .                         # 核心
pip install -r requirements/metrics.txt  # 评估指标（BLEU/ROUGE 等，可选但建议装）
```

可选额外依赖：`metrics`、`deepspeed`（`pip install -r requirements/deepspeed.txt`），其余见 `examples/requirements/`。

**Windows 必装 GPU 版 PyTorch**（默认 pip 装的是 CPU 版，跑不了 CUDA）：

```bash
pip uninstall torch torchvision torchaudio
# 按本机 CUDA 版本装，参考 https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

**（可选）下载渠道**：用 ModelScope 不用翻墙时，`pip install -e .` 已含 `modelscope`；用 HuggingFace 需 `pip install --upgrade huggingface_hub && huggingface-cli login`。

### 0.2 本机环境核对

环境名是 `llama-factory`（不是 `base`）。Git Bash 激活：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llama-factory

python --version
python -c "import datasets, torch; print(datasets.__version__, torch.__version__, torch.cuda.is_available())"
```

本机实测版本：

| 组件 | 本机版本 |
|-----|---------|
| Python | 3.11.14 |
| llamafactory | 0.9.6.dev0 |
| datasets | 3.6.0 ✅ |
| torch | 2.9.1+cu130，CUDA 可用 ✅ |

> ⚠️ **`datasets` 必须在 3.0.0~3.6.0**，否则 modelscope 加载会崩。`pyproject.toml` 允许 `>=2.16,<=4.0`，但本机踩过坑——装新版本前先钉死。

> 💡 **加速 CLI 冷启动**：`conda run -n llama-factory llamafactory-cli ...` 首次会卡 2 分钟以上（重导入 torch/transformers/gradio）。直接调环境内 exe 约需 5 秒：
> `"C:/Users/skype/.conda/envs/llama-factory/Scripts/llamafactory-cli.exe" webui`
> 下文命令统一用激活环境后的 `llamafactory-cli`（最通用）。

### 0.3 启动 WebUI（可选，可视化操作）

```bash
llamafactory-cli webui    # 浏览器打开 http://localhost:7860
```

> WebUI 是常驻服务，`Ctrl+C` 停止。下面所有阶段都能在 WebUI 点出来，也可用命令行（本文用命令行，可复现）。

### 0.4 下载基座模型

LF 所有阶段（PT/SFT/DPO/推理/评估/导出）都需要一个**基座模型**。下到项目根的 `model/` 目录，配置里 `model_name_or_path` 填本地相对路径（如 `model/Qwen3-1.7B`），不依赖每次联网。本文涉及两个基座：

| 用途 | model id | 本地路径 |
|-----|----------|---------|
| 快速验证（小、跑得快） | `Qwen/Qwen3-1.7B` | `model/Qwen3-1.7B` |
| 各章配置蓝本默认 | `Qwen/Qwen3-4B` | `model/Qwen3-4B` |

下载任选其一，**国内首选 ModelScope**：

```bash
# 方式一：ModelScope（快，无需翻墙）
modelscope download --model Qwen/Qwen3-1.7B --local_dir model/Qwen3-1.7B

# 方式二：HuggingFace + 国内镜像
export HF_ENDPOINT=https://hf-mirror.com   # PowerShell: $env:HF_ENDPOINT='https://hf-mirror.com'
huggingface-cli download Qwen/Qwen3-1.7B --local-dir model/Qwen3-1.7B
```

> 💡 **懒人方式**：配置里直接写 hub id（`model_name_or_path: Qwen/Qwen3-1.7B`）+ 设 `export USE_MODELSCOPE=1`，LF 首次跑时自动下到缓存目录。缺点是路径散在缓存里、不便统一管理；正式训练建议显式下到 `model/`。

> ⚠️ 下载本身不踩 datasets 坑，但**别顺手升级 datasets**——本环境必须锁 `3.0.0~3.6.0`（见 0.2）。Qwen3-1.7B（bf16）约 3.4GB。

### 0.5 实测坑：加载模型时 HF 联网超时

用「懒人方式」（配置写 hub id + `HF_ENDPOINT` 镜像）时，模型下到 HF 默认缓存（`~/.cache/huggingface/hub/`）。**train/chat 启动加载阶段**常见这条：

```
'(ReadTimeoutError("HTTPSConnectionPool(host='hf-mirror.com', port=443): Read timed out. (read timeout=10)"), ...)' thrown while requesting HEAD https://hf-mirror.com/Qwen/Qwen3-1.7B/resolve/main/preprocessor_config.json
```

| 现象 | 根因 / 处理 |
|-----|------------|
| 上面那条 `thrown while requesting HEAD ...` | transformers 加载时默认联网发 HEAD 校验 ETag，hf-mirror 10s 超时。**这是 warning，不是错误，程序照常继续**——Qwen3 纯文本模型本就没有 `preprocessor_config.json`（多模态模型才有），缺它对加载无影响。 |
| 彻底消除超时 | 模型已在缓存，**命令前面带** `HF_HUB_OFFLINE=1` 即完全离线、不再联网（推荐，零副作用，只对这条命令生效）：`HF_HUB_OFFLINE=1 llamafactory-cli train xxx.yaml`；PowerShell 先 `$env:HF_HUB_OFFLINE=1` 再跑命令。 |
| 只加大容忍 | `HF_HUB_ETAG_TIMEOUT=30`（治标，镜像抖动仍会偶发）。 |
| 根本上避免 | 用 0.4 的显式方式把模型下到 `model/Qwen3-1.7B`，配置写**本地路径**而非 hub id，根本不触发 HF 联网逻辑。 |

> 💡 **判断真假崩溃**：真崩会带 `OSError` / traceback；只有 `'...' thrown while requesting HEAD` 这种措辞就是 warning，后面会正常继续（出现 `Map:` / `trainable params` / loss 等）。

---

## 1. 预训练 PT（继续预训练 / 知识注入）

> ✅ 已实测（Qwen3-4B + 领域论文/文档语料，45 步、5 epochs 跑通，详见 1.5）。
> 配置蓝本：`examples/train_lora/qwen3_4b_domain_pretrain.yaml`（官方示例 `examples/train_lora/qwen3_lora_pretrain.yaml`）。

**用途**：把**领域纯文本**（非问答对）灌进模型，做知识注入 / 领域适配。学的是「知识 + 语言风格」。

### 1.1 数据格式

PT 是无监督语言建模，只要**纯文本**。在 `data/dataset_info.json` 注册，关键是用 `columns.prompt = "text"` 映射：

```json
"my_corpus": {
  "file_name": "my_corpus.txt",
  "columns": { "prompt": "text" }
}
```

数据文件两种形式皆可（参照 `data/wiki_demo.txt`、`data/c4_demo.jsonl`）：

- **纯文本 `.txt`**：每段一条样本，段落间换行分隔。
- **`.jsonl`**：每行 `{"text": "一段领域文本..."}`。

### 1.2 配置（本机基座）

复制 `examples/train_lora/qwen3_lora_pretrain.yaml` 并把基座改成**本地路径** `model/Qwen3-4B`，数据集改成自己的：

```yaml
### model
model_name_or_path: model/Qwen3-4B      # 改成本地基座（原配置是线上 hub）
trust_remote_code: true

### method
stage: pt
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: my_corpus                      # 你在 dataset_info.json 注册的名字
cutoff_len: 2048
max_samples: 100000
preprocessing_num_workers: 16

### output
output_dir: saves/Qwen3-4B-Thinking/lora/pt
logging_steps: 10
save_steps: 500
save_total_limit: 3
plot_loss: true
overwrite_output_dir: true
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4                   # PT 学习率比 SFT 高（1e-4 ~ 5e-4）
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
```

> 💡 PT **不需要 `template`**（不走 chat 模板，raw text 直接喂）。`packing` 在 PT 阶段自动开启（多条短文本拼进 cutoff 窗口提吞吐）。

### 1.3 训练命令

```bash
llamafactory-cli train examples/train_lora/qwen3_pt.yaml   # 你的配置路径
```

### 1.4 ⚠️ 关键认知

**PT 灌进去的是「知识」，不是「问答能力」。** 做完 PT 模型能续写领域文本，但你直接问它问题，未必会好好回答。若目标是「能回答领域问题」，正确链路是 **PT（注入知识）→ SFT（教问答格式）**，两步缺一不可。

### 1.5 实测记录（Qwen3-4B + 领域论文/文档，2026-08-03）

**语料构建**（`scripts/data/build_domain_corpus.py`）：

- 来源：`E:\AI\Book`（AI演义 36 篇论文、提示词工程手册、Paper2Agent）+ `E:\AI\5-Day-AI-Agents-Intensive-Course-with-Google-2025`（6 份课程 PDF）+ `E:\AI\teach-fish-to-swim`（论文全文/笔记），共 **18 份文档**。
- 处理：PDF/md/html → 清洗（去链接/装饰符号/重复行）→ 按段落切成约 1800 字符的块 → 哈希去重 → 固定种子洗牌，**每 10 块留 1 块做验证**。
- 产出：`data/domain_papers.jsonl`（336 块，约 14.7 万 token）+ `data/domain_papers_eval.jsonl`（38 块，约 1.4 万 token），已注册到 `data/dataset_info.json`（`domain_papers` / `domain_papers_eval`，`columns.prompt = "text"`）。

**训练**（`examples/train_lora/qwen3_4b_domain_pretrain.yaml`）：

| 参数 | 值 | 说明 |
|------|----|------|
| 模型 | `model/Qwen3-4B` | 本地基座，LoRA rank 8 / target all |
| 窗口 | `cutoff_len: 2048`，PT 自动 packing | 多块短文本拼满窗口，提吞吐 |
| 有效 batch | 1 × 8 | 336 块 ≈ 72 个窗口 / 8 ≈ **9 步/epoch** |
| 学习率 | 1e-4，cosine，warmup 0.1 | PT 比 SFT（5e-5）高一档 |
| epochs | 5.0 | 语料小，多过几遍 |
| 时长 | 45 步 ≈ **35 分钟** | ~47 秒/步，16GB 显存占 15.7GB（几乎顶满） |

**结果**：

| 指标 | 训练前（基座） | 训练后（基座+LoRA） | 变化 |
|------|--------------|--------------------|------|
| 验证集 PPL（LLaMA-Factory 内评） | — | 12.56 | — |
| 验证集 PPL（早期 `eval_ppl.py` 测，脚本已删，改用 `stat_utils/cal_ppl.py`） | 18.80 | 12.82 | **↓ 32%** |
| 续写对比（早期 `continue_text.py` 测，脚本已删，改用 `llamafactory-cli chat`） | 臆造模型名 / 质疑工具不存在 | 流畅接续领域文风、术语准确 | 明显变好 |

**关键观察**：eval PPL 在 step 30 后进入平台期（约 2.53 不再降），说明 3~5 epochs 对小语料已足够，**用验证集 PPL 决定停训比拍脑袋设步数更科学**。

**本机两个必踩的坑**：

| 坑 | 现象 | 处理 |
|----|------|------|
| 默认 HF 缓存目录写不进 | `datasets` 卡死 15 分钟+（filelock 无限重试，因为 `C:\Users\skype\.cache\huggingface\datasets` 拒绝创建文件） | 命令前加 `$env:HF_HOME='E:\AI\LLaMA-Factory\hf_cache'`（仓库 `.gitignore` 已忽略 `hf_cache/`）。**所有**用 datasets 的 LF 命令都要带（完整训练命令见 `learning_train_list.md` 第 10 节（动手清单））。 |
| Windows 多进程预处理 | `preprocessing_num_workers: 0` 报 `ValueError`；`>1` 时 spawn 卡死 | 统一设 **1**（语料小，无性能损失） |

---

## 2. 监督微调 SFT（核心）

> ✅ 已实测（Qwen3-4B-Thinking，6554 步、1.0 epoch 跑通）。配置：`examples/train_lora/qwen3_4b_thinking_lora_sft.yaml`。

**用途**：用「指令-回答」对训练，让模型学会按特定格式/风格回答。最常用的微调阶段。

### 2.1 数据格式

**alpaca 格式**（最简单，单轮问答）：

```json
[
  {"instruction": "你好", "output": "你好！有什么可以帮助你的？"},
  {"instruction": "介绍一下Python", "input": "", "output": "Python是一种高级编程语言..."}
]
```

在 `data/dataset_info.json` 注册：

```json
"my_data": {
  "file_name": "my_data.json",
  "columns": { "prompt": "instruction", "query": "input", "response": "output" }
}
```

多轮对话用 **sharegpt 格式**（`"formatting": "sharegpt"`，`columns.messages = "conversations"`），详见 `data/dataset_info.json` 里的 `example` 条目。

### 2.2 配置（实测蓝本）

`examples/train_lora/qwen3_4b_thinking_lora_sft.yaml` 关键项：

| 分块 | 参数 | 值 | 说明 |
|-----|------|----|------|
| model | `model_name_or_path` | `model/Qwen3-4B` | 本地基座 |
| | `template` | `qwen3` | **必须与基座匹配**；关思维链改 `qwen3_nothink` |
| | `enable_thinking` | `true` | 开思维链。**训练与推理必须一致** |
| method | `stage` | `sft` | |
| | `finetuning_type` | `lora` | `full`/`freeze`/`lora` |
| | `lora_rank` / `lora_alpha` | `8` / `16` | 缩放倍率 = alpha/rank = 2 |
| | `lora_target` | `all` | 挂所有线性层（最省心） |
| dataset | `dataset` | `alpaca_zh_demo,...` | 已注册的集名，逗号分隔可混多个 |
| | `cutoff_len` | `2048` | 单条 token 截断，也即 model_max_length |
| | `max_samples` | `100000` | **调试专用**，smoke test 改 `2000` |
| train | `per_device_train_batch_size` × `gradient_accumulation_steps` | `2 × 8` | 有效 batch = 16 |
| | `learning_rate` | `5.0e-05` | 4B+LoRA 建议 `5e-5 ~ 1e-4` |
| | `num_train_epochs` | `1.0` | SFT 通常 1~3，过多易过拟合 |
| | `bf16` | `true` | Ampere+ 用 bf16；旧卡改 `fp16: true` |

### 2.3 训练命令

```bash
llamafactory-cli train examples/train_lora/qwen3_4b_thinking_lora_sft.yaml
```

### 2.4 实测坑

| 现象 | 根因 / 处理 |
|-----|------------|
| 想接着练（增量） | 用 `adapter_name_or_path` 加载已有 adapter + **新 output_dir**（否则 auto-resume 到已完成 checkpoint 立即结束） |
| 训练真被中断 | 把 output_dir 指向未完成目录即可自动 resume；逻辑见 `src/llamafactory/hparams/parser.py:484-500` |
| 显存不够（OOM） | 降 `per_device_train_batch_size` → 降 `cutoff_len` → 降 `lora_rank` → 换 QLoRA（`quantization_bit: 4`） |
| 想快速验证流程 | `max_samples: 2000` + `num_train_epochs: 0.1` |

---

## 3. 偏好对齐 DPO

> ⚠️ 未实测。配置蓝本：`examples/train_lora/qwen3_lora_dpo.yaml`。

**用途**：用「同一问题的好回答 vs 坏回答」对比训练，调模型的回答**风格 / 偏好**（更无害、更符合人类偏好）。**不做新知识注入**——前提是 SFT 已把能力训好。变体：KTO（`stage: kto`，只需好/坏标签不成对）。

### 3.1 数据格式（偏好数据）

sharegpt 格式 + `ranking: true` + `chosen`/`rejected` 字段。在 `data/dataset_info.json` 注册：

```json
"my_pref": {
  "file_name": "my_pref.json",
  "ranking": true,
  "formatting": "sharegpt",
  "columns": { "messages": "conversations", "chosen": "chosen", "rejected": "rejected" }
}
```

数据每条是一个问题 + 两个不同回答（chosen 好 / rejected 差）。参照 `data/dpo_en_demo.json`、`data/dpo_zh_demo.json`。

### 3.2 配置（本机基座）

复制 `examples/train_lora/qwen3_lora_dpo.yaml`，改基座为本地路径：

```yaml
### method
stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
pref_beta: 0.1                         # DPO β，约束偏离参考模型的幅度
pref_loss: sigmoid                      # choices: [sigmoid (dpo), orpo, simpo]

### dataset
dataset: dpo_zh_demo
template: qwen3_nothink                 # DPO 通常关思维链（按需）
cutoff_len: 2048
max_samples: 1000

### output / train 关键差异
output_dir: saves/Qwen3-4B-Thinking/lora/dpo
learning_rate: 5.0e-6                   # ⚠️ 比 SFT 低一个数量级！DPO 必须小 lr
num_train_epochs: 3.0
bf16: true
```

### 3.3 训练命令

```bash
llamafactory-cli train examples/train_lora/qwen3_dpo.yaml
```

> ⚠️ DPO 的 `learning_rate` 必须远小于 SFT（5e-6 级别），否则容易把模型训崩。DPO 通常加载 **SFT 后的 adapter** 作为起点（`adapter_name_or_path` 指向 SFT 产物）。

---

## 4. 推理（加载训练结果）

> ✅ 已实测。配置：`examples/inference/qwen3_lora_chat.yaml`、`examples/inference/qwen3_merged_chat.yaml`。

两种加载方式：

### 4.1 基座 + LoRA adapter（运行时合并）

`qwen3_lora_chat.yaml`：

```yaml
model_name_or_path: model/Qwen3-4B
adapter_name_or_path: saves/Qwen3-4B-Thinking/lora/train_2026-07-27-10-11-28
template: qwen3
enable_thinking: true          # 必须与训练一致
infer_backend: huggingface
trust_remote_code: true
```

### 4.2 直接加载合并后的完整模型

`qwen3_merged_chat.yaml`（无 `adapter_name_or_path`，`model_name_or_path` 直接指向 `merged/`）：

```yaml
model_name_or_path: saves/Qwen3-4B-Thinking/merged
template: qwen3
enable_thinking: true
infer_backend: huggingface
trust_remote_code: true
```

### 4.3 命令（交互式，在自己的终端跑）

```bash
# Git Bash
conda activate llama-factory
PYTHONUTF8=1 llamafactory-cli chat examples/inference/qwen3_lora_chat.yaml
```

```powershell
# PowerShell
conda activate llama-factory
$env:PYTHONUTF8=1
llamafactory-cli chat examples/inference\qwen3_merged_chat.yaml
```

> ⚠️ **中文输入必须 `PYTHONUTF8=1`**（非 UTF-8 终端会把中文损坏成孤立代理项，tokenizers 拒绝；换 fast/slow tokenizer、调 thinking 都没用）。

> Claude Code 非交互，跑不了多轮 REPL——交互式 chat 在自己的终端跑。程序化调用用 `llamafactory-cli api`（起 OpenAI 兼容 API 服务）或换 `infer_backend: vllm` 提吞吐。

---

## 5. 蒸馏（R1 思维链 → 小模型）

> ⚠️ 未实测。

### 5.1 关键认知：LLaMA-Factory 没有「蒸馏」阶段

LF 的 `stage` 只有 `pt / sft / rm / ppo / dpo / kto`，**没有 `distill`**。代码里搜到的 `Distill` 全是**模型名字**（DeepSeek-R1-XXB-Distill，别人蒸馏好的模型，LF 只是能加载），不是训练功能。

所以「蒸馏」在 LF 里的落地方式是 **数据蒸馏**：**用强模型（teacher）生成回答 → 转成 SFT 数据 → 训练弱模型（student）**。本质就是第 2 节的 SFT，只是数据来源不同。

### 5.2 R1 思维链蒸馏实操思路

把 DeepSeek-R1 这类推理模型的「思维链能力」蒸到 Qwen3-4B：

1. **准备 prompt 集**：收集一批你想要 student 学会的问题（数学、代码、推理等）。
2. **teacher 生成**：用 R1（teacher）对每个 prompt 生成带 `<think>...</think>` 思维链的回答。批量生成可用 `llamafactory-cli api` 起 R1 服务 + 脚本调用，或直接调 R1 的 API。
3. **转 SFT 格式**：把 `(prompt, R1 的思维链+回答)` 整理成 alpaca 格式：

   ```json
   [
     {"instruction": "<问题>", "output": "<think>...推理过程...</think>最终答案..."}
   ]
   ```
   注册到 `dataset_info.json`（同 2.1）。
4. **student 训练**：用第 2 节 SFT 配置训 Qwen3-4B，**`enable_thinking: true`**（让 student 学会输出思维链格式）。DeepSeek-R1-Distill-Qwen 系列就是这么来的。

> 💡 关键是**数据质量**：teacher 生成的思维链越准、越清晰，student 蒸出来的推理能力越强。这一步决定蒸馏效果，配置反而是次要的。

---

## 6. 评估（生成式 NLG 评估）

> ✅ 已实测。配置：`examples/extras/nlg_eval/qwen3_4b_thinking_predict.yaml`。

**用途**：让模型在数据集上实际生成回答，对比参考答案，算 BLEU/ROUGE。比训练 loss 更能反映真实生成质量。

### 6.1 配置要点

```yaml
stage: sft                              # 仍走 sft workflow
do_predict: true                        # ⚠️ 关键：预测而非训练
predict_with_generate: true             # ⚠️ 关键：真生成文本（否则只算 loss）
adapter_name_or_path: saves/Qwen3-4B-Thinking/lora/sft
dataset: <评估集>
max_samples: 200
template: qwen3
```

### 6.2 命令

```bash
llamafactory-cli train examples/extras/nlg_eval/qwen3_4b_thinking_predict.yaml
```

### 6.3 产出（`saves/Qwen3-4B-Thinking/lora/predict/`）

- `generated_predictions.jsonl` —— 每行 `label` vs `predict`，**人眼对比最直观**，比 BLEU 更有参考价值。
- `predict_results.json` —— BLEU/ROUGE 指标（中文场景分值普遍偏低，别被低分吓到）。

### 6.4 实测坑（0.9.6 版本）

| 坑 | 处理 |
|----|------|
| `llamafactory-cli eval` **已废弃** | 0.9.6 起 `launcher.py:147` 直接 raise `NotImplementedError`。所有评估统一用 `train` 命令 + 配置里 `do_predict: true`，走 `sft/workflow.py:152` 预测分支。配置不用改，只换子命令。 |
| 命令行覆盖 YAML 用 **`key=value`** | OmegaConf 语法（`parser.py:91`），不是 `--key value`。例：`... cfg.yaml max_samples=500`。用 `--key value` 报 `Some keys are not used by the HfArgumentParser`。 |
| 最稳的覆盖方式 | 直接改配置文件值，跑**无参数**命令。 |

---

## 7. 导出（合并 LoRA → 完整模型）

> ✅ 已实测。配置：`examples/merge_lora/qwen3_4b_thinking_export.yaml`。

**用途**：把 LoRA adapter 烧进基座，导出**自包含的完整模型**（脱离基座单独部署）。

### 7.1 配置要点

```yaml
model_name_or_path: model/Qwen3-4B
adapter_name_or_path: saves/Qwen3-4B-Thinking/lora/sft   # 指训练 output_dir 根目录
template: qwen3
finetuning_type: lora
export_dir: saves/Qwen3-4B-Thinking/merged
export_size: 5                          # 每个分片大小（GB）
export_device: cpu                      # cpu 慢但稳；auto 用 GPU 快但占显存
export_legacy_format: false             # false=safetensors(推荐)；true=pytorch_model.bin
```

### 7.2 命令

```bash
llamafactory-cli export examples/merge_lora/qwen3_4b_thinking_export.yaml
```

导出到 `saves/Qwen3-4B-Thinking/merged/`（Qwen3-4B bf16 约 8GB，按 `export_size: 5` 分 2 个 safetensors 分片）。合并后可直接用 vLLM / Ollama（目录里有 `Modelfile`）/ transformers 加载部署，不必再走 LF。

### 7.3 实测坑

| 坑 | 处理 |
|----|------|
| `Can't find 'adapter_config.json'` | `adapter_name_or_path` 要指向**训练 output_dir 根目录**（含 `adapter_config.json`），**不要**指向 `checkpoint-XXX` 子目录（会被 `save_total_limit` 清理）。 |
| 命令行换合并目标 | `... qwen3_4b_thinking_export.yaml adapter_name_or_path=saves/Qwen3-4B-Thinking/lora/train_xxx`（用 `key=`，非 `--key`）。 |
| 合并报量化错误 | ⚠️ **合并时基座不能是量化模型 / 不能设 `quantization_bit`**（要求全精度基座）。 |
| 内存 / 耗时 | 合并耗内存 ≈ 模型体积（8~16GB），耗时几分钟到十几分钟。`export_device: cpu` 最稳。 |

---

*框架原理见 [frame.md](./frame.md)；各阶段完整示例配置见 `examples/`；数据集格式见 `data/dataset_info.json`。*
