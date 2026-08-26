# LLaMA-Factory 框架原理

> 本文档介绍 LLaMA-Factory 框架本身的结构、模块、配置与核心概念。
> 本机实操：全流程上手见 [learning.md](./learning.md)；领域 PT（4B/8B/9B）与蒸馏闭环的完整实测见 [train_list.md](./train_list.md)。

---

## 1. 项目概述

LLaMA-Factory是一个**统一的大语言模型高效微调框架**，支持100+种模型的开源微调平台。它可以让用户无需编写任何代码，通过配置YAML文件即可完成模型微调。

### 1.1 核心特点
- **零代码微调**：通过YAML配置完成训练
- **模型支持丰富**：支持100+种模型（LLaMA、Qwen、DeepSeek、Mistral等）
- **训练方法多样**：全量、LoRA、QLoRA、冻结等多种微调方式
- **完整训练流程**：从预训练→监督微调→奖励模型→RLHF全流程支持
- **多种推理后端**：HuggingFace、vLLM、SGLang、KTransformers

---

## 2. 项目目录结构

```
LLaMA-Factory/
├── data/                    # 数据集目录
│   ├── dataset_info.json   # 数据集配置（关键！）
│   └── *.json              # 数据文件
├── examples/               # 配置示例（学习重点）
│   ├── train_lora/        # LoRA训练配置
│   ├── train_qlora/       # QLoRA训练配置
│   └── inference/         # 推理配置
├── scripts/               # 数据处理等自写脚本（本项目：.claude/skills/public-data-pipeline/scripts/ 语料构建与蒸馏造数）
├── src/
│   └── llamafactory/
│       ├── cli.py         # CLI入口
│       ├── launcher.py    # 主启动器（核心）
│       ├── data/          # 数据处理模块（loader/parser/template/processor）
│       ├── model/         # 模型加载模块（loader/adapter/patcher）
│       ├── hparams/       # 参数定义与解析（*_args.py + parser.py）
│       ├── train/         # 训练工作流（pt/sft/rm/ppo/dpo/kto 各 workflow）
│       ├── eval/          # 评估模块
│       ├── extras/        # 常量/日志/环境等杂项
│       ├── api/           # API服务
│       ├── chat/          # 聊天引擎
│       └── webui/         # Web界面
├── pyproject.toml          # 安装配置与依赖
└── requirements/           # 分组依赖（metrics/deepspeed 等）
```

---

## 3. 核心模块解析

### 3.1 入口点：launcher.py

**文件位置**: `src/llamafactory/launcher.py`

这是整个项目的调度中心，负责路由不同的命令到对应模块：

```python
# 命令路由逻辑（第38-180行）
├── train      → 启动分布式训练
├── api        → 启动API服务
├── chat       → 启动CLI聊天
├── webui      → 启动LlamaBoard界面
├── export     → 导出合并模型
└── env        → 显示环境信息
```

### 3.2 数据处理模块

| 文件 | 功能 |
|-----|------|
| `data/loader.py` | 数据集加载器 |
| `data/parser.py` | 数据集解析器 |
| `data/template.py` | 模型对话模板定义 |
| `data/processor/` | 不同训练阶段的数据处理器 |

**关键数据流**：
```
原始数据 → parser解析 → loader加载 → processor处理 → 训练
```

### 3.3 模型处理模块

| 文件 | 功能 |
|-----|------|
| `model/loader.py` | 模型加载器 |
| `model/adapter.py` | LoRA等适配器初始化 |
| `model/patcher.py` | 模型补丁（量化、注意力优化） |

**torch 2.9.x Conv3D 防护闸**（`model/loader.py` 的 `load_model`，197-213 行）：当 torch ≥2.9 且 <2.10、模型含视觉 `Conv3d` 模块时直接 raise——该组合有已知的严重性能回归（pytorch#166122）。**纯文本工作负载**（PT/SFT/推理均不执行 Conv3D）可用环境变量 `LF_ALLOW_TORCH29_CONV3D=1` 放行；放行时启动日志出现 bypass WARNING 属预期，不是错误。Qwen3.5 系列自带视觉模块，在 torch 2.9.x 下会触发此闸。

### 3.4 训练工作流模块

```
train/
├── tuner.py          # 训练调度器
├── sft/workflow.py   # 监督微调（最常用）
├── dpo/workflow.py   # DPO训练
├── ppo/workflow.py   # PPO训练
├── pt/workflow.py    # 预训练
└── rm/workflow.py    # 奖励模型训练
```

---

## 4. 配置文件详解

### 4.1 数据集配置：data/dataset_info.json

**这是配置数据集的关键文件**：

```json
{
  "my_dataset": {
    "file_name": "my_data.json",
    "formatting": "alpaca",  // 或sharegpt
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

### 4.2 训练配置：examples/*.yaml

**以LoRA训练为例** (`examples/train_lora/llama3_lora_sft.yaml`):

```yaml
# 模型配置
model_name_or_path: meta-llama/Meta-Llama-3-8B
template: llama3

# 训练方法
stage: sft                # 监督微调
finetuning_type: lora     # LoRA微调
lora_rank: 8              # LoRA秩（越小越省显存）
lora_target: all          # 应用到所有层

# 数据集
dataset: identity,alpaca_en_demo
cutoff_len: 2048          # 最大序列长度

# 输出
output_dir: saves/llama3-8b/lora/sft

# 训练参数
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
bf16: true               # 使用bf16加速
```

---

## 5. 核心技术栈

### 5.1 技术依赖

| 类别 | 技术 |
|-----|-----|
| 深度学习 | PyTorch >= 2.0.0 |
| 模型库 | Transformers, PEFT, TRL |
| 训练加速 | DeepSpeed, FSDP |
| 推理引擎 | vLLM, SGLang |
| 量化 | bitsandbytes, GPTQ, AWQ |

### 5.2 支持的训练方法

| 方法 | 显存需求 | 速度 | 适用场景 |
|-----|---------|------|---------|
| Full（全量） | 很大 | 慢 | 充分资源，完全重训 |
| Freeze（冻结） | 中 | 中 | 只训练部分层 |
| LoRA | 小 | 快 | 最常用，高效微调 |
| QLoRA | 很小 | 快 | 显存受限场景 |

### 5.3 训练阶段（stage）

| stage | 说明 | 数据格式 |
|-------|------|---------|
| `pt` | 预训练 | 纯文本 |
| `sft` | 监督微调 | 指令-响应对 |
| `rm` | 奖励模型 | 评分数据 |
| `ppo` | PPO训练 | 提示-生成-奖励 |
| `dpo` | DPO训练 | 偏好对 |
| `kto` | KTO训练 | 二元反馈 |

stage 的完整定义在 `src/llamafactory/hparams/finetuning_args.py:460`（`Literal["pt", "sft", "rm", "ppo", "dpo", "kto"]`）——**没有 `distill`**，蒸馏在框架中的定位见第 8 节。

---

## 6. 关键概念解析

### 6.1 模板（Template）

模板定义了如何将对话格式化成模型输入。

**位置**: `src/llamafactory/data/template.py`

常见模板：
- `llama3` - LLaMA-3格式
- `qwen` - Qwen格式
- `chatglm` - ChatGLM格式

**qwen3 系模板要点**（2026-08 实测，锚点均在 `src/llamafactory/data/template.py`）：

- `qwen3` / `qwen3_5` 都注册为 **ReasoningTemplate**（`template.py:407` 起定义，处理 `<think>` 块）。SFT 数据不带 `<think>` 块时，模板自动补一个**空 think 块**：`enable_thinking: true` 时拼进 response、**计入 loss**（教模型「先输出空 think 再作答」，`template.py:430-435`）；`enable_thinking: false` 时拼进 prompt、不计 loss。
- 因此**训练与推理必须对齐**：实测的 9B 蒸馏闭环即「SFT 空 think 块计入 loss + 推理 `enable_thinking: false`」的组合（模板在推理侧注入空 think 前缀，与训练分布一致）。
- **`qwen3_5` 必须显式指定**：Qwen3.5 官方 chat_template（tokenizer_config.json 内 Jinja）含多处 `raise_exception`，框架无法从其自动反推解析出模板，训练/推理配置漏写 `template: qwen3_5` 会导致解析失败。
- `qwen3_5_nothink` **未挂 ReasoningTemplate**（`template.py:2168` 起无 `template_class`），不处理 think 块逻辑，**不能**用它替代 `enable_thinking: false` 来关思维链。

### 6.2 LoRA参数

| 参数 | 说明 | 推荐值 |
|-----|------|-------|
| `lora_rank` | LoRA秩 | 8-64 |
| `lora_alpha` | LoRA缩放 | rank的1-2倍 |
| `lora_target` | 目标层 | all/qkv/v等 |

### 6.3 显存优化

当显存不足时：

1. **减小batch size**
2. **增加gradient_accumulation_steps**
3. **使用QLoRA代替LoRA**
4. **减小cutoff_len**
5. **启用DeepSpeed**

**QLoRA 实测锚点**（16GB RTX 5060 Ti，详见 [train_list.md](./train_list.md)）：8B/9B bf16 基座放不进 16GB，必须 4-bit（`quantization_bit: 4` + `method: bnb`，nf4）——量化后基座仅 ~5GB，训练总占用 ~14.3GB，反而比 4B bf16（15.7GB 近顶满）更省；Blackwell 架构（sm_120）需 bitsandbytes ≥0.43。

---

## 7. 核心代码位置速查

| 功能 | 文件位置 |
|-----|---------|
| 命令入口 | `src/llamafactory/launcher.py:38-180` |
| 参数解析 | `src/llamafactory/hparams/parser.py:253-476` |
| 模型注册 | `src/llamafactory/extras/constants.py:153-167` |
| 模板定义 | `src/llamafactory/data/template.py:40-150` |
| ReasoningTemplate（think 块/空 think loss） | `src/llamafactory/data/template.py:407-435` |
| stage 定义（无 distill） | `src/llamafactory/hparams/finetuning_args.py:460` |
| Conv3D 防护闸（LF_ALLOW_TORCH29_CONV3D） | `src/llamafactory/model/loader.py:197-213` |
| 量化基座单 adapter 限制 | `src/llamafactory/hparams/parser.py:198-202` / `src/llamafactory/model/adapter.py:159-161` |
| LoRA初始化 | `src/llamafactory/model/adapter.py` |
| SFT工作流 | `src/llamafactory/train/sft/workflow.py` |
| API服务 | `src/llamafactory/api/app.py` |

---

## 8. 蒸馏闭环的框架定位

LLaMA-Factory **没有独立的蒸馏 stage**：`stage` 只有 `pt/sft/rm/ppo/dpo/kto`（`src/llamafactory/hparams/finetuning_args.py:460`），代码里搜到的 "Distill" 全是模型名（DeepSeek-R1-Distill-*，别人蒸馏好的成品，LF 只是能加载），不是训练功能。

因此「蒸馏」在 LF 生态里的落地方式是 **外部造数 + 原生 SFT**：

```
teacher（外部 API，如 DeepSeek）从语料生成 QA
  → 裁判模型（如 Kimi）评分过筛 + 人工复核        ← 均为 LF 之外的自写脚本
  → 注册为 alpaca 数据集，走 LF 原生 stage: sft   ← LF 只负责这一步
```

本项目落地：`.claude/skills/public-data-pipeline/scripts/generate_domain_qa.py`（出题）→ `judge_domain_qa.py`（裁判/对比评分）→ `ask_compare.py`（留出题自动评测，配合 `llamafactory-cli api`），完整管线与实测见 [train_list.md](./train_list.md) §12。

**量化基座只允许挂 1 个 adapter**（QLoRA 续训的关键约束）：

- `src/llamafactory/hparams/parser.py:198-202`：量化模型上禁止 `create_new_adapter`，且 `adapter_name_or_path` 只接受单个；
- `src/llamafactory/model/adapter.py:159-161`：量化模型上 LoRA 合并不稳定，assert 只挂单 adapter。

因此 QLoRA 下的 **PT → SFT 续训只能走 `adapter_name_or_path` 指向同一 adapter 的 resume 路线**（rank/alpha/target 必须与 PT 完全一致，lr 降一档），不能 `create_new_adapter` 叠第二个，也不能多 adapter 叠加。

---

*本文档持续更新中...*
