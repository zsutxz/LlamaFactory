# LLaMA-Factory 项目学习指南

> 本文档面向新手，介绍LLaMA-Factory项目的关键点和核心技术

## 项目概述

LLaMA-Factory是一个**统一的大语言模型高效微调框架**，支持100+种模型的开源微调平台。它可以让用户无需编写任何代码，通过配置YAML文件即可完成模型微调。

### 核心特点
- **零代码微调**：通过YAML配置完成训练
- **模型支持丰富**：支持100+种模型（LLaMA、Qwen、DeepSeek、Mistral等）
- **训练方法多样**：全量、LoRA、QLoRA、冻结等多种微调方式
- **完整训练流程**：从预训练→监督微调→奖励模型→RLHF全流程支持
- **多种推理后端**：HuggingFace、vLLM、SGLang、KTransformers

---

## 一、项目目录结构

```
LLaMA-Factory/
├── data/                    # 数据集目录
│   ├── dataset_info.json   # 数据集配置（关键！）
│   └── *.json              # 数据文件
├── examples/               # 配置示例（学习重点）
│   ├── train_lora/        # LoRA训练配置
│   ├── train_qlora/       # QLoRA训练配置
│   └── inference/         # 推理配置
├── src/                    # 源代码
│   └── llamafactory/
│       ├── cli.py         # CLI入口
│       ├── launcher.py    # 主启动器（核心）
│       ├── data/          # 数据处理模块
│       ├── model/         # 模型加载模块
│       ├── train/         # 训练工作流
│       ├── api/           # API服务
│       ├── chat/          # 聊天引擎
│       └── webui/         # Web界面
├── setup.py               # 安装配置
└── requirements.txt       # 依赖列表
```

---

## 二、核心模块解析

### 2.1 入口点：launcher.py

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

### 2.2 数据处理模块

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

### 2.3 模型处理模块

| 文件 | 功能 |
|-----|------|
| `model/loader.py` | 模型加载器 |
| `model/adapter.py` | LoRA等适配器初始化 |
| `model/patcher.py` | 模型补丁（量化、注意力优化） |

### 2.4 训练工作流模块

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

## 三、配置文件详解

### 3.1 数据集配置：data/dataset_info.json

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

### 3.2 训练配置：examples/*.yaml

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

## 四、核心技术栈

### 4.1 技术依赖

| 类别 | 技术 |
|-----|------|
| 深度学习 | PyTorch >= 2.0.0 |
| 模型库 | Transformers, PEFT, TRL |
| 训练加速 | DeepSpeed, FSDP |
| 推理引擎 | vLLM, SGLang |
| 量化 | bitsandbytes, GPTQ, AWQ |

### 4.2 支持的训练方法

| 方法 | 显存需求 | 速度 | 适用场景 |
|-----|---------|------|---------|
| Full（全量） | 很大 | 慢 | 充分资源，完全重训 |
| Freeze（冻结） | 中 | 中 | 只训练部分层 |
| LoRA | 小 | 快 | 最常用，高效微调 |
| QLoRA | 很小 | 快 | 显存受限场景 |

### 4.3 训练阶段（stage）

| stage | 说明 | 数据格式 |
|-------|------|---------|
| `pt` | 预训练 | 纯文本 |
| `sft` | 监督微调 | 指令-响应对 |
| `rm` | 奖励模型 | 评分数据 |
| `ppo` | PPO训练 | 提示-生成-奖励 |
| `dpo` | DPO训练 | 偏好对 |
| `kto` | KTO训练 | 二元反馈 |

---

## 五、快速开始

### 5.1 安装

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

### 5.2 准备数据

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

### 5.3 配置训练

复制示例配置并修改：

```bash
cp examples/train_lora/llama3_lora_sft.yaml my_config.yaml
```

### 5.4 开始训练

```bash
llamafactory-cli train my_config.yaml
```

### 5.5 启动WebUI（推荐新手）

```bash
llamafactory-cli webui
```

然后在浏览器打开 http://localhost:7860

---

## 六、关键概念解析

### 6.1 模板（Template）

模板定义了如何将对话格式化成模型输入。

**位置**: `src/llamafactory/data/template.py`

常见模板：
- `llama3` - LLaMA-3格式
- `qwen` - Qwen格式
- `chatglm` - ChatGLM格式

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

---

## 七、学习路径建议

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

## 八、常见问题

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

## 九、资源链接

- **官方文档**: https://github.com/hiyouga/LLaMA-Factory
- **配置示例**: `examples/` 目录
- **示例数据**: `data/` 目录

---

## 十、核心代码位置速查

| 功能 | 文件位置 |
|-----|---------|
| 命令入口 | `src/llamafactory/launcher.py:38-180` |
| 参数解析 | `src/llamafactory/hparams/parser.py:253-476` |
| 模型注册 | `src/llamafactory/extras/constants.py:153-167` |
| 模板定义 | `src/llamafactory/data/template.py:40-150` |
| LoRA初始化 | `src/llamafactory/model/adapter.py` |
| SFT工作流 | `src/llamafactory/train/sft/workflow.py` |
| API服务 | `src/llamafactory/api/app.py` |

---

*本文档持续更新中...*
