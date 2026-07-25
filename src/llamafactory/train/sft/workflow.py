# ==============================================================================
# SFT (监督微调) 工作流程模块
# ==============================================================================
# 功能：实现监督微调的完整训练流程，包括数据加载、模型训练、评估预测等
# 入口：run_sft() 函数
# ==============================================================================

# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Optional

# ============================================================================
# 导入数据处理模块
# ============================================================================
from ...data import (
    SFTDataCollatorWith4DAttentionMask,  # 4D 注意力掩码数据整理器（处理打包序列）
    get_dataset,                         # 加载数据集函数
    get_template_and_fix_tokenizer       # 获取对话模板并修复 tokenizer
)

# ============================================================================
# 导入常量和工具模块
# ============================================================================
from ...extras.constants import IGNORE_INDEX  # 用于忽略 padding token 的损失计算（值为 -100）
from ...extras.logging import get_logger      # 日志记录器
from ...extras.misc import calculate_tps      # 计算 tokens/秒 速度指标
from ...extras.packages import is_transformers_version_greater_than  # 检查 Transformers 版本
from ...extras.ploting import plot_loss       # 绘制损失曲线

# ============================================================================
# 导入模型加载模块
# ============================================================================
from ...model import load_model, load_tokenizer  # 加载模型和 tokenizer

# ============================================================================
# 导入训练工具模块
# ============================================================================
from ..trainer_utils import create_modelcard_and_push  # 创建模型卡片并推送到 Hub

# ============================================================================
# 导入 SFT 特定模块
# ============================================================================
from .metric import (
    ComputeAccuracy,        # 计算 token 级别准确率
    ComputeSimilarity,      # 计算文本相似度（ROUGE、BLEU）
    eval_logit_processor    # 处理 logits 用于评估（减少内存占用）
)
from .trainer import CustomSeq2SeqTrainer  # 自定义 Seq2Seq 训练器


# ============================================================================
# 类型检查（IDE 提示用）
# ============================================================================
if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import (
        DataArguments,         # 数据相关参数
        FinetuningArguments,   # 微调相关参数（LoRA、Full 等）
        GeneratingArguments,   # 生成相关参数（temperature、top_p 等）
        ModelArguments         # 模型相关参数
    )


# 获取当前模块的日志记录器
logger = get_logger(__name__)


# ============================================================================
# SFT 训练主函数
# ============================================================================
def run_sft(
    model_args: "ModelArguments",          # 模型相关参数（路径、量化、模板等）
    data_args: "DataArguments",            # 数据相关参数（数据集、格式、长度等）
    training_args: "Seq2SeqTrainingArguments",  # 训练参数（批大小、学习率、轮数等）
    finetuning_args: "FinetuningArguments",# 微调参数（LoRA rank、target 等）
    generating_args: "GeneratingArguments",# 生成参数（temperature、top_p 等）
    callbacks: Optional[list["TrainerCallback"]] = None,  # 训练回调函数列表
):
    """
    监督微调（SFT）训练的主工作流程

    工作流程：
    1. 加载 tokenizer 和对话模板
    2. 加载数据集（训练集和验证集）
    3. 加载模型（支持量化和 LoRA）
    4. 创建数据整理器（Data Collator）
    5. 配置评估指标（准确率或相似度）
    6. 配置生成参数
    7. 创建训练器（标准或 KTransformers）
    8. 执行训练并保存模型
    9. 执行评估（可选）
    10. 执行预测（可选）
    11. 创建模型卡片
    """
    # ========================================================================
    # 步骤 1: 加载 tokenizer 和对话模板
    # ========================================================================
    tokenizer_module = load_tokenizer(model_args)
    # 加载 tokenizer，返回模块包含 tokenizer 和 processor（多模态用）

    tokenizer = tokenizer_module["tokenizer"]
    # tokenizer：将文本转换为 token IDs 的工具

    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    # 获取对话模板（如 qwen3、llama3 等）
    # 模板负责将用户输入转换为模型能理解的格式

    # ========================================================================
    # 步骤 2: 加载数据集
    # ========================================================================
    dataset_module = get_dataset(
        template,           # 对话模板
        model_args,         # 模型参数
        data_args,          # 数据参数
        training_args,      # 训练参数
        stage="sft",        # 训练阶段：监督微调
        **tokenizer_module  # 传入 tokenizer 和 processor
    )
    # 返回: {"train_dataset": ..., "eval_dataset": ...}

    # ========================================================================
    # 步骤 3: 加载模型
    # ========================================================================
    model = load_model(
        tokenizer,
        model_args,
        finetuning_args,
        training_args.do_train  # 是否需要训练（影响是否加载优化器状态）
    )

    # ========================================================================
    # 特殊处理：量化模型的预测兼容性
    # ========================================================================
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        # 如果模型是量化的且只做预测（不训练）
        # 设置这个标志让模型与预测兼容
        setattr(model, "_hf_peft_config_loaded", True)

    # ========================================================================
    # 步骤 4: 创建数据整理器（Data Collator）
    # ========================================================================
    # 数据整理器负责将多个样本整理成一个 batch，处理 padding 和 attention mask
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        # 对话模板，用于格式化数据

        model=model if not training_args.predict_with_generate else None,
        # 模型对象（生成模式下不需要传入模型）

        pad_to_multiple_of=8 if training_args.do_train else None,
        # 填充到 8 的倍数（用于加速 GPU 计算）
        # 训练时启用，推理时不启用

        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        # label 的填充 token ID
        # IGNORE_INDEX (-100) 表示忽略 padding 的损失计算

        block_diag_attn=model_args.block_diag_attn,
        # 是否使用块对角注意力（长序列优化技术）

        attn_implementation=getattr(model.config, "_attn_implementation", None),
        # 注意力实现方式：eager（原生）、sdpa、flash_attention_2

        compute_dtype=model_args.compute_dtype,
        # 计算数据类型：fp32、fp16、bf16

        **tokenizer_module,
        # 传入 tokenizer 和 processor
    )

    # ========================================================================
    # 步骤 5: 配置评估指标
    # ========================================================================
    metric_module = {}

    # --- KTransformers 特殊处理 ---
    if model_args.use_kt:
        # KTransformers 不支持某些评估模式
        if training_args.predict_with_generate:
            raise NotImplementedError("`predict_with_generate` is not supported in KTransformers SFT yet.")
        elif finetuning_args.compute_accuracy:
            raise NotImplementedError("`compute_accuracy` is not supported in KTransformers SFT yet.")

    # --- 生成评估模式 ---
    if training_args.predict_with_generate:
        # 使用生成模式评估：计算 ROUGE 和 BLEU 相似度
        # 优点：评估生成质量
        # 缺点：速度慢
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)

    # --- 准确率评估模式 ---
    elif finetuning_args.compute_accuracy:
        # 使用准确率评估：计算 token 级别准确率
        # 优点：速度快
        # 缺点：不能反映生成质量
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor
        # eval_logit_processor: 减少 logits 内存占用

    # ========================================================================
    # 步骤 6: 配置生成参数
    # ========================================================================
    # 将生成参数转换为字典，包含 temperature、top_p、top_k、max_new_tokens 等
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)

    # --- 兼容 Transformers v4 和 v5 ---
    if is_transformers_version_greater_than("4.58.0"):
        # Transformers v5+ 处理方式
        extra_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
        if not isinstance(extra_ids, list):
            # 如果没有 additional_special_tokens_ids，从 _extra_special_tokens 获取
            extra_special_tokens = getattr(tokenizer, "_extra_special_tokens", [])
            string_tokens = [str(t) for t in extra_special_tokens]
            extra_ids = tokenizer.convert_tokens_to_ids(string_tokens)

        # 收集所有 EOS token ID（结束符）
        all_eos_ids = [tokenizer.eos_token_id] + [i for i in extra_ids if i != -1]
        unique_eos_ids = list(dict.fromkeys(all_eos_ids))  # 去重
        gen_kwargs["eos_token_id"] = unique_eos_ids
    else:
        # Transformers v4 处理方式
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids

    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # ========================================================================
    # 步骤 7: 创建训练器
    # ========================================================================
    # 根据配置选择使用 KTransformers 训练器或标准训练器

    # --- KTransformers 训练器（用于长序列高效训练）---
    if model_args.use_kt:
        from ktransformers.sft.lora import KTrainer  # type: ignore
        from ktransformers.util.globals import GLOBAL_CONFIG  # type: ignore

        GLOBAL_CONFIG._config["mod"] = "sft"

        trainer = KTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer_module,
            data_collator=data_collator,
            callbacks=callbacks,
            **dataset_module,  # train_dataset, eval_dataset
            **metric_module,   # compute_metrics
        )
        trainer.model_accepts_loss_kwargs = False
        model.config.use_cache = False  # 训练时禁用缓存以节省显存

    # --- 标准 Seq2Seq 训练器 ---
    else:
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            finetuning_args=finetuning_args,
            data_collator=data_collator,
            callbacks=callbacks,
            gen_kwargs=gen_kwargs,
            **dataset_module,  # train_dataset, eval_dataset
            **tokenizer_module,  # tokenizer, processor
            **metric_module,   # compute_metrics
        )

    # ========================================================================
    # 步骤 8: 执行训练
    # ========================================================================
    if training_args.do_train:
        # 开始训练（支持从检查点恢复）
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        # resume_from_checkpoint: 从检查点恢复训练

        # 保存模型（LoRA 权重或完整模型）
        trainer.save_model()

        # 计算有效 tokens/秒（可选）
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"],
                train_result.metrics,
                stage="sft"
            )

        # 记录和保存训练指标
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()  # 保存训练状态（用于恢复）

        # 绘制损失曲线
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            # is_world_process_zero: 只在主进程上执行

            # 确定要绘制的指标
            keys = ["loss"]  # 训练损失

            if isinstance(dataset_module.get("eval_dataset"), dict):
                # 多个评估数据集
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_accuracy"]
                     for key in dataset_module["eval_dataset"].keys()],
                    []
                )
            else:
                # 单个评估数据集
                keys += ["eval_loss", "eval_accuracy"]

            # 绘制并保存损失曲线图
            plot_loss(training_args.output_dir, keys=keys)

    # ========================================================================
    # 生成模式特殊处理
    # ========================================================================
    if training_args.predict_with_generate:
        # 生成模式使用左 padding（因为生成长度不一致）
        tokenizer.padding_side = "left"

    # ========================================================================
    # 步骤 9: 执行评估
    # ========================================================================
    if training_args.do_eval:
        # 在验证集上评估模型性能
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # ========================================================================
    # 步骤 10: 执行预测
    # ========================================================================
    if training_args.do_predict:
        # 警告：批量生成很慢，建议使用 vLLM 进行推理
        logger.warning_rank0_once(
            "Batch generation can be very slow. "
            "Consider using `scripts/vllm_infer.py` instead."
        )

        # 执行预测
        predict_results = trainer.predict(
            dataset_module["eval_dataset"],
            metric_key_prefix="predict",
            **gen_kwargs
        )

        # 记录和保存预测指标
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)

        # 保存预测结果到 JSONL 文件
        trainer.save_predictions(
            dataset_module["eval_dataset"],
            predict_results,
            generating_args.skip_special_tokens
        )

    # ========================================================================
    # 步骤 11: 创建模型卡片
    # ========================================================================
    # 生成 README.md，包含模型信息、训练参数、性能指标等
    create_modelcard_and_push(
        trainer,
        model_args,
        data_args,
        training_args,
        finetuning_args
    )
