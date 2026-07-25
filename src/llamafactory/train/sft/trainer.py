# ==============================================================================
# SFT 自定义训练器模块
# ==============================================================================
# 功能：继承 HuggingFace 的 Seq2SeqTrainer，实现自定义训练逻辑
#
# 主要特性：
# - 支持 FP8 混合精度训练
# - 支持多种优化器（AdamW、GaLore、BAdam、APOLLO 等）
# - 支持自定义学习率调度器
# - 支持禁用数据打乱
# - 支持 DFT Loss（直接偏好调优损失）
# - 保存预测结果到 JSONL
# ==============================================================================

# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX  # 用于忽略 padding token 的损失计算
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback  # 保存多模态 processor 的回调
from ..fp8_utils import configure_fp8_environment, verify_fp8_status  # FP8 配置和验证
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


# ============================================================================
# 类型检查（IDE 提示用）
# ============================================================================
if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments


# 获取日志记录器
logger = logging.get_logger(__name__)


# ============================================================================
# 自定义 Seq2Seq 训练器
# ============================================================================
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    """
    继承 Seq2SeqTrainer，支持生成式指标计算（BLEU、ROUGE）

    扩展功能：
        1. FP8 训练支持
        2. 多种优化器（GaLore、BAdam、APOLLO 等）
        3. 自定义学习率调度
        4. 禁用数据打乱
        5. DFT Loss
        6. 保存预测结果
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",  # 微调参数
        processor: Optional["ProcessorMixin"],    # 多模态 processor（可选）
        model_args: Optional["ModelArguments"] = None,  # 模型参数
        gen_kwargs: Optional[dict[str, Any]] = None,    # 生成参数
        **kwargs,
    ) -> None:
        """
        初始化训练器

        Args:
            finetuning_args: 微调参数（LoRA、Full 等）
            processor: 多模态 processor（图像、音频处理）
            model_args: 模型参数
            gen_kwargs: 生成参数（temperature、top_p 等）
            **kwargs: 传递给父类的其他参数
        """

        # ====================================================================
        # 配置 FP8 环境（如果启用）
        # ====================================================================
        if model_args is not None and model_args.fp8:
            configure_fp8_environment(model_args)

        # ====================================================================
        # 处理 tokenizer/processing_class 参数兼容性
        # ====================================================================
        if is_transformers_version_greater_than("4.46"):
            # Transformers v4.46+ 使用 processing_class
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            # 旧版本使用 tokenizer
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        # 调用父类初始化
        super().__init__(**kwargs)

        # ====================================================================
        # Processor 处理（多模态模型）
        # ====================================================================
        if processor is not None:
            # 避免梯度累积下的损失计算错误
            # 参考: https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        # 保存微调参数
        self.finetuning_args = finetuning_args

        # 保存生成参数
        if gen_kwargs is not None:
            # 参考: https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        # 添加 processor 保存回调
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        # ====================================================================
        # BAdam 优化器配置
        # ====================================================================
        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            # 替换梯度裁剪方法（BAdam 需要旧版本）
            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        # ====================================================================
        # DFT Loss 配置（直接偏好调优损失）
        # ====================================================================
        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            # 替换损失计算函数
            self.compute_loss_func = dft_loss_func

        # ====================================================================
        # 验证 FP8 状态
        # ====================================================================
        # 在训练器初始化后验证（accelerator 应该已可用）
        if model_args is not None and model_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, model_args)

    # ========================================================================
    # 创建自定义优化器
    # ========================================================================
    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        """
        创建自定义优化器

        支持的优化器：
            - AdamW（默认）
            - GaLore（梯度低秩分解）
            - APOLLO
            - BAdam
            - Adam-mini
            - Muon
        """
        if self.optimizer is None:
            # 创建自定义优化器
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    # ========================================================================
    # 创建自定义学习率调度器
    # ========================================================================
    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        """
        创建自定义学习率调度器

        支持的调度器：
            - cosine（余弦退火）
            - linear（线性衰减）
            - polynomial（多项式衰减）
            - constant（恒定）
        """
        # 配置自定义学习率调度器
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    # ========================================================================
    # 获取训练数据采样器
    # ========================================================================
    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        """
        获取训练数据采样器

        Returns:
            - SequentialSampler: 禁用打乱时（顺序采样）
            - RandomSampler: 默认（随机采样）
        """
        if self.finetuning_args.disable_shuffling:
            # 禁用数据打乱（顺序采样）
            return torch.utils.data.SequentialSampler(self.train_dataset)

        # 默认使用父类的随机采样
        return super()._get_train_sampler(*args, **kwargs)

    # ========================================================================
    # 计算损失
    # ========================================================================
    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        """
        计算模型损失

        Args:
            model: 模型
            inputs: 输入数据（input_ids、attention_mask、labels 等）

        Returns:
            损失值

        注意：
            如果启用了 DFT Loss，会使用自定义损失函数
        """
        return super().compute_loss(model, inputs, *args, **kwargs)

    # ========================================================================
    # 预测步骤
    # ========================================================================
    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        """
        执行预测步骤

        Args:
            model: 模型
            inputs: 输入数据
            prediction_loss_only: 是否只计算损失
            ignore_keys: 忽略的键
            **gen_kwargs: 生成参数

        Returns:
            (loss, generated_tokens, labels)

        特殊处理：
            - 移除生成 token 中的输入部分（prompt）
            - 生成模式下不传递 labels
        """
        # ====================================================================
        # 生成模式特殊处理
        # ====================================================================
        if self.args.predict_with_generate:
            # 生成模式下不传递 labels 给模型
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        # 调用父类的预测步骤
        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )

        # ====================================================================
        # 清除输入部分的生成 token
        # ====================================================================
        if generated_tokens is not None and self.args.predict_with_generate:
            # 将输入部分（prompt）替换为 pad_token_id
            # 只保留新生成的部分
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    # ========================================================================
    # 保存预测结果
    # ========================================================================
    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        """
        保存模型预测到 JSONL 文件

        Args:
            dataset: 原始数据集
            predict_results: 预测结果
            skip_special_tokens: 是否跳过特殊 token

        输出格式：
            {"prompt": "用户输入", "predict": "模型预测", "label": "真实标签"}
        """
        # 只在主进程上执行
        if not self.is_world_process_zero:
            return

        # 输出文件路径
        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        # ====================================================================
        # 处理 IGNORE_INDEX
        # ====================================================================
        # 将 IGNORE_INDEX (-100) 替换为 pad_token_id
        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX,
            predict_results.label_ids,
            self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        # ====================================================================
        # 移动 padding token 到末尾
        # ====================================================================
        for i in range(len(preds)):
            # 找到第一个非 padding 的位置
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):
                # 将 padding 移到末尾
                preds[i] = np.concatenate((preds[i][pad_len[0]:], preds[i][:pad_len[0]]), axis=-1)

        # ====================================================================
        # 解码为文本
        # ====================================================================
        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        # ====================================================================
        # 保存到 JSONL 文件
        # ====================================================================
        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                # 每行一个 JSON 对象
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
