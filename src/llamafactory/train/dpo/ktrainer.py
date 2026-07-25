# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
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

"""
基于 KTransformers 的 DPO 训练器

KDPOTrainer 继承自 KTrainer 和 CustomDPOTrainer，利用 KTransformers 框架
提供的优化能力来加速 DPO 训练过程。
"""

from typing import TYPE_CHECKING

import torch
from ktransformers.sft.lora import KTrainer  # type: ignore
from typing_extensions import override

from ..trainer_utils import get_batch_logps, nested_detach
from .trainer import CustomDPOTrainer


if TYPE_CHECKING:
    from transformers import PreTrainedModel


class KDPOTrainer(KTrainer, CustomDPOTrainer):
    """
    KTransformers 加速的 DPO 训练器

    继承 KTrainer 和 CustomDPOTrainer，结合了 KTransformers 的 LoRA 优化
    和 LlamaFactory 的 DPO 训练能力。
    """
    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"], is_ref_model: bool = False
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""计算模型在给定批次上的前向传播输出

        此方法针对 KTransformers 的 CPU 训练优化，将 logits 移动到 CPU 上进行计算。
        计算标签在给定 logits 下的对数概率：
        - 如果 loss_type 不是 IPO、ORPO 或 SimPO，返回对数概率之和
        - 否则返回平均对数概率

        Args:
            model: 要计算前向传播的模型
            batch: 输入批次，包含 input_ids、attention_mask、labels 等
            is_ref_model: 是否为参考模型（参考模型不应用 LD 正则化）

        Returns:
            包含 5 个张量的元组:
            - chosen_logps: 选中样本的对数概率
            - rejected_logps: 拒绝样本的对数概率
            - chosen_logits: 选中样本的 logits
            - rejected_logits: 拒绝样本的 logits
            - chosen_logps_avg: 选中样本的平均对数概率（用于 SFT loss 计算）
        """
        if self.finetuning_args.use_ref_model:
            batch = nested_detach(batch, clone=True)  # 避免梯度计算错误

        labels = batch.pop("labels")  # DPO 不需要在 forward 中计算 loss
        # 在 CPU 上进行计算以利用 KTransformers 的优化
        all_logits: torch.Tensor = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        all_logits = all_logits.to("cpu")
        labels = labels.to(all_logits.device)
        all_logps, valid_length = get_batch_logps(
            logits=all_logits, labels=labels, ld_alpha=(self.ld_alpha if not is_ref_model else None)
        )
        # 对于 IPO、ORPO、SimPO 损失类型，使用平均对数概率
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        # 将批次分为选中（chosen）和拒绝（rejected）两部分
        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)

        if self.loss_type in ["ipo", "orpo", "simpo"]:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps
        else:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length
