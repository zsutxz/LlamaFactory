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
自定义 DPO 训练器

该模块实现了基于 HuggingFace TRL 库的扩展 DPO 训练器，支持：
- 多种损失类型：DPO、IPO、ORPO、SimPO、BCO
- 自定义优化器和调度器
- BAdam 优化器支持
- 参考模型管理
- 丰富的训练指标记录
"""

import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Literal, Optional, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps, nested_detach


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomDPOTrainer(DPOTrainer):
    """
    自定义 DPO 训练器

    继承自 TRL 的 DPOTrainer，提供更多自定义选项：
    - 支持多种偏好优化损失类型
    - 自定义优化器和调度器创建
    - BAdam 优化器支持
    - 可选的参考模型
    - 灵活的损失组合（如 BCO + DPO）
    """
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module"]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        """
        初始化自定义 DPO 训练器

        Args:
            model: 要训练的策略模型
            ref_model: 参考模型（可选），用于计算参考对数概率
            finetuning_args: 微调参数配置
            processor: 处理器（用于多模态模型）
            disable_dropout: 是否禁用 dropout
            **kwargs: 其他传递给父类的参数
        """
        # 兼容 transformers 4.46+ 的 API 变化
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        # 禁用 dropout 以保证训练稳定性
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        # 保存微调参数
        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"
        self.reference_free = False
        self.use_dpo_data_collator = True  # 避免警告
        self.generate_during_eval = False  # 禁用评估时的生成
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # DPO 超参数
        self.beta = finetuning_args.pref_beta  # DPO 的 beta 参数，控制对齐强度
        self.loss_type = finetuning_args.pref_loss  # 损失类型：dpo, ipo, orpo, simpo
        self.ftx_gamma = finetuning_args.pref_ftx  # SFT loss 的权重
        self.bco_gemma = finetuning_args.pref_bco_weight  # BCO loss 的权重
        self.label_smoothing = finetuning_args.dpo_label_smoothing  # 标签平滑
        self.simpo_gamma = finetuning_args.simpo_gamma  # SimPO 的 gamma 参数
        self.ld_alpha = finetuning_args.ld_alpha  # LD (label debias) 正则化系数

        Trainer.__init__(self, model=model, **kwargs)
        self.model_accepts_loss_kwargs = False  # 覆盖 trainer 的默认行为
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # 忽略参考模型的 gc 警告

        # 准备参考模型
        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # 量化模型已经在正确的设备上
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        # 添加处理器保存回调
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        # 添加 BAdam 优化器回调
        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        # 初始化 BCO 的运行统计
        if self.bco_gemma >= 1e-6:
            from trl.trainer import RunningMoments

            self.running = RunningMoments(self.accelerator)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        """创建自定义优化器"""
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        """创建自定义学习率调度器"""
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        """获取训练采样器，支持禁用数据打乱"""
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def get_batch_samples(self, *args, **kwargs):
        r"""使用标准 Trainer 的方法替代 DPO Trainer 的方法"""
        return Trainer.get_batch_samples(self, *args, **kwargs)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""计算 ORPO (Odds Ratio Preference Optimization) 损失

        ORPO 结合了 SFT loss 和 odds ratio loss，无需参考模型。

        Args:
            chosen_logps: 选中样本的对数概率
            rejected_logps: 拒绝样本的对数概率

        Returns:
            ORPO 损失值
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""计算 SimPO (Simple Preference Optimization) 损失

        SimPO 使用固定的奖励差距（gamma）替代参考模型，更简单高效。

        Args:
            chosen_logps: 选中样本的对数概率
            rejected_logps: 拒绝样本的对数概率

        Returns:
            SimPO 损失值
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss

    def bco_loss(
        self,
        chosen_logps: "torch.Tensor",
        rejected_logps: "torch.Tensor",
        reference_chosen_logps: "torch.Tensor",
        reference_rejected_logps: "torch.Tensor",
    ) -> "torch.Tensor":
        r"""计算 BCO (Behavior Clone Optimization) 损失

        BCO 使用动态基线来稳定训练。

        Args:
            chosen_logps: 策略模型的选中样本对数概率
            rejected_logps: 策略模型的拒绝样本对数概率
            reference_chosen_logps: 参考模型的选中样本对数概率
            reference_rejected_logps: 参考模型的拒绝样本对数概率

        Returns:
            BCO 损失值
        """
        chosen_logratios = chosen_logps - reference_chosen_logps
        rejected_logratios = rejected_logps - reference_rejected_logps
        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios
        rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
        self.running.update(rewards)  # 更新基线
        delta = self.running.mean
        bco_loss = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
            -(self.beta * rejected_logratios - delta)
        )
        return bco_loss

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""计算偏好学习的损失

        根据损失类型计算相应的偏好损失和奖励值。

        Args:
            policy_chosen_logps: 策略模型的选中样本对数概率
            policy_rejected_logps: 策略模型的拒绝样本对数概率
            reference_chosen_logps: 参考模型的选中样本对数概率（可选）
            reference_rejected_logps: 参考模型的拒绝样本对数概率（可选）

        Returns:
            包含损失、选中奖励、拒绝奖励的元组
        """
        if not self.finetuning_args.use_ref_model:
            # 不使用参考模型的损失类型
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")

            chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            # 使用参考模型的损失类型
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )

            # 如果启用了 BCO 损失，将其与 DPO 损失结合
            if self.bco_gemma > 1e-6:
                bco_losses = self.bco_loss(
                    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
                )
                losses = (losses + bco_losses * self.bco_gemma) / (1.0 + self.bco_gemma)  # 重新加权 W_p 和 W_q

        return losses, chosen_rewards, rejected_rewards

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"], is_ref_model: bool = False
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""计算模型在给定批次上的前向传播输出

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
        all_logits: torch.Tensor = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
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

    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"]
    ) -> tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""计算参考模型的对数概率

        Args:
            model: 当前模型
            batch: 输入批次

        Returns:
            参考模型的选中/拒绝样本对数概率，如果不使用参考模型则返回 (None, None)
        """
        if not self.finetuning_args.use_ref_model:
            return None, None

        if self.ref_model is None:
            # 使用禁用适配器的当前模型作为参考模型
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(
                ref_model, batch, is_ref_model=True
            )

        return reference_chosen_logps, reference_rejected_logps

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
        r"""计算 DPO 损失和其他指标

        Args:
            model: 要训练的模型
            batch: 输入批次
            train_eval: "train" 或 "eval"

        Returns:
            损失值和指标字典
        """
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
        ) = self.concatenated_forward(model, batch)

        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        sft_loss = -policy_chosen_logps_avg
        # 如果启用了 SFT loss 混合
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        # 记录指标
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()
        if self.loss_type == "orpo":
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
            metrics[f"{prefix}odds_ratio_loss"] = ((losses - sft_loss) / self.beta).mean().item()

        return losses.mean(), metrics

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", tuple["torch.Tensor", list["torch.Tensor"]]]:
        r"""计算损失，接受额外的 kwargs"""
        return super().compute_loss(model, inputs, return_outputs)

    @override
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        r"""记录日志，包括存储的指标

        在多 GPU 训练中进行跨进程的指标平均。

        Args:
            logs: 要记录的日志字典
        """
        # logs 要么包含 "loss" 要么包含 "eval_loss"
        train_eval = "train" if "loss" in logs else "eval"
        # 添加平均存储指标到日志
        key_list, metric_list = [], []
        for key, metrics in self._stored_metrics[train_eval].items():
            key_list.append(key)
            metric_list.append(torch.tensor(metrics, dtype=torch.float).to(self.accelerator.device).mean().item())

        del self._stored_metrics[train_eval]
        if len(metric_list) < 10:  # 填充到 10 以便进行 all reduce
            for i in range(10 - len(metric_list)):
                key_list.append(f"dummy_{i}")
                metric_list.append(0.0)

        metric_list = torch.tensor(metric_list, dtype=torch.float).to(self.accelerator.device)
        metric_list = self.accelerator.reduce(metric_list, "mean").tolist()
        for key, metric in zip(key_list, metric_list):  # 添加剩余项
            if not key.startswith("dummy_"):
                logs[key] = metric

        return Trainer.log(self, logs, *args, **kwargs)
