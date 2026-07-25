# ==============================================================================
# SFT 评估指标模块
# ==============================================================================
# 功能：提供训练过程中的评估指标计算
# - ComputeAccuracy: 计算 token 级别准确率
# - ComputeSimilarity: 计算文本相似度（ROUGE、BLEU）
# ==============================================================================

# Copyright 2025 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from transformers.utils import is_nltk_available

from ...extras.constants import IGNORE_INDEX  # 用于忽略 padding token 的损失计算（-100）
from ...extras.misc import numpify  # 将张量转换为 numpy 数组
from ...extras.packages import is_jieba_available, is_rouge_available


# ============================================================================
# 类型检查（IDE 提示用）
# ============================================================================
if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


# ============================================================================
# 可选依赖导入（根据安装情况）
# ============================================================================
if is_jieba_available():
    import jieba  # type: ignore
    # jieba: 中文分词库，用于计算 ROUGE 指标


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore
    # NLTK: 用于计算 BLEU-4 指标


if is_rouge_available():
    from rouge_chinese import Rouge  # type: ignore
    # rouge_chinese: 中文 ROUGE 指标计算库


# ============================================================================
# Logits 处理函数
# ============================================================================
def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    """
    处理模型输出的 logits，返回最大概率的 token

    用于准确率计算，减少内存占用

    Args:
        logits: 模型输出，形状为 (batch_size, seq_len, vocab_size)
        labels: 标签，形状为 (batch_size, seq_len)

    Returns:
        最大概率的 token IDs，形状为 (batch_size, seq_len)

    说明:
        - logits[0]: 正常的模型输出 (batch_size, seq_len, vocab_size)
        - logits[1]: MoE 模型的辅助损失
    """
    # 处理不同格式的 logits 输出
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss（MoE 模型有辅助损失）
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    # 返回最大概率的 token（沿 vocab_size 维度）
    return torch.argmax(logits, dim=-1)


# ============================================================================
# 准确率计算类
# ============================================================================
@dataclass
class ComputeAccuracy:
    """
    计算 token 级别的准确率

    支持 batch_eval_metrics（批量评估）

    用法:
        >>> compute_accuracy = ComputeAccuracy()
        >>> metrics = compute_accuracy(eval_predictions)
        >>> print(metrics)  # {"accuracy": 0.85}
    """

    def _dump(self) -> Optional[dict[str, float]]:
        """
        输出累积的准确率结果

        Returns:
            平均准确率字典，或 None（首次调用）
        """
        result = None
        if hasattr(self, "score_dict"):
            # 计算所有 batch 的平均准确率
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        # 重置分数字典
        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        # 初始化时重置分数
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        """
        计算准确率

        Args:
            eval_preds: 评估预测结果
                - predictions: 模型预测的 token IDs
                - label_ids: 真实标签的 token IDs
            compute_result: 是否计算并返回结果

        Returns:
            准确率字典 {"accuracy": 0.xxx}

        说明:
            - 预测和标签需要错位对齐：pred[:-1] vs label[1:]
            - 忽略 padding token（IGNORE_INDEX）
        """
        # 将张量转换为 numpy 数组
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        # 逐样本计算准确率
        for i in range(len(preds)):
            # 错位对齐：预测位置 t 对应标签位置 t+1
            pred, label = preds[i, :-1], labels[i, 1:]

            # 创建掩码：只计算非 padding 的位置
            label_mask = label != IGNORE_INDEX

            # 计算准确率（预测正确的比例）
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


# ============================================================================
# 相似度计算类
# ============================================================================
@dataclass
class ComputeSimilarity:
    """
    计算文本相似度分数（ROUGE 和 BLEU）

    支持 batch_eval_metrics（批量评估）

    指标:
        - rouge-1: 单词重叠（unigram）
        - rouge-2: 双词组重叠（bigram）
        - rouge-l: 最长公共子序列
        - bleu-4: 4-gram BLEU 分数

    用法:
        >>> compute_similarity = ComputeSimilarity(tokenizer)
        >>> metrics = compute_similarity(eval_predictions)
        >>> print(metrics)
        # {"rouge-1": 85.2, "rouge-2": 72.1, "rouge-l": 83.5, "bleu-4": 68.9}
    """

    tokenizer: "PreTrainedTokenizer"  # 用于解码 token IDs 为文本

    def _dump(self) -> Optional[dict[str, float]]:
        """
        输出累积的相似度分数

        Returns:
            平均分数字典，或 None（首次调用）
        """
        result = None
        if hasattr(self, "score_dict"):
            # 计算所有 batch 的平均分数
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        # 重置分数字典
        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        # 初始化时重置分数
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        """
        计算文本相似度分数

        Args:
            eval_preds: 评估预测结果
            compute_result: 是否计算并返回结果

        Returns:
            相似度分数字典

        流程:
            1. 将 IGNORE_INDEX 替换为 pad_token_id
            2. 解码预测和标签为文本
            3. 使用 jieba 分词（中文）
            4. 计算 ROUGE 分数
            5. 计算 BLEU-4 分数
        """
        # 将张量转换为 numpy 数组
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        # 将 IGNORE_INDEX (-100) 替换为 pad_token_id，以便解码
        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        # 解码预测和标签为文本
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 逐样本计算相似度分数
        for pred, label in zip(decoded_preds, decoded_labels):
            # 使用 jieba 分词（中文分词）
            hypothesis = list(jieba.cut(pred))      # 预测文本
            reference = list(jieba.cut(label))      # 真实文本

            # 计算 ROUGE 分数
            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                # 空文本处理
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            # 记录 ROUGE 分数（F1 值 × 100）
            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            # 计算 BLEU-4 分数
            bleu_score = sentence_bleu(
                [list(label)],  # 参考翻译（列表的列表）
                list(pred),     # 预测翻译
                smoothing_function=SmoothingFunction().method3  # 平滑方法
            )
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()
