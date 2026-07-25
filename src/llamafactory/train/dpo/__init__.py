# Copyright 2025 the LlamaFactory team.
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
DPO (Direct Preference Optimization，直接偏好优化) 训练模块

该模块提供了基于人类反馈的强化学习（RLHF）简化方案 DPO 的训练实现。
DPO 通过直接优化偏好数据来对齐模型输出，无需训练独立的奖励模型。

主要组件:
- run_dpo: DPO 训练的主工作流函数
- CustomDPOTrainer: 标准 DPO 训练器实现
- KDPOTrainer: KTransformers 加速的 DPO 训练器
"""

from .workflow import run_dpo


__all__ = ["run_dpo"]
