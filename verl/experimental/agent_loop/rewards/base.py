# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput


class RewardScorer(ABC):
    """A single reward scorer that produces a scalar score for a generated output.

    Multiple RewardScorer instances can be composed inside a RewardPipeline
    to support multi-stage reward computation.
    """

    @abstractmethod
    async def score(
        self,
        output: AgentLoopOutput,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        context: dict[str, Any],
    ) -> float:
        """Compute a scalar reward score for one sample.

        Args:
            output: The agent loop output containing generation results.
            prompts: Padded prompt tensor [1, prompt_length].
            responses: Padded response tensor [1, response_length].
            attention_mask: Combined attention mask [1, total_length].
            input_ids: Combined input IDs [1, total_length].
            position_ids: Position IDs tensor.
            context: Additional context (e.g., kwargs from the batch).

        Returns:
            A scalar reward score.
        """
        ...


class RewardPipeline(ABC):
    """Computes reward signals from generated outputs.

    Subclasses define how reward scores are computed and combined.
    The pipeline is called during agent loop postprocessing.
    """

    @abstractmethod
    async def compute_rewards(
        self,
        output: AgentLoopOutput,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        context: dict[str, Any],
    ) -> tuple[Optional[float], dict[str, Any]]:
        """Compute reward signals for one sample.

        Args:
            output: The agent loop output containing generation results.
            prompts: Padded prompt tensor [1, prompt_length].
            responses: Padded response tensor [1, response_length].
            attention_mask: Combined attention mask [1, total_length].
            input_ids: Combined input IDs [1, total_length].
            position_ids: Position IDs tensor.
            context: Additional context (e.g., kwargs from the batch).

        Returns:
            A tuple of (scalar_reward_or_None, extra_info_dict).
            If reward_score is None, downstream code skips rm_scores.
        """
        ...
