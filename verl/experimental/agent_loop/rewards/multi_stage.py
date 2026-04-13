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

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

from verl.experimental.agent_loop.rewards.base import RewardPipeline, RewardScorer

if TYPE_CHECKING:
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput

logger = logging.getLogger(__name__)


class MultiStageRewardPipeline(RewardPipeline):
    """Multi-stage reward pipeline that composes multiple RewardScorer instances.

    Each scorer produces a scalar reward for a specific stage (e.g., format,
    correctness, length penalty). Scores are combined via configurable weights
    and aggregation strategy.

    Args:
        scorers: List of (name, scorer, weight) tuples. Each scorer is called
            independently and the weighted scores are aggregated.
        aggregation: How to combine weighted scores. One of "sum", "min", "product".
            Defaults to "sum".
    """

    def __init__(
        self,
        scorers: list[tuple[str, RewardScorer, float]],
        aggregation: str = "sum",
    ):
        self.scorers = scorers
        self.aggregation = aggregation
        assert aggregation in ("sum", "min", "product"), f"Unknown aggregation: {aggregation}"

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
        # If reward is already computed by the agent loop, use it
        if output.reward_score is not None:
            return output.reward_score, output.extra_fields.get("reward_extra_info", {})

        if not self.scorers:
            return None, {}

        stage_scores: dict[str, float] = {}
        for name, scorer, weight in self.scorers:
            score = await scorer.score(
                output=output,
                prompts=prompts,
                responses=responses,
                attention_mask=attention_mask,
                input_ids=input_ids,
                position_ids=position_ids,
                context=context,
            )
            stage_scores[name] = score

        # Aggregate weighted scores
        weighted_scores = [stage_scores[name] * weight for name, _, weight in self.scorers]

        if self.aggregation == "sum":
            combined = sum(weighted_scores)
        elif self.aggregation == "min":
            combined = min(weighted_scores)
        elif self.aggregation == "product":
            result = 1.0
            for ws in weighted_scores:
                result *= ws
            combined = result

        extra_info = {
            "stage_scores": stage_scores,
            "aggregation": self.aggregation,
            "combined_reward": combined,
        }

        return combined, extra_info
