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

import random
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import ray
import torch
from tensordict import TensorDict

from verl.experimental.agent_loop.rewards.base import RewardPipeline
from verl.protocol import DataProto

if TYPE_CHECKING:
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput


class SingleRewardPipeline(RewardPipeline):
    """Default reward pipeline — delegates to a single async reward worker.

    This extracts the existing ``_compute_score()`` logic from ``AgentLoopWorker``.
    If no reward worker handles are provided, reward_score from the AgentLoopOutput
    is used as-is (e.g., when the agent loop itself computes the reward).
    """

    def __init__(self, reward_loop_worker_handles: Optional[list[ray.actor.ActorHandle]] = None):
        self.reward_loop_worker_handles = reward_loop_worker_handles

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
        if output.reward_score is not None:
            return output.reward_score, output.extra_fields.get("reward_extra_info", {})

        if self.reward_loop_worker_handles is None:
            return None, {}

        batch = TensorDict(
            {
                "prompts": prompts,
                "responses": responses,
                "attention_mask": attention_mask,
                "input_ids": input_ids,
                "position_ids": position_ids,
            },
            batch_size=1,
        )
        non_tensor_batch = {
            **{k: np.array([v]) for k, v in context.items()},
            "__num_turns__": np.array([output.num_turns]),
            "tool_extra_fields": np.array([output.extra_fields], dtype=object),
        }

        data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        selected_handle = random.choice(self.reward_loop_worker_handles)
        result = await selected_handle.compute_score.remote(data)

        reward_score = result["reward_score"]
        reward_extra_info = result["reward_extra_info"]
        return reward_score, reward_extra_info
