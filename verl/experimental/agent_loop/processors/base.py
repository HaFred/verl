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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from verl.workers.config import RolloutConfig

if TYPE_CHECKING:
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput


@dataclass
class ProcessedOutput:
    """Result of token processing — padded tensors ready for batch assembly."""

    prompt_ids: torch.Tensor  # [1, prompt_length]
    response_ids: torch.Tensor  # [1, response_length]
    input_ids: torch.Tensor  # [1, prompt_length + response_length]
    position_ids: torch.Tensor  # [1, seq_len] or [1, ndim, seq_len]
    response_mask: torch.Tensor  # [1, response_length]
    attention_mask: torch.Tensor  # [1, prompt_length + response_length]
    response_logprobs: Optional[torch.Tensor]  # [1, response_length] or None
    routed_experts: Optional[torch.Tensor]  # [1, total_length, layers, topk] or None
    multi_modal_inputs: dict[str, torch.Tensor]


class TokenProcessor(ABC):
    """Handles tokenizer-specific pre/post-processing of generation outputs.

    This includes padding prompt/response to fixed lengths, computing attention
    masks, position IDs, response masks, and multi-modal inputs.
    Subclasses can override for different tokenizer behaviors.
    """

    @abstractmethod
    def pad_and_format(
        self,
        output: AgentLoopOutput,
        rollout_config: RolloutConfig,
        tokenizer: Any,
        processor: Any,
    ) -> ProcessedOutput:
        """Pad and format a single agent loop output into fixed-length tensors.

        Args:
            output: Raw agent loop output with variable-length token lists.
            rollout_config: Rollout configuration (prompt_length, response_length, etc.).
            tokenizer: HuggingFace tokenizer.
            processor: HuggingFace processor (for multimodal models), or None.

        Returns:
            ProcessedOutput with padded tensors ready for batch stacking.
        """
        ...
