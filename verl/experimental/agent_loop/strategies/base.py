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

from abc import ABC, abstractmethod
from typing import Any

from verl.workers.rollout.replica import TokenOutput


class GenerationStrategy(ABC):
    """Controls how LLM generation is configured and how multi-candidate outputs are handled.

    Subclasses modify sampling parameters before generation and select the final
    output when the engine returns multiple candidates (e.g., beam search).
    """

    @abstractmethod
    def build_sampling_params(self, base_params: dict[str, Any], prompt_ids: list[int]) -> dict[str, Any]:
        """Modify or replace sampling parameters before generation.

        Args:
            base_params: Default sampling parameters from rollout config.
            prompt_ids: The prompt token IDs (useful for length-dependent strategies).

        Returns:
            Modified sampling parameters dict.
        """
        ...

    @abstractmethod
    def select_output(self, outputs: list[TokenOutput]) -> TokenOutput:
        """Select the final output from potentially multiple candidates.

        For standard sampling, there is exactly one output. For beam search,
        there may be ``best_of`` candidates to choose from.

        Note:
            This is an extension point for custom AgentLoopBase subclasses.
            The default agent loops (SingleTurnAgentLoop, ToolAgentLoop) do not
            call this method because vLLM's ``best_of`` already selects the best
            beam internally. Subclasses that use ``n > 1`` to retrieve multiple
            candidates should call ``strategy.select_output()`` explicitly.

        Args:
            outputs: List of candidate outputs from the engine.

        Returns:
            The selected TokenOutput.
        """
        ...
