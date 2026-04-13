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

from typing import Any

from verl.experimental.agent_loop.strategies.base import GenerationStrategy
from verl.workers.rollout.replica import TokenOutput


class SamplingStrategy(GenerationStrategy):
    """Default generation strategy — standard sampling with a single output per prompt.

    Passes sampling parameters through unchanged and returns the single output as-is.
    """

    def build_sampling_params(self, base_params: dict[str, Any], prompt_ids: list[int]) -> dict[str, Any]:
        return base_params

    def select_output(self, outputs: list[TokenOutput]) -> TokenOutput:
        assert len(outputs) == 1, f"SamplingStrategy expects exactly 1 output, got {len(outputs)}"
        return outputs[0]
