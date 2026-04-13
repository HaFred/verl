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

import logging
from typing import Any

from verl.experimental.agent_loop.strategies.base import GenerationStrategy
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)


class BeamSearchStrategy(GenerationStrategy):
    """Beam search generation strategy using vLLM's native beam search.

    Sets ``use_beam_search=True`` and ``best_of=num_beams`` in sampling params.
    vLLM internally explores ``best_of`` candidates and returns the top ``n``
    results. With ``n=1`` (default), vLLM selects the best candidate.

    Args:
        num_beams: Number of beams to explore. Maps to vLLM's ``best_of``.
        n: Number of output sequences to return per prompt. Defaults to 1.
        length_penalty: Exponential penalty to sequence length. Values > 1.0
            favor longer sequences, < 1.0 favor shorter. Defaults to 1.0.
    """

    def __init__(self, num_beams: int = 4, n: int = 1, length_penalty: float = 1.0):
        self.num_beams = num_beams
        self.n = n
        self.length_penalty = length_penalty

    def build_sampling_params(self, base_params: dict[str, Any], prompt_ids: list[int]) -> dict[str, Any]:
        params = dict(base_params)
        params["use_beam_search"] = True
        params["best_of"] = self.num_beams
        params["n"] = self.n
        params["length_penalty"] = self.length_penalty
        # Beam search requires temperature=0 in vLLM
        params["temperature"] = 0.0
        # top_p and top_k are not used with beam search
        params.pop("top_p", None)
        params.pop("top_k", None)
        return params

    def select_output(self, outputs: list[TokenOutput]) -> TokenOutput:
        if len(outputs) == 1:
            return outputs[0]
        # When n > 1, select the candidate with highest cumulative log probability.
        # TokenOutput.log_probs is a list of per-token log probs; sum gives cumulative.
        best = outputs[0]
        best_score = sum(best.log_probs) if best.log_probs else float("-inf")
        for candidate in outputs[1:]:
            score = sum(candidate.log_probs) if candidate.log_probs else float("-inf")
            if score > best_score:
                best = candidate
                best_score = score
        return best
