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

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from verl.experimental.agent_loop.processors.base import ProcessedOutput, TokenProcessor
from verl.utils.model import compute_position_id_with_mask
from verl.workers.config import RolloutConfig

if TYPE_CHECKING:
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput


class DefaultTokenProcessor(TokenProcessor):
    """Default token processor — extracts the padding/formatting logic from AgentLoopWorker.

    Handles left-padding prompts, right-padding responses, computing attention masks,
    position IDs, response masks, multi-modal inputs, and routed expert tensors.
    """

    def pad_and_format(
        self,
        output: AgentLoopOutput,
        rollout_config: RolloutConfig,
        tokenizer: Any,
        processor: Any,
    ) -> ProcessedOutput:
        # 1. Left-pad prompt
        tokenizer.padding_side = "left"
        prompt_output = tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=rollout_config.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        # 2. Right-pad response
        tokenizer.padding_side = "right"
        response_output = tokenizer.pad(
            {"input_ids": output.response_ids},
            padding="max_length",
            max_length=rollout_config.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        # 3. Right-pad response mask
        response_mask_output = tokenizer.pad(
            {"input_ids": output.response_mask},
            padding="max_length",
            max_length=rollout_config.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        # 4. Pad response logprobs
        response_logprobs = None
        if output.response_logprobs is not None:
            pad_size = rollout_config.response_length - len(output.response_logprobs)
            response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        # 5. Compute masks and concatenated tensors
        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        # 6. Handle routed experts
        routed_experts = self._pad_routed_experts(output, prompt_output, input_ids)

        # 7. Compute multi-modal inputs
        multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids, tokenizer, processor)

        # 8. Compute position IDs
        position_ids = self._compute_position_ids(input_ids, attention_mask, multi_modal_inputs, processor)

        return ProcessedOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_inputs=multi_modal_inputs,
        )

    def _pad_routed_experts(self, output: AgentLoopOutput, prompt_output, input_ids) -> torch.Tensor | None:
        if output.routed_experts is None:
            return None

        total_length = input_ids.shape[1]
        length, layer_num, topk_num = output.routed_experts.shape

        if isinstance(output.routed_experts, np.ndarray):
            routed_experts_array = output.routed_experts
            if not routed_experts_array.flags.writeable:
                routed_experts_array = routed_experts_array.copy()
            experts_tensor = torch.from_numpy(routed_experts_array)
        elif isinstance(output.routed_experts, torch.Tensor):
            experts_tensor = output.routed_experts
        else:
            raise TypeError(f"Unsupported type for routed_experts: {type(output.routed_experts)}")

        routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)
        start_pos = prompt_output["input_ids"].shape[1] - len(output.prompt_ids)
        end_pos = min(start_pos + length, total_length)

        if start_pos < 0 or end_pos > total_length:
            raise ValueError(
                f"Invalid position range: start_pos={start_pos}, end_pos={end_pos}, total_length={total_length}"
            )
        routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)
        return routed_experts

    def _compute_multi_modal_inputs(
        self, output: AgentLoopOutput, input_ids: torch.Tensor, tokenizer: Any, processor: Any
    ) -> dict[str, torch.Tensor]:
        multi_modal_inputs = {}
        if processor is None:
            return multi_modal_inputs

        images = output.multi_modal_data.get("images")
        videos = output.multi_modal_data.get("videos")
        if videos is not None:
            videos, video_metadatas = zip(*videos, strict=False)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None

        current_text = tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
        multi_modal_inputs = processor(
            text=[current_text],
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_sample_frames=False,
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))

        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            multi_modal_inputs["images_seqlens"] = images_seqlens

        return multi_modal_inputs

    def _compute_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        multi_modal_inputs: dict[str, torch.Tensor],
        processor: Any,
    ) -> torch.Tensor:
        if processor is None:
            return compute_position_id_with_mask(attention_mask)

        multi_modal_kwargs = {
            "image_grid_thw": multi_modal_inputs.get("image_grid_thw"),
            "video_grid_thw": multi_modal_inputs.get("video_grid_thw"),
        }
        mm_token_type_ids = multi_modal_inputs.pop("mm_token_type_ids", None)
        if mm_token_type_ids is not None:
            mm_token_type_ids = torch.zeros_like(input_ids)
            mm_token_type_ids[0][input_ids[0] == processor.image_token_id] = 1
            mm_token_type_ids[0][input_ids[0] == processor.video_token_id] = 2
            multi_modal_kwargs["mm_token_type_ids"] = mm_token_type_ids

        vision_position_ids, _ = processor.get_rope_index(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **multi_modal_kwargs,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)  # (3, 1, seq_len) => (1, 3, seq_len)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
        return position_ids
