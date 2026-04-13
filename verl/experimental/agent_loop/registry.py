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

"""Component registry for composable rollout pipeline.

Each component type (generation strategy, reward pipeline, token processor) has
its own registry. Components are registered by name and can be instantiated from
a RecipeConfig that selects one implementation for each slot.

Usage::

    from verl.experimental.agent_loop.registry import RecipeConfig, build_components

    # From a Hydra/OmegaConf config:
    recipe = RecipeConfig(
        generation_strategy="beam_search",
        generation_strategy_kwargs={"num_beams": 4},
        reward_pipeline="single",
        token_processor="default",
    )
    strategy, pipeline, processor = build_components(recipe, reward_handles=...)

    # Pass to AgentLoopManager.create(..., generation_strategy=strategy, ...)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import ray

from verl.experimental.agent_loop.processors.base import TokenProcessor
from verl.experimental.agent_loop.processors.default import DefaultTokenProcessor
from verl.experimental.agent_loop.rewards.base import RewardPipeline
from verl.experimental.agent_loop.rewards.multi_stage import MultiStageRewardPipeline
from verl.experimental.agent_loop.rewards.single import SingleRewardPipeline
from verl.experimental.agent_loop.strategies.base import GenerationStrategy
from verl.experimental.agent_loop.strategies.beam_search import BeamSearchStrategy
from verl.experimental.agent_loop.strategies.sampling import SamplingStrategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Component registries
# ---------------------------------------------------------------------------

_strategy_registry: dict[str, type[GenerationStrategy]] = {
    "sampling": SamplingStrategy,
    "beam_search": BeamSearchStrategy,
}

_reward_registry: dict[str, type[RewardPipeline]] = {
    "single": SingleRewardPipeline,
    "multi_stage": MultiStageRewardPipeline,
}

_processor_registry: dict[str, type[TokenProcessor]] = {
    "default": DefaultTokenProcessor,
}


def register_strategy(name: str):
    """Decorator to register a GenerationStrategy class."""

    def decorator(cls):
        _strategy_registry[name] = cls
        return cls

    return decorator


def register_reward(name: str):
    """Decorator to register a RewardPipeline class."""

    def decorator(cls):
        _reward_registry[name] = cls
        return cls

    return decorator


def register_processor(name: str):
    """Decorator to register a TokenProcessor class."""

    def decorator(cls):
        _processor_registry[name] = cls
        return cls

    return decorator


# ---------------------------------------------------------------------------
# RecipeConfig
# ---------------------------------------------------------------------------


@dataclass
class RecipeConfig:
    """Configuration for selecting and composing pipeline components.

    Each field selects a registered component by name. The ``*_kwargs`` fields
    are forwarded to the component constructor.

    Attributes:
        generation_strategy: Name of the generation strategy. Default "sampling".
        generation_strategy_kwargs: Constructor kwargs for the strategy.
        reward_pipeline: Name of the reward pipeline. Default "single".
        reward_pipeline_kwargs: Constructor kwargs for the pipeline.
        token_processor: Name of the token processor. Default "default".
        token_processor_kwargs: Constructor kwargs for the processor.
    """

    generation_strategy: str = "sampling"
    generation_strategy_kwargs: dict[str, Any] = field(default_factory=dict)
    reward_pipeline: str = "single"
    reward_pipeline_kwargs: dict[str, Any] = field(default_factory=dict)
    token_processor: str = "default"
    token_processor_kwargs: dict[str, Any] = field(default_factory=dict)


def build_components(
    recipe: RecipeConfig,
    reward_handles: Optional[list[ray.actor.ActorHandle]] = None,
) -> tuple[GenerationStrategy, RewardPipeline, TokenProcessor]:
    """Instantiate pipeline components from a RecipeConfig.

    Args:
        recipe: Configuration selecting which components to use.
        reward_handles: Ray actor handles for async reward workers.
            Passed to reward pipelines that need them (e.g., SingleRewardPipeline).

    Returns:
        A tuple of (GenerationStrategy, RewardPipeline, TokenProcessor).
    """
    # Build generation strategy
    strategy_cls = _strategy_registry.get(recipe.generation_strategy)
    if strategy_cls is None:
        raise ValueError(
            f"Unknown generation strategy '{recipe.generation_strategy}'. "
            f"Available: {list(_strategy_registry.keys())}"
        )
    strategy = strategy_cls(**recipe.generation_strategy_kwargs)

    # Build reward pipeline
    reward_cls = _reward_registry.get(recipe.reward_pipeline)
    if reward_cls is None:
        raise ValueError(
            f"Unknown reward pipeline '{recipe.reward_pipeline}'. "
            f"Available: {list(_reward_registry.keys())}"
        )
    reward_kwargs = dict(recipe.reward_pipeline_kwargs)
    # Inject reward_handles for pipelines that accept it
    if reward_handles is not None and "reward_loop_worker_handles" not in reward_kwargs:
        reward_kwargs["reward_loop_worker_handles"] = reward_handles
    try:
        pipeline = reward_cls(**reward_kwargs)
    except TypeError:
        # Pipeline class doesn't accept reward_loop_worker_handles
        reward_kwargs.pop("reward_loop_worker_handles", None)
        pipeline = reward_cls(**reward_kwargs)

    # Build token processor
    processor_cls = _processor_registry.get(recipe.token_processor)
    if processor_cls is None:
        raise ValueError(
            f"Unknown token processor '{recipe.token_processor}'. "
            f"Available: {list(_processor_registry.keys())}"
        )
    processor = processor_cls(**recipe.token_processor_kwargs)

    return strategy, pipeline, processor
