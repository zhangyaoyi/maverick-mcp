"""Aliyun Bailian (DashScope) LLM provider with intelligent model selection.

This module provides integration with Aliyun's Bailian API (OpenAI-compatible)
for accessing Qwen and other models with automatic model selection based on task requirements.
"""

import logging
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from maverick_mcp.providers.openrouter_provider import TaskType

logger = logging.getLogger(__name__)


class ModelProfile(BaseModel):
    """Profile for a Bailian model with capabilities and costs."""

    model_id: str = Field(description="Bailian model identifier")
    name: str = Field(description="Human-readable model name")
    context_length: int = Field(description="Maximum context length in tokens")
    cost_per_million_input: float = Field(
        description="Cost per million input tokens in USD"
    )
    cost_per_million_output: float = Field(
        description="Cost per million output tokens in USD"
    )
    speed_rating: int = Field(description="Speed rating 1-10 (10 being fastest)")
    quality_rating: int = Field(description="Quality rating 1-10 (10 being best)")
    best_for: list[TaskType] = Field(description="Task types this model excels at")
    temperature: float = Field(default=0.3, description="Default temperature")


# Model profiles for Bailian models
# Pricing in USD per million tokens (approximate DashScope pricing)
BAILIAN_MODEL_PROFILES: dict[str, ModelProfile] = {
    # Primary workhorse — used for all tasks by default
    "qwen3.5-plus": ModelProfile(
        model_id="qwen3.5-plus",
        name="Qwen3.5 Plus",
        context_length=131_072,
        cost_per_million_input=0.50,
        cost_per_million_output=1.50,
        speed_rating=9,
        quality_rating=8,
        best_for=[
            TaskType.GENERAL,
            TaskType.MARKET_ANALYSIS,
            TaskType.TECHNICAL_ANALYSIS,
            TaskType.SENTIMENT_ANALYSIS,
            TaskType.QUICK_ANSWER,
            TaskType.QUERY_CLASSIFICATION,
            TaskType.RESULT_SYNTHESIS,
            TaskType.PORTFOLIO_OPTIMIZATION,
            TaskType.RISK_ASSESSMENT,
        ],
        temperature=0.3,
    ),
    # High-quality tier — deep reasoning and orchestration
    "qwen3-max-2026-01-23": ModelProfile(
        model_id="qwen3-max-2026-01-23",
        name="Qwen3 Max",
        context_length=131_072,
        cost_per_million_input=2.00,
        cost_per_million_output=6.00,
        speed_rating=7,
        quality_rating=9,
        best_for=[
            TaskType.DEEP_RESEARCH,
            TaskType.COMPLEX_REASONING,
            TaskType.MULTI_AGENT_ORCHESTRATION,
        ],
        temperature=0.3,
    ),
    # Balanced tier — analysis and synthesis
    "glm-5": ModelProfile(
        model_id="glm-5",
        name="GLM-5",
        context_length=131_072,
        cost_per_million_input=0.80,
        cost_per_million_output=2.56,
        speed_rating=7,
        quality_rating=9,
        best_for=[
            TaskType.MARKET_ANALYSIS,
            TaskType.TECHNICAL_ANALYSIS,
            TaskType.RESULT_SYNTHESIS,
        ],
        temperature=0.3,
    ),
}

_DEFAULT_MODEL = "qwen3.5-plus"
_FALLBACK_ORDER = ["qwen3.5-plus", "qwen3-max-2026-01-23", "glm-5"]


class BailianProvider:
    """Provider for Aliyun Bailian (DashScope) API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def get_llm(
        self,
        task_type: TaskType = TaskType.GENERAL,
        prefer_fast: bool = False,
        prefer_cheap: bool = True,
        prefer_quality: bool = False,
        model_override: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> ChatOpenAI:
        """Get a ChatOpenAI instance pointed at the Bailian endpoint.

        Falls back through the model list if the primary model is unavailable.
        """
        if model_override:
            model_id = model_override
        elif prefer_quality or task_type in (
            TaskType.DEEP_RESEARCH,
            TaskType.COMPLEX_REASONING,
            TaskType.MULTI_AGENT_ORCHESTRATION,
        ):
            model_id = "qwen3-max-2026-01-23"
        elif task_type in (
            TaskType.MARKET_ANALYSIS,
            TaskType.TECHNICAL_ANALYSIS,
            TaskType.RESULT_SYNTHESIS,
        ) and not prefer_cheap:
            model_id = "glm-5"
        else:
            model_id = _DEFAULT_MODEL

        profile = BAILIAN_MODEL_PROFILES.get(
            model_id, BAILIAN_MODEL_PROFILES[_DEFAULT_MODEL]
        )
        final_temperature = temperature if temperature is not None else profile.temperature

        logger.info(
            "Using Bailian (Aliyun) with model: %s for task: %s", model_id, task_type
        )

        def _make_llm(mid: str, temp: float) -> ChatOpenAI:
            return ChatOpenAI(
                model=mid,
                temperature=temp,
                max_tokens=max_tokens,
                openai_api_base=self.base_url,
                openai_api_key=self.api_key,
                streaming=True,
            )

        primary_llm = _make_llm(model_id, final_temperature)

        # Build fallback chain from remaining models in priority order
        fallback_llms = [
            _make_llm(mid, BAILIAN_MODEL_PROFILES[mid].temperature)
            for mid in _FALLBACK_ORDER
            if mid != model_id and mid in BAILIAN_MODEL_PROFILES
        ]

        if fallback_llms:
            return primary_llm.with_fallbacks(fallback_llms)

        return primary_llm


def get_bailian_llm(
    api_key: str,
    task_type: TaskType = TaskType.GENERAL,
    prefer_fast: bool = False,
    prefer_cheap: bool = True,
    prefer_quality: bool = False,
    **kwargs: Any,
) -> ChatOpenAI:
    """Convenience function to get a Bailian LLM instance."""
    provider = BailianProvider(api_key)
    return provider.get_llm(
        task_type=task_type,
        prefer_fast=prefer_fast,
        prefer_cheap=prefer_cheap,
        prefer_quality=prefer_quality,
        **kwargs,
    )
