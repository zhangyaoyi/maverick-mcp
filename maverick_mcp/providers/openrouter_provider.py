"""OpenRouter LLM provider with intelligent model selection.

This module provides integration with OpenRouter API for accessing various LLMs
with automatic model selection based on task requirements.
"""

import logging
from enum import Enum
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

try:
    from openai import PermissionDeniedError as OpenAIPermissionDeniedError

    _REGION_ERRORS: tuple[type[Exception], ...] = (OpenAIPermissionDeniedError,)
except ImportError:
    _REGION_ERRORS = (Exception,)

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Task types for model selection."""

    # Analysis tasks
    DEEP_RESEARCH = "deep_research"
    MARKET_ANALYSIS = "market_analysis"
    TECHNICAL_ANALYSIS = "technical_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    RISK_ASSESSMENT = "risk_assessment"

    # Synthesis tasks
    RESULT_SYNTHESIS = "result_synthesis"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"

    # Query processing
    QUERY_CLASSIFICATION = "query_classification"
    QUICK_ANSWER = "quick_answer"

    # Complex reasoning
    COMPLEX_REASONING = "complex_reasoning"
    MULTI_AGENT_ORCHESTRATION = "multi_agent_orchestration"

    # Default
    GENERAL = "general"


class ModelProfile(BaseModel):
    """Profile for an LLM model with capabilities and costs."""

    model_id: str = Field(description="OpenRouter model identifier")
    name: str = Field(description="Human-readable model name")
    provider: str = Field(description="Model provider (e.g., anthropic, openai)")
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
    temperature: float = Field(
        default=0.3, description="Default temperature for this model"
    )


# Model profiles for intelligent selection
# Pricing in USD per million tokens (as of March 2026 via OpenRouter)
# Region note: DeepSeek and xAI models excluded — 403 region restrictions observed
MODEL_PROFILES = {
    # ── Primary workhorse ───────────────────────────────────────────────────
    "z-ai/glm-5": ModelProfile(
        model_id="z-ai/glm-5",
        name="GLM-5",
        provider="z-ai",
        context_length=203_000,
        cost_per_million_input=0.80,
        cost_per_million_output=2.56,
        speed_rating=7,
        quality_rating=9,
        best_for=[
            TaskType.DEEP_RESEARCH,
            TaskType.MARKET_ANALYSIS,
            TaskType.TECHNICAL_ANALYSIS,
            TaskType.RESULT_SYNTHESIS,
            TaskType.PORTFOLIO_OPTIMIZATION,
            TaskType.RISK_ASSESSMENT,
            TaskType.COMPLEX_REASONING,
            TaskType.MULTI_AGENT_ORCHESTRATION,
        ],
        temperature=0.3,
    ),
    # ── High-quality tier ───────────────────────────────────────────────────
    "anthropic/claude-sonnet-4.6": ModelProfile(
        model_id="anthropic/claude-sonnet-4.6",
        name="Claude Sonnet 4.6",
        provider="anthropic",
        context_length=1_000_000,
        cost_per_million_input=3.0,
        cost_per_million_output=15.0,
        speed_rating=8,
        quality_rating=9,
        best_for=[
            TaskType.DEEP_RESEARCH,
            TaskType.MARKET_ANALYSIS,
            TaskType.TECHNICAL_ANALYSIS,
            TaskType.MULTI_AGENT_ORCHESTRATION,
            TaskType.RESULT_SYNTHESIS,
            TaskType.PORTFOLIO_OPTIMIZATION,
            TaskType.RISK_ASSESSMENT,
        ],
        temperature=0.3,
    ),
    "google/gemini-3.1-pro-preview": ModelProfile(
        model_id="google/gemini-3.1-pro-preview",
        name="Gemini 3.1 Pro Preview",
        provider="google",
        context_length=1_000_000,
        cost_per_million_input=2.0,
        cost_per_million_output=12.0,
        speed_rating=7,
        quality_rating=9,
        best_for=[
            TaskType.DEEP_RESEARCH,
            TaskType.MARKET_ANALYSIS,
            TaskType.TECHNICAL_ANALYSIS,
            TaskType.COMPLEX_REASONING,
            TaskType.MULTI_AGENT_ORCHESTRATION,
        ],
        temperature=0.3,
    ),
    # ── Fast tier (speed + cost balance) ────────────────────────────────────
    "google/gemini-3-flash-preview": ModelProfile(
        model_id="google/gemini-3-flash-preview",
        name="Gemini 3 Flash Preview",
        provider="google",
        context_length=1_000_000,
        cost_per_million_input=0.50,
        cost_per_million_output=3.0,
        speed_rating=9,
        quality_rating=8,
        best_for=[
            TaskType.DEEP_RESEARCH,
            TaskType.MARKET_ANALYSIS,
            TaskType.SENTIMENT_ANALYSIS,
            TaskType.QUICK_ANSWER,
        ],
        temperature=0.2,
    ),
    "google/gemini-2.5-flash": ModelProfile(
        model_id="google/gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        provider="google",
        context_length=1_000_000,
        cost_per_million_input=0.30,
        cost_per_million_output=2.50,
        speed_rating=10,  # Fastest available
        quality_rating=8,
        best_for=[
            TaskType.DEEP_RESEARCH,
            TaskType.MARKET_ANALYSIS,
            TaskType.SENTIMENT_ANALYSIS,
            TaskType.QUICK_ANSWER,
        ],
        temperature=0.2,
    ),
    "openai/gpt-4o-mini": ModelProfile(
        model_id="openai/gpt-4o-mini",
        name="GPT-4o Mini",
        provider="openai",
        context_length=128_000,
        cost_per_million_input=0.15,
        cost_per_million_output=0.60,
        speed_rating=9,
        quality_rating=7,
        best_for=[
            TaskType.MARKET_ANALYSIS,
            TaskType.TECHNICAL_ANALYSIS,
            TaskType.QUICK_ANSWER,
            TaskType.SENTIMENT_ANALYSIS,
        ],
        temperature=0.2,
    ),
    # ── Lightweight tier (classification / routing) ──────────────────────────
    "anthropic/claude-haiku-4.5": ModelProfile(
        model_id="anthropic/claude-haiku-4.5",
        name="Claude Haiku 4.5",
        provider="anthropic",
        context_length=200_000,
        cost_per_million_input=1.0,
        cost_per_million_output=5.0,
        speed_rating=8,
        quality_rating=7,
        best_for=[
            TaskType.QUERY_CLASSIFICATION,
            TaskType.QUICK_ANSWER,
            TaskType.SENTIMENT_ANALYSIS,
        ],
        temperature=0.2,
    ),
}


class OpenRouterProvider:
    """Provider for OpenRouter API with intelligent model selection."""

    def __init__(self, api_key: str):
        """Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key
        """
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self._model_usage_stats: dict[str, dict[str, int]] = {}

    def get_llm(
        self,
        task_type: TaskType = TaskType.GENERAL,
        prefer_fast: bool = False,
        prefer_cheap: bool = True,  # Default to cost-effective
        prefer_quality: bool = False,  # Override for premium models
        model_override: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        timeout_budget: float | None = None,  # Emergency mode for timeouts
    ) -> ChatOpenAI:
        """Get an LLM instance optimized for the task.

        Args:
            task_type: Type of task to optimize for
            prefer_fast: Prioritize speed over quality
            prefer_cheap: Prioritize cost over quality (default True)
            prefer_quality: Use premium models regardless of cost
            model_override: Override model selection
            temperature: Override default temperature
            max_tokens: Maximum tokens for response
            timeout_budget: Available time budget - triggers emergency mode if < 30s

        Returns:
            Configured ChatOpenAI instance
        """
        # Use override if provided
        if model_override:
            model_id = model_override
            model_profile = MODEL_PROFILES.get(
                model_id,
                ModelProfile(
                    model_id=model_id,
                    name=model_id,
                    provider="unknown",
                    context_length=128000,
                    cost_per_million_input=1.0,
                    cost_per_million_output=1.0,
                    speed_rating=5,
                    quality_rating=5,
                    best_for=[TaskType.GENERAL],
                    temperature=0.3,
                ),
            )
        # Emergency mode for tight timeout budgets
        elif timeout_budget is not None and timeout_budget < 30:
            model_profile = self._select_emergency_model(task_type, timeout_budget)
            model_id = model_profile.model_id
            logger.warning(
                f"EMERGENCY MODE: Selected ultra-fast model '{model_profile.name}' "
                f"for {timeout_budget}s timeout budget"
            )
        else:
            model_profile = self._select_model(
                task_type, prefer_fast, prefer_cheap, prefer_quality
            )
            model_id = model_profile.model_id

        # Use provided temperature or model default
        final_temperature = (
            temperature if temperature is not None else model_profile.temperature
        )

        # Log model selection
        logger.info(
            f"Selected model '{model_profile.name}' for task '{task_type}' "
            f"(speed={model_profile.speed_rating}/10, quality={model_profile.quality_rating}/10, "
            f"cost=${model_profile.cost_per_million_input}/{model_profile.cost_per_million_output} per 1M tokens)"
        )

        # Track usage
        self._track_usage(model_id, task_type)

        def _make_llm(mid: str, temp: float) -> ChatOpenAI:
            return ChatOpenAI(
                model=mid,
                temperature=temp,
                max_tokens=max_tokens,
                openai_api_base=self.base_url,
                openai_api_key=self.api_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/wshobson/maverick-mcp",
                    "X-Title": "Maverick MCP",
                },
                streaming=True,
            )

        primary_llm = _make_llm(model_id, final_temperature)

        # For override/emergency mode, return directly (no fallback chain needed)
        if model_override or (timeout_budget is not None and timeout_budget < 30):
            return primary_llm

        # Build fallback chain so region-restricted models are skipped automatically
        ranked_profiles = self._select_model_ranked(
            task_type, prefer_fast, prefer_cheap, prefer_quality, top_n=4
        )
        fallback_llms = [
            _make_llm(p.model_id, p.temperature)
            for p in ranked_profiles
            if p.model_id != model_id
        ]

        if fallback_llms:
            logger.debug(
                f"Fallback chain: {[p.model_id for p in ranked_profiles if p.model_id != model_id]}"
            )
            return primary_llm.with_fallbacks(
                fallback_llms,
                exceptions_to_handle=_REGION_ERRORS,
            )

        return primary_llm

    def _select_model(
        self,
        task_type: TaskType,
        prefer_fast: bool = False,
        prefer_cheap: bool = True,
        prefer_quality: bool = False,
    ) -> ModelProfile:
        """Select the best model for the task with cost-efficiency in mind.

        Args:
            task_type: Type of task
            prefer_fast: Prioritize speed
            prefer_cheap: Prioritize cost (default True)
            prefer_quality: Use premium models regardless of cost

        Returns:
            Selected model profile
        """
        candidates = []

        # Find models suitable for this task
        for profile in MODEL_PROFILES.values():
            if task_type in profile.best_for or task_type == TaskType.GENERAL:
                candidates.append(profile)

        if not candidates:
            # Fallback to GPT-5 Nano for general tasks
            return MODEL_PROFILES["google/gemini-2.5-flash"]

        # Score and rank candidates
        scored_candidates = []
        for profile in candidates:
            score = 0

            # Calculate average cost for this model
            avg_cost = (
                profile.cost_per_million_input + profile.cost_per_million_output
            ) / 2

            # Quality preference overrides cost considerations
            if prefer_quality:
                # Heavily weight quality for premium mode
                score += profile.quality_rating * 20
                # Task fitness is critical
                if task_type in profile.best_for:
                    score += 40
                # Minimal cost consideration
                score += max(0, 20 - avg_cost)
            else:
                # Cost-efficiency focused scoring (default)
                # Calculate cost-efficiency ratio
                cost_efficiency = profile.quality_rating / max(1, avg_cost)
                score += cost_efficiency * 30

                # Task fitness bonus
                if task_type in profile.best_for:
                    score += 25

                # Base quality (reduced weight)
                score += profile.quality_rating * 5

                # Speed preference
                if prefer_fast:
                    score += profile.speed_rating * 5
                else:
                    score += profile.speed_rating * 2

                # Cost preference adjustment
                if prefer_cheap:
                    # Strong cost preference
                    cost_score = max(0, 100 - avg_cost * 5)
                    score += cost_score
                else:
                    # Balanced cost consideration (default)
                    cost_score = max(0, 60 - avg_cost * 3)
                    score += cost_score

            scored_candidates.append((score, profile))

        # Sort by score and return best
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return scored_candidates[0][1]

    def _select_model_ranked(
        self,
        task_type: TaskType,
        prefer_fast: bool = False,
        prefer_cheap: bool = True,
        prefer_quality: bool = False,
        top_n: int = 3,
    ) -> list[ModelProfile]:
        """Return top-N ranked model profiles for use as primary + fallbacks."""
        candidates = []
        for profile in MODEL_PROFILES.values():
            if task_type in profile.best_for or task_type == TaskType.GENERAL:
                candidates.append(profile)

        if not candidates:
            return [MODEL_PROFILES["openai/gpt-5-nano"]]

        scored_candidates = []
        for profile in candidates:
            score = 0
            avg_cost = (
                profile.cost_per_million_input + profile.cost_per_million_output
            ) / 2
            if prefer_quality:
                score += profile.quality_rating * 20
                if task_type in profile.best_for:
                    score += 40
                score += max(0, 20 - avg_cost)
            else:
                cost_efficiency = profile.quality_rating / max(1, avg_cost)
                score += cost_efficiency * 30
                if task_type in profile.best_for:
                    score += 25
                score += profile.quality_rating * 5
                if prefer_fast:
                    score += profile.speed_rating * 5
                else:
                    score += profile.speed_rating * 2
                if prefer_cheap:
                    score += max(0, 100 - avg_cost * 5)
                else:
                    score += max(0, 60 - avg_cost * 3)
            scored_candidates.append((score, profile))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored_candidates[:top_n]]

    def _select_emergency_model(
        self, task_type: TaskType, timeout_budget: float
    ) -> ModelProfile:
        """Select the fastest model available for emergency timeout situations.

        Emergency mode prioritizes speed above all other considerations.
        Used when timeout_budget < 30 seconds.

        Args:
            task_type: Type of task
            timeout_budget: Available time in seconds (< 30s)

        Returns:
            Fastest available model profile
        """
        # Emergency model priority (speed first)

        # < 25s: absolute fastest
        if timeout_budget < 25:
            return MODEL_PROFILES["google/gemini-2.5-flash"]

        # < 30s: task-aware fast selection
        if task_type in [
            TaskType.DEEP_RESEARCH,
            TaskType.MARKET_ANALYSIS,
            TaskType.TECHNICAL_ANALYSIS,
        ]:
            return MODEL_PROFILES["google/gemini-3-flash-preview"]  # Fast + capable

        return MODEL_PROFILES["google/gemini-2.5-flash"]

    def _track_usage(self, model_id: str, task_type: TaskType):
        """Track model usage for analytics.

        Args:
            model_id: Model identifier
            task_type: Task type
        """
        if model_id not in self._model_usage_stats:
            self._model_usage_stats[model_id] = {}

        task_key = task_type.value
        if task_key not in self._model_usage_stats[model_id]:
            self._model_usage_stats[model_id][task_key] = 0

        self._model_usage_stats[model_id][task_key] += 1

    def get_usage_stats(self) -> dict[str, dict[str, int]]:
        """Get model usage statistics.

        Returns:
            Dictionary of model usage by task type
        """
        return self._model_usage_stats.copy()

    def recommend_models_for_workload(
        self, workload: dict[TaskType, int]
    ) -> dict[str, Any]:
        """Recommend optimal model mix for a given workload.

        Args:
            workload: Dictionary of task types and their frequencies

        Returns:
            Recommendations including models and estimated costs
        """
        recommendations = {}
        total_cost = 0.0

        for task_type, frequency in workload.items():
            # Select best model for this task
            model = self._select_model(task_type)

            # Estimate tokens (rough approximation)
            avg_input_tokens = 2000
            avg_output_tokens = 1000

            # Calculate cost
            input_cost = (
                avg_input_tokens * frequency * model.cost_per_million_input
            ) / 1_000_000
            output_cost = (
                avg_output_tokens * frequency * model.cost_per_million_output
            ) / 1_000_000
            task_cost = input_cost + output_cost

            recommendations[task_type.value] = {
                "model": model.name,
                "model_id": model.model_id,
                "frequency": frequency,
                "estimated_cost": task_cost,
            }

            total_cost += task_cost

        return {
            "recommendations": recommendations,
            "total_estimated_cost": total_cost,
            "cost_per_request": total_cost / sum(workload.values()) if workload else 0,
        }


# Convenience function for backward compatibility
def get_openrouter_llm(
    api_key: str,
    task_type: TaskType = TaskType.GENERAL,
    prefer_fast: bool = False,
    prefer_cheap: bool = True,
    prefer_quality: bool = False,
    **kwargs,
) -> ChatOpenAI:
    """Get an OpenRouter LLM instance with cost-efficiency by default.

    Args:
        api_key: OpenRouter API key
        task_type: Task type for model selection
        prefer_fast: Prioritize speed
        prefer_cheap: Prioritize cost (default True)
        prefer_quality: Use premium models regardless of cost
        **kwargs: Additional arguments for get_llm

    Returns:
        Configured ChatOpenAI instance
    """
    provider = OpenRouterProvider(api_key)
    return provider.get_llm(
        task_type=task_type,
        prefer_fast=prefer_fast,
        prefer_cheap=prefer_cheap,
        prefer_quality=prefer_quality,
        **kwargs,
    )
