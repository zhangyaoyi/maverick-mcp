"""LLM factory for creating language model instances.

This module provides a factory function to create LLM instances with intelligent model selection.
"""

import logging
import os
from typing import Any

from langchain_community.llms import FakeListLLM

from maverick_mcp.providers.bailian_provider import get_bailian_llm
from maverick_mcp.providers.openrouter_provider import (
    TaskType,
    get_openrouter_llm,
)

logger = logging.getLogger(__name__)

# Auth/permission errors that should trigger cross-provider fallback
try:
    from openai import AuthenticationError as _OpenAIAuthError
    from openai import PermissionDeniedError as _OpenAIPermDeniedError

    _AUTH_ERRORS: tuple[type[Exception], ...] = (_OpenAIAuthError, _OpenAIPermDeniedError)
except ImportError:
    _AUTH_ERRORS = (Exception,)


def get_llm(
    task_type: TaskType = TaskType.GENERAL,
    prefer_fast: bool = False,
    prefer_cheap: bool = True,  # Default to cost-effective
    prefer_quality: bool = False,
    model_override: str | None = None,
) -> Any:
    """Create and return an LLM instance with intelligent model selection.

    When multiple provider keys are set, builds a cross-provider fallback chain
    so that a 401/403 from Bailian automatically retries with OpenRouter, etc.

    Priority order:
    1. Bailian (Aliyun) if ALIYUN_API_KEY is available (primary - qwen3.5-plus default)
    2. OpenRouter API if OPENROUTER_API_KEY is available (with smart model selection)
    3. OpenAI ChatOpenAI if OPENAI_API_KEY is available (fallback)
    4. Anthropic ChatAnthropic if ANTHROPIC_API_KEY is available (fallback)
    5. FakeListLLM as fallback for testing
    """
    kwargs = dict(
        task_type=task_type,
        prefer_fast=prefer_fast,
        prefer_cheap=prefer_cheap,
        prefer_quality=prefer_quality,
        model_override=model_override,
    )

    # Build ordered list of available provider LLMs
    provider_llms: list[Any] = []

    aliyun_api_key = os.getenv("ALIYUN_API_KEY")
    if aliyun_api_key:
        provider_llms.append(get_bailian_llm(api_key=aliyun_api_key, **kwargs))

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key:
        provider_llms.append(
            get_openrouter_llm(api_key=openrouter_api_key, **kwargs)
        )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        try:
            from langchain_openai import ChatOpenAI

            provider_llms.append(
                ChatOpenAI(model="gpt-4o-mini", temperature=0.3, streaming=False)
            )
        except ImportError:
            pass

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        try:
            from langchain_anthropic import ChatAnthropic

            provider_llms.append(
                ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.3)
            )
        except ImportError:
            pass

    if not provider_llms:
        logger.warning("No LLM API keys found - using FakeListLLM for testing")
        return FakeListLLM(
            responses=[
                "Mock analysis response for testing purposes.",
                "This is a simulated LLM response.",
                "Market analysis: Moderate bullish sentiment detected.",
            ]
        )

    primary = provider_llms[0]
    fallbacks = provider_llms[1:]

    if fallbacks:
        provider_names = _provider_names(aliyun_api_key, openrouter_api_key, openai_api_key, anthropic_api_key)
        logger.info("LLM chain: %s", " → ".join(provider_names))
        return primary.with_fallbacks(fallbacks, exceptions_to_handle=_AUTH_ERRORS)

    return primary


def _provider_names(aliyun: str | None, openrouter: str | None, openai: str | None, anthropic: str | None) -> list[str]:
    names = []
    if aliyun:
        names.append("Bailian")
    if openrouter:
        names.append("OpenRouter")
    if openai:
        names.append("OpenAI")
    if anthropic:
        names.append("Anthropic")
    return names
