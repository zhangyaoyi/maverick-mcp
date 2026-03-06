"""
Diagnostic script to test Bailian API connectivity and identify fallback reasons.

Usage:
    uv run python scripts/test_bailian.py
    # or in Docker:
    docker compose exec maverick-mcp uv run python scripts/test_bailian.py
"""

import asyncio
import logging
import os
import sys
import traceback

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("bailian_test")


def check_env() -> str | None:
    api_key = os.getenv("ALIYUN_API_KEY")
    if not api_key:
        logger.error("ALIYUN_API_KEY is not set")
        return None
    masked = api_key[:6] + "..." + api_key[-4:]
    logger.info("ALIYUN_API_KEY found: %s (len=%d)", masked, len(api_key))
    return api_key


def test_raw_http(api_key: str) -> None:
    """Test raw HTTP connectivity to the Bailian endpoint."""
    import httpx

    base_url = "https://coding.dashscope.aliyuncs.com/v1"
    logger.info("=== Raw HTTP test → %s ===", base_url)

    payload = {
        "model": "qwen3.5-plus",
        "messages": [{"role": "user", "content": "Say 'hello' in one word."}],
        "max_tokens": 10,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=30, trust_env=False) as client:
            resp = client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
        logger.info("HTTP status: %d", resp.status_code)
        logger.info("Response body: %s", resp.text[:500])

        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            logger.info("SUCCESS — model reply: %r", content)
        else:
            logger.error("FAILED — status=%d body=%s", resp.status_code, resp.text)

    except Exception as e:
        logger.error("HTTP request failed: %s", e)
        traceback.print_exc()


def test_langchain_sync(api_key: str) -> None:
    """Test via LangChain ChatOpenAI (sync invoke)."""
    import httpx
    from langchain_openai import ChatOpenAI

    logger.info("=== LangChain ChatOpenAI sync test ===")
    for model in ["qwen3.5-plus", "qwen3-max-2026-01-23", "glm-5"]:
        logger.info("-- model: %s --", model)
        try:
            llm = ChatOpenAI(
                model=model,
                temperature=0.3,
                max_tokens=20,
                base_url="https://coding.dashscope.aliyuncs.com/v1",
                api_key=api_key,
                streaming=False,
                http_client=httpx.Client(trust_env=False),
                http_async_client=httpx.AsyncClient(trust_env=False),
            )
            result = llm.invoke("Say 'hello' in one word.")
            logger.info("SUCCESS — reply: %r", result.content)
        except Exception as e:
            logger.error("FAILED — %s: %s", type(e).__name__, e)
            traceback.print_exc()


async def test_langchain_async(api_key: str) -> None:
    """Test via LangChain ChatOpenAI (async invoke)."""
    import httpx
    from langchain_openai import ChatOpenAI

    logger.info("=== LangChain ChatOpenAI async test ===")
    try:
        llm = ChatOpenAI(
            model="qwen3.5-plus",
            temperature=0.3,
            max_tokens=20,
            base_url="https://coding.dashscope.aliyuncs.com/v1",
            api_key=api_key,
            streaming=False,
            http_client=httpx.Client(trust_env=False),
            http_async_client=httpx.AsyncClient(trust_env=False),
        )
        result = await llm.ainvoke("Say 'hello' in one word.")
        logger.info("SUCCESS — reply: %r", result.content)
    except Exception as e:
        logger.error("FAILED — %s: %s", type(e).__name__, e)
        traceback.print_exc()


def test_bailian_provider(api_key: str) -> None:
    """Test via BailianProvider (same path as production)."""
    from maverick_mcp.providers.bailian_provider import BailianProvider
    from maverick_mcp.providers.openrouter_provider import TaskType

    logger.info("=== BailianProvider.get_llm() test ===")
    provider = BailianProvider(api_key=api_key)
    for task in [TaskType.GENERAL, TaskType.MULTI_AGENT_ORCHESTRATION]:
        logger.info("-- task: %s --", task)
        try:
            llm = provider.get_llm(task_type=task)
            result = llm.invoke("Say 'hello' in one word.")
            logger.info("SUCCESS — reply: %r", result.content)
        except Exception as e:
            logger.error("FAILED — %s: %s", type(e).__name__, e)
            traceback.print_exc()


def main() -> None:
    logger.info("========== Bailian API Diagnostic ==========")

    api_key = check_env()
    if not api_key:
        sys.exit(1)

    test_raw_http(api_key)
    test_langchain_sync(api_key)
    asyncio.run(test_langchain_async(api_key))
    test_bailian_provider(api_key)

    logger.info("========== Diagnostic complete ==========")


if __name__ == "__main__":
    main()
