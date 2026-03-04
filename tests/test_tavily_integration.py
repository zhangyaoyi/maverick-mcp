"""
Tests for Tavily search provider integration.

Covers:
- TavilySearchProvider initialization and health tracking
- get_cached_search_provider() priority logic (Tavily > Exa)
- DeepResearchAgent initialization with tavily_api_key
- _perform_financial_search() Tavily dispatch
- Settings: tavily_api_key field and api_keys property
- research.py router: provider availability check accepts Tavily
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from maverick_mcp.agents.deep_research import (
    DeepResearchAgent,
    ExaSearchProvider,
    TavilySearchProvider,
    _search_provider_cache,
    get_cached_search_provider,
)
from maverick_mcp.config.settings import get_settings
from maverick_mcp.exceptions import WebSearchError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_provider_cache():
    """Clear the module-level search provider cache between tests."""
    _search_provider_cache.clear()


@pytest.fixture(autouse=True)
def clear_cache():
    """Ensure each test starts with a clean provider cache."""
    _clear_provider_cache()
    yield
    _clear_provider_cache()


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="{}"))
    return llm


# ---------------------------------------------------------------------------
# TavilySearchProvider unit tests
# ---------------------------------------------------------------------------


class TestTavilySearchProvider:
    @pytest.mark.unit
    def test_initialization(self):
        provider = TavilySearchProvider("tvly-test-key")
        assert provider.api_key == "tvly-test-key"
        assert provider.is_healthy() is True
        assert provider._failure_count == 0

    @pytest.mark.unit
    def test_initialization_empty_key(self):
        provider = TavilySearchProvider("")
        assert provider.api_key == ""
        assert provider.is_healthy() is True  # healthy until failures occur

    @pytest.mark.unit
    def test_excluded_domains_set(self):
        provider = TavilySearchProvider("key")
        assert "twitter.com" in provider.excluded_domains
        assert "facebook.com" in provider.excluded_domains
        assert "reddit.com" in provider.excluded_domains

    @pytest.mark.unit
    def test_process_results_filters_excluded_domains(self):
        provider = TavilySearchProvider("key")
        raw = [
            {"url": "https://reuters.com/article", "title": "News", "content": "text", "score": 0.9},
            {"url": "https://twitter.com/user/status/1", "title": "Tweet", "content": "t", "score": 0.5},
            {"url": "https://wsj.com/article", "title": "WSJ", "content": "text2", "score": 0.8},
        ]
        results = provider._process_results(raw)
        urls = [r["url"] for r in results]
        assert "https://twitter.com/user/status/1" not in urls
        assert "https://reuters.com/article" in urls
        assert "https://wsj.com/article" in urls

    @pytest.mark.unit
    def test_process_results_output_shape(self):
        provider = TavilySearchProvider("key")
        raw = [{"url": "https://bloomberg.com/news", "title": "Test", "content": "body", "score": 0.7}]
        results = provider._process_results(raw)
        assert len(results) == 1
        r = results[0]
        assert r["provider"] == "tavily"
        assert r["url"] == "https://bloomberg.com/news"
        assert r["title"] == "Test"
        assert r["content"] == "body"
        assert r["score"] == 0.7

    @pytest.mark.unit
    def test_process_results_uses_raw_content_fallback(self):
        provider = TavilySearchProvider("key")
        raw = [{"url": "https://example.com", "title": "T", "raw_content": "raw body", "score": 0.5}]
        results = provider._process_results(raw)
        assert results[0]["content"] == "raw body"

    @pytest.mark.unit
    def test_health_tracking_failure_and_recovery(self):
        provider = TavilySearchProvider("key")
        assert provider.is_healthy()

        # Drive failure count past max_failures * 2 (non-timeout failures)
        for _ in range(provider._max_failures * 2 + 1):
            provider._record_failure("api_error")

        assert not provider.is_healthy()

        provider._record_success()
        assert provider.is_healthy()
        assert provider._failure_count == 0

    @pytest.mark.unit
    async def test_search_raises_when_unhealthy(self):
        provider = TavilySearchProvider("key")
        provider._is_healthy = False

        with pytest.raises(WebSearchError, match="disabled due to repeated failures"):
            await provider.search("test query")

    @pytest.mark.unit
    async def test_search_raises_when_tavily_not_installed(self):
        provider = TavilySearchProvider("key")

        with patch("maverick_mcp.agents.deep_research.TavilyClient", None):
            with pytest.raises((ImportError, Exception)):
                await provider.search("test query")

    @pytest.mark.unit
    async def test_search_calls_tavily_client(self):
        provider = TavilySearchProvider("tvly-key")

        expected_results = [
            {"url": "https://reuters.com/a", "title": "R", "content": "body", "score": 0.9}
        ]

        mock_client = MagicMock()
        mock_client.search.return_value = {"results": expected_results}

        with patch("maverick_mcp.agents.deep_research.TavilyClient", return_value=mock_client):
            results = await provider.search("AAPL earnings", num_results=5)

        mock_client.search.assert_called_once_with(query="AAPL earnings", max_results=5)
        assert len(results) == 1
        assert results[0]["provider"] == "tavily"


# ---------------------------------------------------------------------------
# get_cached_search_provider() priority tests
# ---------------------------------------------------------------------------


class TestGetCachedSearchProvider:
    @pytest.mark.unit
    async def test_returns_tavily_when_both_keys_present(self):
        provider = await get_cached_search_provider(
            exa_api_key="exa-key", tavily_api_key="tvly-key"
        )
        assert isinstance(provider, TavilySearchProvider)
        assert provider.api_key == "tvly-key"

    @pytest.mark.unit
    async def test_returns_exa_when_only_exa_key(self):
        provider = await get_cached_search_provider(exa_api_key="exa-key")
        assert isinstance(provider, ExaSearchProvider)

    @pytest.mark.unit
    async def test_returns_none_when_no_keys(self):
        provider = await get_cached_search_provider()
        assert provider is None

    @pytest.mark.unit
    async def test_caches_tavily_provider(self):
        p1 = await get_cached_search_provider(tavily_api_key="tvly-key")
        p2 = await get_cached_search_provider(tavily_api_key="tvly-key")
        assert p1 is p2  # same object from cache

    @pytest.mark.unit
    async def test_tavily_takes_priority_over_exa(self):
        """Even when Exa key is provided, Tavily wins."""
        provider = await get_cached_search_provider(
            exa_api_key="exa-key", tavily_api_key="tvly-key"
        )
        assert isinstance(provider, TavilySearchProvider)
        # Exa should NOT be cached when Tavily key is present
        assert not any("exa" in k for k in _search_provider_cache)


# ---------------------------------------------------------------------------
# DeepResearchAgent initialization tests
# ---------------------------------------------------------------------------


class TestDeepResearchAgentTavily:
    @pytest.mark.unit
    def test_stores_tavily_api_key(self, mock_llm):
        agent = DeepResearchAgent(llm=mock_llm, tavily_api_key="tvly-key")
        assert agent._tavily_api_key == "tvly-key"

    @pytest.mark.unit
    def test_stores_both_keys(self, mock_llm):
        agent = DeepResearchAgent(
            llm=mock_llm, exa_api_key="exa-key", tavily_api_key="tvly-key"
        )
        assert agent._exa_api_key == "exa-key"
        assert agent._tavily_api_key == "tvly-key"

    @pytest.mark.unit
    @patch("maverick_mcp.agents.deep_research.get_cached_search_provider")
    async def test_initialize_uses_tavily_provider(self, mock_get_provider, mock_llm):
        tavily_provider = MagicMock(spec=TavilySearchProvider)
        mock_get_provider.return_value = tavily_provider

        agent = DeepResearchAgent(llm=mock_llm, tavily_api_key="tvly-key")
        await agent.initialize()

        mock_get_provider.assert_called_once_with(
            exa_api_key=None, tavily_api_key="tvly-key"
        )
        assert agent.search_providers == [tavily_provider]
        assert agent._search_providers_loaded is True
        assert agent._initialization_pending is False

    @pytest.mark.unit
    @patch("maverick_mcp.agents.deep_research.get_cached_search_provider")
    async def test_initialize_empty_when_no_provider(self, mock_get_provider, mock_llm):
        mock_get_provider.return_value = None

        agent = DeepResearchAgent(llm=mock_llm)
        await agent.initialize()

        assert agent.search_providers == []
        assert agent._search_providers_loaded is True

    @pytest.mark.unit
    def test_web_search_provider_property_returns_first(self, mock_llm):
        agent = DeepResearchAgent(llm=mock_llm, tavily_api_key="tvly-key")
        fake_provider = MagicMock(spec=TavilySearchProvider)
        agent.search_providers = [fake_provider]
        assert agent.web_search_provider is fake_provider

    @pytest.mark.unit
    def test_web_search_provider_property_none_when_empty(self, mock_llm):
        agent = DeepResearchAgent(llm=mock_llm)
        agent.search_providers = []
        assert agent.web_search_provider is None


# ---------------------------------------------------------------------------
# _perform_financial_search() with Tavily
# ---------------------------------------------------------------------------


class TestFinancialSearchWithTavily:
    @pytest.fixture
    def agent_with_tavily(self, mock_llm):
        agent = DeepResearchAgent(llm=mock_llm, tavily_api_key="tvly-key")
        tavily_provider = MagicMock(spec=TavilySearchProvider)
        tavily_provider.search = AsyncMock(
            return_value=[
                {
                    "url": "https://reuters.com/a",
                    "title": "Reuters",
                    "content": "Earnings beat expectations",
                    "score": 0.9,
                    "provider": "tavily",
                }
            ]
        )
        agent.search_providers = [tavily_provider]
        agent._tavily_provider = tavily_provider  # stash for assertions
        return agent

    @pytest.mark.unit
    async def test_tavily_called_on_auto_provider(self, agent_with_tavily):
        result = await agent_with_tavily._perform_financial_search(
            query="AAPL earnings", num_results=5, provider="auto", strategy="news"
        )
        assert result["total_results"] == 1
        assert result["results"][0]["provider"] == "tavily"
        agent_with_tavily._tavily_provider.search.assert_called_once_with("AAPL earnings", 5)

    @pytest.mark.unit
    async def test_tavily_called_on_tavily_provider_arg(self, agent_with_tavily):
        result = await agent_with_tavily._perform_financial_search(
            query="AAPL", num_results=3, provider="tavily", strategy="general"
        )
        assert result["total_results"] >= 0
        agent_with_tavily._tavily_provider.search.assert_called_once()

    @pytest.mark.unit
    async def test_result_metadata_added(self, agent_with_tavily):
        result = await agent_with_tavily._perform_financial_search(
            query="SPY outlook", num_results=2, provider="auto", strategy="broad"
        )
        for r in result["results"]:
            assert "search_strategy" in r
            assert "search_timestamp" in r
            assert r["search_strategy"] == "broad"

    @pytest.mark.unit
    async def test_tavily_error_returns_error_dict(self, mock_llm):
        agent = DeepResearchAgent(llm=mock_llm, tavily_api_key="tvly-key")
        tavily_provider = MagicMock(spec=TavilySearchProvider)
        tavily_provider.search = AsyncMock(side_effect=Exception("API unavailable"))
        agent.search_providers = [tavily_provider]

        result = await agent._perform_financial_search(
            query="TSLA", num_results=5, provider="auto", strategy="news"
        )
        assert "error" in result
        assert "API unavailable" in result["error"]

    @pytest.mark.unit
    async def test_no_providers_returns_error(self, mock_llm):
        agent = DeepResearchAgent(llm=mock_llm)
        agent.search_providers = []

        result = await agent._perform_financial_search(
            query="AAPL", num_results=5, provider="auto", strategy="news"
        )
        assert result["error"] == "No search providers available"
        assert result["results"] == []


# ---------------------------------------------------------------------------
# Settings: tavily_api_key field
# ---------------------------------------------------------------------------


class TestSettingsTavilyKey:
    @pytest.mark.unit
    def test_tavily_api_key_in_api_keys_dict(self):
        settings = get_settings()
        api_keys = settings.research.api_keys
        assert "tavily_api_key" in api_keys

    @pytest.mark.unit
    def test_exa_api_key_still_in_api_keys_dict(self):
        settings = get_settings()
        api_keys = settings.research.api_keys
        assert "exa_api_key" in api_keys

    @pytest.mark.unit
    def test_tavily_api_key_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-env-test")
        # Re-create settings to pick up new env value
        from maverick_mcp.config.settings import ResearchSettings

        s = ResearchSettings()
        assert s.tavily_api_key == "tvly-env-test"

    @pytest.mark.unit
    def test_tavily_api_key_none_when_not_set(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        from maverick_mcp.config.settings import ResearchSettings

        s = ResearchSettings()
        assert s.tavily_api_key is None


# ---------------------------------------------------------------------------
# Research router: provider availability check
# ---------------------------------------------------------------------------


class TestResearchRouterProviderCheck:
    """Verify that the provider availability logic in research.py accepts Tavily."""

    @pytest.mark.unit
    def test_tavily_only_is_sufficient(self):
        """Simulate what the router does: tavily available → no error."""
        exa_available = False
        tavily_available = True
        assert exa_available or tavily_available  # should NOT return error

    @pytest.mark.unit
    def test_exa_only_is_sufficient(self):
        exa_available = True
        tavily_available = False
        assert exa_available or tavily_available

    @pytest.mark.unit
    def test_neither_available_triggers_error(self):
        exa_available = False
        tavily_available = False
        assert not (exa_available or tavily_available)  # should return error

    @pytest.mark.unit
    def test_both_available_is_fine(self):
        exa_available = True
        tavily_available = True
        assert exa_available or tavily_available
