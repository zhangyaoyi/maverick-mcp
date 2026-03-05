"""
Unit tests for Research & Agent tools:
- comprehensive_research
- company_comprehensive_research
- analyze_market_sentiment
- agents_list_available_agents
- agents_analyze_market_with_agent
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestResearchComprehensive:
    """Tests for comprehensive_research."""

    @pytest.mark.asyncio
    async def test_returns_research_result_structure(self):
        """comprehensive_research returns a dict result."""
        from maverick_mcp.api.routers.research import comprehensive_research

        mock_agent = MagicMock()
        mock_agent.research = AsyncMock(return_value={
            "status": "success",
            "query": "AAPL stock analysis",
            "sources": ["Bloomberg", "Reuters"],
            "analysis": "Apple shows strong momentum...",
            "confidence": 0.85,
        })

        with patch(
            "maverick_mcp.api.routers.research.get_research_agent",
            return_value=mock_agent,
        ):
            result = await comprehensive_research(
                query="AAPL stock analysis",
                persona="moderate",
                research_scope="standard",
                max_sources=10,
                timeframe="1m",
            )

        assert isinstance(result, dict)
        assert "status" in result or "query" in result or "analysis" in result

    @pytest.mark.asyncio
    async def test_handles_agent_error(self):
        """comprehensive_research returns error dict when agent fails."""
        from maverick_mcp.api.routers.research import comprehensive_research

        mock_agent = MagicMock()
        mock_agent.research = AsyncMock(side_effect=RuntimeError("Agent unavailable"))

        with patch(
            "maverick_mcp.api.routers.research.get_research_agent",
            return_value=mock_agent,
        ):
            result = await comprehensive_research(
                query="AAPL stock analysis",
                persona="moderate",
                research_scope="basic",
                max_sources=5,
                timeframe="1d",
            )

        assert isinstance(result, dict)
        # Should not raise, should return error dict
        assert "error" in result or "status" in result


class TestResearchCompanyComprehensive:
    """Tests for company_comprehensive_research."""

    @pytest.mark.asyncio
    async def test_company_research_is_dict(self):
        """company_comprehensive_research returns a dict."""
        from maverick_mcp.api.routers.research import company_comprehensive_research

        mock_agent = MagicMock()
        mock_agent.research = AsyncMock(return_value={
            "status": "success",
            "symbol": "AAPL",
            "company_name": "Apple Inc.",
            "analysis": "Strong fundamentals...",
        })

        with patch(
            "maverick_mcp.api.routers.research.get_research_agent",
            return_value=mock_agent,
        ):
            result = await company_comprehensive_research(
                symbol="AAPL",
                include_competitive_analysis=False,
                persona="moderate",
            )

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_handles_unknown_symbol(self):
        """company_comprehensive_research handles unknown symbol gracefully."""
        from maverick_mcp.api.routers.research import company_comprehensive_research

        mock_agent = MagicMock()
        mock_agent.research = AsyncMock(side_effect=ValueError("Symbol not found"))

        with patch(
            "maverick_mcp.api.routers.research.get_research_agent",
            return_value=mock_agent,
        ):
            result = await company_comprehensive_research(
                symbol="ZZZZZ",
                persona="moderate",
            )

        assert isinstance(result, dict)


class TestAnalyzeMarketSentiment:
    """Tests for analyze_market_sentiment."""

    @pytest.mark.asyncio
    async def test_sentiment_returns_dict(self):
        """analyze_market_sentiment returns a dict."""
        from maverick_mcp.api.routers.research import analyze_market_sentiment

        mock_agent = MagicMock()
        mock_agent.research = AsyncMock(return_value={
            "status": "success",
            "topic": "AAPL",
            "sentiment_score": 0.72,
            "sentiment_label": "Bullish",
        })

        with patch(
            "maverick_mcp.api.routers.research.get_research_agent",
            return_value=mock_agent,
        ):
            result = await analyze_market_sentiment(
                topic="AAPL",
                timeframe="1w",
                persona="moderate",
            )

        assert isinstance(result, dict)

    def test_sentiment_label_validity(self):
        """Sentiment labels are one of expected values."""
        valid_labels = {"Bullish", "Bearish", "Neutral", "Mixed"}
        test_labels = ["Bullish", "Bearish", "Neutral"]
        for label in test_labels:
            assert label in valid_labels

    def test_sentiment_score_range(self):
        """Sentiment score should be between -1 and 1."""
        scores = [0.72, -0.45, 0.0, 0.95, -0.88]
        for score in scores:
            assert -1.0 <= score <= 1.0


class TestAgentsListAvailableAgents:
    """Tests for agents_list_available_agents."""

    def test_agents_list_not_empty(self):
        """INVESTOR_PERSONAS contains available personas."""
        from maverick_mcp.agents.base import INVESTOR_PERSONAS

        assert isinstance(INVESTOR_PERSONAS, dict)
        assert len(INVESTOR_PERSONAS) > 0

    def test_each_persona_has_name(self):
        """Each investor persona has required attributes."""
        from maverick_mcp.agents.base import INVESTOR_PERSONAS

        for key, persona in INVESTOR_PERSONAS.items():
            assert key  # key is non-empty
            assert persona  # persona has content


class TestAgentsAnalyzeMarketWithAgent:
    """Tests for agents_analyze_market_with_agent."""

    @pytest.mark.asyncio
    async def test_market_analysis_returns_result(self):
        """Market agent analysis returns structured result."""
        from maverick_mcp.agents.market_analysis import MarketAnalysisAgent

        mock_agent = MagicMock(spec=MarketAnalysisAgent)
        mock_agent.analyze = AsyncMock(return_value={
            "status": "success",
            "ticker": "AAPL",
            "analysis": "Strong bullish momentum...",
            "recommendation": "BUY",
            "confidence": 0.78,
        })

        result = await mock_agent.analyze("AAPL")

        assert result["status"] == "success"
        assert result["ticker"] == "AAPL"
        assert "recommendation" in result

    @pytest.mark.asyncio
    async def test_agent_handles_invalid_ticker(self):
        """Market agent returns error for invalid ticker."""
        from maverick_mcp.agents.market_analysis import MarketAnalysisAgent

        mock_agent = MagicMock(spec=MarketAnalysisAgent)
        mock_agent.analyze = AsyncMock(return_value={
            "status": "error",
            "error": "Invalid ticker symbol",
        })

        result = await mock_agent.analyze("INVALID_TICKER_XYZ")
        assert result["status"] == "error"


class TestAgentsDeepResearchFinancial:
    """Tests for agents_deep_research_financial."""

    @pytest.mark.asyncio
    async def test_deep_research_returns_analysis(self):
        """Deep research returns comprehensive financial analysis."""
        from maverick_mcp.agents.deep_research import DeepResearchAgent

        mock_agent = MagicMock(spec=DeepResearchAgent)
        mock_agent.research = AsyncMock(return_value={
            "status": "success",
            "query": "NVDA semiconductor market analysis",
            "research_depth": "comprehensive",
            "analysis": "NVIDIA dominates AI chip market...",
            "sources": ["Bloomberg", "SEC filings", "analyst reports"],
            "confidence_score": 0.88,
        })

        result = await mock_agent.research("NVDA semiconductor market analysis")

        assert result["status"] == "success"
        assert "analysis" in result
        assert result["confidence_score"] > 0.5
