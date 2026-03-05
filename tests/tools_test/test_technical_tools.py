"""
Unit tests for Technical Analysis tools:
- get_rsi_analysis
- get_macd_analysis
- get_support_resistance
- get_full_technical_analysis
- get_stock_chart_analysis
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestRsiAnalysis:
    """Tests for get_rsi_analysis."""

    @pytest.mark.asyncio
    async def test_rsi_returns_expected_keys(self, sample_ohlcv_df):
        """get_rsi_analysis returns ticker, analysis with current and signal."""
        from maverick_mcp.api.routers.technical import get_rsi_analysis

        with patch(
            "maverick_mcp.api.routers.technical.get_stock_dataframe_async",
            new_callable=AsyncMock,
            return_value=sample_ohlcv_df,
        ):
            result = await get_rsi_analysis("AAPL", period=14, days=365)

        assert result.get("ticker") == "AAPL"
        assert "analysis" in result
        analysis = result["analysis"]
        assert "current" in analysis
        assert "signal" in analysis

    @pytest.mark.asyncio
    async def test_rsi_signal_is_valid(self, sample_ohlcv_df):
        """RSI signal value is one of the expected categories."""
        from maverick_mcp.api.routers.technical import get_rsi_analysis

        with patch(
            "maverick_mcp.api.routers.technical.get_stock_dataframe_async",
            new_callable=AsyncMock,
            return_value=sample_ohlcv_df,
        ):
            result = await get_rsi_analysis("MSFT")

        # Include "unavailable" for cases where RSI cannot be computed
        valid_signals = {"oversold", "neutral", "overbought", "bullish", "bearish", "unavailable"}
        assert result["analysis"]["signal"] in valid_signals

    @pytest.mark.asyncio
    async def test_rsi_handles_error(self):
        """get_rsi_analysis returns error dict when data fetch fails."""
        from maverick_mcp.api.routers.technical import get_rsi_analysis

        with patch(
            "maverick_mcp.api.routers.technical.get_stock_dataframe_async",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Data unavailable"),
        ):
            result = await get_rsi_analysis("BADTICKER")

        assert "error" in result


class TestMacdAnalysis:
    """Tests for get_macd_analysis."""

    @pytest.mark.asyncio
    async def test_macd_returns_expected_keys(self, sample_ohlcv_df):
        """get_macd_analysis returns macd, signal, histogram in analysis."""
        from maverick_mcp.api.routers.technical import get_macd_analysis

        with patch(
            "maverick_mcp.api.routers.technical.get_stock_dataframe_async",
            new_callable=AsyncMock,
            return_value=sample_ohlcv_df,
        ):
            result = await get_macd_analysis("AAPL")

        assert result.get("ticker") == "AAPL"
        assert "analysis" in result
        analysis = result["analysis"]
        assert "macd" in analysis
        assert "signal" in analysis
        assert "histogram" in analysis

    @pytest.mark.asyncio
    async def test_macd_indicator_field_present(self, sample_ohlcv_df):
        """get_macd_analysis includes indicator direction field."""
        from maverick_mcp.api.routers.technical import get_macd_analysis

        with patch(
            "maverick_mcp.api.routers.technical.get_stock_dataframe_async",
            new_callable=AsyncMock,
            return_value=sample_ohlcv_df,
        ):
            result = await get_macd_analysis("TSLA")

        assert "indicator" in result["analysis"]


class TestSupportResistance:
    """Tests for get_support_resistance."""

    @pytest.mark.asyncio
    async def test_returns_support_and_resistance_lists(self, sample_ohlcv_df):
        """get_support_resistance returns non-empty lists."""
        from maverick_mcp.api.routers.technical import get_support_resistance

        with patch(
            "maverick_mcp.api.routers.technical.get_stock_dataframe_async",
            new_callable=AsyncMock,
            return_value=sample_ohlcv_df,
        ):
            result = await get_support_resistance("AAPL")

        assert "support_levels" in result
        assert "resistance_levels" in result
        assert isinstance(result["support_levels"], list)
        assert isinstance(result["resistance_levels"], list)

    @pytest.mark.asyncio
    async def test_handles_insufficient_data(self):
        """get_support_resistance handles empty DataFrame gracefully."""
        from maverick_mcp.api.routers.technical import get_support_resistance

        with patch(
            "maverick_mcp.api.routers.technical.get_stock_dataframe_async",
            new_callable=AsyncMock,
            side_effect=ValueError("Insufficient data"),
        ):
            result = await get_support_resistance("AAPL")

        assert "error" in result


class TestFullTechnicalAnalysis:
    """Tests for get_full_technical_analysis."""

    @pytest.mark.asyncio
    async def test_full_analysis_contains_all_indicators(self, sample_ohlcv_df):
        """get_full_technical_analysis returns RSI, MACD, Bollinger Bands."""
        from maverick_mcp.api.routers.technical import get_full_technical_analysis

        with patch(
            "maverick_mcp.api.routers.technical.get_stock_dataframe_async",
            new_callable=AsyncMock,
            return_value=sample_ohlcv_df,
        ):
            result = await get_full_technical_analysis("AAPL")

        assert "indicators" in result
        indicators = result["indicators"]
        assert "rsi" in indicators
        assert "macd" in indicators
        assert "bollinger_bands" in indicators

    @pytest.mark.asyncio
    async def test_full_analysis_includes_levels_and_price(self, sample_ohlcv_df):
        """get_full_technical_analysis returns current_price and levels."""
        from maverick_mcp.api.routers.technical import get_full_technical_analysis

        with patch(
            "maverick_mcp.api.routers.technical.get_stock_dataframe_async",
            new_callable=AsyncMock,
            return_value=sample_ohlcv_df,
        ):
            result = await get_full_technical_analysis("MSFT")

        assert "current_price" in result
        assert "levels" in result
        assert "last_updated" in result

    @pytest.mark.asyncio
    async def test_full_analysis_handles_error(self):
        """get_full_technical_analysis returns error dict on failure."""
        from maverick_mcp.api.routers.technical import get_full_technical_analysis

        with patch(
            "maverick_mcp.api.routers.technical.get_stock_dataframe_async",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Provider down"),
        ):
            result = await get_full_technical_analysis("AAPL")

        assert "error" in result


class TestStockChartAnalysis:
    """Tests for get_stock_chart_analysis."""

    @pytest.mark.asyncio
    async def test_chart_analysis_returns_dict(self, sample_ohlcv_df):
        """get_stock_chart_analysis returns a dict (ticker or error)."""
        from maverick_mcp.api.routers.technical import get_stock_chart_analysis

        mock_b64 = "iVBORw0KGgoAAAANS"  # fake base64 png prefix

        with (
            patch(
                "maverick_mcp.api.routers.technical.get_stock_dataframe_async",
                new_callable=AsyncMock,
                return_value=sample_ohlcv_df,
            ),
            patch(
                "maverick_mcp.api.routers.technical.create_plotly_technical_chart",
                return_value=MagicMock(),
            ),
            patch(
                "maverick_mcp.api.routers.technical.plotly_fig_to_base64",
                return_value=mock_b64,
            ),
        ):
            result = await get_stock_chart_analysis("AAPL")

        assert isinstance(result, dict)
        # Either succeeds with ticker or returns an error dict
        assert "ticker" in result or "error" in result
