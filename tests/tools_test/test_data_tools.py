"""
Unit tests for Data tools:
- fetch_stock_data
- fetch_stock_data_batch
- get_stock_info
- get_cached_price_data
- clear_cache
- get_news_sentiment
"""

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tests.unit.conftest import make_mock_stock


class TestFetchStockData:
    """Tests for fetch_stock_data function."""

    def test_returns_ticker_and_record_count(self, sample_ohlcv_df):
        """fetch_stock_data returns expected keys including ticker and record_count."""
        from maverick_mcp.api.routers.data import fetch_stock_data

        mock_service = MagicMock()
        mock_service.get_stock_data.return_value = sample_ohlcv_df

        @contextmanager
        def fake_session():
            yield MagicMock()

        with (
            patch(
                "maverick_mcp.api.routers.data.StockDataFetchingService",
                return_value=MagicMock(),
            ),
            patch(
                "maverick_mcp.api.routers.data.get_db_session_read_only",
                fake_session,
            ),
            patch(
                "maverick_mcp.api.routers.data.StockAnalysisService",
                return_value=mock_service,
            ),
            patch(
                "maverick_mcp.api.routers.data.CacheManagementService",
                return_value=MagicMock(),
            ),
        ):
            result = fetch_stock_data("AAPL", "2024-01-01", "2024-12-31")

        assert result["ticker"] == "AAPL"
        assert result["record_count"] == len(sample_ohlcv_df)
        assert "columns" in result or "data" in result

    def test_handles_exception_gracefully(self):
        """fetch_stock_data returns error dict on exception."""
        from maverick_mcp.api.routers.data import fetch_stock_data

        mock_service = MagicMock()
        mock_service.get_stock_data.side_effect = RuntimeError("Connection failed")

        @contextmanager
        def fake_session():
            yield MagicMock()

        with (
            patch(
                "maverick_mcp.api.routers.data.StockDataFetchingService",
                return_value=MagicMock(),
            ),
            patch(
                "maverick_mcp.api.routers.data.get_db_session_read_only",
                fake_session,
            ),
            patch(
                "maverick_mcp.api.routers.data.StockAnalysisService",
                return_value=mock_service,
            ),
            patch(
                "maverick_mcp.api.routers.data.CacheManagementService",
                return_value=MagicMock(),
            ),
        ):
            result = fetch_stock_data("AAPL")

        assert "error" in result
        assert result["ticker"] == "AAPL"


class TestFetchStockDataBatch:
    """Tests for fetch_stock_data_batch function."""

    def test_batch_returns_all_tickers(self, sample_ohlcv_df):
        """fetch_stock_data_batch returns results for every requested ticker."""
        from maverick_mcp.api.routers.data import fetch_stock_data_batch

        mock_service = MagicMock()
        mock_service.get_stock_data.return_value = sample_ohlcv_df

        @contextmanager
        def fake_session():
            yield MagicMock()

        with (
            patch(
                "maverick_mcp.api.routers.data.StockDataFetchingService",
                return_value=MagicMock(),
            ),
            patch(
                "maverick_mcp.api.routers.data.get_db_session_read_only",
                fake_session,
            ),
            patch(
                "maverick_mcp.api.routers.data.StockAnalysisService",
                return_value=mock_service,
            ),
            patch(
                "maverick_mcp.api.routers.data.CacheManagementService",
                return_value=MagicMock(),
            ),
        ):
            result = fetch_stock_data_batch(
                ["AAPL", "MSFT", "GOOGL"], "2024-01-01", "2024-12-31"
            )

        assert "results" in result
        assert result["success_count"] == 3
        assert result["error_count"] == 0
        assert set(result["tickers"]) == {"AAPL", "MSFT", "GOOGL"}

    def test_batch_handles_partial_failures(self, sample_ohlcv_df):
        """fetch_stock_data_batch correctly counts errors on partial failures."""
        from maverick_mcp.api.routers.data import fetch_stock_data_batch

        mock_service = MagicMock()
        call_count = {"n": 0}

        def side_effect(ticker, *args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("API error")
            return sample_ohlcv_df

        mock_service.get_stock_data.side_effect = side_effect

        @contextmanager
        def fake_session():
            yield MagicMock()

        with (
            patch(
                "maverick_mcp.api.routers.data.StockDataFetchingService",
                return_value=MagicMock(),
            ),
            patch(
                "maverick_mcp.api.routers.data.get_db_session_read_only",
                fake_session,
            ),
            patch(
                "maverick_mcp.api.routers.data.StockAnalysisService",
                return_value=mock_service,
            ),
            patch(
                "maverick_mcp.api.routers.data.CacheManagementService",
                return_value=MagicMock(),
            ),
        ):
            result = fetch_stock_data_batch(["AAPL", "FAIL", "MSFT"])

        assert result["success_count"] == 2
        assert result["error_count"] == 1


class TestGetStockInfo:
    """Tests for get_stock_info function."""

    def test_returns_structured_info(self, mock_stock_provider):
        """get_stock_info returns company and market_data sections."""
        from maverick_mcp.api.routers.data import get_stock_info

        @contextmanager
        def fake_session():
            yield MagicMock()

        with (
            patch(
                "maverick_mcp.api.routers.data.get_db_session_read_only",
                fake_session,
            ),
            patch(
                "maverick_mcp.api.routers.data.StockDataProvider",
                return_value=mock_stock_provider,
            ),
        ):
            result = get_stock_info("AAPL")

        assert result["ticker"] == "AAPL"
        assert "company" in result
        assert "market_data" in result
        assert result["company"]["name"] == "Apple Inc."

    def test_handles_missing_fields(self):
        """get_stock_info handles provider returning empty dict."""
        from maverick_mcp.api.routers.data import get_stock_info

        mock_provider = MagicMock()
        mock_provider.get_stock_info.return_value = {}

        @contextmanager
        def fake_session():
            yield MagicMock()

        with (
            patch(
                "maverick_mcp.api.routers.data.get_db_session_read_only",
                fake_session,
            ),
            patch(
                "maverick_mcp.api.routers.data.StockDataProvider",
                return_value=mock_provider,
            ),
        ):
            result = get_stock_info("AAPL")

        assert result["ticker"] == "AAPL"
        assert result["company"]["name"] is None


class TestGetNewsSentiment:
    """Tests for get_news_sentiment function."""

    def test_returns_dict_with_ticker(self):
        """get_news_sentiment always returns a dict with 'ticker' key."""
        from maverick_mcp.api.routers.data import get_news_sentiment

        result = get_news_sentiment("AAPL")
        assert isinstance(result, dict)
        assert result.get("ticker") == "AAPL"

    def test_returns_sentiment_or_articles(self):
        """get_news_sentiment returns either articles or a sentiment/status key."""
        from maverick_mcp.api.routers.data import get_news_sentiment

        result = get_news_sentiment("MSFT")
        assert isinstance(result, dict)
        # Either full articles or fallback/error dict
        has_expected_key = any(
            k in result for k in ("articles", "sentiment", "status", "error")
        )
        assert has_expected_key

    def test_handles_request_exception(self):
        """get_news_sentiment handles network errors gracefully."""
        import requests
        from maverick_mcp.api.routers.data import get_news_sentiment

        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("timeout")):
            result = get_news_sentiment("AAPL")

        # Should not raise; return error or fallback
        assert isinstance(result, dict)
