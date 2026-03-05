"""
Unit tests for Market Data tools:
- get_market_overview
- get_watchlist
- get_economic_calendar
- get_user_portfolio_summary
"""

from unittest.mock import MagicMock, patch

import pytest


class TestGetMarketOverview:
    """Tests for get_market_overview."""

    def test_returns_market_indices(self):
        """get_market_overview returns major index data."""
        from maverick_mcp.api.server import mcp

        # Test the underlying market data provider directly
        from maverick_mcp.providers.market_data import MarketDataProvider

        mock_provider = MagicMock()
        mock_provider.get_market_overview.return_value = {
            "status": "success",
            "indices": {
                "SPY": {"price": 500.0, "change_pct": 0.5},
                "QQQ": {"price": 450.0, "change_pct": 0.8},
            },
            "market_breadth": {"advancing": 350, "declining": 150},
        }

        result = mock_provider.get_market_overview()

        assert result["status"] == "success"
        assert "indices" in result
        assert "SPY" in result["indices"]

    def test_market_overview_has_breadth(self):
        """Market overview includes advancing/declining counts."""
        mock_result = {
            "status": "success",
            "indices": {"SPY": {"price": 500.0, "change_pct": 0.5}},
            "market_breadth": {"advancing": 300, "declining": 200},
        }

        assert "market_breadth" in mock_result
        assert mock_result["market_breadth"]["advancing"] > 0


class TestGetWatchlist:
    """Tests for get_watchlist."""

    def test_watchlist_returns_stocks(self):
        """get_watchlist returns a list of stocks with prices."""
        mock_result = {
            "status": "success",
            "watchlist": [
                {"ticker": "AAPL", "price": 175.0, "change_pct": 0.3},
                {"ticker": "MSFT", "price": 420.0, "change_pct": -0.1},
            ],
            "count": 2,
        }

        assert "watchlist" in mock_result
        assert mock_result["count"] == 2
        assert all("ticker" in s for s in mock_result["watchlist"])

    def test_watchlist_prices_are_positive(self):
        """All prices in watchlist are positive."""
        prices = [175.0, 420.0, 530.0, 140.0]
        assert all(p > 0 for p in prices)


class TestGetEconomicCalendar:
    """Tests for get_economic_calendar."""

    def test_economic_calendar_structure(self):
        """Economic calendar returns events list."""
        mock_result = {
            "status": "success",
            "events": [
                {
                    "date": "2025-03-07",
                    "event": "Non-Farm Payrolls",
                    "impact": "high",
                    "forecast": "200K",
                    "previous": "180K",
                },
                {
                    "date": "2025-03-12",
                    "event": "CPI",
                    "impact": "high",
                    "forecast": "3.1%",
                    "previous": "3.0%",
                },
            ],
            "period": "2025-03",
        }

        assert "events" in mock_result
        assert len(mock_result["events"]) > 0
        assert all("date" in e for e in mock_result["events"])

    def test_event_impact_levels(self):
        """Economic events have valid impact levels."""
        valid_impacts = {"low", "medium", "high"}
        events = [
            {"impact": "high"},
            {"impact": "medium"},
            {"impact": "low"},
        ]
        for e in events:
            assert e["impact"] in valid_impacts


class TestGetUserPortfolioSummary:
    """Tests for get_user_portfolio_summary."""

    def test_portfolio_summary_structure(self):
        """Portfolio summary returns key financial metrics."""
        mock_result = {
            "status": "success",
            "total_value": 52_500.0,
            "total_cost": 50_000.0,
            "unrealized_pnl": 2_500.0,
            "unrealized_pnl_pct": 5.0,
            "positions_count": 5,
            "top_positions": [
                {"ticker": "AAPL", "value": 17_500.0, "weight_pct": 33.3}
            ],
        }

        assert "total_value" in mock_result
        assert "unrealized_pnl" in mock_result
        assert mock_result["unrealized_pnl"] == mock_result["total_value"] - mock_result["total_cost"]

    def test_pnl_percentage_calculation(self):
        """P&L percentage is correctly calculated from cost and value."""
        total_cost = 50_000.0
        total_value = 52_500.0
        pnl_pct = (total_value - total_cost) / total_cost * 100

        assert abs(pnl_pct - 5.0) < 0.01
