"""
Unit tests for Stock Screening tools:
- get_maverick_stocks
- get_maverick_bear_stocks
- get_supply_demand_breakouts
- get_all_screening_recommendations
- get_screening_by_criteria

Note: screening.py uses lazy imports inside functions, so we patch at the
maverick_mcp.data.models level rather than at the router module level.
"""

from unittest.mock import MagicMock, patch

import pytest

from tests.unit.conftest import make_mock_stock


class TestGetMaverickStocks:
    """Tests for get_maverick_stocks."""

    def test_returns_bullish_stocks(self):
        """get_maverick_stocks returns status=success and stocks list."""
        from maverick_mcp.api.routers.screening import get_maverick_stocks

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        stocks = [make_mock_stock("AAPL", 92), make_mock_stock("MSFT", 88)]

        with (
            patch(
                "maverick_mcp.data.models.MaverickStocks.get_top_stocks",
                return_value=stocks,
            ),
            patch(
                "maverick_mcp.data.models.SessionLocal",
                return_value=mock_session,
            ),
        ):
            result = get_maverick_stocks(limit=10)

        assert result["status"] == "success"
        assert result["count"] == 2
        assert result["stocks"][0]["stock"] == "AAPL"
        assert result["screening_type"] == "maverick_bullish"

    def test_handles_db_error(self):
        """get_maverick_stocks returns error status on database failure."""
        from maverick_mcp.api.routers.screening import get_maverick_stocks

        with patch(
            "maverick_mcp.data.models.SessionLocal",
            side_effect=RuntimeError("DB down"),
        ):
            result = get_maverick_stocks()

        assert result["status"] == "error"
        assert "error" in result

    def test_returns_empty_list_when_no_stocks(self):
        """get_maverick_stocks handles empty result from database."""
        from maverick_mcp.api.routers.screening import get_maverick_stocks

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "maverick_mcp.data.models.MaverickStocks.get_top_stocks",
                return_value=[],
            ),
            patch(
                "maverick_mcp.data.models.SessionLocal",
                return_value=mock_session,
            ),
        ):
            result = get_maverick_stocks(limit=10)

        assert result["status"] == "success"
        assert result["count"] == 0
        assert result["stocks"] == []


class TestGetMaverickBearStocks:
    """Tests for get_maverick_bear_stocks."""

    def test_returns_bearish_stocks(self):
        """get_maverick_bear_stocks returns bearish screening type."""
        from maverick_mcp.api.routers.screening import get_maverick_bear_stocks

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        stocks = [make_mock_stock("TSLA", 20)]

        with (
            patch(
                "maverick_mcp.data.models.MaverickBearStocks.get_top_stocks",
                return_value=stocks,
            ),
            patch(
                "maverick_mcp.data.models.SessionLocal",
                return_value=mock_session,
            ),
        ):
            result = get_maverick_bear_stocks(limit=5)

        assert result["status"] == "success"
        assert result["screening_type"] == "maverick_bearish"
        assert len(result["stocks"]) == 1

    def test_handles_db_error(self):
        """get_maverick_bear_stocks returns error on failure."""
        from maverick_mcp.api.routers.screening import get_maverick_bear_stocks

        with patch(
            "maverick_mcp.data.models.SessionLocal",
            side_effect=Exception("Connection reset"),
        ):
            result = get_maverick_bear_stocks()

        assert result["status"] == "error"


class TestGetSupplyDemandBreakouts:
    """Tests for get_supply_demand_breakouts."""

    def test_returns_breakout_stocks(self):
        """get_supply_demand_breakouts returns breakout screening type."""
        from maverick_mcp.api.routers.screening import get_supply_demand_breakouts

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        stocks = [make_mock_stock("NVDA", 95), make_mock_stock("AMD", 90)]

        with (
            patch(
                "maverick_mcp.data.models.SupplyDemandBreakoutStocks.get_top_stocks",
                return_value=stocks,
            ),
            patch(
                "maverick_mcp.data.models.SessionLocal",
                return_value=mock_session,
            ),
        ):
            result = get_supply_demand_breakouts(limit=10)

        assert result["status"] == "success"
        assert result["screening_type"] == "supply_demand_breakout"
        assert result["count"] == 2

    def test_filter_moving_averages_flag(self):
        """get_supply_demand_breakouts uses filtered query when flag is True."""
        from maverick_mcp.api.routers.screening import get_supply_demand_breakouts

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        stocks = [make_mock_stock("AAPL", 95)]

        with (
            patch(
                "maverick_mcp.data.models.SupplyDemandBreakoutStocks.get_stocks_above_moving_averages",
                return_value=stocks,
            ),
            patch(
                "maverick_mcp.data.models.SessionLocal",
                return_value=mock_session,
            ),
        ):
            result = get_supply_demand_breakouts(filter_moving_averages=True, limit=5)

        assert result["status"] == "success"
        assert result["count"] == 1

    def test_handles_db_error(self):
        """get_supply_demand_breakouts returns error on failure."""
        from maverick_mcp.api.routers.screening import get_supply_demand_breakouts

        with patch(
            "maverick_mcp.data.models.SessionLocal",
            side_effect=Exception("DB error"),
        ):
            result = get_supply_demand_breakouts()

        assert result["status"] == "error"


class TestGetAllScreeningRecommendations:
    """Tests for get_all_screening_recommendations."""

    def test_returns_all_three_categories(self, mock_stock_provider):
        """get_all_screening_recommendations returns all strategy categories."""
        from maverick_mcp.api.routers.screening import get_all_screening_recommendations

        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider.get_all_screening_recommendations",
            return_value=mock_stock_provider.get_all_screening_recommendations(),
        ):
            result = get_all_screening_recommendations()

        assert "maverick_stocks" in result
        assert "maverick_bear_stocks" in result
        assert "supply_demand_breakouts" in result

    def test_handles_provider_error(self):
        """get_all_screening_recommendations returns error dict on failure."""
        from maverick_mcp.api.routers.screening import get_all_screening_recommendations

        with patch(
            "maverick_mcp.providers.stock_data.StockDataProvider.get_all_screening_recommendations",
            side_effect=RuntimeError("API down"),
        ):
            result = get_all_screening_recommendations()

        assert result["status"] == "error"


class TestGetScreeningByCriteria:
    """Tests for get_screening_by_criteria."""

    def test_no_criteria_returns_results(self):
        """get_screening_by_criteria with no filters returns all results."""
        from maverick_mcp.api.routers.screening import get_screening_by_criteria

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [make_mock_stock("AAPL", 90)]

        with patch(
            "maverick_mcp.data.models.SessionLocal",
            return_value=mock_session,
        ):
            result = get_screening_by_criteria(limit=10)

        assert result["status"] == "success"

    def test_converts_string_inputs(self):
        """get_screening_by_criteria converts string numeric inputs correctly."""
        from maverick_mcp.api.routers.screening import get_screening_by_criteria

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []

        with patch(
            "maverick_mcp.data.models.SessionLocal",
            return_value=mock_session,
        ):
            result = get_screening_by_criteria(
                min_momentum_score="75", max_price="200", limit="15"
            )

        # Criteria should be parsed as float/int
        assert result["criteria"]["min_momentum_score"] == 75.0
        assert result["criteria"]["max_price"] == 200.0

    def test_handles_db_error(self):
        """get_screening_by_criteria returns error on database failure."""
        from maverick_mcp.api.routers.screening import get_screening_by_criteria

        with patch(
            "maverick_mcp.data.models.SessionLocal",
            side_effect=Exception("DB error"),
        ):
            result = get_screening_by_criteria(min_momentum_score=80.0)

        assert result["status"] == "error"
