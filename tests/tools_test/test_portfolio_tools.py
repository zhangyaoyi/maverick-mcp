"""
Unit tests for Portfolio Management tools:
- _normalize_ticker
- _validate_ticker
- risk_adjusted_analysis
- add_portfolio_position
- get_my_portfolio
- remove_portfolio_position
- clear_my_portfolio
- compare_tickers
- portfolio_correlation_analysis
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tests.unit.conftest import make_mock_stock


class TestNormalizeTicker:
    """Tests for _normalize_ticker helper."""

    def test_converts_to_uppercase(self):
        from maverick_mcp.api.routers.portfolio import _normalize_ticker
        assert _normalize_ticker("aapl") == "AAPL"

    def test_strips_whitespace(self):
        from maverick_mcp.api.routers.portfolio import _normalize_ticker
        assert _normalize_ticker("  MSFT  ") == "MSFT"

    def test_mixed_case(self):
        from maverick_mcp.api.routers.portfolio import _normalize_ticker
        assert _normalize_ticker("gOoGl") == "GOOGL"


class TestValidateTicker:
    """Tests for _validate_ticker helper."""

    def test_valid_ticker(self):
        from maverick_mcp.api.routers.portfolio import _validate_ticker
        valid, error = _validate_ticker("AAPL")
        assert valid is True
        assert error is None

    def test_empty_ticker_is_invalid(self):
        from maverick_mcp.api.routers.portfolio import _validate_ticker
        valid, error = _validate_ticker("")
        assert valid is False
        assert error is not None

    def test_whitespace_only_is_invalid(self):
        from maverick_mcp.api.routers.portfolio import _validate_ticker
        valid, error = _validate_ticker("   ")
        assert valid is False

    def test_too_long_ticker_is_invalid(self):
        from maverick_mcp.api.routers.portfolio import _validate_ticker
        valid, error = _validate_ticker("TOOLONGTICKER")
        assert valid is False
        assert "too long" in error.lower()

    def test_special_characters_are_invalid(self):
        from maverick_mcp.api.routers.portfolio import _validate_ticker
        valid, error = _validate_ticker("AA-PL")
        assert valid is False

    def test_numeric_ticker_is_valid(self):
        from maverick_mcp.api.routers.portfolio import _validate_ticker
        valid, error = _validate_ticker("1234")
        assert valid is True


class TestRiskAdjustedAnalysis:
    """Tests for risk_adjusted_analysis."""

    def test_returns_risk_analysis_structure(self, sample_ohlcv_df):
        """risk_adjusted_analysis returns expected nested structure."""
        from maverick_mcp.api.routers.portfolio import risk_adjusted_analysis

        mock_provider = MagicMock()
        mock_provider.get_stock_data.return_value = sample_ohlcv_df

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        with (
            patch(
                "maverick_mcp.api.routers.portfolio.stock_provider",
                mock_provider,
            ),
            patch(
                "maverick_mcp.api.routers.portfolio.get_db",
                return_value=iter([mock_db]),
            ),
        ):
            result = risk_adjusted_analysis("AAPL", risk_level=50.0)

        assert result["ticker"] == "AAPL"
        assert "position_sizing" in result
        assert "risk_management" in result
        assert "entry_strategy" in result
        assert "targets" in result

    def test_converts_string_risk_level(self, sample_ohlcv_df):
        """risk_adjusted_analysis accepts risk_level as string."""
        from maverick_mcp.api.routers.portfolio import risk_adjusted_analysis

        mock_provider = MagicMock()
        mock_provider.get_stock_data.return_value = sample_ohlcv_df
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        with (
            patch("maverick_mcp.api.routers.portfolio.stock_provider", mock_provider),
            patch(
                "maverick_mcp.api.routers.portfolio.get_db",
                return_value=iter([mock_db]),
            ),
        ):
            result = risk_adjusted_analysis("AAPL", risk_level="75")

        assert result["risk_level"] == 75.0

    def test_handles_empty_dataframe(self):
        """risk_adjusted_analysis returns error when data is empty."""
        from maverick_mcp.api.routers.portfolio import risk_adjusted_analysis

        mock_provider = MagicMock()
        mock_provider.get_stock_data.return_value = pd.DataFrame()

        with patch("maverick_mcp.api.routers.portfolio.stock_provider", mock_provider):
            result = risk_adjusted_analysis("AAPL")

        assert "error" in result

    def test_aggressive_strategy_at_high_risk(self, sample_ohlcv_df):
        """risk_adjusted_analysis returns aggressive strategy for risk_level > 70."""
        from maverick_mcp.api.routers.portfolio import risk_adjusted_analysis

        mock_provider = MagicMock()
        mock_provider.get_stock_data.return_value = sample_ohlcv_df
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        with (
            patch("maverick_mcp.api.routers.portfolio.stock_provider", mock_provider),
            patch(
                "maverick_mcp.api.routers.portfolio.get_db",
                return_value=iter([mock_db]),
            ),
        ):
            result = risk_adjusted_analysis("AAPL", risk_level=80.0)

        assert result["analysis"]["strategy_type"] == "aggressive"

    def test_conservative_strategy_at_low_risk(self, sample_ohlcv_df):
        """risk_adjusted_analysis returns conservative strategy for risk_level < 30."""
        from maverick_mcp.api.routers.portfolio import risk_adjusted_analysis

        mock_provider = MagicMock()
        mock_provider.get_stock_data.return_value = sample_ohlcv_df
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        with (
            patch("maverick_mcp.api.routers.portfolio.stock_provider", mock_provider),
            patch(
                "maverick_mcp.api.routers.portfolio.get_db",
                return_value=iter([mock_db]),
            ),
        ):
            result = risk_adjusted_analysis("AAPL", risk_level=10.0)

        assert result["analysis"]["strategy_type"] == "conservative"


class TestAddPortfolioPosition:
    """Tests for add_portfolio_position (the actual function name in portfolio.py)."""

    def test_add_invalid_ticker_returns_error(self):
        """add_portfolio_position returns error for empty ticker."""
        from maverick_mcp.api.routers.portfolio import add_portfolio_position

        result = add_portfolio_position(ticker="", shares=10.0, purchase_price=150.0)
        assert "error" in result

    def test_add_negative_shares_returns_error(self):
        """add_portfolio_position returns error for negative shares."""
        from maverick_mcp.api.routers.portfolio import add_portfolio_position

        result = add_portfolio_position(ticker="AAPL", shares=-5.0, purchase_price=150.0)
        assert "error" in result

    def test_add_zero_cost_returns_error(self):
        """add_portfolio_position returns error for zero purchase_price."""
        from maverick_mcp.api.routers.portfolio import add_portfolio_position

        result = add_portfolio_position(ticker="AAPL", shares=10.0, purchase_price=0.0)
        assert "error" in result

    def test_add_valid_position_succeeds(self):
        """add_portfolio_position creates portfolio and position on valid input."""
        from maverick_mcp.api.routers.portfolio import add_portfolio_position

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        with patch(
            "maverick_mcp.api.routers.portfolio.get_db",
            return_value=iter([mock_db]),
        ):
            result = add_portfolio_position(
                ticker="AAPL",
                shares=10.0,
                purchase_price=150.0,
            )

        assert "status" in result or "ticker" in result


class TestGetMyPortfolio:
    """Tests for get_my_portfolio."""

    def test_returns_portfolio_structure(self):
        """get_my_portfolio returns portfolio with positions list."""
        from maverick_mcp.api.routers.portfolio import get_my_portfolio

        mock_portfolio = MagicMock()
        mock_portfolio.id = 1
        mock_portfolio.name = "My Portfolio"
        mock_portfolio.user_id = "default"

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_portfolio
        )
        mock_db.query.return_value.filter.return_value.all.return_value = []

        with patch(
            "maverick_mcp.api.routers.portfolio.get_db",
            return_value=iter([mock_db]),
        ):
            result = get_my_portfolio()

        assert "portfolio" in result or "status" in result

    def test_returns_empty_when_no_portfolio(self):
        """get_my_portfolio returns empty structure when no portfolio found."""
        from maverick_mcp.api.routers.portfolio import get_my_portfolio

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        with patch(
            "maverick_mcp.api.routers.portfolio.get_db",
            return_value=iter([mock_db]),
        ):
            result = get_my_portfolio()

        assert "portfolio" in result or "positions" in result or "status" in result


class TestRemovePortfolioPosition:
    """Tests for remove_portfolio_position."""

    def test_remove_invalid_ticker_returns_error(self):
        """remove_portfolio_position returns error for empty ticker."""
        from maverick_mcp.api.routers.portfolio import remove_portfolio_position

        result = remove_portfolio_position("")
        assert "error" in result

    def test_remove_existing_position(self):
        """remove_portfolio_position handles removal when position found."""
        from maverick_mcp.api.routers.portfolio import remove_portfolio_position

        mock_portfolio = MagicMock()
        mock_portfolio.id = 1

        mock_position = MagicMock()
        mock_position.ticker = "AAPL"
        mock_position.shares = Decimal("10")

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            mock_portfolio,
            mock_position,
        ]

        with patch(
            "maverick_mcp.api.routers.portfolio.get_db",
            return_value=iter([mock_db]),
        ):
            result = remove_portfolio_position("AAPL")

        assert "status" in result


class TestClearMyPortfolio:
    """Tests for clear_my_portfolio."""

    def test_clear_requires_confirmation(self):
        """clear_my_portfolio returns warning without confirm=True."""
        from maverick_mcp.api.routers.portfolio import clear_my_portfolio

        result = clear_my_portfolio(confirm=False)
        # Should not clear — return warning or informational message
        assert any(k in result for k in ("warning", "confirm", "error", "message"))

    def test_clear_with_confirmation(self):
        """clear_my_portfolio deletes positions when confirm=True."""
        from maverick_mcp.api.routers.portfolio import clear_my_portfolio

        mock_portfolio = MagicMock()
        mock_portfolio.id = 1

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_portfolio
        )
        mock_db.query.return_value.filter.return_value.delete.return_value = 3

        with patch(
            "maverick_mcp.api.routers.portfolio.get_db",
            return_value=iter([mock_db]),
        ):
            result = clear_my_portfolio(confirm=True)

        assert "status" in result
