"""
Shared fixtures for MCP tool unit tests.

This conftest provides lightweight, mock-based fixtures that do not
require Docker, PostgreSQL, or Redis.
"""

import os

# Set test environment before any other imports
os.environ.setdefault("MAVERICK_TEST_ENV", "true")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("TIINGO_API_KEY", "test_tiingo_key")
os.environ.setdefault("OPENROUTER_API_KEY", "test_openrouter_key")

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_df():
    """Standard OHLCV DataFrame with 250 trading days."""
    dates = pd.date_range(end="2025-01-31", periods=250, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.normal(0, 1, 250))
    df = pd.DataFrame(
        {
            "open": close * np.random.uniform(0.99, 1.01, 250),
            "high": close * np.random.uniform(1.00, 1.03, 250),
            "low": close * np.random.uniform(0.97, 1.00, 250),
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, 250),
        },
        index=dates,
    )
    # Also provide uppercase aliases so both styles work
    df["Open"] = df["open"]
    df["High"] = df["high"]
    df["Low"] = df["low"]
    df["Close"] = df["close"]
    df["Volume"] = df["volume"]
    return df


@pytest.fixture
def mock_stock_provider(sample_ohlcv_df):
    """Mock StockDataProvider that returns sample data."""
    provider = MagicMock()
    provider.get_stock_data.return_value = sample_ohlcv_df
    provider.get_stock_info.return_value = {
        "longName": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "currentPrice": 175.0,
        "marketCap": 2_700_000_000_000,
        "trailingPE": 28.5,
        "website": "https://www.apple.com",
    }
    provider.get_all_screening_recommendations.return_value = {
        "maverick_stocks": [{"stock": "AAPL", "combined_score": 92}],
        "maverick_bear_stocks": [{"stock": "META", "combined_score": 30}],
        "supply_demand_breakouts": [{"stock": "MSFT", "combined_score": 88}],
    }
    return provider


@pytest.fixture
def mock_db_session():
    """Mock SQLAlchemy database session."""
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    return session


def make_mock_stock(ticker: str, score: float = 85.0) -> MagicMock:
    """Helper to create a mock screening stock object."""
    stock = MagicMock()
    stock.to_dict.return_value = {
        "stock": ticker,
        "close": 150.0,
        "combined_score": score,
        "momentum_score": score - 5,
        "adr_pct": 2.5,
        "sector": "Technology",
    }
    return stock
