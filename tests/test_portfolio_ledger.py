"""Portfolio Ledger behavior tests.

Uses local SQLite in-memory databases — no Docker required.
Tests run in 5-10 seconds.

Known bugs documented inline (marked with BUG):
  Bug #1  portfolio_ledger_service.py:85  – first BUY with fee: avg_cost = price (excludes fee)
  Bug #2  portfolio.py:1145               – remove_position SELL uses avg_cost as price (P&L always 0)
  Bug #3  portfolio_ledger_service.py:79  – every BUY overwrites purchase_date (should keep earliest)
  Bug #4  lot_method field stored but ignored; always uses average-cost method
"""
from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from maverick_mcp.api.routers import portfolio as portfolio_router
from maverick_mcp.api.services.portfolio_ledger_service import (
    PortfolioLedgerService,
    RecordTradeInput,
)
from maverick_mcp.data.models import (
    PortfolioPosition,
    PortfolioTransaction,
    UserPortfolio,
)
from maverick_mcp.database.base import Base

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_session() -> Session:
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)()


def _override_get_db(db_session: Session):
    def _gen():
        yield db_session

    return _gen


def _mock_stock_provider(price: float = 150.0):
    """Return a mock StockDataProvider that returns a single-row DataFrame."""
    mock = MagicMock()
    df = pd.DataFrame({"Close": [price]})
    mock.get_stock_data.return_value = df
    return mock


# ---------------------------------------------------------------------------
# Group 1: PortfolioLedgerService — BUY logic (11 tests)
# ---------------------------------------------------------------------------


def test_service_buy_creates_portfolio_and_position():
    """First BUY auto-creates UserPortfolio + PortfolioPosition."""
    db = _make_session()
    svc = PortfolioLedgerService(db)
    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL",
            side="BUY",
            quantity=Decimal("10"),
            price=Decimal("100"),
        ),
    )
    db.commit()

    portfolio = db.query(UserPortfolio).filter_by(user_id="u1", name="p1").first()
    assert portfolio is not None

    position = (
        db.query(PortfolioPosition)
        .filter_by(portfolio_id=portfolio.id, ticker="AAPL")
        .first()
    )
    assert position is not None
    db.close()


def test_service_buy_creates_transaction_record():
    """BUY writes a PortfolioTransaction with correct field values."""
    db = _make_session()
    svc = PortfolioLedgerService(db)
    txn = svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL",
            side="BUY",
            quantity=Decimal("5"),
            price=Decimal("200"),
            fee=Decimal("2.50"),
        ),
    )
    db.commit()

    assert txn.ticker == "AAPL"
    assert txn.side == "BUY"
    assert Decimal(str(txn.quantity)) == Decimal("5")
    assert Decimal(str(txn.price)) == Decimal("200")
    assert Decimal(str(txn.fee)) == Decimal("2.50")
    db.close()


def test_service_buy_no_fee_avg_cost_equals_price():
    """No fee: average_cost_basis == price (correct behavior)."""
    db = _make_session()
    svc = PortfolioLedgerService(db)
    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL",
            side="BUY",
            quantity=Decimal("10"),
            price=Decimal("100"),
            fee=Decimal("0"),
        ),
    )
    db.commit()

    portfolio = db.query(UserPortfolio).filter_by(user_id="u1", name="p1").first()
    position = (
        db.query(PortfolioPosition)
        .filter_by(portfolio_id=portfolio.id, ticker="AAPL")
        .first()
    )
    assert Decimal(str(position.average_cost_basis)) == Decimal("100")
    db.close()


def test_service_buy_with_fee_initial_avg_cost_bug():
    """BUG #1: First BUY with fee — avg_cost = price (fee excluded).

    Expected (correct): avg_cost = (qty*price + fee) / qty = (10*100 + 10) / 10 = 101.00
    Actual (buggy):     avg_cost = price = 100.00

    This test documents the current broken behavior. Fix: line 85 in
    portfolio_ledger_service.py should set average_cost_basis=total_cost/quantity.
    """
    db = _make_session()
    svc = PortfolioLedgerService(db)
    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL",
            side="BUY",
            quantity=Decimal("10"),
            price=Decimal("100"),
            fee=Decimal("10"),
        ),
    )
    db.commit()

    portfolio = db.query(UserPortfolio).filter_by(user_id="u1", name="p1").first()
    position = (
        db.query(PortfolioPosition)
        .filter_by(portfolio_id=portfolio.id, ticker="AAPL")
        .first()
    )
    # BUG: currently equals price, not (qty*price+fee)/qty
    assert Decimal(str(position.average_cost_basis)) == Decimal("100")
    # The correct value would be 101.00 — uncomment after fix:
    # assert Decimal(str(position.average_cost_basis)) == Decimal("101.00")
    db.close()


def test_service_buy_second_purchase_weighted_avg():
    """Two buys: weighted average cost correctly computed on second BUY."""
    db = _make_session()
    svc = PortfolioLedgerService(db)

    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL",
            side="BUY",
            quantity=Decimal("10"),
            price=Decimal("100"),
        ),
    )
    db.commit()

    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL",
            side="BUY",
            quantity=Decimal("10"),
            price=Decimal("120"),
        ),
    )
    db.commit()

    portfolio = db.query(UserPortfolio).filter_by(user_id="u1", name="p1").first()
    position = (
        db.query(PortfolioPosition)
        .filter_by(portfolio_id=portfolio.id, ticker="AAPL")
        .first()
    )
    # (10*100 + 10*120) / 20 = 110
    assert Decimal(str(position.average_cost_basis)) == Decimal("110")
    db.close()


def test_service_buy_with_fee_total_cost_correct():
    """total_cost = qty*price + fee (includes fee correctly)."""
    db = _make_session()
    svc = PortfolioLedgerService(db)
    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL",
            side="BUY",
            quantity=Decimal("10"),
            price=Decimal("100"),
            fee=Decimal("10"),
        ),
    )
    db.commit()

    portfolio = db.query(UserPortfolio).filter_by(user_id="u1", name="p1").first()
    position = (
        db.query(PortfolioPosition)
        .filter_by(portfolio_id=portfolio.id, ticker="AAPL")
        .first()
    )
    assert Decimal(str(position.total_cost)) == Decimal("1010")
    db.close()


def test_service_buy_shares_accumulate_across_purchases():
    """Three buys: shares correctly accumulate."""
    db = _make_session()
    svc = PortfolioLedgerService(db)

    for qty in [5, 3, 7]:
        svc.record_trade(
            user_id="u1",
            portfolio_name="p1",
            payload=RecordTradeInput(
                ticker="AAPL",
                side="BUY",
                quantity=Decimal(str(qty)),
                price=Decimal("100"),
            ),
        )
        db.commit()

    portfolio = db.query(UserPortfolio).filter_by(user_id="u1", name="p1").first()
    position = (
        db.query(PortfolioPosition)
        .filter_by(portfolio_id=portfolio.id, ticker="AAPL")
        .first()
    )
    assert Decimal(str(position.shares)) == Decimal("15")
    db.close()


def test_service_buy_uses_existing_portfolio():
    """Second BUY reuses the same UserPortfolio row, not a new one."""
    db = _make_session()
    svc = PortfolioLedgerService(db)

    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL",
            side="BUY",
            quantity=Decimal("5"),
            price=Decimal("100"),
        ),
    )
    db.commit()

    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="MSFT",
            side="BUY",
            quantity=Decimal("3"),
            price=Decimal("200"),
        ),
    )
    db.commit()

    count = db.query(UserPortfolio).filter_by(user_id="u1", name="p1").count()
    assert count == 1
    db.close()


def test_service_buy_ticker_normalized_uppercase():
    """ticker is stored in uppercase regardless of input case."""
    db = _make_session()
    svc = PortfolioLedgerService(db)
    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="aapl",
            side="BUY",
            quantity=Decimal("5"),
            price=Decimal("100"),
        ),
    )
    db.commit()

    portfolio = db.query(UserPortfolio).filter_by(user_id="u1", name="p1").first()
    position = (
        db.query(PortfolioPosition).filter_by(portfolio_id=portfolio.id).first()
    )
    assert position.ticker == "AAPL"
    db.close()


def test_service_buy_fractional_shares_precision():
    """0.5 fractional shares stored without precision loss."""
    db = _make_session()
    svc = PortfolioLedgerService(db)
    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL",
            side="BUY",
            quantity=Decimal("0.5"),
            price=Decimal("100"),
        ),
    )
    db.commit()

    portfolio = db.query(UserPortfolio).filter_by(user_id="u1", name="p1").first()
    position = (
        db.query(PortfolioPosition)
        .filter_by(portfolio_id=portfolio.id, ticker="AAPL")
        .first()
    )
    assert Decimal(str(position.shares)) == Decimal("0.5")
    db.close()


def test_service_buy_updates_purchase_date_bug():
    """BUG #3: Subsequent BUY overwrites purchase_date instead of keeping earliest.

    Expected (correct): purchase_date stays at the first buy date.
    Actual (buggy):     purchase_date is overwritten by every BUY.

    Fix: line 79 in portfolio_ledger_service.py — only set purchase_date when
    pos.purchase_date is None or trade_time < pos.purchase_date.
    """
    db = _make_session()
    svc = PortfolioLedgerService(db)

    first_date = datetime(2024, 1, 1, tzinfo=UTC)
    second_date = datetime(2024, 6, 1, tzinfo=UTC)

    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL",
            side="BUY",
            quantity=Decimal("5"),
            price=Decimal("100"),
            trade_time=first_date,
        ),
    )
    db.commit()

    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL",
            side="BUY",
            quantity=Decimal("5"),
            price=Decimal("110"),
            trade_time=second_date,
        ),
    )
    db.commit()

    portfolio = db.query(UserPortfolio).filter_by(user_id="u1", name="p1").first()
    position = (
        db.query(PortfolioPosition)
        .filter_by(portfolio_id=portfolio.id, ticker="AAPL")
        .first()
    )
    # BUG: currently overwritten to second_date
    # After fix this should equal first_date
    # SQLite strips tzinfo, so compare naive datetimes
    assert position.purchase_date.replace(tzinfo=None) == second_date.replace(tzinfo=None)
    # Correct behavior would be: assert position.purchase_date.replace(tzinfo=None) == first_date.replace(tzinfo=None)
    db.close()


# ---------------------------------------------------------------------------
# Group 2: PortfolioLedgerService — SELL logic (7 tests)
# ---------------------------------------------------------------------------


def test_service_sell_partial_reduces_shares():
    db = _make_session()
    svc = PortfolioLedgerService(db)
    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL", side="BUY", quantity=Decimal("10"), price=Decimal("100")
        ),
    )
    db.commit()

    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL", side="SELL", quantity=Decimal("4"), price=Decimal("110")
        ),
    )
    db.commit()

    portfolio = db.query(UserPortfolio).filter_by(user_id="u1", name="p1").first()
    position = (
        db.query(PortfolioPosition)
        .filter_by(portfolio_id=portfolio.id, ticker="AAPL")
        .first()
    )
    assert Decimal(str(position.shares)) == Decimal("6")
    db.close()


def test_service_sell_partial_total_cost_recalculated():
    """After partial sell, total_cost = remaining_shares * avg_cost."""
    db = _make_session()
    svc = PortfolioLedgerService(db)
    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL", side="BUY", quantity=Decimal("10"), price=Decimal("100")
        ),
    )
    db.commit()

    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL", side="SELL", quantity=Decimal("4"), price=Decimal("110")
        ),
    )
    db.commit()

    portfolio = db.query(UserPortfolio).filter_by(user_id="u1", name="p1").first()
    position = (
        db.query(PortfolioPosition)
        .filter_by(portfolio_id=portfolio.id, ticker="AAPL")
        .first()
    )
    expected_total = Decimal("6") * Decimal(str(position.average_cost_basis))
    assert Decimal(str(position.total_cost)) == expected_total
    db.close()


def test_service_sell_full_deletes_position():
    """Selling all shares removes the PortfolioPosition row."""
    db = _make_session()
    svc = PortfolioLedgerService(db)
    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL", side="BUY", quantity=Decimal("5"), price=Decimal("100")
        ),
    )
    db.commit()

    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL", side="SELL", quantity=Decimal("5"), price=Decimal("120")
        ),
    )
    db.commit()

    portfolio = db.query(UserPortfolio).filter_by(user_id="u1", name="p1").first()
    position = (
        db.query(PortfolioPosition)
        .filter_by(portfolio_id=portfolio.id, ticker="AAPL")
        .first()
    )
    assert position is None
    db.close()


def test_service_sell_creates_transaction_record():
    """SELL writes a PortfolioTransaction with side='SELL'."""
    db = _make_session()
    svc = PortfolioLedgerService(db)
    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL", side="BUY", quantity=Decimal("10"), price=Decimal("100")
        ),
    )
    db.commit()

    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL", side="SELL", quantity=Decimal("3"), price=Decimal("120")
        ),
    )
    db.commit()

    txns = (
        db.query(PortfolioTransaction)
        .filter_by(ticker="AAPL", side="SELL")
        .all()
    )
    assert len(txns) == 1
    assert Decimal(str(txns[0].quantity)) == Decimal("3")
    db.close()


def test_service_sell_no_position_raises_value_error():
    """SELL on ticker with no position raises ValueError."""
    db = _make_session()
    svc = PortfolioLedgerService(db)
    # Create portfolio so it exists
    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="MSFT", side="BUY", quantity=Decimal("5"), price=Decimal("200")
        ),
    )
    db.commit()

    with pytest.raises(ValueError, match="No existing position"):
        svc.record_trade(
            user_id="u1",
            portfolio_name="p1",
            payload=RecordTradeInput(
                ticker="AAPL",
                side="SELL",
                quantity=Decimal("3"),
                price=Decimal("150"),
            ),
        )
    db.close()


def test_service_sell_oversell_raises_value_error():
    """Selling more shares than held raises ValueError."""
    db = _make_session()
    svc = PortfolioLedgerService(db)
    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL", side="BUY", quantity=Decimal("5"), price=Decimal("100")
        ),
    )
    db.commit()

    with pytest.raises(ValueError, match="Cannot sell more shares"):
        svc.record_trade(
            user_id="u1",
            portfolio_name="p1",
            payload=RecordTradeInput(
                ticker="AAPL",
                side="SELL",
                quantity=Decimal("10"),
                price=Decimal("120"),
            ),
        )
    db.close()


def test_service_sell_exact_shares_closes_position():
    """Selling exactly the held quantity closes the position cleanly."""
    db = _make_session()
    svc = PortfolioLedgerService(db)
    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL", side="BUY", quantity=Decimal("7"), price=Decimal("100")
        ),
    )
    db.commit()

    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL", side="SELL", quantity=Decimal("7"), price=Decimal("110")
        ),
    )
    db.commit()

    portfolio = db.query(UserPortfolio).filter_by(user_id="u1", name="p1").first()
    position = (
        db.query(PortfolioPosition)
        .filter_by(portfolio_id=portfolio.id, ticker="AAPL")
        .first()
    )
    assert position is None
    db.close()


# ---------------------------------------------------------------------------
# Group 3: add_portfolio_position — input validation (6 tests)
# ---------------------------------------------------------------------------


def test_add_empty_ticker_returns_error(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.add_portfolio_position(
        ticker="", shares=10, purchase_price=100.0
    )
    assert result["status"] == "error"
    db.close()


def test_add_negative_shares_returns_error(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=-5, purchase_price=100.0
    )
    assert result["status"] == "error"
    db.close()


def test_add_zero_price_returns_error(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=10, purchase_price=0.0
    )
    assert result["status"] == "error"
    db.close()


def test_add_excessive_shares_returns_error(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=2_000_000_000, purchase_price=100.0
    )
    assert result["status"] == "error"
    db.close()


def test_add_excessive_price_returns_error(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=10, purchase_price=2_000_000.0
    )
    assert result["status"] == "error"
    db.close()


def test_add_invalid_date_format_returns_error(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=10, purchase_price=100.0, purchase_date="13/31/2024"
    )
    assert result["status"] == "error"
    db.close()


# ---------------------------------------------------------------------------
# Group 4: add_portfolio_position — normal flows (8 tests)
# ---------------------------------------------------------------------------


def test_add_first_buy_returns_success_status(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.add_portfolio_position(
        ticker="AAPL",
        shares=10,
        purchase_price=150.0,
        user_id="u1",
        portfolio_name="p1",
    )
    assert result["status"] == "success"
    db.close()


def test_add_position_data_round_trip(monkeypatch):
    """Returned position data matches what was submitted."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.add_portfolio_position(
        ticker="MSFT",
        shares=5,
        purchase_price=200.0,
        user_id="u1",
        portfolio_name="p1",
    )
    assert result["status"] == "success"
    pos = result["position"]
    assert pos["ticker"] == "MSFT"
    assert pos["shares"] == 5.0
    assert pos["average_cost_basis"] == 200.0
    assert pos["total_cost"] == 1000.0
    db.close()


def test_add_second_buy_averages_cost_basis(monkeypatch):
    """Two buys at different prices produce correct weighted average."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))

    portfolio_router.add_portfolio_position(
        ticker="AAPL",
        shares=10,
        purchase_price=100.0,
        user_id="u1",
        portfolio_name="p1",
    )
    result = portfolio_router.add_portfolio_position(
        ticker="AAPL",
        shares=10,
        purchase_price=120.0,
        user_id="u1",
        portfolio_name="p1",
    )
    assert result["status"] == "success"
    # (10*100 + 10*120) / 20 = 110
    assert result["position"]["average_cost_basis"] == 110.0
    assert result["position"]["shares"] == 20.0
    db.close()


def test_add_with_purchase_date_iso_format(monkeypatch):
    """ISO date string is accepted and round-tripped in response."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.add_portfolio_position(
        ticker="AAPL",
        shares=5,
        purchase_price=100.0,
        purchase_date="2024-01-15",
        user_id="u1",
        portfolio_name="p1",
    )
    assert result["status"] == "success"
    assert "2024-01-15" in result["position"]["purchase_date"]
    db.close()


def test_add_without_purchase_date_uses_now(monkeypatch):
    """No purchase_date defaults to current UTC time."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    before = datetime.now(UTC)
    result = portfolio_router.add_portfolio_position(
        ticker="AAPL",
        shares=5,
        purchase_price=100.0,
        user_id="u1",
        portfolio_name="p1",
    )
    after = datetime.now(UTC)
    assert result["status"] == "success"
    # purchase_date should be between before and after
    purchase_date_str = result["position"]["purchase_date"]
    purchase_date = datetime.fromisoformat(purchase_date_str)
    if purchase_date.tzinfo is None:
        from datetime import timezone

        purchase_date = purchase_date.replace(tzinfo=timezone.utc)
    assert before <= purchase_date <= after
    db.close()


def test_add_with_notes_preserved(monkeypatch):
    """Notes string is stored and returned unchanged."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.add_portfolio_position(
        ticker="AAPL",
        shares=5,
        purchase_price=100.0,
        notes="Long-term hold",
        user_id="u1",
        portfolio_name="p1",
    )
    assert result["status"] == "success"
    assert result["position"]["notes"] == "Long-term hold"
    db.close()


def test_add_ticker_case_insensitive(monkeypatch):
    """Lowercase ticker input is normalized to uppercase."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.add_portfolio_position(
        ticker="aapl",
        shares=5,
        purchase_price=100.0,
        user_id="u1",
        portfolio_name="p1",
    )
    assert result["status"] == "success"
    assert result["position"]["ticker"] == "AAPL"
    db.close()


def test_add_creates_portfolio_automatically(monkeypatch):
    """add_portfolio_position creates the portfolio if it does not exist."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.add_portfolio_position(
        ticker="AAPL",
        shares=5,
        purchase_price=100.0,
        user_id="new_user",
        portfolio_name="My IRA",
    )
    assert result["status"] == "success"
    assert result["portfolio"]["name"] == "My IRA"
    assert result["portfolio"]["user_id"] == "new_user"
    db.close()


# ---------------------------------------------------------------------------
# Group 5: get_my_portfolio — no-price mode (9 tests)
# ---------------------------------------------------------------------------


def test_get_empty_portfolio_returns_empty_status(monkeypatch):
    """Querying a non-existent portfolio returns status='empty'."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.get_my_portfolio(
        user_id="nobody",
        portfolio_name="Ghost",
        include_current_prices=False,
    )
    assert result["status"] == "empty"
    assert result["positions"] == []
    db.close()


def test_get_portfolio_with_one_position_no_prices(monkeypatch):
    """After one BUY, get_my_portfolio returns that position."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    portfolio_router.add_portfolio_position(
        ticker="AAPL",
        shares=10,
        purchase_price=100.0,
        user_id="u1",
        portfolio_name="p1",
    )
    result = portfolio_router.get_my_portfolio(
        user_id="u1",
        portfolio_name="p1",
        include_current_prices=False,
    )
    assert result["status"] == "success"
    assert len(result["positions"]) == 1
    assert result["positions"][0]["ticker"] == "AAPL"
    db.close()


def test_get_portfolio_positions_count(monkeypatch):
    """metrics.number_of_positions matches actual position count."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    for ticker in ["AAPL", "MSFT", "GOOGL"]:
        portfolio_router.add_portfolio_position(
            ticker=ticker,
            shares=5,
            purchase_price=100.0,
            user_id="u1",
            portfolio_name="p1",
        )
    result = portfolio_router.get_my_portfolio(
        user_id="u1",
        portfolio_name="p1",
        include_current_prices=False,
    )
    assert result["metrics"]["number_of_positions"] == 3
    db.close()


def test_get_portfolio_metrics_total_invested(monkeypatch):
    """total_invested = sum of (shares * avg_cost) across all positions."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=10, purchase_price=100.0, user_id="u1", portfolio_name="p1"
    )
    portfolio_router.add_portfolio_position(
        ticker="MSFT", shares=5, purchase_price=200.0, user_id="u1", portfolio_name="p1"
    )
    result = portfolio_router.get_my_portfolio(
        user_id="u1",
        portfolio_name="p1",
        include_current_prices=False,
    )
    # 10*100 + 5*200 = 2000
    assert result["metrics"]["total_invested"] == 2000.0
    db.close()


def test_get_portfolio_ledger_mode_flag(monkeypatch):
    """When transactions exist, response includes calculation_mode='ledger_transactions'."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=5, purchase_price=100.0, user_id="u1", portfolio_name="p1"
    )
    result = portfolio_router.get_my_portfolio(
        user_id="u1",
        portfolio_name="p1",
        include_current_prices=False,
    )
    assert result.get("calculation_mode") == "ledger_transactions"
    db.close()


def test_get_portfolio_avg_cost_from_ledger(monkeypatch):
    """Ledger re-calculates weighted average cost from transactions."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=10, purchase_price=100.0, user_id="u1", portfolio_name="p1"
    )
    portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=10, purchase_price=120.0, user_id="u1", portfolio_name="p1"
    )
    result = portfolio_router.get_my_portfolio(
        user_id="u1",
        portfolio_name="p1",
        include_current_prices=False,
    )
    pos = result["positions"][0]
    assert pos["average_cost_basis"] == 110.0
    db.close()


def test_get_portfolio_realized_pnl_after_sell(monkeypatch):
    """Realized P&L is computed after a partial sell via portfolio_record_trade."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))

    # Buy 10 @ 100
    portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=10, purchase_price=100.0, user_id="u1", portfolio_name="p1"
    )
    # Sell 5 @ 120 via record_trade (supports market price)
    portfolio_router.portfolio_record_trade(
        ticker="AAPL",
        side="SELL",
        quantity=5,
        price=120.0,
        user_id="u1",
        portfolio_name="p1",
    )
    result = portfolio_router.get_my_portfolio(
        user_id="u1",
        portfolio_name="p1",
        include_current_prices=False,
    )
    # realized = 5*(120-100) = 100
    assert result["metrics"]["total_realized_gain_loss"] == 100.0
    db.close()


def test_get_portfolio_first_purchase_date_in_ledger_mode(monkeypatch):
    """first_purchase_date is the date of the first BUY transaction."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    portfolio_router.add_portfolio_position(
        ticker="AAPL",
        shares=5,
        purchase_price=100.0,
        purchase_date="2024-01-01",
        user_id="u1",
        portfolio_name="p1",
    )
    portfolio_router.add_portfolio_position(
        ticker="AAPL",
        shares=5,
        purchase_price=120.0,
        purchase_date="2024-06-01",
        user_id="u1",
        portfolio_name="p1",
    )
    result = portfolio_router.get_my_portfolio(
        user_id="u1",
        portfolio_name="p1",
        include_current_prices=False,
    )
    pos = result["positions"][0]
    assert "2024-01-01" in pos["purchase_date"]
    db.close()


def test_get_portfolio_after_full_sell_shows_empty(monkeypatch):
    """After selling all shares, portfolio shows no open positions."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=5, purchase_price=100.0, user_id="u1", portfolio_name="p1"
    )
    portfolio_router.remove_portfolio_position(
        ticker="AAPL", shares=None, user_id="u1", portfolio_name="p1"
    )
    result = portfolio_router.get_my_portfolio(
        user_id="u1",
        portfolio_name="p1",
        include_current_prices=False,
    )
    assert result["positions"] == []
    db.close()


# ---------------------------------------------------------------------------
# Group 6: get_my_portfolio — mock price mode (5 tests)
# ---------------------------------------------------------------------------


def test_get_portfolio_with_mocked_price_unrealized_pnl(monkeypatch):
    """Mocked current price produces correct unrealized P&L."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    monkeypatch.setattr(portfolio_router, "stock_provider", _mock_stock_provider(150.0))

    portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=10, purchase_price=100.0, user_id="u1", portfolio_name="p1"
    )
    result = portfolio_router.get_my_portfolio(
        user_id="u1",
        portfolio_name="p1",
        include_current_prices=True,
    )
    pos = result["positions"][0]
    # 10*(150-100) = 500
    assert pos["unrealized_gain_loss"] == 500.0
    db.close()


def test_get_portfolio_pnl_positive_above_cost(monkeypatch):
    """Current price above cost: unrealized P&L is positive."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    monkeypatch.setattr(portfolio_router, "stock_provider", _mock_stock_provider(120.0))

    portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=5, purchase_price=100.0, user_id="u1", portfolio_name="p1"
    )
    result = portfolio_router.get_my_portfolio(
        user_id="u1",
        portfolio_name="p1",
        include_current_prices=True,
    )
    assert result["positions"][0]["unrealized_gain_loss"] > 0
    db.close()


def test_get_portfolio_pnl_negative_below_cost(monkeypatch):
    """Current price below cost: unrealized P&L is negative."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    monkeypatch.setattr(portfolio_router, "stock_provider", _mock_stock_provider(80.0))

    portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=5, purchase_price=100.0, user_id="u1", portfolio_name="p1"
    )
    result = portfolio_router.get_my_portfolio(
        user_id="u1",
        portfolio_name="p1",
        include_current_prices=True,
    )
    assert result["positions"][0]["unrealized_gain_loss"] < 0
    db.close()


def test_get_portfolio_pnl_percent_calculation(monkeypatch):
    """unrealized_gain_loss_percent = unrealized / total_cost * 100."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    monkeypatch.setattr(portfolio_router, "stock_provider", _mock_stock_provider(110.0))

    portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=10, purchase_price=100.0, user_id="u1", portfolio_name="p1"
    )
    result = portfolio_router.get_my_portfolio(
        user_id="u1",
        portfolio_name="p1",
        include_current_prices=True,
    )
    pos = result["positions"][0]
    # gain = 100, total_cost = 1000, pct = 10%
    assert abs(pos["unrealized_gain_loss_percent"] - 10.0) < 0.01
    db.close()


def test_get_portfolio_price_fetch_failure_graceful(monkeypatch):
    """If stock_provider raises an exception, portfolio still returns without P&L."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))

    failing_provider = MagicMock()
    failing_provider.get_stock_data.side_effect = Exception("Network error")
    monkeypatch.setattr(portfolio_router, "stock_provider", failing_provider)

    portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=5, purchase_price=100.0, user_id="u1", portfolio_name="p1"
    )
    result = portfolio_router.get_my_portfolio(
        user_id="u1",
        portfolio_name="p1",
        include_current_prices=True,
    )
    # Should succeed gracefully without P&L data
    assert result["status"] == "success"
    assert len(result["positions"]) == 1
    assert "unrealized_gain_loss" not in result["positions"][0]
    db.close()


# ---------------------------------------------------------------------------
# Group 7: remove_portfolio_position (9 tests)
# ---------------------------------------------------------------------------


def test_remove_empty_ticker_returns_error(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.remove_portfolio_position(ticker="")
    assert result["status"] == "error"
    db.close()


def test_remove_nonexistent_portfolio_returns_error(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.remove_portfolio_position(
        ticker="AAPL", user_id="ghost", portfolio_name="nothing"
    )
    assert result["status"] == "error"
    assert "not found" in result["error"].lower()
    db.close()


def test_remove_nonexistent_ticker_returns_error(monkeypatch):
    """Portfolio exists but ticker not in it."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    portfolio_router.add_portfolio_position(
        ticker="MSFT", shares=5, purchase_price=200.0, user_id="u1", portfolio_name="p1"
    )
    result = portfolio_router.remove_portfolio_position(
        ticker="AAPL", user_id="u1", portfolio_name="p1"
    )
    assert result["status"] == "error"
    db.close()


def test_remove_partial_position_success(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=10, purchase_price=100.0, user_id="u1", portfolio_name="p1"
    )
    result = portfolio_router.remove_portfolio_position(
        ticker="AAPL", shares=4, user_id="u1", portfolio_name="p1"
    )
    assert result["status"] == "success"
    assert result["position_fully_closed"] is False
    assert result["remaining_position"]["shares"] == 6.0
    db.close()


def test_remove_full_position_by_none(monkeypatch):
    """shares=None removes entire position."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=5, purchase_price=100.0, user_id="u1", portfolio_name="p1"
    )
    result = portfolio_router.remove_portfolio_position(
        ticker="AAPL", shares=None, user_id="u1", portfolio_name="p1"
    )
    assert result["status"] == "success"
    assert result["position_fully_closed"] is True
    db.close()


def test_remove_full_position_by_exact_count(monkeypatch):
    """Specifying exact share count also closes position."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=5, purchase_price=100.0, user_id="u1", portfolio_name="p1"
    )
    result = portfolio_router.remove_portfolio_position(
        ticker="AAPL", shares=5, user_id="u1", portfolio_name="p1"
    )
    assert result["status"] == "success"
    assert result["position_fully_closed"] is True
    db.close()


def test_remove_records_sell_transaction(monkeypatch):
    """remove_portfolio_position writes a SELL transaction to the ledger."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=10, purchase_price=100.0, user_id="u1", portfolio_name="p1"
    )
    portfolio_router.remove_portfolio_position(
        ticker="AAPL", shares=4, user_id="u1", portfolio_name="p1"
    )
    sell_txns = db.query(PortfolioTransaction).filter_by(ticker="AAPL", side="SELL").all()
    assert len(sell_txns) == 1
    assert Decimal(str(sell_txns[0].quantity)) == Decimal("4")
    db.close()


def test_remove_uses_cost_basis_as_price_bug(monkeypatch):
    """BUG #2: SELL transaction price == avg_cost_basis (not market price).

    remove_portfolio_position does not accept a market_price parameter — it
    hardcodes `price=position_db.average_cost_basis`.  This means the SELL
    transaction records cost-basis price, making realized P&L always 0 in the
    ledger when using remove_portfolio_position.

    Fix: portfolio.py line 1145 should use a real market price parameter.
    """
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=10, purchase_price=100.0, user_id="u1", portfolio_name="p1"
    )
    portfolio_router.remove_portfolio_position(
        ticker="AAPL", shares=5, user_id="u1", portfolio_name="p1"
    )
    sell_txn = (
        db.query(PortfolioTransaction).filter_by(ticker="AAPL", side="SELL").first()
    )
    # BUG: sell price == avg_cost (100) instead of market price
    assert float(sell_txn.price) == 100.0
    # After fix: the sell price would reflect actual market price
    db.close()


def test_remove_oversell_clamps_to_position_size(monkeypatch):
    """remove_portfolio_position silently caps shares to position size via min().

    When shares > position.shares, the router uses min(shares, position.shares),
    so selling 10 when only 5 are held sells all 5 (position_fully_closed=True)
    rather than returning an error.
    """
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=5, purchase_price=100.0, user_id="u1", portfolio_name="p1"
    )
    result = portfolio_router.remove_portfolio_position(
        ticker="AAPL", shares=10, user_id="u1", portfolio_name="p1"
    )
    # Silently closes the full position rather than erroring
    assert result["status"] == "success"
    assert result["position_fully_closed"] is True
    db.close()


# ---------------------------------------------------------------------------
# Group 8: portfolio_record_trade — validation (6 tests)
# ---------------------------------------------------------------------------


def test_record_trade_invalid_side_returns_error(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.portfolio_record_trade(
        ticker="AAPL", side="HOLD", quantity=5, price=100.0
    )
    assert result["status"] == "error"
    assert "BUY or SELL" in result["error"]
    db.close()


def test_record_trade_zero_quantity_returns_error(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.portfolio_record_trade(
        ticker="AAPL", side="BUY", quantity=0, price=100.0
    )
    assert result["status"] == "error"
    db.close()


def test_record_trade_zero_price_returns_error(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.portfolio_record_trade(
        ticker="AAPL", side="BUY", quantity=5, price=0.0
    )
    assert result["status"] == "error"
    db.close()


def test_record_trade_negative_fee_returns_error(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.portfolio_record_trade(
        ticker="AAPL", side="BUY", quantity=5, price=100.0, fee=-1.0
    )
    assert result["status"] == "error"
    db.close()


def test_record_trade_invalid_lot_method_returns_error(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.portfolio_record_trade(
        ticker="AAPL", side="BUY", quantity=5, price=100.0, lot_method="RANDOM"
    )
    assert result["status"] == "error"
    db.close()


def test_record_trade_invalid_trade_time_returns_error(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.portfolio_record_trade(
        ticker="AAPL",
        side="BUY",
        quantity=5,
        price=100.0,
        trade_time="not-a-date",
    )
    assert result["status"] == "error"
    db.close()


# ---------------------------------------------------------------------------
# Group 9: portfolio_record_trade — normal flows (7 tests)
# ---------------------------------------------------------------------------


def test_record_trade_buy_returns_success(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.portfolio_record_trade(
        ticker="AAPL",
        side="BUY",
        quantity=10,
        price=100.0,
        user_id="u1",
        portfolio_name="p1",
    )
    assert result["status"] == "success"
    assert result["transaction"]["side"] == "BUY"
    db.close()


def test_record_trade_sell_returns_success(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    portfolio_router.portfolio_record_trade(
        ticker="AAPL", side="BUY", quantity=10, price=100.0, user_id="u1", portfolio_name="p1"
    )
    result = portfolio_router.portfolio_record_trade(
        ticker="AAPL", side="SELL", quantity=5, price=120.0, user_id="u1", portfolio_name="p1"
    )
    assert result["status"] == "success"
    assert result["transaction"]["side"] == "SELL"
    db.close()


def test_record_trade_sell_oversell_returns_error(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    portfolio_router.portfolio_record_trade(
        ticker="AAPL", side="BUY", quantity=3, price=100.0, user_id="u1", portfolio_name="p1"
    )
    result = portfolio_router.portfolio_record_trade(
        ticker="AAPL", side="SELL", quantity=5, price=100.0, user_id="u1", portfolio_name="p1"
    )
    assert result["status"] == "error"
    assert "more shares" in result["error"].lower()
    db.close()


def test_record_trade_sell_no_position_returns_error(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.portfolio_record_trade(
        ticker="AAPL", side="SELL", quantity=5, price=100.0, user_id="u1", portfolio_name="p1"
    )
    assert result["status"] == "error"
    db.close()


def test_record_trade_with_fee_stored_in_transaction(monkeypatch):
    """fee value is persisted in the PortfolioTransaction row."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    portfolio_router.portfolio_record_trade(
        ticker="AAPL",
        side="BUY",
        quantity=10,
        price=100.0,
        fee=9.99,
        user_id="u1",
        portfolio_name="p1",
    )
    txn = db.query(PortfolioTransaction).filter_by(ticker="AAPL").first()
    assert abs(float(txn.fee) - 9.99) < 0.001
    db.close()


def test_record_trade_lot_method_stored_but_ignored_bug(monkeypatch):
    """BUG #4: lot_method is stored in transactions but never used in calculation.

    Whether FIFO, LIFO, or AVG is specified, cost-basis computation always
    uses the same average-cost method.  The lot_method field is a placeholder
    for future functionality.

    This test verifies the field is stored correctly while confirming that
    different lot_methods produce identical position values (proof of ignored logic).
    """
    db_fifo = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db_fifo))
    portfolio_router.portfolio_record_trade(
        ticker="AAPL", side="BUY", quantity=10, price=100.0, lot_method="FIFO",
        user_id="u1", portfolio_name="p1"
    )
    portfolio_router.portfolio_record_trade(
        ticker="AAPL", side="BUY", quantity=10, price=120.0, lot_method="FIFO",
        user_id="u1", portfolio_name="p1"
    )
    portfolio_router.portfolio_record_trade(
        ticker="AAPL", side="SELL", quantity=5, price=130.0, lot_method="FIFO",
        user_id="u1", portfolio_name="p1"
    )

    db_lifo = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db_lifo))
    portfolio_router.portfolio_record_trade(
        ticker="AAPL", side="BUY", quantity=10, price=100.0, lot_method="LIFO",
        user_id="u1", portfolio_name="p1"
    )
    portfolio_router.portfolio_record_trade(
        ticker="AAPL", side="BUY", quantity=10, price=120.0, lot_method="LIFO",
        user_id="u1", portfolio_name="p1"
    )
    portfolio_router.portfolio_record_trade(
        ticker="AAPL", side="SELL", quantity=5, price=130.0, lot_method="LIFO",
        user_id="u1", portfolio_name="p1"
    )

    # Verify lot_method was stored
    fifo_txn = db_fifo.query(PortfolioTransaction).filter_by(side="BUY").first()
    assert fifo_txn.lot_method == "FIFO"
    lifo_txn = db_lifo.query(PortfolioTransaction).filter_by(side="BUY").first()
    assert lifo_txn.lot_method == "LIFO"

    # BUG: both produce identical avg_cost (FIFO/LIFO distinction ignored)
    port_fifo = db_fifo.query(UserPortfolio).filter_by(user_id="u1").first()
    pos_fifo = db_fifo.query(PortfolioPosition).filter_by(portfolio_id=port_fifo.id).first()
    port_lifo = db_lifo.query(UserPortfolio).filter_by(user_id="u1").first()
    pos_lifo = db_lifo.query(PortfolioPosition).filter_by(portfolio_id=port_lifo.id).first()

    assert float(pos_fifo.average_cost_basis) == float(pos_lifo.average_cost_basis)

    db_fifo.close()
    db_lifo.close()


def test_record_trade_creates_portfolio_if_not_exists(monkeypatch):
    """portfolio_record_trade creates the portfolio on first BUY."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.portfolio_record_trade(
        ticker="AAPL",
        side="BUY",
        quantity=5,
        price=100.0,
        user_id="brand_new",
        portfolio_name="Fresh Start",
    )
    assert result["status"] == "success"
    assert result["portfolio"]["name"] == "Fresh Start"
    db.close()


# ---------------------------------------------------------------------------
# Group 10: clear_my_portfolio (4 tests)
# ---------------------------------------------------------------------------


def test_clear_without_confirm_returns_error(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.clear_my_portfolio(confirm=False)
    assert result["status"] == "error"
    db.close()


def test_clear_nonexistent_portfolio_returns_error(monkeypatch):
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.clear_my_portfolio(
        user_id="ghost", portfolio_name="nothing", confirm=True
    )
    assert result["status"] == "error"
    db.close()


def test_clear_with_positions_deletes_all(monkeypatch):
    """All positions are removed after clear."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    for ticker in ["AAPL", "MSFT", "GOOGL"]:
        portfolio_router.add_portfolio_position(
            ticker=ticker, shares=5, purchase_price=100.0, user_id="u1", portfolio_name="p1"
        )
    result = portfolio_router.clear_my_portfolio(
        user_id="u1", portfolio_name="p1", confirm=True
    )
    assert result["status"] == "success"
    assert result["positions_cleared"] == 3
    db.close()


def test_clear_empty_portfolio_returns_success(monkeypatch):
    """Clearing an already-empty portfolio succeeds gracefully."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    # Create a portfolio with no positions by using record_trade then clearing
    portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=5, purchase_price=100.0, user_id="u1", portfolio_name="p1"
    )
    portfolio_router.remove_portfolio_position(
        ticker="AAPL", shares=None, user_id="u1", portfolio_name="p1"
    )
    result = portfolio_router.clear_my_portfolio(
        user_id="u1", portfolio_name="p1", confirm=True
    )
    assert result["status"] == "success"
    db.close()


# ---------------------------------------------------------------------------
# Group 11: Math precision (5 tests)
# ---------------------------------------------------------------------------


def test_weighted_avg_decimal_no_float_error():
    """Decimal arithmetic in ledger produces exact result, no float rounding error."""
    db = _make_session()
    svc = PortfolioLedgerService(db)

    # Use values that trip up float arithmetic
    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL",
            side="BUY",
            quantity=Decimal("3"),
            price=Decimal("33.33"),
        ),
    )
    db.commit()
    svc.record_trade(
        user_id="u1",
        portfolio_name="p1",
        payload=RecordTradeInput(
            ticker="AAPL",
            side="BUY",
            quantity=Decimal("3"),
            price=Decimal("66.67"),
        ),
    )
    db.commit()

    portfolio = db.query(UserPortfolio).filter_by(user_id="u1", name="p1").first()
    position = (
        db.query(PortfolioPosition)
        .filter_by(portfolio_id=portfolio.id, ticker="AAPL")
        .first()
    )
    # Should use Decimal math: (3*33.33 + 3*66.67) / 6 = 50.00
    avg = Decimal(str(position.average_cost_basis))
    assert abs(avg - Decimal("50.00")) < Decimal("0.01")
    db.close()


def test_cost_basis_with_cents_precision(monkeypatch):
    """Cost basis with cents (e.g., $150.75) is stored precisely."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    result = portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=4, purchase_price=150.75, user_id="u1", portfolio_name="p1"
    )
    assert result["status"] == "success"
    assert abs(result["position"]["average_cost_basis"] - 150.75) < 0.001
    db.close()


def test_realized_pnl_math_correct(monkeypatch):
    """buy 10@100, sell 5@120 → realized = 5*(120-100) = 100."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))

    portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=10, purchase_price=100.0, user_id="u1", portfolio_name="p1"
    )
    portfolio_router.portfolio_record_trade(
        ticker="AAPL",
        side="SELL",
        quantity=5,
        price=120.0,
        user_id="u1",
        portfolio_name="p1",
    )
    result = portfolio_router.get_my_portfolio(
        user_id="u1",
        portfolio_name="p1",
        include_current_prices=False,
    )
    assert result["metrics"]["total_realized_gain_loss"] == 100.0
    db.close()


def test_total_invested_sums_all_fees(monkeypatch):
    """total_invested in ledger mode = sum of all BUY costs (qty*price+fee)."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))

    # Use portfolio_record_trade to pass fees
    portfolio_router.portfolio_record_trade(
        ticker="AAPL", side="BUY", quantity=10, price=100.0, fee=5.0,
        user_id="u1", portfolio_name="p1"
    )
    portfolio_router.portfolio_record_trade(
        ticker="AAPL", side="BUY", quantity=5, price=110.0, fee=3.0,
        user_id="u1", portfolio_name="p1"
    )
    result = portfolio_router.get_my_portfolio(
        user_id="u1",
        portfolio_name="p1",
        include_current_prices=False,
    )
    # (10*100+5) + (5*110+3) = 1005 + 553 = 1558
    assert abs(result["metrics"]["total_invested"] - 1558.0) < 0.01
    db.close()


def test_pnl_percent_formula(monkeypatch):
    """total_return_percent = total_unrealized / total_invested * 100."""
    db = _make_session()
    monkeypatch.setattr(portfolio_router, "get_db", _override_get_db(db))
    monkeypatch.setattr(portfolio_router, "stock_provider", _mock_stock_provider(110.0))

    portfolio_router.add_portfolio_position(
        ticker="AAPL", shares=10, purchase_price=100.0, user_id="u1", portfolio_name="p1"
    )
    result = portfolio_router.get_my_portfolio(
        user_id="u1", portfolio_name="p1", include_current_prices=True
    )
    # unrealized = 10*(110-100) = 100, invested = 1000, pct = 10%
    assert abs(result["metrics"]["total_return_percent"] - 10.0) < 0.1
    db.close()
