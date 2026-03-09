"""Portfolio Ledger schema tests.

Validates SQLAlchemy model definitions for:
  - PortfolioTransaction
  - PortfolioCashTransaction
  - PortfolioValuationSnapshot

All tests use SQLite in-memory databases — no Docker required.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker

from maverick_mcp.data.models import (
    PortfolioCashTransaction,
    PortfolioTransaction,
    PortfolioValuationSnapshot,
    UserPortfolio,
)
from maverick_mcp.database.base import Base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session() -> Session:
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)()


def _make_portfolio(db: Session, user_id: str = "u1", name: str = "p1") -> UserPortfolio:
    p = UserPortfolio(user_id=user_id, name=name)
    db.add(p)
    db.flush()
    return p


# ---------------------------------------------------------------------------
# PortfolioTransaction — 9 tests
# ---------------------------------------------------------------------------


def test_transaction_tablename():
    assert PortfolioTransaction.__tablename__ == "mcp_portfolio_transactions"


def test_transaction_create_buy_record():
    """All BUY fields are persisted and retrieved correctly."""
    db = _make_session()
    portfolio = _make_portfolio(db)
    trade_time = datetime(2024, 3, 15, 12, 0, 0, tzinfo=UTC)

    txn = PortfolioTransaction(
        portfolio_id=portfolio.id,
        ticker="AAPL",
        side="BUY",
        quantity=Decimal("10"),
        price=Decimal("150.00"),
        fee=Decimal("2.50"),
        trade_time=trade_time,
        lot_method="FIFO",
        notes="Initial purchase",
    )
    db.add(txn)
    db.commit()

    saved = db.query(PortfolioTransaction).first()
    assert saved.ticker == "AAPL"
    assert saved.side == "BUY"
    assert Decimal(str(saved.quantity)) == Decimal("10")
    assert Decimal(str(saved.price)) == Decimal("150.00")
    assert Decimal(str(saved.fee)) == Decimal("2.50")
    assert saved.lot_method == "FIFO"
    assert saved.notes == "Initial purchase"
    db.close()


def test_transaction_create_sell_record():
    db = _make_session()
    portfolio = _make_portfolio(db)

    txn = PortfolioTransaction(
        portfolio_id=portfolio.id,
        ticker="MSFT",
        side="SELL",
        quantity=Decimal("5"),
        price=Decimal("300.00"),
        fee=Decimal("0"),
        trade_time=datetime.now(UTC),
        lot_method="AVG",
    )
    db.add(txn)
    db.commit()

    saved = db.query(PortfolioTransaction).filter_by(ticker="MSFT").first()
    assert saved.side == "SELL"
    db.close()


def test_transaction_cascade_delete_with_portfolio():
    """Deleting portfolio cascades to transactions."""
    db = _make_session()
    portfolio = _make_portfolio(db)

    db.add(
        PortfolioTransaction(
            portfolio_id=portfolio.id,
            ticker="AAPL",
            side="BUY",
            quantity=Decimal("5"),
            price=Decimal("100"),
            fee=Decimal("0"),
            trade_time=datetime.now(UTC),
            lot_method="FIFO",
        )
    )
    db.commit()

    db.delete(portfolio)
    db.commit()

    count = db.query(PortfolioTransaction).count()
    assert count == 0
    db.close()


def test_transaction_uuid_primary_key():
    """Transaction id is a UUID."""
    db = _make_session()
    portfolio = _make_portfolio(db)

    txn = PortfolioTransaction(
        portfolio_id=portfolio.id,
        ticker="AAPL",
        side="BUY",
        quantity=Decimal("1"),
        price=Decimal("100"),
        fee=Decimal("0"),
        trade_time=datetime.now(UTC),
        lot_method="FIFO",
    )
    db.add(txn)
    db.commit()

    saved = db.query(PortfolioTransaction).first()
    # id should be a UUID (either uuid.UUID or its string representation)
    try:
        uuid.UUID(str(saved.id))
    except (ValueError, AttributeError):
        pytest.fail(f"Transaction id is not a valid UUID: {saved.id!r}")
    db.close()


def test_transaction_quantity_high_precision():
    """Numeric(20,8) supports 8 decimal places for fractional shares."""
    db = _make_session()
    portfolio = _make_portfolio(db)

    qty = Decimal("0.12345678")
    txn = PortfolioTransaction(
        portfolio_id=portfolio.id,
        ticker="AAPL",
        side="BUY",
        quantity=qty,
        price=Decimal("100"),
        fee=Decimal("0"),
        trade_time=datetime.now(UTC),
        lot_method="FIFO",
    )
    db.add(txn)
    db.commit()

    saved = db.query(PortfolioTransaction).first()
    assert abs(Decimal(str(saved.quantity)) - qty) < Decimal("0.000001")
    db.close()


def test_transaction_default_fee_is_zero():
    """fee column defaults to 0 when not specified."""
    db = _make_session()
    portfolio = _make_portfolio(db)

    txn = PortfolioTransaction(
        portfolio_id=portfolio.id,
        ticker="AAPL",
        side="BUY",
        quantity=Decimal("5"),
        price=Decimal("100"),
        trade_time=datetime.now(UTC),
        lot_method="FIFO",
    )
    db.add(txn)
    db.commit()

    saved = db.query(PortfolioTransaction).first()
    assert Decimal(str(saved.fee)) == Decimal("0")
    db.close()


def test_transaction_default_lot_method_is_fifo():
    """lot_method column defaults to 'FIFO'."""
    db = _make_session()
    portfolio = _make_portfolio(db)

    txn = PortfolioTransaction(
        portfolio_id=portfolio.id,
        ticker="AAPL",
        side="BUY",
        quantity=Decimal("5"),
        price=Decimal("100"),
        fee=Decimal("0"),
        trade_time=datetime.now(UTC),
    )
    db.add(txn)
    db.commit()

    saved = db.query(PortfolioTransaction).first()
    assert saved.lot_method == "FIFO"
    db.close()


def test_transaction_portfolio_relationship():
    """PortfolioTransaction.portfolio back-references the owning UserPortfolio."""
    db = _make_session()
    portfolio = _make_portfolio(db)

    txn = PortfolioTransaction(
        portfolio_id=portfolio.id,
        ticker="AAPL",
        side="BUY",
        quantity=Decimal("5"),
        price=Decimal("100"),
        fee=Decimal("0"),
        trade_time=datetime.now(UTC),
        lot_method="FIFO",
    )
    db.add(txn)
    db.commit()
    db.refresh(txn)

    assert txn.portfolio.id == portfolio.id
    assert txn.portfolio.name == "p1"
    db.close()


# ---------------------------------------------------------------------------
# PortfolioCashTransaction — 7 tests
# ---------------------------------------------------------------------------


def test_cash_transaction_tablename():
    assert PortfolioCashTransaction.__tablename__ == "mcp_portfolio_cash_transactions"


def test_cash_transaction_deposit_type():
    """DEPOSIT cash transaction persists correctly."""
    db = _make_session()
    portfolio = _make_portfolio(db)

    ct = PortfolioCashTransaction(
        portfolio_id=portfolio.id,
        type="DEPOSIT",
        amount=Decimal("10000.00"),
        event_time=datetime.now(UTC),
    )
    db.add(ct)
    db.commit()

    saved = db.query(PortfolioCashTransaction).first()
    assert saved.type == "DEPOSIT"
    assert Decimal(str(saved.amount)) == Decimal("10000.00")
    db.close()


def test_cash_transaction_default_currency_usd():
    """currency column defaults to 'USD'."""
    db = _make_session()
    portfolio = _make_portfolio(db)

    ct = PortfolioCashTransaction(
        portfolio_id=portfolio.id,
        type="DEPOSIT",
        amount=Decimal("500"),
        event_time=datetime.now(UTC),
    )
    db.add(ct)
    db.commit()

    saved = db.query(PortfolioCashTransaction).first()
    assert saved.currency == "USD"
    db.close()


def test_cash_transaction_negative_amount_allowed():
    """Negative amount is allowed (e.g., WITHDRAW or FEE)."""
    db = _make_session()
    portfolio = _make_portfolio(db)

    ct = PortfolioCashTransaction(
        portfolio_id=portfolio.id,
        type="FEE",
        amount=Decimal("-25.00"),
        event_time=datetime.now(UTC),
    )
    db.add(ct)
    db.commit()

    saved = db.query(PortfolioCashTransaction).first()
    assert Decimal(str(saved.amount)) == Decimal("-25.00")
    db.close()


def test_cash_transaction_cascade_delete():
    """Deleting portfolio cascades to cash transactions."""
    db = _make_session()
    portfolio = _make_portfolio(db)

    db.add(
        PortfolioCashTransaction(
            portfolio_id=portfolio.id,
            type="DEPOSIT",
            amount=Decimal("1000"),
            event_time=datetime.now(UTC),
        )
    )
    db.commit()

    db.delete(portfolio)
    db.commit()

    count = db.query(PortfolioCashTransaction).count()
    assert count == 0
    db.close()


def test_cash_transaction_relationship_to_portfolio():
    """PortfolioCashTransaction.portfolio back-references the owning portfolio."""
    db = _make_session()
    portfolio = _make_portfolio(db)

    ct = PortfolioCashTransaction(
        portfolio_id=portfolio.id,
        type="DIVIDEND",
        amount=Decimal("50"),
        event_time=datetime.now(UTC),
    )
    db.add(ct)
    db.commit()
    db.refresh(ct)

    assert ct.portfolio.id == portfolio.id
    db.close()


def test_cash_transaction_all_type_values_storable():
    """All documented type values (DEPOSIT/WITHDRAW/DIVIDEND/FEE/INTEREST) can be stored."""
    db = _make_session()
    portfolio = _make_portfolio(db)

    for tx_type in ["DEPOSIT", "WITHDRAW", "DIVIDEND", "FEE", "INTEREST"]:
        db.add(
            PortfolioCashTransaction(
                portfolio_id=portfolio.id,
                type=tx_type,
                amount=Decimal("10"),
                event_time=datetime.now(UTC),
            )
        )
    db.commit()

    types_stored = {
        row.type for row in db.query(PortfolioCashTransaction).all()
    }
    assert types_stored == {"DEPOSIT", "WITHDRAW", "DIVIDEND", "FEE", "INTEREST"}
    db.close()


# ---------------------------------------------------------------------------
# PortfolioValuationSnapshot — 7 tests
# ---------------------------------------------------------------------------


def test_valuation_snapshot_tablename():
    assert PortfolioValuationSnapshot.__tablename__ == "mcp_portfolio_valuation_snapshots"


def test_valuation_snapshot_create_record():
    """All financial fields are stored and retrieved correctly."""
    db = _make_session()
    portfolio = _make_portfolio(db)
    as_of = datetime(2024, 12, 31, 23, 59, tzinfo=UTC)

    snap = PortfolioValuationSnapshot(
        portfolio_id=portfolio.id,
        as_of=as_of,
        invested=Decimal("10000.00"),
        market_value=Decimal("11500.00"),
        cash_value=Decimal("500.00"),
        total_equity=Decimal("12000.00"),
        unrealized_pnl=Decimal("1500.00"),
        realized_pnl=Decimal("200.00"),
    )
    db.add(snap)
    db.commit()

    saved = db.query(PortfolioValuationSnapshot).first()
    assert Decimal(str(saved.invested)) == Decimal("10000.00")
    assert Decimal(str(saved.market_value)) == Decimal("11500.00")
    assert Decimal(str(saved.cash_value)) == Decimal("500.00")
    assert Decimal(str(saved.total_equity)) == Decimal("12000.00")
    assert Decimal(str(saved.unrealized_pnl)) == Decimal("1500.00")
    assert Decimal(str(saved.realized_pnl)) == Decimal("200.00")
    db.close()


def test_valuation_snapshot_unique_constraint_portfolio_asof():
    """Duplicate (portfolio_id, as_of) raises an integrity error."""
    from sqlalchemy.exc import IntegrityError

    db = _make_session()
    portfolio = _make_portfolio(db)
    as_of = datetime(2024, 1, 1, tzinfo=UTC)

    def _snap():
        return PortfolioValuationSnapshot(
            portfolio_id=portfolio.id,
            as_of=as_of,
            invested=Decimal("1000"),
            market_value=Decimal("1000"),
            cash_value=Decimal("0"),
            total_equity=Decimal("1000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
        )

    db.add(_snap())
    db.commit()

    db.add(_snap())
    with pytest.raises(IntegrityError):
        db.commit()
    db.close()


def test_valuation_snapshot_cascade_delete():
    """Deleting portfolio cascades to valuation snapshots."""
    db = _make_session()
    portfolio = _make_portfolio(db)

    db.add(
        PortfolioValuationSnapshot(
            portfolio_id=portfolio.id,
            as_of=datetime.now(UTC),
            invested=Decimal("1000"),
            market_value=Decimal("1000"),
            cash_value=Decimal("0"),
            total_equity=Decimal("1000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
        )
    )
    db.commit()

    db.delete(portfolio)
    db.commit()

    count = db.query(PortfolioValuationSnapshot).count()
    assert count == 0
    db.close()


def test_valuation_snapshot_meta_json_field():
    """meta JSON field stores and retrieves arbitrary dict data."""
    db = _make_session()
    portfolio = _make_portfolio(db)

    meta_data = {"source": "nightly_job", "version": 2}
    snap = PortfolioValuationSnapshot(
        portfolio_id=portfolio.id,
        as_of=datetime.now(UTC),
        invested=Decimal("1000"),
        market_value=Decimal("1000"),
        cash_value=Decimal("0"),
        total_equity=Decimal("1000"),
        unrealized_pnl=Decimal("0"),
        realized_pnl=Decimal("0"),
        meta=meta_data,
    )
    db.add(snap)
    db.commit()

    saved = db.query(PortfolioValuationSnapshot).first()
    assert saved.meta["source"] == "nightly_job"
    assert saved.meta["version"] == 2
    db.close()


def test_valuation_snapshot_no_updated_at():
    """PortfolioValuationSnapshot has no updated_at column (append-only log)."""
    columns = {col.name for col in PortfolioValuationSnapshot.__table__.columns}
    assert "updated_at" not in columns


def test_valuation_snapshot_created_at_auto_set():
    """created_at is automatically set on insert."""
    db = _make_session()
    portfolio = _make_portfolio(db)

    before = datetime.now(UTC)
    snap = PortfolioValuationSnapshot(
        portfolio_id=portfolio.id,
        as_of=datetime.now(UTC),
        invested=Decimal("1000"),
        market_value=Decimal("1000"),
        cash_value=Decimal("0"),
        total_equity=Decimal("1000"),
        unrealized_pnl=Decimal("0"),
        realized_pnl=Decimal("0"),
    )
    db.add(snap)
    db.commit()
    after = datetime.now(UTC)

    saved = db.query(PortfolioValuationSnapshot).first()
    # created_at should be set (not None)
    assert saved.created_at is not None
    db.close()
