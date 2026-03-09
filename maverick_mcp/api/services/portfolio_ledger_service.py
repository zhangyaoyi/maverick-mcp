"""Portfolio ledger service for transaction-centric portfolio updates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy.orm import Session

from maverick_mcp.data.models import PortfolioPosition, PortfolioTransaction, UserPortfolio


@dataclass
class RecordTradeInput:
    ticker: str
    side: str
    quantity: Decimal
    price: Decimal
    fee: Decimal = Decimal("0")
    trade_time: datetime | None = None
    lot_method: str = "FIFO"
    notes: str | None = None


class PortfolioLedgerService:
    """Service that records trade ledger entries and keeps legacy positions in sync."""

    def __init__(self, db: Session):
        self.db = db

    def record_trade(
        self,
        *,
        user_id: str,
        portfolio_name: str,
        payload: RecordTradeInput,
    ) -> PortfolioTransaction:
        portfolio = (
            self.db.query(UserPortfolio)
            .filter_by(user_id=user_id, name=portfolio_name)
            .first()
        )
        if not portfolio:
            portfolio = UserPortfolio(user_id=user_id, name=portfolio_name)
            self.db.add(portfolio)
            self.db.flush()

        trade_time = payload.trade_time or datetime.now(UTC)
        side = payload.side.upper()
        txn = PortfolioTransaction(
            portfolio_id=portfolio.id,
            ticker=payload.ticker.upper(),
            side=side,
            quantity=payload.quantity,
            price=payload.price,
            fee=payload.fee,
            trade_time=trade_time,
            lot_method=payload.lot_method.upper(),
            notes=payload.notes,
        )
        self.db.add(txn)

        pos = (
            self.db.query(PortfolioPosition)
            .filter_by(portfolio_id=portfolio.id, ticker=payload.ticker.upper())
            .first()
        )

        if side == "BUY":
            total_cost = payload.quantity * payload.price + payload.fee
            if pos:
                old_total = pos.shares * pos.average_cost_basis
                new_total = old_total + total_cost
                new_shares = pos.shares + payload.quantity
                pos.shares = new_shares
                pos.average_cost_basis = new_total / new_shares
                pos.total_cost = new_total
                pos.purchase_date = trade_time
            else:
                self.db.add(
                    PortfolioPosition(
                        portfolio_id=portfolio.id,
                        ticker=payload.ticker.upper(),
                        shares=payload.quantity,
                        average_cost_basis=payload.price,
                        total_cost=total_cost,
                        purchase_date=trade_time,
                        notes=payload.notes,
                    )
                )
        else:
            if not pos:
                raise ValueError(f"No existing position for {payload.ticker.upper()}")
            if payload.quantity > pos.shares:
                raise ValueError("Cannot sell more shares than currently held")

            remaining = pos.shares - payload.quantity
            if remaining == 0:
                self.db.delete(pos)
            else:
                pos.shares = remaining
                pos.total_cost = remaining * pos.average_cost_basis

        self.db.flush()
        return txn
