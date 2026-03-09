"""add portfolio ledger tables

Revision ID: 003_add_portfolio_ledger_tables
Revises: 002_add_missing_indexes
Create Date: 2026-03-08
"""

from alembic import op
import sqlalchemy as sa

revision = "003_add_portfolio_ledger_tables"
down_revision = "002_add_missing_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "mcp_portfolio_transactions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("portfolio_id", sa.UUID(), nullable=False),
        sa.Column("ticker", sa.String(length=20), nullable=False),
        sa.Column("side", sa.String(length=10), nullable=False),
        sa.Column("quantity", sa.Numeric(20, 8), nullable=False),
        sa.Column("price", sa.Numeric(12, 4), nullable=False),
        sa.Column("fee", sa.Numeric(20, 4), nullable=False, server_default="0"),
        sa.Column("trade_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lot_method", sa.String(length=10), nullable=False, server_default="FIFO"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["portfolio_id"], ["mcp_portfolios.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_portfolio_transactions_portfolio_time",
        "mcp_portfolio_transactions",
        ["portfolio_id", "trade_time"],
    )
    op.create_index(
        "idx_portfolio_transactions_portfolio_ticker",
        "mcp_portfolio_transactions",
        ["portfolio_id", "ticker"],
    )

    op.create_table(
        "mcp_portfolio_cash_transactions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("portfolio_id", sa.UUID(), nullable=False),
        sa.Column("type", sa.String(length=20), nullable=False),
        sa.Column("amount", sa.Numeric(20, 4), nullable=False),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=False, server_default="USD"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["portfolio_id"], ["mcp_portfolios.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_portfolio_cash_transactions_portfolio_time",
        "mcp_portfolio_cash_transactions",
        ["portfolio_id", "event_time"],
    )

    op.create_table(
        "mcp_portfolio_valuation_snapshots",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("portfolio_id", sa.UUID(), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("invested", sa.Numeric(20, 4), nullable=False),
        sa.Column("market_value", sa.Numeric(20, 4), nullable=False),
        sa.Column("cash_value", sa.Numeric(20, 4), nullable=False),
        sa.Column("total_equity", sa.Numeric(20, 4), nullable=False),
        sa.Column("unrealized_pnl", sa.Numeric(20, 4), nullable=False),
        sa.Column("realized_pnl", sa.Numeric(20, 4), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=False, server_default="USD"),
        sa.Column("meta", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["portfolio_id"], ["mcp_portfolios.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("portfolio_id", "as_of", name="uq_portfolio_valuation_snapshot_portfolio_asof"),
    )
    op.create_index(
        "idx_portfolio_valuation_snapshots_portfolio_asof",
        "mcp_portfolio_valuation_snapshots",
        ["portfolio_id", "as_of"],
    )


def downgrade() -> None:
    op.drop_index("idx_portfolio_valuation_snapshots_portfolio_asof", table_name="mcp_portfolio_valuation_snapshots")
    op.drop_table("mcp_portfolio_valuation_snapshots")

    op.drop_index("idx_portfolio_cash_transactions_portfolio_time", table_name="mcp_portfolio_cash_transactions")
    op.drop_table("mcp_portfolio_cash_transactions")

    op.drop_index("idx_portfolio_transactions_portfolio_ticker", table_name="mcp_portfolio_transactions")
    op.drop_index("idx_portfolio_transactions_portfolio_time", table_name="mcp_portfolio_transactions")
    op.drop_table("mcp_portfolio_transactions")
