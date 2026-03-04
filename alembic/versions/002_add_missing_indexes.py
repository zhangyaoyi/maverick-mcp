"""Add missing database indexes for performance optimization

Revision ID: 002_add_missing_indexes
Revises: 001_create_all_tables
Create Date: 2026-03-04

Changes:
- mcp_price_cache: add volume DESC index (used by get_high_volume_stocks)
- mcp_price_cache: add date single-column index (sync with model)
- mcp_stocks: add sector and exchange indexes (sync with model)
- mcp_technical_cache: add (stock_id, date, indicator_type) 3-col composite index
- mcp_maverick_stocks: replace combined_score ASC with DESC index
- mcp_maverick_bear_stocks: replace score ASC with DESC index
- mcp_supply_demand_breakouts: replace momentum_score ASC with DESC index
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision = "002_add_missing_indexes"
down_revision = "001_create_all_tables"
branch_labels = None
depends_on = None


def _index_exists(index_name: str) -> bool:
    """Check if an index already exists in the database."""
    conn = op.get_bind()
    result = conn.execute(
        sa.text(
            "SELECT 1 FROM pg_indexes WHERE indexname = :name"
        ),
        {"name": index_name},
    )
    return result.fetchone() is not None


def upgrade() -> None:
    """Add missing indexes. Safe to run on existing databases (existence-checked)."""

    # ── 1. mcp_price_cache: volume DESC ──────────────────────────────────────
    # Used by: get_high_volume_stocks() WHERE volume >= :min_volume ORDER BY volume DESC
    if not _index_exists("mcp_price_cache_volume_idx"):
        # CONCURRENTLY cannot run inside a transaction; use plain CREATE INDEX
        op.create_index(
            "mcp_price_cache_volume_idx",
            "mcp_price_cache",
            [sa.text("volume DESC")],
        )
        print("✅ Created mcp_price_cache_volume_idx (volume DESC)")

    # ── 2. mcp_price_cache: date single-column ────────────────────────────────
    # Used by: date-only lookups; also sync gap between migration and model
    if not _index_exists("mcp_price_cache_date_idx"):
        op.create_index(
            "mcp_price_cache_date_idx",
            "mcp_price_cache",
            ["date"],
        )
        print("✅ Created mcp_price_cache_date_idx")

    # ── 3. mcp_stocks: sector ─────────────────────────────────────────────────
    # Sync gap: migration 001 created this but models.py __table_args__ missed it
    if not _index_exists("mcp_stocks_sector_idx"):
        op.create_index("mcp_stocks_sector_idx", "mcp_stocks", ["sector"])
        print("✅ Created mcp_stocks_sector_idx")

    # ── 4. mcp_stocks: exchange ───────────────────────────────────────────────
    if not _index_exists("mcp_stocks_exchange_idx"):
        op.create_index("mcp_stocks_exchange_idx", "mcp_stocks", ["exchange"])
        print("✅ Created mcp_stocks_exchange_idx")

    # ── 5. mcp_technical_cache: 3-col composite ───────────────────────────────
    # Used by: WHERE stock_id=? AND date>=? AND indicator_type=?
    if not _index_exists("mcp_technical_cache_stock_date_indicator_idx"):
        op.create_index(
            "mcp_technical_cache_stock_date_indicator_idx",
            "mcp_technical_cache",
            ["stock_id", "date", "indicator_type"],
        )
        print("✅ Created mcp_technical_cache_stock_date_indicator_idx")

    # ── 6. mcp_maverick_stocks: combined_score DESC ───────────────────────────
    # ORDER BY combined_score DESC is the primary sort for recommendations
    if _index_exists("mcp_maverick_stocks_combined_score_idx"):
        op.drop_index("mcp_maverick_stocks_combined_score_idx", table_name="mcp_maverick_stocks")
        print("🗑️  Dropped mcp_maverick_stocks_combined_score_idx (ASC)")

    if not _index_exists("mcp_maverick_stocks_combined_score_idx"):
        op.create_index(
            "mcp_maverick_stocks_combined_score_idx",
            "mcp_maverick_stocks",
            [sa.text("combined_score DESC")],
        )
        print("✅ Recreated mcp_maverick_stocks_combined_score_idx (DESC)")

    # ── 7. mcp_maverick_bear_stocks: score DESC ───────────────────────────────
    if _index_exists("mcp_maverick_bear_stocks_score_idx"):
        op.drop_index("mcp_maverick_bear_stocks_score_idx", table_name="mcp_maverick_bear_stocks")
        print("🗑️  Dropped mcp_maverick_bear_stocks_score_idx (ASC)")

    if not _index_exists("mcp_maverick_bear_stocks_score_idx"):
        op.create_index(
            "mcp_maverick_bear_stocks_score_idx",
            "mcp_maverick_bear_stocks",
            [sa.text("score DESC")],
        )
        print("✅ Recreated mcp_maverick_bear_stocks_score_idx (DESC)")

    # ── 8. mcp_supply_demand_breakouts: momentum_score DESC ───────────────────
    if _index_exists("mcp_supply_demand_breakouts_momentum_score_idx"):
        op.drop_index(
            "mcp_supply_demand_breakouts_momentum_score_idx",
            table_name="mcp_supply_demand_breakouts",
        )
        print("🗑️  Dropped mcp_supply_demand_breakouts_momentum_score_idx (ASC)")

    if not _index_exists("mcp_supply_demand_breakouts_momentum_score_idx"):
        op.create_index(
            "mcp_supply_demand_breakouts_momentum_score_idx",
            "mcp_supply_demand_breakouts",
            [sa.text("momentum_score DESC")],
        )
        print("✅ Recreated mcp_supply_demand_breakouts_momentum_score_idx (DESC)")

    print("🎉 Index optimization migration completed!")


def downgrade() -> None:
    """Revert index changes."""

    # Restore ASC score indexes
    if _index_exists("mcp_maverick_stocks_combined_score_idx"):
        op.drop_index("mcp_maverick_stocks_combined_score_idx", table_name="mcp_maverick_stocks")
    op.create_index(
        "mcp_maverick_stocks_combined_score_idx",
        "mcp_maverick_stocks",
        ["combined_score"],
    )

    if _index_exists("mcp_maverick_bear_stocks_score_idx"):
        op.drop_index("mcp_maverick_bear_stocks_score_idx", table_name="mcp_maverick_bear_stocks")
    op.create_index(
        "mcp_maverick_bear_stocks_score_idx",
        "mcp_maverick_bear_stocks",
        ["score"],
    )

    if _index_exists("mcp_supply_demand_breakouts_momentum_score_idx"):
        op.drop_index(
            "mcp_supply_demand_breakouts_momentum_score_idx",
            table_name="mcp_supply_demand_breakouts",
        )
    op.create_index(
        "mcp_supply_demand_breakouts_momentum_score_idx",
        "mcp_supply_demand_breakouts",
        ["momentum_score"],
    )

    # Drop newly added indexes
    for idx, tbl in [
        ("mcp_technical_cache_stock_date_indicator_idx", "mcp_technical_cache"),
        ("mcp_stocks_exchange_idx", "mcp_stocks"),
        ("mcp_stocks_sector_idx", "mcp_stocks"),
        ("mcp_price_cache_date_idx", "mcp_price_cache"),
        ("mcp_price_cache_volume_idx", "mcp_price_cache"),
    ]:
        if _index_exists(idx):
            op.drop_index(idx, table_name=tbl)
