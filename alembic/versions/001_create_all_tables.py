"""Create all tables - clean single migration

Revision ID: 001_create_all_tables
Revises:
Create Date: 2026-03-03

This is a consolidated migration that creates all tables for maverick-mcp.
Replaces the previous broken migration chain.

Tables created:
- mcp_stocks
- mcp_price_cache
- mcp_maverick_stocks
- mcp_maverick_bear_stocks
- mcp_supply_demand_breakouts
- mcp_technical_cache
- mcp_backtest_results
- mcp_backtest_trades
- mcp_optimization_results
- mcp_walk_forward_tests
- mcp_backtest_portfolios
- mcp_portfolios
- mcp_portfolio_positions
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "001_create_all_tables"
down_revision = None
branch_labels = None
depends_on = None


def _table_exists(table_name: str) -> bool:
    """Check if a table already exists."""
    conn = op.get_bind()
    return sa.inspect(conn).has_table(table_name)


def upgrade() -> None:
    """Create all tables with existence checks."""

    # ── 1. mcp_stocks ────────────────────────────────────────────────────────
    if not _table_exists("mcp_stocks"):
        op.create_table(
            "mcp_stocks",
            sa.Column("stock_id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("ticker_symbol", sa.String(10), nullable=False, unique=True),
            sa.Column("company_name", sa.String(255)),
            sa.Column("description", sa.Text()),
            sa.Column("sector", sa.String(100)),
            sa.Column("industry", sa.String(100)),
            sa.Column("exchange", sa.String(50)),
            sa.Column("country", sa.String(50)),
            sa.Column("currency", sa.String(3)),
            sa.Column("isin", sa.String(12)),
            sa.Column("market_cap", sa.BigInteger()),
            sa.Column("shares_outstanding", sa.BigInteger()),
            sa.Column("is_etf", sa.Boolean(), server_default="false"),
            sa.Column("is_active", sa.Boolean(), server_default="true"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("mcp_stocks_ticker_idx", "mcp_stocks", ["ticker_symbol"])
        op.create_index("mcp_stocks_sector_idx", "mcp_stocks", ["sector"])
        op.create_index("mcp_stocks_exchange_idx", "mcp_stocks", ["exchange"])
        op.create_index("mcp_stocks_is_active_idx", "mcp_stocks", ["is_active"])
        print("✅ Created mcp_stocks")

    # ── 2. mcp_price_cache ───────────────────────────────────────────────────
    if not _table_exists("mcp_price_cache"):
        op.create_table(
            "mcp_price_cache",
            sa.Column("price_cache_id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column(
                "stock_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("mcp_stocks.stock_id"),
                nullable=False,
            ),
            sa.Column("date", sa.Date(), nullable=False),
            sa.Column("open_price", sa.Numeric(12, 4)),
            sa.Column("high_price", sa.Numeric(12, 4)),
            sa.Column("low_price", sa.Numeric(12, 4)),
            sa.Column("close_price", sa.Numeric(12, 4)),
            sa.Column("volume", sa.BigInteger()),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_unique_constraint(
            "mcp_price_cache_stock_date_unique", "mcp_price_cache", ["stock_id", "date"]
        )
        op.create_index(
            "mcp_price_cache_stock_id_date_idx", "mcp_price_cache", ["stock_id", "date"]
        )
        op.create_index(
            "mcp_price_cache_ticker_date_idx", "mcp_price_cache", ["stock_id", "date"]
        )
        op.create_index("mcp_price_cache_date_idx", "mcp_price_cache", ["date"])
        print("✅ Created mcp_price_cache")

    # ── 3. mcp_maverick_stocks ───────────────────────────────────────────────
    if not _table_exists("mcp_maverick_stocks"):
        op.create_table(
            "mcp_maverick_stocks",
            sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column(
                "stock_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("mcp_stocks.stock_id"),
                nullable=False,
                index=True,
            ),
            sa.Column("date_analyzed", sa.Date(), nullable=False),
            sa.Column("open_price", sa.Numeric(12, 4), server_default="0"),
            sa.Column("high_price", sa.Numeric(12, 4), server_default="0"),
            sa.Column("low_price", sa.Numeric(12, 4), server_default="0"),
            sa.Column("close_price", sa.Numeric(12, 4), server_default="0"),
            sa.Column("volume", sa.BigInteger(), server_default="0"),
            sa.Column("ema_21", sa.Numeric(12, 4), server_default="0"),
            sa.Column("sma_50", sa.Numeric(12, 4), server_default="0"),
            sa.Column("sma_150", sa.Numeric(12, 4), server_default="0"),
            sa.Column("sma_200", sa.Numeric(12, 4), server_default="0"),
            sa.Column("momentum_score", sa.Numeric(5, 2), server_default="0"),
            sa.Column("avg_vol_30d", sa.Numeric(15, 2), server_default="0"),
            sa.Column("adr_pct", sa.Numeric(5, 2), server_default="0"),
            sa.Column("atr", sa.Numeric(12, 4), server_default="0"),
            sa.Column("pattern_type", sa.String(50)),
            sa.Column("squeeze_status", sa.String(50)),
            sa.Column("consolidation_status", sa.String(50)),
            sa.Column("entry_signal", sa.String(50)),
            sa.Column("compression_score", sa.Integer(), server_default="0"),
            sa.Column("pattern_detected", sa.Integer(), server_default="0"),
            sa.Column("combined_score", sa.Integer(), server_default="0"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index(
            "mcp_maverick_stocks_combined_score_idx", "mcp_maverick_stocks", ["combined_score"]
        )
        op.create_index(
            "mcp_maverick_stocks_momentum_score_idx", "mcp_maverick_stocks", ["momentum_score"]
        )
        op.create_index(
            "mcp_maverick_stocks_date_analyzed_idx", "mcp_maverick_stocks", ["date_analyzed"]
        )
        op.create_index(
            "mcp_maverick_stocks_stock_date_idx",
            "mcp_maverick_stocks",
            ["stock_id", "date_analyzed"],
        )
        print("✅ Created mcp_maverick_stocks")

    # ── 4. mcp_maverick_bear_stocks ──────────────────────────────────────────
    if not _table_exists("mcp_maverick_bear_stocks"):
        op.create_table(
            "mcp_maverick_bear_stocks",
            sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column(
                "stock_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("mcp_stocks.stock_id"),
                nullable=False,
                index=True,
            ),
            sa.Column("date_analyzed", sa.Date(), nullable=False),
            sa.Column("open_price", sa.Numeric(12, 4), server_default="0"),
            sa.Column("high_price", sa.Numeric(12, 4), server_default="0"),
            sa.Column("low_price", sa.Numeric(12, 4), server_default="0"),
            sa.Column("close_price", sa.Numeric(12, 4), server_default="0"),
            sa.Column("volume", sa.BigInteger(), server_default="0"),
            sa.Column("momentum_score", sa.Numeric(5, 2), server_default="0"),
            sa.Column("ema_21", sa.Numeric(12, 4), server_default="0"),
            sa.Column("sma_50", sa.Numeric(12, 4), server_default="0"),
            sa.Column("sma_200", sa.Numeric(12, 4), server_default="0"),
            sa.Column("rsi_14", sa.Numeric(5, 2), server_default="0"),
            sa.Column("macd", sa.Numeric(12, 6), server_default="0"),
            sa.Column("macd_signal", sa.Numeric(12, 6), server_default="0"),
            sa.Column("macd_histogram", sa.Numeric(12, 6), server_default="0"),
            sa.Column("dist_days_20", sa.Integer(), server_default="0"),
            sa.Column("adr_pct", sa.Numeric(5, 2), server_default="0"),
            sa.Column("atr_contraction", sa.Boolean(), server_default="false"),
            sa.Column("atr", sa.Numeric(12, 4), server_default="0"),
            sa.Column("avg_vol_30d", sa.Numeric(15, 2), server_default="0"),
            sa.Column("big_down_vol", sa.Boolean(), server_default="false"),
            sa.Column("squeeze_status", sa.String(50)),
            sa.Column("consolidation_status", sa.String(50)),
            sa.Column("score", sa.Integer(), server_default="0"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index(
            "mcp_maverick_bear_stocks_score_idx", "mcp_maverick_bear_stocks", ["score"]
        )
        op.create_index(
            "mcp_maverick_bear_stocks_momentum_score_idx",
            "mcp_maverick_bear_stocks",
            ["momentum_score"],
        )
        op.create_index(
            "mcp_maverick_bear_stocks_date_analyzed_idx",
            "mcp_maverick_bear_stocks",
            ["date_analyzed"],
        )
        op.create_index(
            "mcp_maverick_bear_stocks_stock_date_idx",
            "mcp_maverick_bear_stocks",
            ["stock_id", "date_analyzed"],
        )
        print("✅ Created mcp_maverick_bear_stocks")

    # ── 5. mcp_supply_demand_breakouts ───────────────────────────────────────
    if not _table_exists("mcp_supply_demand_breakouts"):
        op.create_table(
            "mcp_supply_demand_breakouts",
            sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column(
                "stock_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("mcp_stocks.stock_id"),
                nullable=False,
                index=True,
            ),
            sa.Column("date_analyzed", sa.Date(), nullable=False),
            sa.Column("open_price", sa.Numeric(12, 4), server_default="0"),
            sa.Column("high_price", sa.Numeric(12, 4), server_default="0"),
            sa.Column("low_price", sa.Numeric(12, 4), server_default="0"),
            sa.Column("close_price", sa.Numeric(12, 4), server_default="0"),
            sa.Column("volume", sa.BigInteger(), server_default="0"),
            sa.Column("ema_21", sa.Numeric(12, 4), server_default="0"),
            sa.Column("sma_50", sa.Numeric(12, 4), server_default="0"),
            sa.Column("sma_150", sa.Numeric(12, 4), server_default="0"),
            sa.Column("sma_200", sa.Numeric(12, 4), server_default="0"),
            sa.Column("momentum_score", sa.Numeric(5, 2), server_default="0"),
            sa.Column("avg_volume_30d", sa.Numeric(15, 2), server_default="0"),
            sa.Column("adr_pct", sa.Numeric(5, 2), server_default="0"),
            sa.Column("atr", sa.Numeric(12, 4), server_default="0"),
            sa.Column("pattern_type", sa.String(50)),
            sa.Column("squeeze_status", sa.String(50)),
            sa.Column("consolidation_status", sa.String(50)),
            sa.Column("entry_signal", sa.String(50)),
            sa.Column("accumulation_rating", sa.Numeric(5, 2), server_default="0"),
            sa.Column("distribution_rating", sa.Numeric(5, 2), server_default="0"),
            sa.Column("breakout_strength", sa.Numeric(5, 2), server_default="0"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index(
            "mcp_supply_demand_breakouts_momentum_score_idx",
            "mcp_supply_demand_breakouts",
            ["momentum_score"],
        )
        op.create_index(
            "mcp_supply_demand_breakouts_date_analyzed_idx",
            "mcp_supply_demand_breakouts",
            ["date_analyzed"],
        )
        op.create_index(
            "mcp_supply_demand_breakouts_stock_date_idx",
            "mcp_supply_demand_breakouts",
            ["stock_id", "date_analyzed"],
        )
        op.create_index(
            "mcp_supply_demand_breakouts_ma_filter_idx",
            "mcp_supply_demand_breakouts",
            ["close_price", "sma_50", "sma_150", "sma_200"],
        )
        print("✅ Created mcp_supply_demand_breakouts")

    # ── 6. mcp_technical_cache ───────────────────────────────────────────────
    if not _table_exists("mcp_technical_cache"):
        op.create_table(
            "mcp_technical_cache",
            sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column(
                "stock_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("mcp_stocks.stock_id"),
                nullable=False,
            ),
            sa.Column("date", sa.Date(), nullable=False),
            sa.Column("indicator_type", sa.String(50), nullable=False),
            sa.Column("value", sa.Numeric(20, 8)),
            sa.Column("value_2", sa.Numeric(20, 8)),
            sa.Column("value_3", sa.Numeric(20, 8)),
            sa.Column("metadata", sa.Text()),
            sa.Column("period", sa.Integer()),
            sa.Column("parameters", sa.Text()),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_unique_constraint(
            "mcp_technical_cache_stock_date_indicator_unique",
            "mcp_technical_cache",
            ["stock_id", "date", "indicator_type"],
        )
        op.create_index(
            "mcp_technical_cache_stock_date_idx", "mcp_technical_cache", ["stock_id", "date"]
        )
        op.create_index(
            "mcp_technical_cache_indicator_idx", "mcp_technical_cache", ["indicator_type"]
        )
        op.create_index("mcp_technical_cache_date_idx", "mcp_technical_cache", ["date"])
        print("✅ Created mcp_technical_cache")

    # ── 7. mcp_backtest_results ──────────────────────────────────────────────
    if not _table_exists("mcp_backtest_results"):
        op.create_table(
            "mcp_backtest_results",
            sa.Column("backtest_id", sa.Uuid(), nullable=False, primary_key=True),
            sa.Column("symbol", sa.String(length=10), nullable=False),
            sa.Column("strategy_type", sa.String(length=50), nullable=False),
            sa.Column("backtest_date", sa.DateTime(timezone=True), nullable=False),
            sa.Column("start_date", sa.Date(), nullable=False),
            sa.Column("end_date", sa.Date(), nullable=False),
            sa.Column("initial_capital", sa.Numeric(precision=15, scale=2), server_default="10000.0"),
            sa.Column("fees", sa.Numeric(precision=6, scale=4), server_default="0.001"),
            sa.Column("slippage", sa.Numeric(precision=6, scale=4), server_default="0.001"),
            sa.Column("parameters", sa.JSON()),
            sa.Column("total_return", sa.Numeric(precision=10, scale=4)),
            sa.Column("annualized_return", sa.Numeric(precision=10, scale=4)),
            sa.Column("sharpe_ratio", sa.Numeric(precision=8, scale=4)),
            sa.Column("sortino_ratio", sa.Numeric(precision=8, scale=4)),
            sa.Column("calmar_ratio", sa.Numeric(precision=8, scale=4)),
            sa.Column("max_drawdown", sa.Numeric(precision=8, scale=4)),
            sa.Column("max_drawdown_duration", sa.Integer()),
            sa.Column("volatility", sa.Numeric(precision=8, scale=4)),
            sa.Column("downside_volatility", sa.Numeric(precision=8, scale=4)),
            sa.Column("total_trades", sa.Integer(), server_default="0"),
            sa.Column("winning_trades", sa.Integer(), server_default="0"),
            sa.Column("losing_trades", sa.Integer(), server_default="0"),
            sa.Column("win_rate", sa.Numeric(precision=5, scale=4)),
            sa.Column("profit_factor", sa.Numeric(precision=8, scale=4)),
            sa.Column("average_win", sa.Numeric(precision=12, scale=4)),
            sa.Column("average_loss", sa.Numeric(precision=12, scale=4)),
            sa.Column("largest_win", sa.Numeric(precision=12, scale=4)),
            sa.Column("largest_loss", sa.Numeric(precision=12, scale=4)),
            sa.Column("final_portfolio_value", sa.Numeric(precision=15, scale=2)),
            sa.Column("peak_portfolio_value", sa.Numeric(precision=15, scale=2)),
            sa.Column("beta", sa.Numeric(precision=8, scale=4)),
            sa.Column("alpha", sa.Numeric(precision=8, scale=4)),
            sa.Column("equity_curve", sa.JSON()),
            sa.Column("drawdown_series", sa.JSON()),
            sa.Column("execution_time_seconds", sa.Numeric(precision=8, scale=3)),
            sa.Column("data_points", sa.Integer()),
            sa.Column("status", sa.String(length=20), server_default="completed"),
            sa.Column("error_message", sa.Text()),
            sa.Column("notes", sa.Text()),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("mcp_backtest_results_symbol_idx", "mcp_backtest_results", ["symbol"])
        op.create_index("mcp_backtest_results_strategy_idx", "mcp_backtest_results", ["strategy_type"])
        op.create_index("mcp_backtest_results_date_idx", "mcp_backtest_results", ["backtest_date"])
        op.create_index("mcp_backtest_results_sharpe_idx", "mcp_backtest_results", ["sharpe_ratio"])
        op.create_index("mcp_backtest_results_total_return_idx", "mcp_backtest_results", ["total_return"])
        op.create_index(
            "mcp_backtest_results_symbol_strategy_idx",
            "mcp_backtest_results",
            ["symbol", "strategy_type"],
        )
        print("✅ Created mcp_backtest_results")

    # ── 8. mcp_backtest_trades ───────────────────────────────────────────────
    if not _table_exists("mcp_backtest_trades"):
        op.create_table(
            "mcp_backtest_trades",
            sa.Column("trade_id", sa.Uuid(), nullable=False, primary_key=True),
            sa.Column("backtest_id", sa.Uuid(), nullable=False),
            sa.Column("trade_number", sa.Integer(), nullable=False),
            sa.Column("entry_date", sa.Date(), nullable=False),
            sa.Column("entry_price", sa.Numeric(precision=12, scale=4), nullable=False),
            sa.Column("entry_time", sa.DateTime(timezone=True)),
            sa.Column("exit_date", sa.Date()),
            sa.Column("exit_price", sa.Numeric(precision=12, scale=4)),
            sa.Column("exit_time", sa.DateTime(timezone=True)),
            sa.Column("position_size", sa.Numeric(precision=15, scale=2)),
            sa.Column("direction", sa.String(length=5), nullable=False),
            sa.Column("pnl", sa.Numeric(precision=12, scale=4)),
            sa.Column("pnl_percent", sa.Numeric(precision=8, scale=4)),
            sa.Column("mae", sa.Numeric(precision=8, scale=4)),
            sa.Column("mfe", sa.Numeric(precision=8, scale=4)),
            sa.Column("duration_days", sa.Integer()),
            sa.Column("duration_hours", sa.Numeric(precision=8, scale=2)),
            sa.Column("exit_reason", sa.String(length=50)),
            sa.Column("fees_paid", sa.Numeric(precision=10, scale=4), server_default="0"),
            sa.Column("slippage_cost", sa.Numeric(precision=10, scale=4), server_default="0"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(
                ["backtest_id"], ["mcp_backtest_results.backtest_id"], ondelete="CASCADE"
            ),
        )
        op.create_index("mcp_backtest_trades_backtest_idx", "mcp_backtest_trades", ["backtest_id"])
        op.create_index("mcp_backtest_trades_entry_date_idx", "mcp_backtest_trades", ["entry_date"])
        op.create_index("mcp_backtest_trades_exit_date_idx", "mcp_backtest_trades", ["exit_date"])
        op.create_index("mcp_backtest_trades_pnl_idx", "mcp_backtest_trades", ["pnl"])
        op.create_index(
            "mcp_backtest_trades_backtest_entry_idx",
            "mcp_backtest_trades",
            ["backtest_id", "entry_date"],
        )
        print("✅ Created mcp_backtest_trades")

    # ── 9. mcp_optimization_results ─────────────────────────────────────────
    if not _table_exists("mcp_optimization_results"):
        op.create_table(
            "mcp_optimization_results",
            sa.Column("optimization_id", sa.Uuid(), nullable=False, primary_key=True),
            sa.Column("backtest_id", sa.Uuid(), nullable=False),
            sa.Column("optimization_date", sa.DateTime(timezone=True), nullable=False),
            sa.Column("parameter_set", sa.Integer(), nullable=False),
            sa.Column("parameters", sa.JSON(), nullable=False),
            sa.Column("objective_function", sa.String(length=50)),
            sa.Column("objective_value", sa.Numeric(precision=12, scale=6)),
            sa.Column("total_return", sa.Numeric(precision=10, scale=4)),
            sa.Column("sharpe_ratio", sa.Numeric(precision=8, scale=4)),
            sa.Column("max_drawdown", sa.Numeric(precision=8, scale=4)),
            sa.Column("win_rate", sa.Numeric(precision=5, scale=4)),
            sa.Column("profit_factor", sa.Numeric(precision=8, scale=4)),
            sa.Column("total_trades", sa.Integer()),
            sa.Column("rank", sa.Integer()),
            sa.Column("is_statistically_significant", sa.Boolean(), server_default="false"),
            sa.Column("p_value", sa.Numeric(precision=8, scale=6)),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(
                ["backtest_id"], ["mcp_backtest_results.backtest_id"], ondelete="CASCADE"
            ),
        )
        op.create_index(
            "mcp_optimization_results_backtest_idx", "mcp_optimization_results", ["backtest_id"]
        )
        op.create_index(
            "mcp_optimization_results_param_set_idx",
            "mcp_optimization_results",
            ["parameter_set"],
        )
        op.create_index(
            "mcp_optimization_results_objective_idx",
            "mcp_optimization_results",
            ["objective_value"],
        )
        print("✅ Created mcp_optimization_results")

    # ── 10. mcp_walk_forward_tests ───────────────────────────────────────────
    if not _table_exists("mcp_walk_forward_tests"):
        op.create_table(
            "mcp_walk_forward_tests",
            sa.Column("walk_forward_id", sa.Uuid(), nullable=False, primary_key=True),
            sa.Column("parent_backtest_id", sa.Uuid(), nullable=False),
            sa.Column("test_date", sa.DateTime(timezone=True), nullable=False),
            sa.Column("window_size_months", sa.Integer(), nullable=False),
            sa.Column("step_size_months", sa.Integer(), nullable=False),
            sa.Column("training_start", sa.Date(), nullable=False),
            sa.Column("training_end", sa.Date(), nullable=False),
            sa.Column("test_period_start", sa.Date(), nullable=False),
            sa.Column("test_period_end", sa.Date(), nullable=False),
            sa.Column("optimal_parameters", sa.JSON()),
            sa.Column("training_performance", sa.Numeric(precision=10, scale=4)),
            sa.Column("out_of_sample_return", sa.Numeric(precision=10, scale=4)),
            sa.Column("out_of_sample_sharpe", sa.Numeric(precision=8, scale=4)),
            sa.Column("out_of_sample_drawdown", sa.Numeric(precision=8, scale=4)),
            sa.Column("out_of_sample_trades", sa.Integer()),
            sa.Column("performance_ratio", sa.Numeric(precision=8, scale=4)),
            sa.Column("degradation_factor", sa.Numeric(precision=8, scale=4)),
            sa.Column("is_profitable", sa.Boolean()),
            sa.Column("is_statistically_significant", sa.Boolean(), server_default="false"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(
                ["parent_backtest_id"],
                ["mcp_backtest_results.backtest_id"],
                ondelete="CASCADE",
            ),
        )
        op.create_index(
            "mcp_walk_forward_tests_parent_idx",
            "mcp_walk_forward_tests",
            ["parent_backtest_id"],
        )
        op.create_index(
            "mcp_walk_forward_tests_period_idx",
            "mcp_walk_forward_tests",
            ["test_period_start"],
        )
        op.create_index(
            "mcp_walk_forward_tests_performance_idx",
            "mcp_walk_forward_tests",
            ["out_of_sample_return"],
        )
        print("✅ Created mcp_walk_forward_tests")

    # ── 11. mcp_backtest_portfolios ──────────────────────────────────────────
    if not _table_exists("mcp_backtest_portfolios"):
        op.create_table(
            "mcp_backtest_portfolios",
            sa.Column("portfolio_backtest_id", sa.Uuid(), nullable=False, primary_key=True),
            sa.Column("portfolio_name", sa.String(length=100), nullable=False),
            sa.Column("description", sa.Text()),
            sa.Column("backtest_date", sa.DateTime(timezone=True), nullable=False),
            sa.Column("start_date", sa.Date(), nullable=False),
            sa.Column("end_date", sa.Date(), nullable=False),
            sa.Column("symbols", sa.JSON(), nullable=False),
            sa.Column("weights", sa.JSON()),
            sa.Column("rebalance_frequency", sa.String(length=20)),
            sa.Column("initial_capital", sa.Numeric(precision=15, scale=2), server_default="100000.0"),
            sa.Column("max_positions", sa.Integer()),
            sa.Column("position_sizing_method", sa.String(length=50)),
            sa.Column("portfolio_stop_loss", sa.Numeric(precision=6, scale=4)),
            sa.Column("max_sector_allocation", sa.Numeric(precision=5, scale=4)),
            sa.Column("correlation_threshold", sa.Numeric(precision=5, scale=4)),
            sa.Column("total_return", sa.Numeric(precision=10, scale=4)),
            sa.Column("annualized_return", sa.Numeric(precision=10, scale=4)),
            sa.Column("sharpe_ratio", sa.Numeric(precision=8, scale=4)),
            sa.Column("sortino_ratio", sa.Numeric(precision=8, scale=4)),
            sa.Column("max_drawdown", sa.Numeric(precision=8, scale=4)),
            sa.Column("volatility", sa.Numeric(precision=8, scale=4)),
            sa.Column("diversification_ratio", sa.Numeric(precision=8, scale=4)),
            sa.Column("concentration_index", sa.Numeric(precision=8, scale=4)),
            sa.Column("turnover_rate", sa.Numeric(precision=8, scale=4)),
            sa.Column("component_backtest_ids", sa.JSON()),
            sa.Column("portfolio_equity_curve", sa.JSON()),
            sa.Column("portfolio_weights_history", sa.JSON()),
            sa.Column("status", sa.String(length=20), server_default="completed"),
            sa.Column("notes", sa.Text()),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index(
            "mcp_backtest_portfolios_name_idx", "mcp_backtest_portfolios", ["portfolio_name"]
        )
        op.create_index(
            "mcp_backtest_portfolios_date_idx", "mcp_backtest_portfolios", ["backtest_date"]
        )
        op.create_index(
            "mcp_backtest_portfolios_return_idx", "mcp_backtest_portfolios", ["total_return"]
        )
        print("✅ Created mcp_backtest_portfolios")

    # ── 12. mcp_portfolios ───────────────────────────────────────────────────
    if not _table_exists("mcp_portfolios"):
        op.create_table(
            "mcp_portfolios",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("user_id", sa.String(100), nullable=False, server_default="default"),
            sa.Column("name", sa.String(200), nullable=False, server_default="My Portfolio"),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
        )
        op.create_index("idx_portfolio_user", "mcp_portfolios", ["user_id"])
        op.create_unique_constraint("uq_user_portfolio_name", "mcp_portfolios", ["user_id", "name"])
        print("✅ Created mcp_portfolios")

    # ── 13. mcp_portfolio_positions ──────────────────────────────────────────
    if not _table_exists("mcp_portfolio_positions"):
        op.create_table(
            "mcp_portfolio_positions",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("portfolio_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("ticker", sa.String(20), nullable=False),
            sa.Column("shares", sa.Numeric(20, 8), nullable=False),
            sa.Column("average_cost_basis", sa.Numeric(12, 4), nullable=False),
            sa.Column("total_cost", sa.Numeric(20, 4), nullable=False),
            sa.Column("purchase_date", sa.DateTime(timezone=True), nullable=False),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(
                ["portfolio_id"], ["mcp_portfolios.id"], ondelete="CASCADE"
            ),
        )
        op.create_index("idx_position_portfolio", "mcp_portfolio_positions", ["portfolio_id"])
        op.create_index("idx_position_ticker", "mcp_portfolio_positions", ["ticker"])
        op.create_index(
            "idx_position_portfolio_ticker",
            "mcp_portfolio_positions",
            ["portfolio_id", "ticker"],
        )
        op.create_unique_constraint(
            "uq_portfolio_position_ticker",
            "mcp_portfolio_positions",
            ["portfolio_id", "ticker"],
        )
        print("✅ Created mcp_portfolio_positions")

    print("🎉 All tables created successfully!")


def downgrade() -> None:
    """Drop all tables in reverse order."""
    tables = [
        "mcp_portfolio_positions",
        "mcp_portfolios",
        "mcp_backtest_portfolios",
        "mcp_walk_forward_tests",
        "mcp_optimization_results",
        "mcp_backtest_trades",
        "mcp_backtest_results",
        "mcp_technical_cache",
        "mcp_supply_demand_breakouts",
        "mcp_maverick_bear_stocks",
        "mcp_maverick_stocks",
        "mcp_price_cache",
        "mcp_stocks",
    ]
    for table in tables:
        if _table_exists(table):
            op.drop_table(table)
            print(f"🗑️  Dropped {table}")
