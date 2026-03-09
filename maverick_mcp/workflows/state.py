"""
State definitions for LangGraph workflows using TypedDict pattern.
"""

from datetime import datetime
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


def take_latest_status(current: str, new: str) -> str:
    """Reducer function that takes the latest status update."""
    return new if new else current


class BaseAgentState(TypedDict):
    """Base state for all agents with comprehensive tracking."""

    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    persona: str
    timestamp: datetime
    token_count: int
    error: str | None

    # Enhanced tracking
    analyzed_stocks: dict[str, dict[str, Any]]  # symbol -> analysis data
    key_price_levels: dict[str, dict[str, float]]  # symbol -> support/resistance
    last_analysis_time: dict[str, datetime]  # symbol -> timestamp
    conversation_context: dict[str, Any]  # Additional context

    # Performance tracking
    execution_time_ms: float | None
    api_calls_made: int
    cache_hits: int
    cache_misses: int


class BacktestingWorkflowState(BaseAgentState):
    """State for intelligent backtesting workflows with market regime analysis."""

    # Input parameters
    symbol: str  # Stock symbol to backtest
    start_date: str  # Start date for analysis (YYYY-MM-DD)
    end_date: str  # End date for analysis (YYYY-MM-DD)
    initial_capital: float  # Starting capital for backtest
    requested_strategy: str | None  # User-requested strategy (optional)

    # Market regime analysis
    market_regime: str  # bull, bear, sideways, volatile, low_volume
    regime_confidence: float  # Confidence in regime detection (0-1)
    regime_indicators: dict[str, float]  # Supporting indicators for regime
    regime_analysis_time_ms: float  # Time spent on regime analysis
    volatility_percentile: float  # Current volatility vs historical
    trend_strength: float  # Strength of current trend (-1 to 1)

    # Market conditions context
    market_conditions: dict[str, Any]  # Overall market environment
    sector_performance: dict[str, float]  # Sector relative performance
    correlation_to_market: float  # Stock correlation to broad market
    volume_profile: dict[str, float]  # Volume characteristics
    support_resistance_levels: list[float]  # Key price levels

    # Strategy selection process
    candidate_strategies: list[dict[str, Any]]  # List of potential strategies
    strategy_rankings: dict[str, float]  # Strategy -> fitness score
    selected_strategies: list[str]  # Final selected strategies for testing
    strategy_selection_reasoning: str  # Why these strategies were chosen
    strategy_selection_confidence: float  # Confidence in selection (0-1)

    # Parameter optimization
    optimization_config: dict[str, Any]  # Optimization configuration
    parameter_grids: dict[str, dict[str, list]]  # Strategy -> parameter grid
    optimization_results: dict[str, dict[str, Any]]  # Strategy -> optimization results
    best_parameters: dict[str, dict[str, Any]]  # Strategy -> best parameters
    optimization_time_ms: float  # Time spent on optimization
    optimization_iterations: int  # Number of parameter combinations tested

    # Validation and robustness
    walk_forward_results: dict[str, dict[str, Any]]  # Strategy -> WF results
    monte_carlo_results: dict[str, dict[str, Any]]  # Strategy -> MC results
    out_of_sample_performance: dict[str, dict[str, float]]  # OOS metrics
    robustness_score: dict[str, float]  # Strategy -> robustness score (0-1)
    validation_warnings: list[str]  # Validation warnings and concerns

    # Final recommendations
    final_strategy_ranking: list[dict[str, Any]]  # Ranked strategy recommendations
    recommended_strategy: str  # Top recommended strategy
    recommended_parameters: dict[str, Any]  # Recommended parameter set
    recommendation_confidence: float  # Overall confidence (0-1)
    risk_assessment: dict[str, Any]  # Risk analysis of recommendation

    # Performance metrics aggregation
    comparative_metrics: dict[str, dict[str, float]]  # Strategy -> metrics
    benchmark_comparison: dict[str, float]  # Comparison to buy-and-hold
    risk_adjusted_performance: dict[str, float]  # Strategy -> risk-adj returns
    drawdown_analysis: dict[str, dict[str, float]]  # Drawdown characteristics

    # Workflow status and control
    workflow_status: Annotated[
        str, take_latest_status
    ]  # analyzing_regime, selecting_strategies, optimizing, validating, completed
    current_step: str  # Current workflow step for progress tracking
    steps_completed: list[str]  # Completed workflow steps
    total_execution_time_ms: float  # Total workflow execution time

    # Error handling and recovery
    errors_encountered: list[dict[str, Any]]  # Errors with context
    fallback_strategies_used: list[str]  # Fallback strategies activated
    data_quality_issues: list[str]  # Data quality concerns identified

    # Caching and performance
    cached_results: dict[str, Any]  # Cached intermediate results
    cache_hit_rate: float  # Cache effectiveness
    api_calls_made: int  # Number of external API calls

    # Advanced analysis features
    regime_transition_analysis: dict[str, Any]  # Analysis of regime changes
    multi_timeframe_analysis: dict[str, dict[str, Any]]  # Analysis across timeframes
    correlation_analysis: dict[str, float]  # Inter-asset correlations
    macroeconomic_context: dict[str, Any]  # Macro environment factors
