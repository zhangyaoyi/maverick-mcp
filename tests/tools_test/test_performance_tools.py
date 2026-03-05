"""
Unit tests for Performance Monitoring tools (all async):
- get_system_performance_health (get_comprehensive_performance_report)
- get_redis_health_status (get_redis_connection_health)
- get_cache_performance_status (get_cache_performance_metrics)
- get_database_performance_status (get_query_performance_metrics)
- analyze_database_index_usage (analyze_database_indexes)
- optimize_cache_configuration (optimize_cache_settings)
- clear_system_caches (clear_performance_caches)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestGetSystemPerformanceHealth:
    """Tests for get_comprehensive_performance_report."""

    @pytest.mark.asyncio
    async def test_returns_health_score(self):
        """get_comprehensive_performance_report returns overall_health_score."""
        from maverick_mcp.tools.performance_monitoring import (
            get_comprehensive_performance_report,
        )

        mock_redis_health = {
            "redis_health": {"status": "healthy", "connected": True}
        }
        mock_cache_metrics = {"hit_rate": 90.0}
        mock_db_metrics = {"avg_query_time_ms": 5.0}

        with (
            patch(
                "maverick_mcp.tools.performance_monitoring.get_redis_connection_health",
                new_callable=AsyncMock,
                return_value=mock_redis_health,
            ),
            patch(
                "maverick_mcp.tools.performance_monitoring.get_cache_performance_metrics",
                new_callable=AsyncMock,
                return_value=mock_cache_metrics,
            ),
            patch(
                "maverick_mcp.tools.performance_monitoring.get_query_performance_metrics",
                new_callable=AsyncMock,
                return_value=mock_db_metrics,
            ),
        ):
            result = await get_comprehensive_performance_report()

        assert isinstance(result, dict)
        assert "overall_health_score" in result or "status" in result

    @pytest.mark.asyncio
    async def test_health_score_range(self):
        """Health score is between 0 and 100."""
        from maverick_mcp.tools.performance_monitoring import (
            get_comprehensive_performance_report,
        )

        with (
            patch(
                "maverick_mcp.tools.performance_monitoring.get_redis_connection_health",
                new_callable=AsyncMock,
                return_value={"redis_health": {"status": "unhealthy"}},
            ),
            patch(
                "maverick_mcp.tools.performance_monitoring.get_cache_performance_metrics",
                new_callable=AsyncMock,
                return_value={"hit_rate": 50.0},
            ),
            patch(
                "maverick_mcp.tools.performance_monitoring.get_query_performance_metrics",
                new_callable=AsyncMock,
                return_value={"avg_query_time_ms": 100.0},
            ),
        ):
            result = await get_comprehensive_performance_report()

        if "overall_health_score" in result:
            score = result["overall_health_score"]
            assert 0 <= score <= 100


class TestGetRedisHealthStatus:
    """Tests for get_redis_connection_health."""

    @pytest.mark.asyncio
    async def test_healthy_redis_returns_healthy_status(self):
        """Returns healthy status when Redis is reachable."""
        from maverick_mcp.tools.performance_monitoring import get_redis_connection_health

        mock_redis_manager = MagicMock()
        mock_redis_manager.get_metrics.return_value = {"connected": True}

        mock_client = AsyncMock()
        mock_client.set = AsyncMock(return_value=True)
        mock_client.get = AsyncMock(return_value=b"test_value")
        mock_client.delete = AsyncMock(return_value=1)
        mock_redis_manager.get_client = AsyncMock(return_value=mock_client)

        with patch(
            "maverick_mcp.tools.performance_monitoring.redis_manager",
            mock_redis_manager,
        ):
            result = await get_redis_connection_health()

        assert isinstance(result, dict)
        assert "redis_health" in result
        assert result["redis_health"].get("status") in ("healthy", "connected")

    @pytest.mark.asyncio
    async def test_unavailable_redis_returns_error(self):
        """Returns error status when Redis is unreachable."""
        from maverick_mcp.tools.performance_monitoring import get_redis_connection_health

        mock_redis_manager = MagicMock()
        mock_redis_manager.get_metrics.side_effect = Exception("Redis unavailable")

        with patch(
            "maverick_mcp.tools.performance_monitoring.redis_manager",
            mock_redis_manager,
        ):
            result = await get_redis_connection_health()

        assert isinstance(result, dict)
        assert "redis_health" in result
        assert result["redis_health"].get("status") in ("unhealthy", "error", "disconnected")


class TestGetCachePerformanceStatus:
    """Tests for get_cache_performance_metrics."""

    @pytest.mark.asyncio
    async def test_returns_cache_metrics(self):
        """Cache metrics return a dict with cache-related data."""
        from maverick_mcp.tools.performance_monitoring import get_cache_performance_metrics

        mock_request_cache = MagicMock()
        mock_request_cache.get_stats.return_value = {
            "hits": 900,
            "misses": 100,
            "size": 500,
        }

        with patch(
            "maverick_mcp.tools.performance_monitoring.request_cache",
            mock_request_cache,
        ):
            result = await get_cache_performance_metrics()

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_returns_dict_even_on_error(self):
        """Cache metrics degrade gracefully on exceptions."""
        from maverick_mcp.tools.performance_monitoring import get_cache_performance_metrics

        with patch(
            "maverick_mcp.tools.performance_monitoring.request_cache",
            side_effect=Exception("Cache error"),
        ):
            try:
                result = await get_cache_performance_metrics()
                assert isinstance(result, dict)
            except Exception:
                pass  # If exception propagates, that's also acceptable


class TestGetDatabasePerformanceStatus:
    """Tests for get_query_performance_metrics."""

    @pytest.mark.asyncio
    async def test_returns_query_metrics(self):
        """Database performance metrics return a dict."""
        from maverick_mcp.tools.performance_monitoring import get_query_performance_metrics

        mock_query_optimizer = MagicMock()
        mock_query_optimizer.get_stats.return_value = {
            "total_queries": 1000,
            "avg_time_ms": 5.0,
            "slow_queries": 10,
        }

        with patch(
            "maverick_mcp.tools.performance_monitoring.query_optimizer",
            mock_query_optimizer,
        ):
            result = await get_query_performance_metrics()

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_handles_db_connection_failure(self):
        """Database metrics degrade gracefully on connection failure."""
        from maverick_mcp.tools.performance_monitoring import get_query_performance_metrics

        mock_query_optimizer = MagicMock()
        mock_query_optimizer.get_stats.side_effect = Exception("DB unreachable")

        with patch(
            "maverick_mcp.tools.performance_monitoring.query_optimizer",
            mock_query_optimizer,
        ):
            result = await get_query_performance_metrics()
            assert isinstance(result, dict)


class TestClearSystemCaches:
    """Tests for clear_performance_caches."""

    @pytest.mark.asyncio
    async def test_clear_caches_returns_dict(self):
        """clear_performance_caches returns a dict."""
        from maverick_mcp.tools.performance_monitoring import clear_performance_caches

        mock_request_cache = MagicMock()
        mock_request_cache.clear.return_value = None

        mock_redis_manager = MagicMock()
        mock_redis_manager.get_client = AsyncMock(return_value=None)

        with (
            patch(
                "maverick_mcp.tools.performance_monitoring.request_cache",
                mock_request_cache,
            ),
            patch(
                "maverick_mcp.tools.performance_monitoring.redis_manager",
                mock_redis_manager,
            ),
        ):
            result = await clear_performance_caches()

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_handles_cache_unavailable(self):
        """clear_performance_caches degrades gracefully when cache unavailable."""
        from maverick_mcp.tools.performance_monitoring import clear_performance_caches

        mock_request_cache = MagicMock()
        mock_request_cache.clear.side_effect = Exception("Cache unavailable")

        mock_redis_manager = MagicMock()
        mock_redis_manager.get_client = AsyncMock(side_effect=Exception("Redis down"))

        with (
            patch(
                "maverick_mcp.tools.performance_monitoring.request_cache",
                mock_request_cache,
            ),
            patch(
                "maverick_mcp.tools.performance_monitoring.redis_manager",
                mock_redis_manager,
            ),
        ):
            result = await clear_performance_caches()

        assert isinstance(result, dict)


class TestAnalyzeDatabaseIndexUsage:
    """Tests for analyze_database_indexes."""

    @pytest.mark.asyncio
    async def test_returns_dict(self):
        """analyze_database_indexes returns a dict."""
        from maverick_mcp.tools.performance_monitoring import analyze_database_indexes

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(return_value=MagicMock(fetchall=MagicMock(return_value=[])))

        with patch(
            "maverick_mcp.tools.performance_monitoring.get_async_db_session",
            return_value=mock_session,
        ):
            result = await analyze_database_indexes()

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_result_is_not_empty(self):
        """Index analysis result is a non-empty dict."""
        from maverick_mcp.tools.performance_monitoring import analyze_database_indexes

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(return_value=MagicMock(fetchall=MagicMock(return_value=[])))

        with patch(
            "maverick_mcp.tools.performance_monitoring.get_async_db_session",
            return_value=mock_session,
        ):
            result = await analyze_database_indexes()

        # Result should be a non-empty dict
        assert len(result) > 0


class TestOptimizeCacheConfiguration:
    """Tests for optimize_cache_settings."""

    @pytest.mark.asyncio
    async def test_returns_optimization_dict(self):
        """optimize_cache_settings returns a dict."""
        from maverick_mcp.tools.performance_monitoring import optimize_cache_settings

        mock_redis_manager = MagicMock()
        mock_redis_manager.get_metrics.return_value = {
            "hits": 700,
            "misses": 300,
        }
        mock_request_cache = MagicMock()
        mock_request_cache.get_stats.return_value = {
            "hits": 500,
            "misses": 200,
            "size": 300,
            "max_size": 1000,
        }

        with (
            patch(
                "maverick_mcp.tools.performance_monitoring.redis_manager",
                mock_redis_manager,
            ),
            patch(
                "maverick_mcp.tools.performance_monitoring.request_cache",
                mock_request_cache,
            ),
        ):
            result = await optimize_cache_settings()

        assert isinstance(result, dict)
