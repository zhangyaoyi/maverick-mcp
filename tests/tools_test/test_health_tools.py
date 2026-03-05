"""
Unit tests for Health & System tools:
- get_system_health
- get_component_status
- get_circuit_breaker_status
- get_resource_usage
- get_status_dashboard
- reset_circuit_breaker
- get_health_history
- run_health_diagnostics
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _get_tool_fn(test_mcp, name: str):
    """Helper to retrieve a registered tool function by name."""
    for tool in test_mcp._tool_manager._tools.values():
        if tool.name == name:
            return tool.fn
    return None


def make_mock_health_status(overall="healthy"):
    """Build a minimal health status dict."""
    comp = MagicMock()
    comp.status = "healthy"
    comp.__dict__ = {"status": "healthy", "latency_ms": 5}
    return {
        "status": overall,
        "components": {"database": comp, "redis": comp},
        "resource_usage": {
            "cpu_percent": 20.0,
            "memory_percent": 40.0,
            "disk_percent": 30.0,
        },
    }


class TestGetSystemHealth:
    """Tests for get_system_health."""

    @pytest.mark.asyncio
    async def test_returns_success_on_healthy_system(self):
        """get_system_health returns status=success with data and timestamp."""
        from fastmcp import FastMCP
        from maverick_mcp.api.routers.health_tools import register_health_tools

        test_mcp = FastMCP("Test")
        register_health_tools(test_mcp)
        tool_fn = _get_tool_fn(test_mcp, "get_system_health")
        assert tool_fn, "get_system_health tool not registered"

        mock_health = make_mock_health_status()

        with patch(
            "maverick_mcp.api.routers.health_enhanced._get_detailed_health_status",
            new_callable=AsyncMock,
            return_value=mock_health,
        ):
            result = await tool_fn()

        assert result["status"] == "success"
        assert "data" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_returns_error_on_exception(self):
        """get_system_health returns status=error when health check fails."""
        from fastmcp import FastMCP
        from maverick_mcp.api.routers.health_tools import register_health_tools

        test_mcp = FastMCP("Test")
        register_health_tools(test_mcp)
        tool_fn = _get_tool_fn(test_mcp, "get_system_health")
        assert tool_fn

        with patch(
            "maverick_mcp.api.routers.health_enhanced._get_detailed_health_status",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Health check failed"),
        ):
            result = await tool_fn()

        assert result["status"] == "error"
        assert "error" in result
        assert "timestamp" in result


class TestGetComponentStatus:
    """Tests for get_component_status."""

    @pytest.mark.asyncio
    async def test_returns_all_components_when_no_name_given(self):
        """get_component_status returns all components when no name specified."""
        from fastmcp import FastMCP
        from maverick_mcp.api.routers.health_tools import register_health_tools

        test_mcp = FastMCP("Test")
        register_health_tools(test_mcp)
        tool_fn = _get_tool_fn(test_mcp, "get_component_status")
        assert tool_fn

        mock_health = make_mock_health_status()

        with patch(
            "maverick_mcp.api.routers.health_enhanced._get_detailed_health_status",
            new_callable=AsyncMock,
            return_value=mock_health,
        ):
            result = await tool_fn()

        assert result["status"] == "success"
        assert "total_components" in result

    @pytest.mark.asyncio
    async def test_returns_error_for_unknown_component(self):
        """get_component_status returns error for unknown component name."""
        from fastmcp import FastMCP
        from maverick_mcp.api.routers.health_tools import register_health_tools

        test_mcp = FastMCP("Test")
        register_health_tools(test_mcp)
        tool_fn = _get_tool_fn(test_mcp, "get_component_status")
        assert tool_fn

        mock_health = make_mock_health_status()

        with patch(
            "maverick_mcp.api.routers.health_enhanced._get_detailed_health_status",
            new_callable=AsyncMock,
            return_value=mock_health,
        ):
            result = await tool_fn(component_name="nonexistent_component")

        assert result["status"] == "error"


class TestGetCircuitBreakerStatus:
    """Tests for get_circuit_breaker_status."""

    @pytest.mark.asyncio
    async def test_returns_breaker_summary(self):
        """get_circuit_breaker_status returns summary with state counts."""
        from fastmcp import FastMCP
        from maverick_mcp.api.routers.health_tools import register_health_tools

        test_mcp = FastMCP("Test")
        register_health_tools(test_mcp)
        tool_fn = _get_tool_fn(test_mcp, "get_circuit_breaker_status")
        assert tool_fn

        mock_cb_status = {
            "tiingo_api": {"state": "closed", "failure_count": 0},
            "openrouter_api": {"state": "open", "failure_count": 5},
        }

        # Patch at the source module where it's imported from
        with patch(
            "maverick_mcp.utils.circuit_breaker.get_all_circuit_breaker_status",
            return_value=mock_cb_status,
        ):
            result = await tool_fn()

        assert result["status"] == "success"
        assert "summary" in result
        assert result["summary"]["total_breakers"] == 2

    @pytest.mark.asyncio
    async def test_state_counts_are_correct(self):
        """get_circuit_breaker_status counts open/closed/half_open correctly."""
        from fastmcp import FastMCP
        from maverick_mcp.api.routers.health_tools import register_health_tools

        test_mcp = FastMCP("Test")
        register_health_tools(test_mcp)
        tool_fn = _get_tool_fn(test_mcp, "get_circuit_breaker_status")
        assert tool_fn

        mock_cb_status = {
            "api_a": {"state": "closed"},
            "api_b": {"state": "open"},
            "api_c": {"state": "half_open"},
        }

        with patch(
            "maverick_mcp.utils.circuit_breaker.get_all_circuit_breaker_status",
            return_value=mock_cb_status,
        ):
            result = await tool_fn()

        states = result["summary"]["states"]
        assert states["closed"] == 1
        assert states["open"] == 1
        assert states["half_open"] == 1


class TestResetCircuitBreaker:
    """Tests for reset_circuit_breaker."""

    @pytest.mark.asyncio
    async def test_reset_existing_breaker(self):
        """reset_circuit_breaker returns success for known breaker."""
        from fastmcp import FastMCP
        from maverick_mcp.api.routers.health_tools import register_health_tools

        test_mcp = FastMCP("Test")
        register_health_tools(test_mcp)
        tool_fn = _get_tool_fn(test_mcp, "reset_circuit_breaker")
        assert tool_fn

        mock_manager = MagicMock()
        mock_manager.reset_breaker.return_value = True

        with patch(
            "maverick_mcp.utils.circuit_breaker.get_circuit_breaker_manager",
            return_value=mock_manager,
        ):
            result = await tool_fn(breaker_name="tiingo_api")

        assert result["status"] == "success"
        assert result["breaker_name"] == "tiingo_api"

    @pytest.mark.asyncio
    async def test_reset_unknown_breaker_returns_error(self):
        """reset_circuit_breaker returns error for unknown breaker name."""
        from fastmcp import FastMCP
        from maverick_mcp.api.routers.health_tools import register_health_tools

        test_mcp = FastMCP("Test")
        register_health_tools(test_mcp)
        tool_fn = _get_tool_fn(test_mcp, "reset_circuit_breaker")
        assert tool_fn

        mock_manager = MagicMock()
        mock_manager.reset_breaker.return_value = False

        with patch(
            "maverick_mcp.utils.circuit_breaker.get_circuit_breaker_manager",
            return_value=mock_manager,
        ):
            result = await tool_fn(breaker_name="nonexistent")

        assert result["status"] == "error"


class TestGetResourceUsage:
    """Tests for get_resource_usage."""

    @pytest.mark.asyncio
    async def test_returns_resource_data_and_alerts(self):
        """get_resource_usage returns data with alert flags."""
        from fastmcp import FastMCP
        from maverick_mcp.api.routers.health_tools import register_health_tools

        test_mcp = FastMCP("Test")
        register_health_tools(test_mcp)
        tool_fn = _get_tool_fn(test_mcp, "get_resource_usage")
        assert tool_fn

        mock_usage = MagicMock()
        mock_usage.cpu_percent = 25.0
        mock_usage.memory_percent = 50.0
        mock_usage.disk_percent = 40.0
        mock_usage.__dict__ = {
            "cpu_percent": 25.0,
            "memory_percent": 50.0,
            "disk_percent": 40.0,
        }

        with patch(
            "maverick_mcp.api.routers.health_enhanced._get_resource_usage",
            return_value=mock_usage,
        ):
            result = await tool_fn()

        assert result["status"] == "success"
        assert "alerts" in result
        assert result["alerts"]["high_cpu"] is False
        assert result["alerts"]["high_memory"] is False

    @pytest.mark.asyncio
    async def test_alerts_triggered_at_high_usage(self):
        """get_resource_usage flags alerts when thresholds exceeded."""
        from fastmcp import FastMCP
        from maverick_mcp.api.routers.health_tools import register_health_tools

        test_mcp = FastMCP("Test")
        register_health_tools(test_mcp)
        tool_fn = _get_tool_fn(test_mcp, "get_resource_usage")
        assert tool_fn

        mock_usage = MagicMock()
        mock_usage.cpu_percent = 90.0
        mock_usage.memory_percent = 92.0
        mock_usage.disk_percent = 95.0
        mock_usage.__dict__ = {
            "cpu_percent": 90.0,
            "memory_percent": 92.0,
            "disk_percent": 95.0,
        }

        with patch(
            "maverick_mcp.api.routers.health_enhanced._get_resource_usage",
            return_value=mock_usage,
        ):
            result = await tool_fn()

        assert result["alerts"]["high_cpu"] is True
        assert result["alerts"]["high_memory"] is True
        assert result["alerts"]["high_disk"] is True
