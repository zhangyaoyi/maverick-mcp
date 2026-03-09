# Maverick-MCP Makefile
# Central command interface for agent-friendly development

.PHONY: help dev dev-sse dev-http dev-stdio stop test test-all test-watch test-specific test-parallel test-cov test-speed test-speed-quick test-speed-emergency test-speed-comparison test-strategies test-portfolio-ledger lint format typecheck clean tail-log backend check migrate migrate-dev migrate-test migrate-prod migrate-all setup create-dev-db create-dbs redis-start redis-stop experiment experiment-once benchmark-parallel benchmark-speed docker-up docker-down docker-logs docker-infra-up docker-infra-down infra-up infra-down infra-reset

# Default target
help:
	@echo "Maverick-MCP Development Commands:"
	@echo ""
	@echo "  make dev          - Start development environment (SSE transport, default)"
	@echo "  make dev-sse      - Start with SSE transport (same as dev)"
	@echo "  make dev-http     - Start with Streamable-HTTP transport (for curl/testing)"
	@echo "  make dev-stdio    - Start with STDIO transport (for direct connections)"
	@echo "  make backend      - Start backend MCP server (dev mode)"
	@echo "  make stop         - Stop all services"
	@echo ""
	@echo "  make test         - Run unit tests (fast)"
	@echo "  make test-all     - Run all tests including integration"
	@echo "  make test-watch   - Auto-run tests on file changes"
	@echo "  make test-specific TEST=name - Run specific test"
	@echo "  make test-parallel - Run tests in parallel"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo "  make test-fixes   - Validate MCP tool fixes are working"
	@echo "  make test-speed   - Run speed optimization validation tests"
	@echo "  make test-speed-quick - Quick speed validation for CI"
	@echo "  make test-speed-emergency - Emergency mode speed tests"
	@echo "  make test-speed-comparison - Before/after performance comparison"
	@echo "  make test-strategies - Validate ALL backtesting strategies"
	@echo ""
	@echo "  make lint         - Run code quality checks"
	@echo "  make format       - Auto-format code"
	@echo "  make typecheck    - Run type checking"
	@echo "  make check        - Run all checks (lint + type check)"
	@echo ""
	@echo "  make tail-log     - Follow backend logs"
	@echo ""
	@echo "  make experiment   - Watch and auto-run .py files"
	@echo "  make benchmark-parallel - Test parallel screening"
	@echo "  make benchmark-speed - Run comprehensive speed benchmark"
	@echo "  make migrate      - Run database migrations"
	@echo "  make migrate-dev  - Run migrations on dev database (uses .env.development)"
	@echo "  make create-dev-db - Create maverick_mcp_dev database in Docker PostgreSQL"
	@echo "  make setup        - Initial project setup"
	@echo "  make clean        - Clean up generated files"
	@echo ""
	@echo "  make infra-up     - Start infrastructure (Postgres + Redis) with --env-file .env"
	@echo "  make infra-down   - Stop infrastructure"
	@echo "  make infra-reset  - Wipe volumes and restart infrastructure"
	@echo "  make migrate-dev  - Run migrations on maverick_mcp_dev"
	@echo "  make migrate-test - Run migrations on maverick_mcp_test"
	@echo "  make migrate-prod - Run migrations on maverick_mcp (with confirmation)"
	@echo "  make migrate-all  - Run migrations on all three databases"
	@echo "  make docker-infra-up  - Start infrastructure (Postgres + Redis)"
	@echo "  make docker-infra-down - Stop infrastructure"
	@echo "  make docker-up    - Start app (requires infra running)"
	@echo "  make docker-down  - Stop app services"
	@echo "  make docker-logs  - View Docker logs"

# Development commands
dev:
	@echo "Starting Maverick-MCP development environment (SSE transport)..."
	@set -a; [ -f .env ] && . ./.env; [ -f .env.dev ] && . ./.env.dev; set +a; ./scripts/dev.sh

dev-sse:
	@echo "Starting Maverick-MCP development environment (SSE transport)..."
	@set -a; [ -f .env ] && . ./.env; [ -f .env.dev ] && . ./.env.dev; set +a; ./scripts/dev.sh

dev-http:
	@echo "Starting Maverick-MCP development environment (Streamable-HTTP transport)..."
	@MAVERICK_TRANSPORT=streamable-http ./scripts/dev.sh

dev-stdio:
	@echo "Starting Maverick-MCP development environment (STDIO transport)..."
	@MAVERICK_TRANSPORT=stdio ./scripts/dev.sh

backend:
	@echo "Starting backend in development mode..."
	@./scripts/start-backend.sh --dev

stop:
	@echo "Stopping all services..."
	@pkill -f "maverick_mcp.api.server" || true
	@echo "All services stopped."

# Testing commands
test:
	@echo "Running unit tests..."
	@uv run pytest -v

test-all:
	@echo "Running all tests (including integration)..."
	@uv run pytest -v -m ""

test-watch:
	@echo "Starting test watcher..."
	@if ! uv pip show pytest-watch > /dev/null 2>&1; then \
		echo "Installing pytest-watch..."; \
		uv pip install pytest-watch; \
	fi
	@uv run ptw -- -v

test-specific:
	@if [ -z "$(TEST)" ]; then \
		echo "Usage: make test-specific TEST=test_name"; \
		exit 1; \
	fi
	@echo "Running specific test: $(TEST)"
	@uv run pytest -v -k "$(TEST)"

test-portfolio-ledger:
	@echo "Running portfolio ledger focused local tests (no Docker required)..."
	@uv run pytest -v maverick_mcp/tests/test_portfolio_ledger_local.py

test-parallel:
	@echo "Running tests in parallel..."
	@if ! uv pip show pytest-xdist > /dev/null 2>&1; then \
		echo "Installing pytest-xdist..."; \
		uv pip install pytest-xdist; \
	fi
	@uv run pytest -v -n auto

test-cov:
	@echo "Running tests with coverage..."
	@uv run pytest --cov=maverick_mcp --cov-report=html --cov-report=term

test-fixes:
	@echo "Running MCP tool fixes validation..."
	@uv run python maverick_mcp/tests/test_mcp_tool_fixes.py

test-fixes-verbose:
	@echo "Running MCP tool fixes validation (verbose)..."
	@uv run python -u maverick_mcp/tests/test_mcp_tool_fixes.py

# Speed optimization testing commands
test-speed:
	@echo "Running speed optimization validation tests..."
	@uv run pytest -v tests/test_speed_optimization_validation.py

test-speed-quick:
	@echo "Running quick speed validation for CI..."
	@uv run python scripts/speed_benchmark.py --mode quick

test-speed-emergency:
	@echo "Running emergency mode speed tests..."
	@uv run python scripts/speed_benchmark.py --mode emergency

test-speed-comparison:
	@echo "Running before/after performance comparison..."
	@uv run python scripts/speed_benchmark.py --mode comparison

test-strategies:
	@echo "Validating ALL backtesting strategies with real market data..."
	@uv run python scripts/test_all_strategies.py

# Code quality commands
lint:
	@echo "Running linter..."
	@uv run ruff check .

format:
	@echo "Formatting code..."
	@uv run ruff format .
	@uv run ruff check . --fix

typecheck:
	@echo "Running type checker..."
	@uv run pyright

check: lint typecheck
	@echo "All checks passed!"

# Utility commands
tail-log:
	@echo "Following backend logs (Ctrl+C to stop)..."
	@tail -f backend.log

experiment:
	@echo "Starting experiment harness..."
	@python tools/experiment.py

experiment-once:
	@echo "Running experiments once..."
	@python tools/experiment.py --once

migrate:
	@echo "Running database migrations..."
	@./scripts/run-migrations.sh upgrade

create-dev-db:
	@echo "Creating development database maverick_mcp_dev..."
	@docker exec $$(docker ps -qf "name=postgres") \
		psql -U $${POSTGRES_USER:-postgres} \
		-c "CREATE DATABASE maverick_mcp_dev;" 2>/dev/null || \
		echo "Database may already exist or Docker container not found"

migrate-dev:
	@echo "Running migrations on dev database (maverick_mcp_dev)..."
	@set -a; [ -f .env ] && . ./.env; [ -f .env.dev ] && . ./.env.dev; set +a; uv run alembic upgrade head

migrate-test:
	@echo "Running migrations on test database (maverick_mcp_test)..."
	@set -a; [ -f .env ] && . ./.env; [ -f .env.test ] && . ./.env.test; set +a; uv run alembic upgrade head

migrate-prod:
	@echo "Running migrations on production database (maverick_mcp)..."
	@read -p "Confirm production migration? [y/N] " ans && [ "$$ans" = "y" ] || exit 1
	@set -a; [ -f .env ] && . ./.env; [ -f .env.prod ] && . ./.env.prod; set +a; uv run alembic upgrade head

migrate-all: migrate-dev migrate-test migrate-prod
	@echo "All database migrations complete."

setup:
	@echo "Setting up Maverick-MCP..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file - please update with your configuration"; \
	fi
	@uv sync
	@echo "Setup complete! Run 'make dev' to start development."

clean:
	@echo "Cleaning up..."
	@rm -rf .pytest_cache
	@rm -rf htmlcov
	@rm -rf .coverage
	@rm -rf .ruff_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete."

# Service management
redis-start:
	@echo "Starting Redis..."
	@if command -v brew &> /dev/null; then \
		brew services start redis; \
	else \
		redis-server --daemonize yes; \
	fi

redis-stop:
	@echo "Stopping Redis..."
	@if command -v brew &> /dev/null; then \
		brew services stop redis; \
	else \
		pkill redis-server || true; \
	fi

# Quick shortcuts
d: dev
dh: dev-http
ds: dev-stdio
b: backend
t: test
l: lint
c: check

# Performance testing
benchmark-parallel:
	@echo "Benchmarking parallel screening performance..."
	@python -c "from tools.quick_test import test_parallel_screening; import asyncio; asyncio.run(test_parallel_screening())"

benchmark-speed:
	@echo "Running comprehensive speed benchmark..."
	@uv run python scripts/speed_benchmark.py --mode full


# Infrastructure commands (with --env-file .env for credentials)
create-dbs:
	@echo "Creating databases (idempotent — safe to re-run)..."
	@for i in $$(seq 1 30); do \
		docker exec maverick-infra-postgres-1 pg_isready 2>/dev/null && break; \
		echo "Waiting for Postgres... ($$i/30)"; sleep 1; \
	done
	@docker exec maverick-infra-postgres-1 \
		bash /docker-entrypoint-initdb.d/init-db.sh
	@echo "Databases ready: maverick_mcp, maverick_mcp_dev, maverick_mcp_test"

infra-up:
	@echo "Starting infrastructure (Postgres + Redis)..."
	@docker compose --env-file .env -f docker-compose.infra.yml up -d
	@$(MAKE) --no-print-directory create-dbs

infra-down:
	@echo "Stopping infrastructure..."
	@docker compose --env-file .env -f docker-compose.infra.yml down

infra-reset:
	@echo "Resetting infrastructure (wipes volumes)..."
	@docker compose --env-file .env -f docker-compose.infra.yml down -v
	@docker compose --env-file .env -f docker-compose.infra.yml up -d
	@$(MAKE) --no-print-directory create-dbs

# Docker commands
docker-infra-up:
	@echo "Starting infrastructure (Postgres + Redis)..."
	@docker compose -f docker-compose.infra.yml up -d

docker-infra-down:
	@echo "Stopping infrastructure..."
	@docker compose -f docker-compose.infra.yml down

docker-up: docker-infra-up
	@echo "Starting app services..."
	@docker compose up --build -d

docker-down:
	@echo "Stopping app services..."
	@docker compose down

docker-logs:
	@echo "Following Docker logs (Ctrl+C to stop)..."
	@docker compose logs -f