# Infrastructure Setup

Single Docker Compose stack (`docker-compose.infra.yml`) hosting one PostgreSQL instance and one Redis instance, serving three environments.

## Environment Map

| Environment | PostgreSQL DB         | Redis DB |
|-------------|-----------------------|----------|
| dev         | `maverick_mcp_dev`    | 1        |
| test        | `maverick_mcp_test`   | 2        |
| prod        | `maverick_mcp`        | 0        |

## Required `.env` Variables

These must be set in `.env` before running any infra commands:

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=changeme
```

All three environment files (`.env.dev`, `.env.test`, `.env.prod`) reference these via `${POSTGRES_USER}` / `${POSTGRES_PASSWORD}` substitution, so credentials are defined in one place.

## Environment Files

| File        | Purpose                                      | Git-tracked |
|-------------|----------------------------------------------|-------------|
| `.env`      | Credentials + API keys (source of truth)     | No          |
| `.env.dev`  | Dev overrides (DB, Redis DB, debug flags)    | No          |
| `.env.test` | Test overrides (DB, Redis DB, cache off)     | No          |
| `.env.prod` | Prod overrides (DB, Redis DB, cache TTL)     | No          |
| `.env.example` | Template for `.env`                       | Yes         |

## Makefile Commands

```bash
# Start Postgres + Redis, then create all 3 databases
make infra-up

# Stop containers (preserves volumes/data)
make infra-down

# Wipe volumes and restart (destroys all data, re-runs init)
make infra-reset

# Create databases in already-running container (idempotent)
make create-dbs

# Run migrations per environment
make migrate-dev    # → maverick_mcp_dev
make migrate-test   # → maverick_mcp_test
make migrate-prod   # → maverick_mcp  (prompts for confirmation)
make migrate-all    # → all three in sequence
```

## How Database Initialization Works

`scripts/init-db.sh` is mounted into `/docker-entrypoint-initdb.d/` and runs automatically on first container start with an empty volume. It creates all three databases idempotently.

If the Postgres volume already exists (container restarted, not reset), the init script does **not** run automatically. Use `make create-dbs` to create any missing databases manually — it is safe to re-run.

## Migrations

`migrate-dev` / `migrate-test` / `migrate-prod` merge `.env` (credentials) with the environment-specific file, then call `uv run alembic upgrade head` directly. Alembic reads `DATABASE_URL` from the environment, so the correct database is targeted without touching `.env`.

Migrations are **incremental** — they never drop or truncate existing data. Only `make infra-reset` (which wipes Docker volumes) will destroy data.
