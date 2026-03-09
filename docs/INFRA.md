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

All three environment files reference these via `${POSTGRES_USER}` / `${POSTGRES_PASSWORD}` substitution, so credentials are defined in one place.

## Environment Files

| File           | Purpose                                   | Git-tracked |
|----------------|-------------------------------------------|-------------|
| `.env`         | Credentials + API keys (source of truth)  | No          |
| `.env.dev`     | Dev overrides (DB, Redis DB, debug flags) | No          |
| `.env.test`    | Test overrides (DB, Redis DB, cache off)  | No          |
| `.env.prod`    | Prod overrides (DB, Redis DB, cache TTL)  | No          |
| `.env.example` | Template for `.env`                       | Yes         |

## Makefile Commands

### Infrastructure

```bash
# Start Postgres + Redis, then create all 3 databases
make infra-up

# Stop containers (preserves volumes/data)
make infra-down

# Wipe volumes and restart (destroys all data, re-runs init)
make infra-reset

# Create databases in already-running container (idempotent, safe to re-run)
make create-dbs
```

### Migrations

```bash
make migrate-dev    # → maverick_mcp_dev
make migrate-test   # → maverick_mcp_test
make migrate-prod   # → maverick_mcp  (prompts for confirmation)
make migrate-all    # → all three in sequence
```

Migrations are **incremental** — they never drop or truncate existing data. Only `make infra-reset` (which wipes Docker volumes) will destroy data.

### Data Sync

```bash
make db-sync        # Copy prod data → dev and test
```

## Data Sync: prod → dev/test

`scripts/sync-db.sh` dumps the prod database and restores it into dev and test. The entire operation runs inside the Postgres container.

```
prod (maverick_mcp)
     │ pg_dump -Fc
     ▼
 /tmp/prod.dump  (inside container)
     │ pg_restore
     ├──▶ maverick_mcp_dev
     └──▶ maverick_mcp_test
```

Before each restore, active connections to the target database are terminated via `pg_terminate_backend` so the drop succeeds cleanly.

**Caveats:**
- Sync **fully overwrites** dev/test — any data not in prod will be lost
- Avoid writing to dev/test while sync is running
- Prod is read-only during the process and is never affected

### Scheduled Sync (cron)

```bash
crontab -e
```

Add (runs daily at 02:00):

```cron
0 2 * * * cd /Users/overman/projects/maverick-mcp && make db-sync >> /tmp/maverick-db-sync.log 2>&1
```

## How Database Initialization Works

`scripts/init-db.sh` is mounted into `/docker-entrypoint-initdb.d/` and runs automatically on first container start with an empty volume. It creates all three databases idempotently.

If the Postgres volume already exists (container restarted, not reset), the init script does **not** run automatically. Use `make create-dbs` to create any missing databases — it is safe to re-run.
