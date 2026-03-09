#!/bin/bash
# Dump prod DB and restore into dev + test (runs entirely inside the Postgres container)
set -e

CONTAINER=maverick-infra-postgres-1
PG_USER=${POSTGRES_USER:-postgres}

echo "→ Checking container is running..."
if ! docker exec "$CONTAINER" pg_isready -U "$PG_USER" >/dev/null 2>&1; then
    echo "Error: Postgres container '$CONTAINER' is not ready" >&2
    exit 1
fi

echo "→ Dumping maverick_mcp (prod)..."
docker exec "$CONTAINER" pg_dump -U "$PG_USER" -Fc -f /tmp/prod.dump maverick_mcp

echo "→ Restoring to maverick_mcp_dev..."
docker exec "$CONTAINER" bash -c "
    psql -U $PG_USER -d postgres -c \"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='maverick_mcp_dev' AND pid <> pg_backend_pid();\" &&
    dropdb -U $PG_USER --if-exists maverick_mcp_dev &&
    createdb -U $PG_USER maverick_mcp_dev &&
    pg_restore -U $PG_USER -d maverick_mcp_dev /tmp/prod.dump
"

echo "→ Restoring to maverick_mcp_test..."
docker exec "$CONTAINER" bash -c "
    psql -U $PG_USER -d postgres -c \"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='maverick_mcp_test' AND pid <> pg_backend_pid();\" &&
    dropdb -U $PG_USER --if-exists maverick_mcp_test &&
    createdb -U $PG_USER maverick_mcp_test &&
    pg_restore -U $PG_USER -d maverick_mcp_test /tmp/prod.dump
"

docker exec "$CONTAINER" rm /tmp/prod.dump

echo "✓ Sync complete: prod → dev, test"
