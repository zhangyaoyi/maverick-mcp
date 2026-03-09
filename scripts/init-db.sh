#!/bin/bash
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
  SELECT 'CREATE DATABASE maverick_mcp'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'maverick_mcp')\gexec
  SELECT 'CREATE DATABASE maverick_mcp_dev'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'maverick_mcp_dev')\gexec
  SELECT 'CREATE DATABASE maverick_mcp_test'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'maverick_mcp_test')\gexec
EOSQL
