#!/usr/bin/env bash
# Container entrypoint: wait for the database, run migrations, start the API.
set -euo pipefail

echo "Waiting for database at ${POSTGRES_HOST:-db}:${POSTGRES_PORT:-5432}..."
python - <<'PY'
import os
import sys
import time

import psycopg2

host = os.getenv("POSTGRES_HOST", "db")
port = int(os.getenv("POSTGRES_PORT", "5432"))
user = os.getenv("POSTGRES_USER", "coastwatch")
password = os.getenv("POSTGRES_PASSWORD", "coastwatch")
dbname = os.getenv("POSTGRES_DB", "coastwatch")

for attempt in range(60):
    try:
        psycopg2.connect(host=host, port=port, user=user,
                         password=password, dbname=dbname, connect_timeout=3).close()
        print("Database is ready.")
        sys.exit(0)
    except Exception:
        print(f"  ...not ready yet ({attempt + 1}/60)")
        time.sleep(2)
print("Database did not become ready in time.", file=sys.stderr)
sys.exit(1)
PY

echo "Applying Alembic migrations..."
alembic upgrade head

echo "Starting API..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
