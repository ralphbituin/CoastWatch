"""Verify database connectivity and extension versions.

Usage::

    python -m scripts.check_db
"""

from __future__ import annotations

import sys

from sqlalchemy import text

from app.core.config import settings
from app.database.session import dispose_engine, get_engine


def main() -> int:
    print(f"Connecting to: {settings.sync_database_url.split('@')[-1]}")
    try:
        with get_engine().connect() as conn:
            version = conn.execute(text("SHOW server_version")).scalar_one()
            print(f"server_version        : {version}")
            for ext in ("timescaledb", "postgis"):
                ver = conn.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = :n"),
                    {"n": ext},
                ).scalar_one_or_none()
                print(f"{ext:<22}: {ver or 'NOT INSTALLED'}")
    except Exception as exc:  # noqa: BLE001 - CLI reporting
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        dispose_engine()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
