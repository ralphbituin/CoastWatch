# CoastWatch backend scripts

Operational utilities. Run from the `backend/` directory with the virtualenv
active, e.g. `python -m scripts.check_db`.

| Script | Purpose |
|---|---|
| `check_db.py` | Verify DB connectivity and extension versions |
| `import_tide_csv.py` | CLI wrapper around CSV tide import (Phase 2 stub) |
| `entrypoint.sh` | Container entrypoint: wait for DB, run migrations, start API |
