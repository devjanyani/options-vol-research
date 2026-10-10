"""
Save today's option chains to the local SQLite database.

Run once per trading day, ideally between 3:30 and 3:55 pm New York time:

    python scripts/collect_snapshot.py
    python scripts/collect_snapshot.py --tickers SPY QQQ --max-expirations 6

Weekends are skipped unless you pass --force (weekend quotes are just
Friday's stale quotes).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.data.market_data import get_default_provider  # noqa: E402
from app.data.repository import Repository  # noqa: E402
from app.data.snapshots import collect_snapshot  # noqa: E402
from app.utils.dates import is_market_open, market_now  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=config.DEFAULT_TICKERS)
    ap.add_argument("--max-expirations", type=int, default=8)
    ap.add_argument("--max-days", type=int, default=120, help="Only expirations within this many days")
    ap.add_argument("--db", default=str(config.DB_PATH))
    ap.add_argument("--force", action="store_true", help="Collect even on weekends")
    args = ap.parse_args()

    now = market_now()
    if now.weekday() >= 5 and not args.force:
        print(f"{now:%A}: markets are closed, nothing collected (use --force to override).")
        return 0
    if not is_market_open(now):
        print("Note: the market is not open right now; bid/ask quotes may be stale or zero.")

    provider, repo = get_default_provider(), Repository(args.db)
    exit_code = 0
    for t in args.tickers:
        try:
            rep = collect_snapshot(provider, repo, t, args.max_expirations, args.max_days)
            print(f"{rep.ticker:6s} {rep.snapshot_date}: {rep.expirations_saved} expirations, "
                  f"{rep.contracts_saved} contracts ({rep.liquid_contracts} liquid)")
            for e in rep.errors:
                print(f"   ! {e}")
        except Exception as exc:  # noqa: BLE001
            print(f"{t:6s} FAILED: {type(exc).__name__}: {exc}")
            exit_code = 1
    print(f"Database: {args.db}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
