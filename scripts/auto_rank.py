"""
Daily ranking auto-runner — called hourly by launchd.

Runs the ranking scan only when:
  1. Current JST time is >= 15:30 (market close)
  2. Today's ranking data has NOT already been stored
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

JST      = timezone(timedelta(hours=9))
NOW_JST  = datetime.now(JST)
TODAY    = NOW_JST.strftime("%Y-%m-%d")
HH_MM    = NOW_JST.hour * 60 + NOW_JST.minute

# Only run after 15:30 JST on weekdays
if HH_MM < 930 or NOW_JST.weekday() >= 5:
    sys.exit(0)

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import duckdb

DB_PATH = PROJECT_DIR / "data" / "market.duckdb"
LOG     = PROJECT_DIR / "data" / "auto_rank.log"

# Skip if already fetched today
try:
    conn = duckdb.connect(str(DB_PATH))
    n = conn.execute(
        "SELECT COUNT(*) FROM rankings WHERE date = ?", [TODAY]
    ).fetchone()[0]
    conn.close()
    if n > 0:
        sys.exit(0)
except Exception:
    pass

# Run the scan
import os
os.chdir(PROJECT_DIR)

from rich.console import Console
console = Console(file=open(LOG, "a"))

console.print(f"\n[{NOW_JST.strftime('%Y-%m-%d %H:%M')} JST] Running rank scan …")

try:
    from bot.ranking.scraper import fetch_all
    from bot.ranking.tracker import RankingTracker

    rankings = fetch_all(delay=1.5)
    tracker  = RankingTracker(str(DB_PATH))

    # Guard: on JP market holidays Yahoo serves the prior session's ranking.
    # If today's scrape is identical to the last stored day, skip recording it
    # so holidays don't pollute streak/trajectory with duplicate days.
    stale_of = tracker.matches_last_snapshot(rankings, exclude_date=TODAY)
    if stale_of:
        console.print(f"  [yellow]Stale data — identical to {stale_of} "
                      f"(market likely closed). Skipping store.[/]")
        sys.exit(0)

    written  = tracker.store(rankings, as_of=TODAY)
    totals   = {k: len(v) for k, v in rankings.items()}
    console.print(
        f"  gainers={totals['gainers']}  volume={totals['volume']}  hot={totals['hot']}"
        f"  → {written} rows stored"
    )

    # Score candidates, auto-screen the top ones, and write the HTML report
    from bot.ranking.report import save_candidates_html
    from bot.ranking.screen import screen_candidates
    TOP_N, MIN_DAYS, SCREEN_N = 50, 3, 12
    df    = tracker.get_candidates(top_n=TOP_N, min_days=MIN_DAYS, lookback_days=30)
    dates = tracker.available_dates()

    screen_df = None
    limit_status = {}
    if not df.empty:
        top_syms = list(df["symbol"].astype(str).head(SCREEN_N))
        console.print(f"  Screening {len(top_syms)} candidates (fetch + backtest + OOS) …")
        try:
            screen_df = screen_candidates(top_syms, db_path=str(DB_PATH))
            console.print(f"  Screened {len(screen_df)} symbols")
        except Exception as exc:
            console.print(f"  [yellow]screen failed: {exc}[/]")
        try:
            from bot.ranking.limit import fetch_limit_status
            limit_status = fetch_limit_status(list(df["symbol"].astype(str)))
        except Exception as exc:
            console.print(f"  [yellow]limit status failed: {exc}[/]")

    out = save_candidates_html(
        df, dates, str(PROJECT_DIR / "data" / "candidates.html"),
        top_n=TOP_N, min_days=MIN_DAYS, screen_df=screen_df, limit_status=limit_status,
    )
    console.print(f"  Report → {out} ({len(df)} candidates)")

    # Encrypted personal portfolio report (only if holdings + password exist)
    try:
        from bot.holdings.store  import HoldingsStore
        from bot.holdings.report import save_portfolio_html
        password = os.getenv("PORTFOLIO_PASSWORD", "")
        hdf = HoldingsStore(str(DB_PATH)).list_all()
        if password and not hdf.empty:
            pout = save_portfolio_html(hdf, password, out_path=str(PROJECT_DIR / "data" / "portfolio.html"))
            console.print(f"  Portfolio → {pout} ({len(hdf)} holdings, encrypted)")
        elif not hdf.empty:
            console.print("  [yellow]Portfolio skipped: PORTFOLIO_PASSWORD not set in .env[/]")
    except Exception as exc:
        console.print(f"  [yellow]portfolio failed: {exc}[/]")
except Exception as e:
    console.print(f"  [red]ERROR: {e}[/]")
