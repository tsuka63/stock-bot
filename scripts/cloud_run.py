"""
Cloud daily run (GitHub Actions) — rankings + screening only.

Unlike scripts/auto_rank.py (which runs locally on a schedule and also builds
the encrypted portfolio), this:
  • uses a SEPARATE rankings-only DB (data/rankings.duckdb) so no personal
    holdings ever reach the public repo,
  • writes the public report to docs/index.html (served by GitHub Pages),
  • does NOT touch the portfolio,
  • has no time-of-day guard (the workflow's cron handles timing).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from rich.console import Console

console = Console()

JST     = timezone(timedelta(hours=9))
TODAY   = datetime.now(JST).strftime("%Y-%m-%d")
DB_PATH = str(PROJECT_DIR / "data" / "rankings.duckdb")
OUT     = str(PROJECT_DIR / "docs" / "index.html")

TOP_N, MIN_DAYS, SCREEN_N = 50, 3, 12


def main() -> int:
    from bot.ranking.scraper import fetch_all
    from bot.ranking.tracker import RankingTracker
    from bot.ranking.report  import save_candidates_html
    from bot.ranking.screen  import screen_candidates
    from bot.ranking.limit   import fetch_limit_status

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    tracker = RankingTracker(DB_PATH)

    console.print(f"[{TODAY}] Scraping Yahoo Finance Japan rankings …")
    rankings = fetch_all(delay=1.5)

    # Holiday / stale guard: skip storing a duplicate of the last session
    stale_of = tracker.matches_last_snapshot(rankings, exclude_date=TODAY)
    if stale_of:
        console.print(f"  Stale data — identical to {stale_of} (market closed). Not storing.")
    else:
        written = tracker.store(rankings, as_of=TODAY)
        totals  = {k: len(v) for k, v in rankings.items()}
        console.print(
            f"  gainers={totals['gainers']} volume={totals['volume']} "
            f"hot={totals['hot']} → {written} rows stored"
        )

    df    = tracker.get_candidates(top_n=TOP_N, min_days=MIN_DAYS, lookback_days=30)
    dates = tracker.available_dates()

    screen_df = None
    limit_status = {}
    if not df.empty:
        top_syms = list(df["symbol"].astype(str).head(SCREEN_N))
        console.print(f"  Screening {len(top_syms)} candidates (yfinance + backtest + OOS) …")
        try:
            screen_df = screen_candidates(top_syms, db_path=DB_PATH)
            console.print(f"  Screened {len(screen_df)} symbols")
        except Exception as exc:
            console.print(f"  [yellow]screen failed: {exc}[/]")

        # ストップ高/安 status for every candidate shown
        try:
            limit_status = fetch_limit_status(list(df["symbol"].astype(str)))
            n_high = sum(1 for v in limit_status.values() if v == "high")
            console.print(f"  Limit status: {len(limit_status)} checked, {n_high} ストップ高")
        except Exception as exc:
            console.print(f"  [yellow]limit status failed: {exc}[/]")

    save_candidates_html(df, dates, OUT, top_n=TOP_N, min_days=MIN_DAYS,
                         screen_df=screen_df, limit_status=limit_status)
    console.print(f"  Report → {OUT} ({len(df)} candidates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
