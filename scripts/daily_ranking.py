"""
Daily ranking pipeline.

Steps:
  1. Scrape today's Yahoo Finance Japan rankings
     (値上がり率, 出来高, 注目株)
  2. Store snapshots in DuckDB
  3. Identify candidate stocks (top 50 in any ranking for 3+ days)
  4. Auto-fetch price data from J-Quants for new candidates
  5. Run backtest comparison and print results

Usage:
  python scripts/daily_ranking.py
  python scripts/daily_ranking.py --min-days 5 --top-n 30 --backtest
  python scripts/daily_ranking.py --show-only   # just show stored candidates
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table   import Table
from rich         import box

console = Console()

from datetime import date, timedelta

DB_PATH    = "data/market.duckdb"
# Backtest window: a rolling ~2-year lookback ending today (no hardcoded date)
SINCE_DATE = (date.today() - timedelta(days=730)).isoformat()
PRICE_SOURCE = "yfinance"   # free, up-to-date — closes the J-Quants 3-month gap


def _scrape_and_store(tracker) -> dict[str, list[dict]]:
    from bot.ranking.scraper import fetch_all

    console.rule("[bold cyan]Scraping Yahoo Finance Japan Rankings[/]")
    with console.status("Fetching rankings …"):
        rankings = fetch_all(delay=1.5)

    totals = {k: len(v) for k, v in rankings.items()}
    console.print(
        f"  [green]gainers[/]: {totals['gainers']} entries  "
        f"[green]volume[/]: {totals['volume']} entries  "
        f"[green]hot[/]: {totals['hot']} entries"
    )

    today   = date.today().isoformat()
    written = tracker.store(rankings, as_of=today)
    console.print(f"  Stored [bold]{written}[/] rows for {today}")
    return rankings


def _show_candidates(tracker, top_n: int, min_days: int) -> list[str]:
    console.rule("[bold cyan]Ranking Candidates[/]")
    df = tracker.get_candidates(top_n=top_n, min_days=min_days)

    if df.empty:
        console.print(
            f"[yellow]No candidates found "
            f"(top {top_n}, {min_days}+ days in rankings).[/]\n"
            "Tip: run without --show-only for a few days to build history."
        )
        return []

    tbl = Table(
        title=f"Candidates — top {top_n}, ≥{min_days} days",
        box=box.SIMPLE_HEAVY,
    )
    tbl.add_column("Symbol",      style="bold cyan", no_wrap=True)
    tbl.add_column("Name",        max_width=22)
    tbl.add_column("Streak",      justify="right")
    tbl.add_column("Appearances", justify="right")
    tbl.add_column("Avg Rank",    justify="right")
    tbl.add_column("Best Rank",   justify="right")
    tbl.add_column("Types")
    tbl.add_column("Last Seen")

    for _, row in df.iterrows():
        tbl.add_row(
            str(row["symbol"]),
            str(row["name"]) if row["name"] else "—",
            f"[bold green]{int(row['streak'])}d[/]",
            str(int(row["appearances"])),
            f"{row['avg_rank']:.1f}",
            str(int(row["best_rank"])),
            str(row["rank_types"]),
            str(row["last_date"]),
        )

    console.print(tbl)
    return list(df["symbol"].astype(str))


def _fetch_price_data(symbols: list[str]) -> list[str]:
    """Fetch up-to-date price data (Yahoo Finance) for candidate symbols."""
    from bot.data.fetcher import DataFetcher
    from bot.data.store   import DataStore

    fetcher = DataFetcher(PRICE_SOURCE)
    store   = DataStore(DB_PATH)
    fetched = []

    console.rule(f"[bold cyan]Fetching Price Data ({PRICE_SOURCE})[/]")
    for sym in symbols:
        console.print(f"  Fetching [cyan]{sym}[/] …", end=" ")
        try:
            # Always refresh so the most recent bars are present
            df = fetcher.fetch(sym, "1d", since=SINCE_DATE)
            if df.empty:
                console.print("[yellow]no data[/]")
            else:
                store.save(df, sym, "1d")
                console.print(f"[green]{len(df)} bars[/]")
                fetched.append(sym)
        except Exception as exc:
            console.print(f"[red]{exc}[/]")

    return fetched


def _run_backtests(symbols: list[str]) -> None:
    """Run all strategies on candidate symbols and print comparison."""
    import bot.ml  # noqa: F401  — registers ML strategies
    from bot.backtest.engine import BacktestEngine
    from bot.backtest.report import print_comparison
    from bot.config          import SimConfig
    from bot.strategy        import REGISTRY

    if not symbols:
        return

    console.rule("[bold cyan]Strategy Backtest — Candidate Stocks[/]")
    cfg    = SimConfig(market="jp", fee_rate=0.001)
    engine = BacktestEngine(config=cfg, db_path=DB_PATH)
    strats = list(REGISTRY.values())

    for sym in symbols:
        console.print(f"\n[bold]{sym}[/]")
        results = []
        for cls in strats:
            try:
                r = engine.run(cls(), sym, "1d", SINCE_DATE, None)
                results.append(r)
            except Exception:
                pass

        if results:
            print_comparison(results)
        else:
            console.print("  [yellow]Not enough data for backtest[/]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily ranking pipeline")
    parser.add_argument("--top-n",     type=int,  default=50,    help="Top N threshold (default: 50)")
    parser.add_argument("--min-days",  type=int,  default=3,     help="Min days in ranking (default: 3)")
    parser.add_argument("--lookback",  type=int,  default=30,    help="Lookback window in days (default: 30)")
    parser.add_argument("--backtest",  action="store_true",       help="Auto-backtest candidates")
    parser.add_argument("--show-only", action="store_true",       help="Skip scraping, just show candidates")
    parser.add_argument("--no-fetch",  action="store_true",       help="Skip J-Quants data fetch")
    args = parser.parse_args()

    from bot.ranking.tracker import RankingTracker
    tracker = RankingTracker(DB_PATH)

    if not args.show_only:
        _scrape_and_store(tracker)

    candidates = _show_candidates(tracker, top_n=args.top_n, min_days=args.min_days)

    if candidates and args.backtest and not args.no_fetch:
        fetched = _fetch_price_data(candidates)
        _run_backtests(fetched)
    elif candidates and args.backtest:
        _run_backtests(candidates)


if __name__ == "__main__":
    main()
