#!/usr/bin/env python3
"""
Fetch and store historical stock data.

Examples:
    python scripts/fetch_data.py                               # AAPL 1d, 2023 (Alpaca)
    python scripts/fetch_data.py TSLA 1h 2023-01-01           # US stock, hourly
    python scripts/fetch_data.py 7203 1d 2023-01-01 jquants   # Toyota (J-Quants)
    python scripts/fetch_data.py NVDA 1d 2020-01-01 2024-01-01 alpaca
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from bot.data.fetcher import DataFetcher
from bot.data.store   import DataStore

console = Console()


def main():
    symbol    = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "1d"
    since     = sys.argv[3] if len(sys.argv) > 3 else "2023-01-01"
    # argv[4] can be either 'until' date or 'source' — detect by format
    until     = None
    source    = "alpaca"
    if len(sys.argv) > 4:
        arg4 = sys.argv[4]
        if arg4 in ("alpaca", "jquants"):
            source = arg4
        else:
            until = arg4
    if len(sys.argv) > 5:
        source = sys.argv[5]

    console.rule(f"[bold cyan]Fetching {symbol} [{timeframe}] from {source}[/]")
    console.print(f"Period: {since} → {until or 'now'}")

    fetcher = DataFetcher(source)
    df      = fetcher.fetch(symbol, timeframe, since=since, until=until)

    if df.empty:
        console.print("[red]No data returned.[/]")
        return

    console.print(f"Fetched [green]{len(df):,}[/] bars  "
                  f"({df['timestamp'].min()} … {df['timestamp'].max()})")

    store   = DataStore()
    written = store.save(df, symbol, timeframe)
    console.print(f"[green]Saved {written:,} bars to data/market.duckdb[/]")


if __name__ == "__main__":
    main()
