#!/usr/bin/env python3
"""
Scan a universe of symbols with a strategy and rank by performance.

Examples:
    python scripts/scan.py --universe tech --strategy sma_crossover --since 2023-01-01
    python scripts/scan.py --symbols AAPL,MSFT,GOOG --strategy macd --since 2023-01-01
    python scripts/scan.py --universe n225 --strategy macd --since 2023-01-01 --market jp
    python scripts/scan.py --universe tech --strategy all --since 2023-01-01 --top-n 10
    python scripts/scan.py --universe tech --strategy sma_crossover --since 2023-01-01 --fetch
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

from bot.backtest.engine import BacktestEngine
from bot.config import SimConfig
import bot.ml  # noqa: F401 — registers ML strategies
from bot.strategy import REGISTRY
from bot.universes import UNIVERSES

console = Console()


def parse_args():
    p = argparse.ArgumentParser(
        description="Scan a universe of symbols and rank by strategy performance."
    )
    p.add_argument("--symbols",   help="Comma-separated list of symbols (e.g. AAPL,MSFT)")
    p.add_argument("--universe",  help=f"Predefined universe: {list(UNIVERSES)}")
    p.add_argument("--strategy",  default="sma_crossover",
                   help="Strategy name or 'all' to run every registered strategy")
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--since",     required=True, help="Start date ISO-8601 (e.g. 2023-01-01)")
    p.add_argument("--until",     default=None,  help="End date ISO-8601 (default: now)")
    p.add_argument("--market",    default="us",  help="'us' or 'jp'")
    p.add_argument("--top-n",     type=int, default=0,
                   help="Show top N results (0 = all)")
    p.add_argument("--sort-by",   default="sharpe",
                   help="Column to sort by: sharpe|return_pct|calmar|max_dd_pct")
    p.add_argument("--fetch",     action="store_true",
                   help="Auto-fetch missing symbols from the data source")
    p.add_argument("--source",    default="alpaca",
                   help="Data source for --fetch: 'alpaca' or 'jquants'")
    p.add_argument("--db-path",   default="data/market.duckdb")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve symbol list ────────────────────────────────────────────────
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.universe:
        if args.universe not in UNIVERSES:
            console.print(
                f"[red]Unknown universe '{args.universe}'. "
                f"Available: {list(UNIVERSES)}[/]"
            )
            sys.exit(1)
        symbols = UNIVERSES[args.universe]
    else:
        console.print("[red]Provide --symbols or --universe.[/]")
        sys.exit(1)

    # ── Resolve strategy list ──────────────────────────────────────────────
    if args.strategy == "all":
        strategy_names = list(REGISTRY.keys())
    elif args.strategy not in REGISTRY:
        console.print(
            f"[red]Unknown strategy '{args.strategy}'. "
            f"Available: {list(REGISTRY)}[/]"
        )
        sys.exit(1)
    else:
        strategy_names = [args.strategy]

    # ── Optional auto-fetch ────────────────────────────────────────────────
    if args.fetch:
        from bot.data.fetcher import DataFetcher
        from bot.data.store import DataStore
        fetcher = DataFetcher(args.source)
        store   = DataStore(args.db_path)
        for sym in symbols:
            console.print(f"  Fetching [cyan]{sym}[/] …", end=" ")
            try:
                df = fetcher.fetch(sym, args.timeframe,
                                   since=args.since, until=args.until)
                if df.empty:
                    console.print("[yellow]no data[/]")
                else:
                    written = store.save(df, sym, args.timeframe)
                    console.print(f"[green]{written} bars[/]")
            except Exception as exc:
                console.print(f"[red]{exc}[/]")

    # ── Run backtests ──────────────────────────────────────────────────────
    cfg    = SimConfig(market=args.market)
    engine = BacktestEngine(config=cfg, db_path=args.db_path)

    rows: list[dict] = []
    skipped = 0

    with console.status(
        f"Running {len(strategy_names)} strategy × {len(symbols)} symbols …"
    ):
        for sym in symbols:
            for strat_name in strategy_names:
                try:
                    strat  = REGISTRY[strat_name]()
                    result = engine.run(
                        strat, sym, args.timeframe, args.since, args.until
                    )
                    m = result.metrics
                    rows.append({
                        "symbol":     sym,
                        "strategy":   strat_name,
                        "return_pct": m["total_return_pct"],
                        "cagr_pct":   m["cagr_pct"],
                        "sharpe":     m["sharpe"],
                        "calmar":     m["calmar"],
                        "max_dd_pct": m["max_drawdown_pct"],
                        "win_rate":   m["win_rate_pct"],
                        "trades":     m["total_trades"],
                        "bh_return":  m["bh_return_pct"],
                    })
                except ValueError as exc:
                    if "No data" in str(exc):
                        skipped += 1
                    else:
                        console.print(f"[yellow]  {sym}/{strat_name}: {exc}[/]")
                except Exception as exc:
                    console.print(f"[yellow]  {sym}/{strat_name}: {exc}[/]")

    if skipped:
        console.print(
            f"[yellow]{skipped} symbol(s) skipped (no data — run with --fetch)[/]"
        )

    if not rows:
        console.print("[red]No results.[/]")
        return

    # ── Sort and optionally truncate ───────────────────────────────────────
    sort_col = args.sort_by
    if sort_col not in rows[0]:
        console.print(
            f"[yellow]Unknown sort column '{sort_col}', defaulting to 'sharpe'.[/]"
        )
        sort_col = "sharpe"

    ascending = sort_col == "max_dd_pct"   # lower drawdown is better
    df_res = pd.DataFrame(rows).sort_values(sort_col, ascending=ascending)
    if args.top_n > 0:
        df_res = df_res.head(args.top_n)

    # ── Print table ────────────────────────────────────────────────────────
    multi_strat = len(strategy_names) > 1
    table = Table(
        title=f"Scan Results — sorted by {sort_col}",
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("Symbol",   style="bold cyan")
    if multi_strat:
        table.add_column("Strategy")
    table.add_column("Return",   justify="right")
    table.add_column("CAGR",     justify="right")
    table.add_column("Sharpe",   justify="right")
    table.add_column("Calmar",   justify="right")
    table.add_column("MaxDD",    justify="right")
    table.add_column("WinRate",  justify="right")
    table.add_column("Trades",   justify="right")
    table.add_column("B&H",      justify="right")

    def _c(val: float, suffix: str = "", pos_good: bool = True) -> str:
        color = "green" if (val > 0) == pos_good else "red"
        return f"[{color}]{val:+.2f}{suffix}[/]"

    for _, row in df_res.iterrows():
        cells = [row["symbol"]]
        if multi_strat:
            cells.append(row["strategy"])
        cells += [
            _c(row["return_pct"], "%"),
            _c(row["cagr_pct"], "%"),
            _c(row["sharpe"]),
            _c(row["calmar"]),
            f"[red]{row['max_dd_pct']:.2f}%[/]",
            f"{row['win_rate']:.1f}%",
            str(int(row["trades"])),
            _c(row["bh_return"], "%"),
        ]
        table.add_row(*cells)

    console.print("\n")
    console.print(table)
    console.print(
        f"[dim]{len(df_res)} result(s) shown"
        + (f" of {len(rows)}" if len(rows) > len(df_res) else "")
        + "[/]"
    )


if __name__ == "__main__":
    main()
