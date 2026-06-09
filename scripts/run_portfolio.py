#!/usr/bin/env python3
"""
Run a portfolio backtest across multiple symbols.

Examples:
    python scripts/run_portfolio.py --strategy sma_crossover --universe tech --since 2023-01-01
    python scripts/run_portfolio.py --strategy macd --symbols AAPL,MSFT,NVDA --since 2023-01-01
    python scripts/run_portfolio.py --strategy macd --universe tech --since 2023-01-01 --allocation risk_parity --report
    python scripts/run_portfolio.py --strategy sma_crossover --universe n225 --since 2023-01-01 --market jp
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from bot.backtest.engine    import BacktestEngine
from bot.backtest.portfolio import PortfolioEngine, print_portfolio_report, save_portfolio_html
from bot.config             import SimConfig
import bot.ml  # noqa: F401 — registers ML strategies
from bot.strategy           import REGISTRY
from bot.universes          import UNIVERSES

console = Console()


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-symbol portfolio backtest."
    )
    p.add_argument("--symbols",    help="Comma-separated symbols (e.g. AAPL,MSFT)")
    p.add_argument("--universe",   help=f"Predefined universe: {list(UNIVERSES)}")
    p.add_argument("--strategy",   default="sma_crossover")
    p.add_argument("--timeframe",  default="1d")
    p.add_argument("--since",      required=True)
    p.add_argument("--until",      default=None)
    p.add_argument("--capital",    type=float, default=10_000.0)
    p.add_argument("--market",     default="us", help="'us' or 'jp'")
    p.add_argument("--allocation", default="equal_weight",
                   help="equal_weight | risk_parity")
    p.add_argument("--report",     action="store_true",
                   help="Save an HTML equity-curve report")
    p.add_argument("--db-path",    default="data/market.duckdb")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve symbol list ────────────────────────────────────────────────
    if args.symbols:
        sym_list = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.universe:
        if args.universe not in UNIVERSES:
            console.print(
                f"[red]Unknown universe '{args.universe}'. "
                f"Available: {list(UNIVERSES)}[/]"
            )
            sys.exit(1)
        sym_list = UNIVERSES[args.universe]
    else:
        console.print("[red]Provide --symbols or --universe.[/]")
        sys.exit(1)

    if args.strategy not in REGISTRY:
        console.print(
            f"[red]Unknown strategy '{args.strategy}'. "
            f"Available: {list(REGISTRY)}[/]"
        )
        sys.exit(1)

    # ── Run portfolio backtest ─────────────────────────────────────────────
    cfg         = SimConfig(initial_capital=args.capital, market=args.market)
    bt_engine   = BacktestEngine(config=cfg, db_path=args.db_path)
    port_engine = PortfolioEngine(bt_engine)

    console.print(
        f"Portfolio: [cyan]{args.strategy}[/] · {len(sym_list)} symbols "
        f"· allocation=[bold]{args.allocation}[/]"
    )

    with console.status("Running portfolio backtest …"):
        result = port_engine.run(
            strategy_factory = lambda: REGISTRY[args.strategy](),
            symbols          = sym_list,
            timeframe        = args.timeframe,
            since            = args.since,
            until            = args.until,
            allocation       = args.allocation,
        )

    print_portfolio_report(result)

    if args.report:
        Path("data").mkdir(exist_ok=True)
        tag = args.universe or "custom"
        out = f"data/portfolio_{args.strategy}_{tag}_{args.timeframe}.html"
        save_portfolio_html(result, out)


if __name__ == "__main__":
    main()
