#!/usr/bin/env python3
"""
Run a backtest and print results.

Examples:
    python scripts/run_backtest.py                                          # all strategies, AAPL 1d
    python scripts/run_backtest.py sma_crossover
    python scripts/run_backtest.py macd AAPL 1d 2023-01-01 2024-01-01 --report
    python scripts/run_backtest.py macd AAPL 1d 2023-01-01 --walkforward
    python scripts/run_backtest.py supervised AAPL 1d 2023-01-01 --report
    # Japanese stocks
    python scripts/run_backtest.py sma_crossover 7203 1d 2023-01-01 --market jp
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from bot.backtest.engine      import BacktestEngine
from bot.backtest.report      import print_comparison, print_report, save_comparison_html, save_html_report
from bot.backtest.walkforward import WalkForwardValidator
from bot.config               import SimConfig
import bot.ml  # registers ML strategies into bot.strategy.REGISTRY
from bot.strategy             import REGISTRY

console = Console()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("strategy",        nargs="?", default="all")
    p.add_argument("symbol",          nargs="?", default="AAPL")
    p.add_argument("timeframe",       nargs="?", default="1d")
    p.add_argument("since",           nargs="?", default="2023-01-01")
    p.add_argument("until",           nargs="?", default=None)
    p.add_argument("--market",        default="us",  help="'us' or 'jp'")
    p.add_argument("--sizing",        default="full",
                   help="Position sizing: full | fixed_frac | kelly")
    p.add_argument("--risk-per-trade", type=float, default=0.02,
                   help="Fraction of equity per trade (fixed_frac/kelly)")
    p.add_argument("--stop-loss",     type=float, default=0.0,
                   help="Stop-loss fraction below entry (0=off, e.g. 0.05=5%%)")
    p.add_argument("--take-profit",   type=float, default=0.0,
                   help="Take-profit fraction above entry (0=off, e.g. 0.10=10%%)")
    p.add_argument("--report",        action="store_true")
    p.add_argument("--walkforward",   action="store_true", help="Run walk-forward validation")
    p.add_argument("--wf-windows",    type=int, default=6)
    p.add_argument("--wf-anchored",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    cfg    = SimConfig(
        market=args.market,
        position_sizing=args.sizing,
        risk_per_trade=args.risk_per_trade,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
    )
    engine = BacktestEngine(config=cfg)

    if args.strategy == "all":
        strategies = [cls() for cls in REGISTRY.values()]
        with console.status(f"Running {len(strategies)} strategies …"):
            results = engine.compare(strategies, args.symbol, args.timeframe, args.since, args.until)
        print_comparison(results)
        if args.report:
            Path("data").mkdir(exist_ok=True)
            out = f"data/comparison_{args.symbol}_{args.timeframe}.html"
            save_comparison_html(results, out)
    else:
        if args.strategy not in REGISTRY:
            console.print(f"[red]Unknown strategy. Available: {list(REGISTRY)}[/]")
            sys.exit(1)

        strat  = REGISTRY[args.strategy]()
        result = engine.run(strat, args.symbol, args.timeframe, args.since, args.until)
        print_report(result)

        if args.report:
            Path("data").mkdir(exist_ok=True)
            out = f"data/report_{args.strategy}_{args.symbol}_{args.timeframe}.html"
            save_html_report(result, out)

        if args.walkforward:
            console.rule("[bold cyan]Walk-Forward Validation[/]")
            validator = WalkForwardValidator(
                engine,
                n_windows=args.wf_windows,
                anchored=args.wf_anchored,
            )
            wf = validator.run(
                strategy_factory = lambda: REGISTRY[args.strategy](),
                symbol           = args.symbol,
                timeframe        = args.timeframe,
                since            = args.since,
                until            = args.until or "2099-01-01",
            )
            validator.print_report(wf)
            if args.report:
                Path("data").mkdir(exist_ok=True)
                out = f"data/wf_{args.strategy}_{args.symbol}.html"
                validator.save_html_report(wf, out)


if __name__ == "__main__":
    main()
