"""
Multi-symbol portfolio backtester.

Runs the same strategy on every symbol in a universe, splits initial capital
by allocation method, and combines individual equity curves into a single
portfolio equity curve.

Allocation methods
------------------
equal_weight   1/N per symbol (default)
risk_parity    inversely proportional to realised price volatility
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

from bot.config import ANNUAL_BARS, ANNUAL_BARS_JP

if TYPE_CHECKING:
    from bot.backtest.engine import BacktestEngine, BacktestResult
    from bot.strategy.base import BaseStrategy

console = Console()

_ANNUAL_BARS_BY_MARKET = {"us": ANNUAL_BARS, "jp": ANNUAL_BARS_JP}


@dataclass
class PortfolioResult:
    strategy_name:    str
    symbols:          list[str]
    timeframe:        str
    since:            str
    end:              str
    allocation:       str
    weights:          dict[str, float]
    per_symbol:       dict[str, "BacktestResult"]
    portfolio_values: np.ndarray
    initial_capital:  float
    metrics:          dict = field(default_factory=dict)


class PortfolioEngine:
    """
    Wraps a BacktestEngine, running the same strategy on multiple symbols
    and combining results according to the chosen allocation scheme.
    """

    def __init__(self, backtest_engine: "BacktestEngine"):
        self.engine = backtest_engine

    def run(
        self,
        strategy_factory: Callable[[], "BaseStrategy"],
        symbols: list[str],
        timeframe: str,
        since: str,
        until: Optional[str] = None,
        allocation: str = "equal_weight",
    ) -> PortfolioResult:
        cfg = self.engine.cfg

        # ── Collect a strategy name without consuming the factory ───────────
        _probe = strategy_factory()
        strat_name = _probe.name

        # ── Individual backtests (each using full initial_capital) ──────────
        per_symbol: dict[str, BacktestResult] = {}
        for sym in symbols:
            try:
                result = self.engine.run(
                    strategy_factory(), sym, timeframe, since, until
                )
                per_symbol[sym] = result
            except ValueError:
                console.print(f"[yellow]  Skip {sym}: no data[/]")

        if not per_symbol:
            raise ValueError("No data available for any of the requested symbols.")

        active = list(per_symbol.keys())

        # ── Compute allocation weights ──────────────────────────────────────
        weights = self._weights(per_symbol, allocation)

        # ── Scale and combine equity curves ────────────────────────────────
        # Each symbol's equity curve starts at cfg.initial_capital.
        # We rescale to weight * initial_capital, then sum.
        # Because pv[0] == initial_capital always (no trade at bar 0),
        # the scaling factor is simply the weight.
        frames: dict[str, pd.Series] = {}
        for sym, r in per_symbol.items():
            ts  = r.df["timestamp"].to_numpy(dtype=np.int64)
            pv  = r.portfolio_values * weights[sym]
            frames[sym] = pd.Series(pv, index=ts, name=sym)

        pv_df  = pd.DataFrame(frames).sort_index().ffill()
        port_pv = pv_df[active].sum(axis=1).to_numpy(dtype=float)

        # ── Portfolio metrics ───────────────────────────────────────────────
        from bot.backtest.report import compute_metrics

        bars_map    = _ANNUAL_BARS_BY_MARKET.get(cfg.market, ANNUAL_BARS)
        annual_bars = bars_map.get(timeframe, cfg.annual_bars)

        all_trades = []
        for r in per_symbol.values():
            all_trades.extend(r.trades)
        all_trades.sort(key=lambda t: t["timestamp"])

        # Equal-weight buy-and-hold B&H return
        bh_parts = []
        for sym in active:
            df_s   = per_symbol[sym].df
            bh_ret = df_s["close"].iloc[-1] / df_s["close"].iloc[0]
            bh_parts.append(bh_ret * weights[sym])
        bh_pct = (sum(bh_parts) - 1) * 100

        first_df = next(iter(per_symbol.values())).df
        metrics  = compute_metrics(
            port_pv, all_trades, first_df, annual_bars,
            bh_return_pct=bh_pct,
        )

        end = until or str(max(r.df["timestamp"].max() for r in per_symbol.values()))

        return PortfolioResult(
            strategy_name    = strat_name,
            symbols          = active,
            timeframe        = timeframe,
            since            = since,
            end              = end,
            allocation       = allocation,
            weights          = weights,
            per_symbol       = per_symbol,
            portfolio_values = port_pv,
            initial_capital  = cfg.initial_capital,
            metrics          = metrics,
        )

    # ── Weight calculators ─────────────────────────────────────────────────

    def _weights(
        self,
        per_symbol: dict[str, "BacktestResult"],
        allocation: str,
    ) -> dict[str, float]:
        syms = list(per_symbol.keys())
        n    = len(syms)

        if allocation == "equal_weight" or n == 1:
            return {s: 1 / n for s in syms}

        if allocation == "risk_parity":
            vols: dict[str, float] = {}
            for sym, r in per_symbol.items():
                close = r.df["close"].to_numpy(dtype=float)
                rets  = np.diff(close) / np.where(close[:-1] > 0, close[:-1], 1)
                vols[sym] = float(np.std(rets)) if len(rets) > 1 else 1.0
            inv   = {s: 1 / max(v, 1e-9) for s, v in vols.items()}
            total = sum(inv.values())
            return {s: v / total for s, v in inv.items()}

        # Unknown allocation → fallback to equal
        console.print(f"[yellow]Unknown allocation '{allocation}', using equal_weight.[/]")
        return {s: 1 / n for s in syms}


# ── Terminal report ────────────────────────────────────────────────────────────

def print_portfolio_report(result: PortfolioResult) -> None:
    m = result.metrics
    console.print(
        f"\n[bold cyan]{result.strategy_name}[/] "
        f"· {len(result.symbols)} symbols · {result.timeframe} "
        f"· {result.since} → {result.end} "
        f"· allocation=[bold]{result.allocation}[/]\n"
    )

    def _color(val: float, pos_good: bool = True) -> str:
        return "[green]" if (val > 0) == pos_good else "[red]"

    # Portfolio-level metrics
    perf = Table(title="Portfolio Performance", box=box.SIMPLE_HEAVY, show_header=False)
    perf.add_column("Metric", style="dim")
    perf.add_column("Value",  justify="right")
    rows = [
        ("Total return",           f"{_color(m['total_return_pct'])}{m['total_return_pct']:+.2f}%[/]"),
        ("CAGR",                   f"{_color(m['cagr_pct'])}{m['cagr_pct']:+.2f}%[/]"),
        ("Equal-weight B&H",       f"{_color(m['bh_return_pct'])}{m['bh_return_pct']:+.2f}%[/]"),
        ("Sharpe ratio",           f"{_color(m['sharpe'])}{m['sharpe']:.3f}[/]"),
        ("Sortino ratio",          f"{_color(m['sortino'])}{m['sortino']:.3f}[/]"),
        ("Calmar ratio",           f"{_color(m['calmar'])}{m['calmar']:.3f}[/]"),
        ("Max drawdown",           f"[red]{m['max_drawdown_pct']:.2f}%[/]"),
        ("Win rate (all trades)",  f"{m['win_rate_pct']:.2f}%"),
        ("Profit factor",          f"{m['profit_factor']:.3f}"),
        ("Total trades",           str(m["total_trades"])),
        ("Final equity",           f"${m['final_equity']:,.2f}"),
    ]
    for label, value in rows:
        perf.add_row(label, value)
    console.print(perf)

    # Per-symbol summary
    sym_tbl = Table(title="Per-Symbol Summary", box=box.SIMPLE_HEAVY)
    sym_tbl.add_column("Symbol",  style="bold")
    sym_tbl.add_column("Weight",  justify="right")
    sym_tbl.add_column("Return",  justify="right")
    sym_tbl.add_column("Sharpe",  justify="right")
    sym_tbl.add_column("MaxDD",   justify="right")
    sym_tbl.add_column("Trades",  justify="right")
    sym_tbl.add_column("B&H",     justify="right")

    def _c(v: float, sfx: str = "", pos_good: bool = True) -> str:
        color = "green" if (v > 0) == pos_good else "red"
        return f"[{color}]{v:+.2f}{sfx}[/]"

    for sym in result.symbols:
        r  = result.per_symbol[sym]
        sm = r.metrics
        w  = result.weights[sym]
        sym_tbl.add_row(
            sym,
            f"{w:.1%}",
            _c(sm["total_return_pct"], "%"),
            _c(sm["sharpe"]),
            f"[red]{sm['max_drawdown_pct']:.2f}%[/]",
            str(sm["total_trades"]),
            _c(sm["bh_return_pct"], "%"),
        )

    console.print(sym_tbl)


# ── HTML report ────────────────────────────────────────────────────────────────

def save_portfolio_html(result: PortfolioResult, path: str = "portfolio.html") -> str:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        console.print("[yellow]plotly not installed — skipping HTML report[/]")
        return ""

    # Collect timestamps from the first symbol (all aligned to same after ffill)
    first_sym = result.symbols[0]
    xs = result.per_symbol[first_sym].df["timestamp"].tolist()
    pv = result.portfolio_values

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=["Equity Curves (indexed to 100)", "Portfolio Drawdown"],
        vertical_spacing=0.08,
    )

    # Portfolio equity indexed to 100
    start = pv[0] if pv[0] > 0 else 1
    fig.add_trace(go.Scatter(
        x=xs, y=(pv / start * 100).tolist(),
        name="Portfolio", line=dict(color="#2196F3", width=3),
    ), row=1, col=1)

    # Individual symbols (thinner lines)
    palette = [
        "#ef5350", "#26a69a", "#ab47bc", "#ff7043",
        "#66bb6a", "#ffa726", "#29b6f6", "#ec407a",
        "#8d6e63", "#78909c",
    ]
    for idx, sym in enumerate(result.symbols):
        r      = result.per_symbol[sym]
        sym_xs = r.df["timestamp"].tolist()
        sym_pv = r.portfolio_values * result.weights[sym]
        s0     = sym_pv[0] if sym_pv[0] > 0 else 1
        fig.add_trace(go.Scatter(
            x=sym_xs, y=(sym_pv / s0 * 100).tolist(),
            name=sym,
            line=dict(color=palette[idx % len(palette)], width=1, dash="dot"),
            opacity=0.7,
        ), row=1, col=1)

    # Portfolio drawdown
    running_max = np.maximum.accumulate(pv)
    dd = (pv - running_max) / np.where(running_max > 0, running_max, 1) * 100
    fig.add_trace(go.Scatter(
        x=xs, y=dd.tolist(), name="Drawdown %",
        line=dict(color="#ef5350", width=1),
        fill="tozeroy", fillcolor="rgba(239,83,80,0.2)",
    ), row=2, col=1)

    m = result.metrics
    fig.update_layout(
        title=(
            f"Portfolio: {result.strategy_name} | {len(result.symbols)} symbols "
            f"({result.allocation}) | "
            f"Return: {m['total_return_pct']:+.2f}% | "
            f"Sharpe: {m['sharpe']:.3f} | "
            f"MaxDD: {m['max_drawdown_pct']:.2f}%"
        ),
        template="plotly_dark",
        height=800,
        showlegend=True,
    )

    fig.write_html(path)
    console.print(f"[green]Portfolio report saved → {path}[/]")
    return path
