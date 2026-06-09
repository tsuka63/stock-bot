"""
Performance metrics and visualisation for backtest results.
"""

from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from bot.backtest.engine import BacktestResult

console = Console()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    portfolio_values: np.ndarray,
    trades: list[dict],
    df: pd.DataFrame,
    annual_bars: int = 252,
    bh_return_pct: Optional[float] = None,   # override for portfolio-level B&H
) -> dict:
    pv = np.asarray(portfolio_values, dtype=float)
    n  = len(pv)
    if n == 0:
        return {}

    initial = pv[0]
    final   = pv[-1]

    total_return_pct = (final / initial - 1) * 100 if initial else 0.0
    bar_returns      = np.diff(pv) / pv[:-1]
    bar_returns      = np.where(np.isfinite(bar_returns), bar_returns, 0)

    years = n / annual_bars
    cagr  = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 and initial > 0 else 0.0

    mean_r = np.mean(bar_returns)
    std_r  = np.std(bar_returns, ddof=1)
    sharpe = (mean_r / std_r * math.sqrt(annual_bars)) if std_r > 0 else 0.0

    neg_returns = bar_returns[bar_returns < 0]
    down_std    = np.std(neg_returns, ddof=1) if len(neg_returns) > 1 else 0.0
    sortino     = (mean_r / down_std * math.sqrt(annual_bars)) if down_std > 0 else 0.0

    running_max = np.maximum.accumulate(pv)
    drawdowns   = (pv - running_max) / np.where(running_max > 0, running_max, 1)
    max_dd_pct  = float(drawdowns.min()) * 100

    calmar = cagr / abs(max_dd_pct) if max_dd_pct != 0 else 0.0

    in_dd = drawdowns < 0
    if in_dd.any():
        dd_lengths, cur = [], 0
        for x in in_dd:
            cur = cur + 1 if x else 0
            dd_lengths.append(cur)
        max_dd_bars = int(max(dd_lengths))
    else:
        max_dd_bars = 0

    if bh_return_pct is None:
        bh_return_pct = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

    closed = [t for t in trades if t["side"] in ("sell", "liquidate", "close_short", "stop_loss", "take_profit")]
    pnls   = [t["pnl"] for t in closed]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate      = (len(wins) / len(pnls) * 100) if pnls else 0.0
    avg_win       = float(np.mean(wins))   if wins   else 0.0
    avg_loss      = float(np.mean(losses)) if losses else 0.0
    gross_profit  = sum(wins)
    gross_loss    = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_fees    = sum(t["fee"] for t in trades)

    sl_hits = sum(1 for t in trades if t["side"] == "stop_loss")
    tp_hits = sum(1 for t in trades if t["side"] == "take_profit")

    return {
        "total_return_pct": round(total_return_pct, 2),
        "cagr_pct":         round(cagr, 2),
        "sharpe":           round(sharpe, 3),
        "sortino":          round(sortino, 3),
        "calmar":           round(calmar, 3),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "max_dd_bars":      max_dd_bars,
        "win_rate_pct":     round(win_rate, 2),
        "profit_factor":    round(profit_factor, 3),
        "avg_win":          round(avg_win, 4),
        "avg_loss":         round(avg_loss, 4),
        "total_trades":     len(trades),
        "sl_hits":          sl_hits,
        "tp_hits":          tp_hits,
        "total_fees":       round(total_fees, 4),
        "final_equity":     round(final, 2),
        "initial_equity":   round(initial, 2),
        "bh_return_pct":    round(bh_return_pct, 2),
    }


# ---------------------------------------------------------------------------
# Rich terminal report
# ---------------------------------------------------------------------------

def print_report(result: "BacktestResult") -> None:
    m = result.metrics
    title = (
        f"[bold cyan]{result.strategy_name}[/] "
        f"· {result.symbol} {result.timeframe} "
        f"· {result.start} → {result.end}"
    )
    console.print(f"\n{title}\n")

    perf = Table(title="Performance", box=box.SIMPLE_HEAVY, show_header=False)
    perf.add_column("Metric", style="dim")
    perf.add_column("Value", justify="right")

    def _color(val: float, pos_good: bool = True) -> str:
        if val > 0:
            return "[green]" if pos_good else "[red]"
        return "[red]" if pos_good else "[green]"

    rows = [
        ("Total return",        f"{_color(m['total_return_pct'])}{m['total_return_pct']:+.2f}%[/]"),
        ("CAGR",                f"{_color(m['cagr_pct'])}{m['cagr_pct']:+.2f}%[/]"),
        ("Buy-and-hold return", f"{_color(m['bh_return_pct'])}{m['bh_return_pct']:+.2f}%[/]"),
        ("Sharpe ratio",        f"{_color(m['sharpe'])}{m['sharpe']:.3f}[/]"),
        ("Sortino ratio",       f"{_color(m['sortino'])}{m['sortino']:.3f}[/]"),
        ("Calmar ratio",        f"{_color(m['calmar'])}{m['calmar']:.3f}[/]"),
        ("Max drawdown",        f"[red]{m['max_drawdown_pct']:.2f}%[/]"),
        ("Max DD duration",     f"{m['max_dd_bars']} bars"),
        ("Win rate",            f"{m['win_rate_pct']:.2f}%"),
        ("Profit factor",       f"{m['profit_factor']:.3f}"),
        ("Avg win / avg loss",  f"{m['avg_win']:.4f} / {m['avg_loss']:.4f}"),
        ("Total trades",        str(m["total_trades"])),
        ("Stop-loss hits",      str(m["sl_hits"])),
        ("Take-profit hits",    str(m["tp_hits"])),
        ("Total fees paid",     f"{m['total_fees']:.4f}"),
        ("Final equity",        f"${m['final_equity']:,.2f}"),
    ]
    for label, value in rows:
        perf.add_row(label, value)

    console.print(perf)


def print_comparison(results: list["BacktestResult"]) -> None:
    table = Table(title="Strategy Comparison", box=box.SIMPLE_HEAVY)
    table.add_column("Strategy", style="bold")
    table.add_column("Return",  justify="right")
    table.add_column("CAGR",    justify="right")
    table.add_column("Sharpe",  justify="right")
    table.add_column("MaxDD",   justify="right")
    table.add_column("WinRate", justify="right")
    table.add_column("Trades",  justify="right")
    table.add_column("B&H",     justify="right")

    def _fmt(val: float, suffix: str = "", pos_good: bool = True) -> str:
        color = "green" if (val > 0) == pos_good else "red"
        return f"[{color}]{val:+.2f}{suffix}[/]"

    for r in results:
        m = r.metrics
        table.add_row(
            r.strategy_name,
            _fmt(m["total_return_pct"], "%"),
            _fmt(m["cagr_pct"], "%"),
            _fmt(m["sharpe"]),
            _fmt(m["max_drawdown_pct"], "%", pos_good=False),
            f"{m['win_rate_pct']:.1f}%",
            str(m["total_trades"]),
            _fmt(m["bh_return_pct"], "%"),
        )

    console.print("\n")
    console.print(table)


# ---------------------------------------------------------------------------
# Plotly HTML report
# ---------------------------------------------------------------------------

def save_html_report(result: "BacktestResult", path: str = "report.html") -> str:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        console.print("[yellow]plotly not installed — skipping HTML report[/]")
        return ""

    df   = result.df
    pv   = result.portfolio_values
    m    = result.metrics
    buys  = [t for t in result.trades if t["side"] == "buy"]
    sells = [t for t in result.trades if t["side"] in ("sell", "liquidate")]

    xs = df["timestamp"].tolist()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=["Price + Trades", "Equity Curve", "Drawdown"],
        vertical_spacing=0.06,
    )

    fig.add_trace(go.Candlestick(
        x=xs, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="Price", increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    if buys:
        fig.add_trace(go.Scatter(
            x=[b["timestamp"] for b in buys],
            y=[b["price"] for b in buys],
            mode="markers", marker=dict(symbol="triangle-up", size=10, color="lime"),
            name="Buy",
        ), row=1, col=1)
    if sells:
        fig.add_trace(go.Scatter(
            x=[s["timestamp"] for s in sells],
            y=[s["price"] for s in sells],
            mode="markers", marker=dict(symbol="triangle-down", size=10, color="red"),
            name="Sell",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=xs, y=pv.tolist(), name="Equity",
        line=dict(color="#2196F3", width=2), fill="tozeroy", fillcolor="rgba(33,150,243,0.1)",
    ), row=2, col=1)

    running_max = np.maximum.accumulate(pv)
    dd = (pv - running_max) / np.where(running_max > 0, running_max, 1) * 100
    fig.add_trace(go.Scatter(
        x=xs, y=dd.tolist(), name="Drawdown %",
        line=dict(color="#ef5350", width=1), fill="tozeroy", fillcolor="rgba(239,83,80,0.2)",
    ), row=3, col=1)

    title_text = (
        f"{result.strategy_name} | {result.symbol} {result.timeframe} | "
        f"Return: {m['total_return_pct']:+.2f}% | Sharpe: {m['sharpe']:.3f} | "
        f"MaxDD: {m['max_drawdown_pct']:.2f}%"
    )
    fig.update_layout(
        title=title_text,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=900,
        showlegend=True,
    )

    fig.write_html(path)
    console.print(f"[green]Report saved → {path}[/]")
    return path


def save_comparison_html(results: list["BacktestResult"], path: str = "comparison.html") -> str:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return ""

    fig = go.Figure()
    for r in results:
        xs = r.df["timestamp"].tolist()
        fig.add_trace(go.Scatter(
            x=xs, y=(r.portfolio_values / r.portfolio_values[0] * 100).tolist(),
            name=r.strategy_name, mode="lines",
        ))

    fig.update_layout(
        title="Equity Curves (indexed to 100)",
        template="plotly_dark",
        yaxis_title="Equity (indexed)",
        height=600,
    )
    fig.write_html(path)
    console.print(f"[green]Comparison chart saved → {path}[/]")
    return path
