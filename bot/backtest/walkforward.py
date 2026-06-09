"""
Walk-Forward Validation

過去データで戦略を評価するとき、同じ期間を「最適化」と「テスト」に
使ってしまうと過学習になる。ウォークフォワードはこれを防ぐ仕組み:

  ┌─ Train ──────────┐┌─ Test ─┐
  ├─── Train ────────────┤┌─ Test ─┐
  ├──── Train ──────────────┤┌─ Test ─┐  (Rolling)
  ...

各 Test ウィンドウは完全に未来のデータ → 実運用に近い性能評価ができる。

Usage:
    from bot.backtest.walkforward import WalkForwardValidator

    validator = WalkForwardValidator(engine, n_windows=6, train_frac=0.7)
    wf_result = validator.run(
        strategy_factory=lambda: SmaCrossover(fast=10, slow=50),
        symbol="AAPL", timeframe="1d",
        since="2021-01-01", until="2024-12-31",
    )
    validator.print_report(wf_result)
    validator.save_html_report(wf_result, "data/walkforward.html")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

from bot.backtest.report import compute_metrics
from bot.config import ANNUAL_BARS, ANNUAL_BARS_JP

if TYPE_CHECKING:
    from bot.backtest.engine import BacktestEngine
    from bot.strategy.base import BaseStrategy

console = Console()


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    window_idx:   int
    train_since:  str
    train_until:  str
    test_since:   str
    test_until:   str
    n_train_bars: int
    n_test_bars:  int
    metrics:      dict
    portfolio_values: np.ndarray
    trades:           list[dict]


@dataclass
class WalkForwardResult:
    strategy_name: str
    symbol:        str
    timeframe:     str
    windows:       list[WindowResult]

    @property
    def summary_df(self) -> pd.DataFrame:
        rows = []
        for w in self.windows:
            m = w.metrics
            rows.append({
                "window":        w.window_idx + 1,
                "test_period":   f"{w.test_since} ~ {w.test_until}",
                "n_train":       w.n_train_bars,
                "n_test":        w.n_test_bars,
                "return_pct":    m.get("total_return_pct", 0),
                "sharpe":        m.get("sharpe", 0),
                "max_dd_pct":    m.get("max_drawdown_pct", 0),
                "win_rate_pct":  m.get("win_rate_pct", 0),
                "trades":        m.get("total_trades", 0),
                "bh_return_pct": m.get("bh_return_pct", 0),
            })
        return pd.DataFrame(rows)

    @property
    def aggregate_metrics(self) -> dict:
        df = self.summary_df
        agg = {}
        for col in ("return_pct", "sharpe", "max_dd_pct", "win_rate_pct"):
            vals = df[col].dropna()
            agg[col] = {
                "mean": round(float(vals.mean()), 3),
                "std":  round(float(vals.std()),  3),
                "min":  round(float(vals.min()),  3),
                "max":  round(float(vals.max()),  3),
            }
        profitable = (df["return_pct"] > 0).sum()
        agg["profitable_windows"] = f"{profitable}/{len(self.windows)}"
        agg["consistency_pct"]    = round(profitable / max(1, len(self.windows)) * 100, 1)
        return agg


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """
    Parameters
    ----------
    engine:         BacktestEngine (データストアも内包)
    n_windows:      テスト期間の数
    train_frac:     各ウィンドウ内の訓練データ割合
    anchored:       True = アンカード（訓練は常に先頭から）
                    False = ローリング（訓練ウィンドウもスライド）
    min_train_bars: 訓練データの最低バー数（これ未満はスキップ）
    """

    def __init__(
        self,
        engine: "BacktestEngine",
        n_windows: int = 6,
        train_frac: float = 0.70,
        anchored: bool = False,
        min_train_bars: int = 100,
    ):
        self.engine         = engine
        self.n_windows      = n_windows
        self.train_frac     = train_frac
        self.anchored       = anchored
        self.min_train_bars = min_train_bars

    def _compute_windows(self, df: pd.DataFrame) -> list[tuple[int, int, int, int]]:
        n = len(df)
        test_bars  = n // self.n_windows
        train_bars = round(test_bars * self.train_frac / (1 - self.train_frac))

        windows = []
        for i in range(self.n_windows):
            test_start  = i * test_bars
            test_end    = min(test_start + test_bars, n)
            train_start = 0 if self.anchored else max(0, test_start - train_bars)
            train_end   = test_start

            if (train_end - train_start) < self.min_train_bars:
                continue
            windows.append((train_start, train_end, test_start, test_end))

        return windows

    @staticmethod
    def _ts_to_str(ts_ms: int) -> str:
        return pd.Timestamp(ts_ms, unit="ms").strftime("%Y-%m-%d")

    def run(
        self,
        strategy_factory: Callable[[], "BaseStrategy"],
        symbol: str,
        timeframe: str,
        since: str,
        until: str,
    ) -> WalkForwardResult:
        df = self.engine.store.load(symbol, timeframe, since=since, until=until)
        if df.empty:
            raise ValueError(f"No data for {symbol}/{timeframe} {since}→{until}")

        # Market-aware annualisation
        market = getattr(self.engine.cfg, "market", "us")
        bars_map    = ANNUAL_BARS_JP if market == "jp" else ANNUAL_BARS
        annual_bars = bars_map.get(timeframe, 252 if market == "us" else 245)

        idx_windows = self._compute_windows(df)
        if not idx_windows:
            raise ValueError(
                "有効なウィンドウがありません。"
                "min_train_bars を下げるか n_windows を減らしてください。"
            )

        strat_name = strategy_factory().name
        console.rule(
            f"[bold cyan]Walk-Forward: {strat_name} | {symbol} {timeframe} | "
            f"{len(idx_windows)} windows ({'anchored' if self.anchored else 'rolling'})[/]"
        )

        window_results = []
        for i, (tr_s, tr_e, te_s, te_e) in enumerate(idx_windows):
            train_df = df.iloc[tr_s:tr_e].reset_index(drop=True)
            test_df  = df.iloc[te_s:te_e].reset_index(drop=True)
            full_df  = pd.concat([train_df, test_df]).reset_index(drop=True)

            strat = strategy_factory()
            actual_frac = len(train_df) / len(full_df)
            if hasattr(strat, "train_frac"):
                strat.train_frac = actual_frac

            console.print(
                f"  Window {i+1}/{len(idx_windows)} | "
                f"train {self._ts_to_str(int(df['timestamp'].iloc[tr_s]))} → "
                f"{self._ts_to_str(int(df['timestamp'].iloc[tr_e-1]))} "
                f"({len(train_df):,} bars) | "
                f"test  {self._ts_to_str(int(df['timestamp'].iloc[te_s]))} → "
                f"{self._ts_to_str(int(df['timestamp'].iloc[te_e-1]))} "
                f"({len(test_df):,} bars)"
            )

            signals_full = strat.generate_signals(full_df)
            test_signals = signals_full[len(train_df):]
            sim = self.engine._simulate(test_df, test_signals)

            test_pv     = np.array(sim["portfolio_values"])
            test_trades = sim["trades"]
            metrics     = compute_metrics(test_pv, test_trades, test_df, annual_bars)

            window_results.append(WindowResult(
                window_idx       = i,
                train_since      = self._ts_to_str(int(df["timestamp"].iloc[tr_s])),
                train_until      = self._ts_to_str(int(df["timestamp"].iloc[tr_e - 1])),
                test_since       = self._ts_to_str(int(df["timestamp"].iloc[te_s])),
                test_until       = self._ts_to_str(int(df["timestamp"].iloc[te_e - 1])),
                n_train_bars     = len(train_df),
                n_test_bars      = len(test_df),
                metrics          = metrics,
                portfolio_values = test_pv,
                trades           = test_trades,
            ))

        return WalkForwardResult(
            strategy_name = strat_name,
            symbol        = symbol,
            timeframe     = timeframe,
            windows       = window_results,
        )

    def print_report(self, result: WalkForwardResult) -> None:
        agg = result.aggregate_metrics
        df  = result.summary_df

        t = Table(
            title=f"Walk-Forward: {result.strategy_name}  [{result.symbol} {result.timeframe}]",
            box=box.SIMPLE_HEAVY,
        )
        t.add_column("Win",    justify="center", style="dim")
        t.add_column("Test Period")
        t.add_column("Return", justify="right")
        t.add_column("Sharpe", justify="right")
        t.add_column("Max DD", justify="right")
        t.add_column("WinRate",justify="right")
        t.add_column("Trades", justify="right")
        t.add_column("B&H",    justify="right")

        for _, row in df.iterrows():
            color = "green" if row.return_pct > 0 else "red"
            t.add_row(
                str(int(row.window)),
                row.test_period,
                f"[{color}]{row.return_pct:+.2f}%[/]",
                f"{row.sharpe:+.3f}",
                f"[red]{row.max_dd_pct:.2f}%[/]",
                f"{row.win_rate_pct:.1f}%",
                str(int(row.trades)),
                f"{row.bh_return_pct:+.2f}%",
            )

        m_ret = agg["return_pct"]
        t.add_row(
            "[bold]AVG[/]", "",
            f"[bold]{m_ret['mean']:+.2f}% ±{m_ret['std']:.2f}[/]",
            f"[bold]{agg['sharpe']['mean']:+.3f}[/]",
            f"[bold]{agg['max_dd_pct']['mean']:.2f}%[/]",
            "", "", "",
        )

        console.print()
        console.print(t)
        console.print(
            f"  [bold]一貫性スコア[/]: "
            f"[cyan]{agg['consistency_pct']}%[/] のウィンドウで黒字 "
            f"({agg['profitable_windows']})"
        )
        console.print()

    def save_html_report(
        self, result: WalkForwardResult, path: str = "data/walkforward.html"
    ) -> str:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            console.print("[yellow]plotly not installed — HTML 出力スキップ[/]")
            return ""

        n   = len(result.windows)
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[
                "テスト期間 Equity Curve（ウィンドウ別、100 基準）",
                "ウィンドウ別リターン",
            ],
            row_heights=[0.6, 0.4],
            vertical_spacing=0.12,
        )

        colors = ["#2196F3","#4CAF50","#FF9800","#E91E63","#9C27B0","#00BCD4"]
        for i, w in enumerate(result.windows):
            pv = w.portfolio_values
            fig.add_trace(go.Scatter(
                x=list(range(len(pv))),
                y=(pv / pv[0] * 100).tolist(),
                name=f"Win {i+1} ({w.test_since})",
                line=dict(color=colors[i % len(colors)], width=2),
            ), row=1, col=1)

        df  = result.summary_df
        ret = df["return_pct"].tolist()
        fig.add_trace(go.Bar(
            x=[f"Win {i+1}" for i in range(n)],
            y=ret,
            marker_color=["green" if r > 0 else "red" for r in ret],
            name="Return %",
            text=[f"{r:+.2f}%" for r in ret],
            textposition="outside",
        ), row=2, col=1)

        agg = result.aggregate_metrics
        fig.update_layout(
            title=(
                f"Walk-Forward: {result.strategy_name}  |  {result.symbol} {result.timeframe}  |  "
                f"一貫性: {agg['consistency_pct']}%  |  平均 Sharpe: {agg['sharpe']['mean']:+.3f}"
            ),
            template="plotly_dark", height=800, showlegend=True,
        )
        fig.add_hline(y=100, row=1, col=1, line_dash="dot", line_color="gray")
        fig.add_hline(y=0,   row=2, col=1, line_dash="dot", line_color="gray")

        fig.write_html(path)
        console.print(f"[green]HTML レポート保存 → {path}[/]")
        return path
