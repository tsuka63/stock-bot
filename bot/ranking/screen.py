"""
Auto-screen ranking candidates: fetch recent prices and find, per symbol,
the strategy that holds up out-of-sample.

For each candidate this fetches up-to-date prices (Yahoo Finance), runs the
technical strategies, and scores each on both the full window (in-sample)
and the most recent `oos_frac` tail (out-of-sample). The strategy with the
best *OOS* Sharpe is reported, so a candidate is judged on what actually
generalises — not on an in-sample number that may be overfit.

ML strategies (XGBoost/RL) are intentionally excluded here to keep the daily
run fast; they can still be run manually via the CLI.
"""

from __future__ import annotations

from datetime import date, timedelta

import math
import pandas as pd

from bot.backtest.engine import BacktestEngine
from bot.backtest.report import compute_metrics
from bot.config          import SimConfig
from bot.data.fetcher    import DataFetcher
from bot.data.store      import DataStore
from bot.strategy        import REGISTRY   # technical only (bot.ml not imported)


def _hold_flag(is_sharpe: float, oos_sharpe: float) -> str:
    if oos_sharpe is None or math.isnan(oos_sharpe):
        return "—"
    if is_sharpe > 0 and oos_sharpe <= 0:
        return "✗"          # sign flip → overfit
    if is_sharpe > 0 and oos_sharpe < is_sharpe * 0.5:
        return "△"          # decayed a lot
    if is_sharpe > 0:
        return "✓"          # holds up
    return "·"


def screen_candidates(
    symbols:   list[str],
    db_path:   str = "data/market.duckdb",
    lookback_days: int = 730,
    oos_frac:  float = 0.30,
    source:    str = "yfinance",
    market:    str = "jp",
    fee_rate:  float = 0.001,
) -> pd.DataFrame:
    """
    Returns one row per symbol describing its most robust strategy.

    Columns: symbol, best_strategy, is_sharpe, oos_sharpe, hold,
             return_pct, oos_return_pct, trades, n_bars
    """
    since   = (date.today() - timedelta(days=lookback_days)).isoformat()
    fetcher = DataFetcher(source)
    store   = DataStore(db_path)
    engine  = BacktestEngine(config=SimConfig(market=market, fee_rate=fee_rate), db_path=db_path)
    strat_names = list(REGISTRY.keys())

    rows: list[dict] = []
    for sym in symbols:
        # Refresh prices so the freshest bars are present
        try:
            df = fetcher.fetch(sym, "1d", since=since)
            if not df.empty:
                store.save(df, sym, "1d")
        except Exception:
            pass

        best = None
        for sn in strat_names:
            try:
                r = engine.run(REGISTRY[sn](), sym, "1d", since, None)
            except Exception:
                continue
            m  = r.metrics
            pv = r.portfolio_values
            n  = len(pv)
            split = int(n * (1 - oos_frac))
            if n - split < 5:
                continue
            tail = [t for t in r.trades if t.get("bar_index", 0) >= split]
            om   = compute_metrics(pv[split:], tail, r.df.iloc[split:], r.annual_bars)

            cand = {
                "symbol":         sym,
                "best_strategy":  sn,
                "is_sharpe":      m["sharpe"],
                "oos_sharpe":     om.get("sharpe", float("nan")),
                "return_pct":     m["total_return_pct"],
                "oos_return_pct": om.get("total_return_pct", float("nan")),
                "trades":         m["total_trades"],
                "n_bars":         n,
                "equity":         [float(v) for v in pv],   # for the report sparkline
                "split":          split,
            }
            # Robustness-first: rank by OOS Sharpe
            key = cand["oos_sharpe"] if not math.isnan(cand["oos_sharpe"]) else -math.inf
            if best is None or key > best[0]:
                best = (key, cand)

        if best is not None:
            cand = best[1]
            cand["hold"] = _hold_flag(cand["is_sharpe"], cand["oos_sharpe"])
            rows.append(cand)

    if not rows:
        return pd.DataFrame(columns=[
            "symbol", "best_strategy", "is_sharpe", "oos_sharpe", "hold",
            "return_pct", "oos_return_pct", "trades", "n_bars", "equity", "split",
        ])

    df = pd.DataFrame(rows)
    # Best generalisers first
    return df.sort_values("oos_sharpe", ascending=False).reset_index(drop=True)
