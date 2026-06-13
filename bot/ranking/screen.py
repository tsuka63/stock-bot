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


def consensus_signal(df) -> dict:
    """
    Aggregate today's stance across all technical strategies for one symbol.

    Each strategy emits buy(1)/sell(-1)/hold(0) per bar. We track each one's
    long/flat position to the latest bar, then count:
      • long_count  — strategies currently holding a long position
      • buy_today   — strategies that fired a fresh BUY on the latest bar
      • sell_today  — strategies that fired a fresh SELL on the latest bar
      • total       — number of strategies evaluated

    A high long_count / a cluster of buy_today is broad agreement; relying on
    several strategies rather than one is more robust on choppy small-caps.
    """
    long_count = buy_today = sell_today = total = 0
    for cls in REGISTRY.values():
        try:
            sig = cls().generate_signals(df)
        except Exception:
            continue
        if len(sig) == 0:
            continue
        total += 1
        pos = 0
        for s in sig:
            if s == 1:
                pos = 1
            elif s == -1:
                pos = 0
        long_count += pos
        if sig[-1] == 1:
            buy_today += 1
        elif sig[-1] == -1:
            sell_today += 1
    return {
        "long_count": long_count,
        "total":      total,
        "buy_today":  buy_today,
        "sell_today": sell_today,
    }


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
            # Today's buy/sell consensus across all strategies for this symbol
            try:
                cons = consensus_signal(store.load(sym, "1d", since=since))
            except Exception:
                cons = {"long_count": 0, "total": 0, "buy_today": 0, "sell_today": 0}
            cand.update({
                "long_count": cons["long_count"],
                "strat_total": cons["total"],
                "buy_today":  cons["buy_today"],
                "sell_today": cons["sell_today"],
            })
            rows.append(cand)

    if not rows:
        return pd.DataFrame(columns=[
            "symbol", "best_strategy", "is_sharpe", "oos_sharpe", "hold",
            "return_pct", "oos_return_pct", "trades", "n_bars", "equity", "split",
            "long_count", "strat_total", "buy_today", "sell_today",
        ])

    df = pd.DataFrame(rows)
    # Best generalisers first
    return df.sort_values("oos_sharpe", ascending=False).reset_index(drop=True)
