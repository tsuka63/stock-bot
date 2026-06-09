"""
BacktestEngine: runs a strategy against stored OHLCV data.

Pure-Python simulation — no external extension required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from bot.config import ANNUAL_BARS, ANNUAL_BARS_JP, SimConfig
from bot.data.store import DataStore
from bot.strategy.base import BaseStrategy

_ANNUAL_BARS_BY_MARKET = {
    "us": ANNUAL_BARS,
    "jp": ANNUAL_BARS_JP,
}


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    timeframe: str
    start: str
    end: str

    df: pd.DataFrame
    signals: np.ndarray
    portfolio_values: np.ndarray
    cash_values: np.ndarray
    position_sizes: np.ndarray
    trades: list[dict]
    final_value: float
    total_fees: float
    total_trades: int

    engine: str = "python"
    annual_bars: int = 0       # 0 = auto-detect from timeframe (US default)
    metrics: dict = field(default_factory=dict)

    def __post_init__(self):
        from bot.backtest.report import compute_metrics
        ab = self.annual_bars or ANNUAL_BARS.get(self.timeframe, 252)
        self.metrics = compute_metrics(
            self.portfolio_values, self.trades, self.df, ab
        )


class BacktestEngine:
    def __init__(
        self,
        config: Optional[SimConfig] = None,
        db_path: str = "data/market.duckdb",
    ):
        self.cfg   = config or SimConfig()
        self.store = DataStore(db_path)

    def run(
        self,
        strategy: BaseStrategy,
        symbol: str,
        timeframe: str,
        since: str,
        until: Optional[str] = None,
    ) -> BacktestResult:
        df = self.store.load(symbol, timeframe, since=since, until=until)
        if df.empty:
            raise ValueError(
                f"No data for {symbol}/{timeframe} between {since} and {until}. "
                "Run `fetch_data.py` first."
            )

        signals = strategy.generate_signals(df)
        result  = self._simulate(df, signals)
        end     = until or str(df["timestamp"].max())

        bars_map    = _ANNUAL_BARS_BY_MARKET.get(self.cfg.market, ANNUAL_BARS)
        annual_bars = bars_map.get(timeframe, self.cfg.annual_bars)

        return BacktestResult(
            strategy_name    = strategy.name,
            symbol           = symbol,
            timeframe        = timeframe,
            start            = since,
            end              = end,
            df               = df,
            signals          = signals,
            portfolio_values = np.array(result["portfolio_values"]),
            cash_values      = np.array(result["cash_values"]),
            position_sizes   = np.array(result["position_sizes"]),
            trades           = result["trades"],
            final_value      = result["final_value"],
            total_fees       = result["total_fees"],
            total_trades     = result["total_trades"],
            annual_bars      = annual_bars,
        )

    def compare(
        self,
        strategies: list[BaseStrategy],
        symbol: str,
        timeframe: str,
        since: str,
        until: Optional[str] = None,
    ) -> list[BacktestResult]:
        return [self.run(s, symbol, timeframe, since, until) for s in strategies]

    @staticmethod
    def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
        """Wilder's Average True Range, forward-filled over the warmup window."""
        n = len(closes)
        if n == 0:
            return np.zeros(0)
        prev_close = np.concatenate(([closes[0]], closes[:-1]))
        tr = np.maximum.reduce([
            highs - lows,
            np.abs(highs - prev_close),
            np.abs(lows - prev_close),
        ])
        atr = np.zeros(n)
        if n <= period:
            atr[:] = tr.mean() if n else 0.0
            return atr
        atr[period - 1] = tr[:period].mean()
        alpha = 1.0 / period
        for i in range(period, n):
            atr[i] = (1 - alpha) * atr[i - 1] + alpha * tr[i]
        atr[:period - 1] = atr[period - 1]
        return atr

    def _simulate(self, df: pd.DataFrame, signals: np.ndarray) -> dict:
        opens   = df["open"].to_numpy(dtype=float)
        highs   = df["high"].to_numpy(dtype=float)
        lows    = df["low"].to_numpy(dtype=float)
        closes  = df["close"].to_numpy(dtype=float)
        volumes = df["volume"].to_numpy(dtype=float)
        ts      = df["timestamp"].to_numpy(dtype=np.int64)
        n       = len(opens)

        cfg         = self.cfg
        cash        = cfg.initial_capital
        position    = 0.0
        avg_entry   = 0.0
        total_fees  = 0.0
        trades: list[dict] = []
        closed_pnls: list[float] = []   # rolling history for Kelly sizing
        pv      = np.zeros(n)
        cv      = np.zeros(n)
        ps      = np.zeros(n)
        pending = 0

        sl_pct = cfg.stop_loss_pct
        tp_pct = cfg.take_profit_pct

        # ── Volatility-scaled slippage (ATR component) ─────────────────────
        atr = (
            self._atr(highs, lows, closes, cfg.atr_period)
            if cfg.slippage_atr_mult > 0 else np.zeros(n)
        )

        def _slip_frac(i: int) -> float:
            """One-way slippage as a fraction of fill price for bar i."""
            frac = cfg.slippage_pct
            if cfg.slippage_atr_mult > 0 and closes[i] > 0:
                frac += cfg.slippage_atr_mult * (atr[i] / closes[i])
            return frac

        def _trade_val(equity: float) -> float:
            sizing = cfg.position_sizing
            if sizing == "fixed_frac":
                frac = cfg.risk_per_trade
            elif sizing == "kelly" and len(closed_pnls) >= cfg.kelly_lookback:
                recent = closed_pnls[-cfg.kelly_lookback :]
                wins   = [p for p in recent if p > 0]
                losses = [p for p in recent if p <= 0]
                if wins and losses:
                    p = len(wins) / len(recent)
                    b = (sum(wins) / len(wins)) / (abs(sum(losses)) / len(losses))
                    k = (b * p - (1 - p)) / b
                    frac = max(0.0, min(k * 0.5, cfg.risk_per_trade))  # half-Kelly, capped
                else:
                    frac = cfg.risk_per_trade
            elif sizing == "kelly":
                frac = cfg.risk_per_trade   # fallback before enough data
            else:                           # "full"
                frac = cfg.position_size_pct
            return equity * frac / (1 + cfg.fee_rate)

        for i in range(n):
            # ── Execute pending order at open ──────────────────────────────
            if pending != 0:
                slip = 1 + pending * _slip_frac(i)
                fill = opens[i] * slip

                if pending > 0 and position <= 0.0:
                    if position < 0.0:
                        size = -position
                        cost = size * fill
                        fee  = cost * cfg.fee_rate
                        pnl  = (avg_entry - fill) * size - fee
                        cash -= cost + fee
                        total_fees += fee
                        trades.append({"bar_index": i, "timestamp": int(ts[i]),
                                       "side": "close_short", "price": fill,
                                       "size": size, "fee": fee, "pnl": pnl})
                        closed_pnls.append(pnl)
                        position, avg_entry = 0.0, 0.0
                    tv  = _trade_val(cash)
                    if fill > 0 and cash > 1e-9:
                        size = tv / fill
                        # Cap fill size to a fraction of the bar's volume (liquidity limit)
                        if cfg.max_volume_participation > 0:
                            max_shares = volumes[i] * cfg.max_volume_participation
                            size = min(size, max_shares)
                        cost = size * fill
                        fee  = cost * cfg.fee_rate
                        if size > 0:
                            cash -= cost + fee
                            position, avg_entry = size, fill
                            total_fees += fee
                            trades.append({"bar_index": i, "timestamp": int(ts[i]),
                                           "side": "buy", "price": fill,
                                           "size": size, "fee": fee, "pnl": 0.0})

                elif pending < 0 and position > 0.0:
                    size     = position
                    proceeds = size * fill
                    fee      = proceeds * cfg.fee_rate
                    pnl      = (fill - avg_entry) * size - fee
                    cash    += proceeds - fee
                    total_fees += fee
                    trades.append({"bar_index": i, "timestamp": int(ts[i]),
                                   "side": "sell", "price": fill,
                                   "size": size, "fee": fee, "pnl": pnl})
                    closed_pnls.append(pnl)
                    position, avg_entry = 0.0, 0.0

                pending = 0

            # ── Stop-loss / take-profit (intra-bar) ────────────────────────
            if position > 0.0 and (sl_pct > 0 or tp_pct > 0):
                sl_level = avg_entry * (1 - sl_pct) if sl_pct > 0 else 0.0
                tp_level = avg_entry * (1 + tp_pct) if tp_pct > 0 else float("inf")

                exit_price: float | None = None
                exit_side: str           = ""

                if cfg.conservative_intrabar:
                    # Gap-aware: if the bar opens beyond a level, fill at the open
                    # (worse than the idealised level). Then, when both the stop and
                    # the target lie inside the bar's range, assume the STOP triggers
                    # first — the pessimistic ordering, since intrabar path is unknown.
                    if sl_pct > 0 and opens[i] <= sl_level:
                        exit_price, exit_side = opens[i], "stop_loss"
                    elif tp_pct > 0 and opens[i] >= tp_level:
                        exit_price, exit_side = opens[i], "take_profit"
                    elif sl_pct > 0 and lows[i] <= sl_level:
                        exit_price, exit_side = sl_level, "stop_loss"
                    elif tp_pct > 0 and highs[i] >= tp_level:
                        exit_price, exit_side = tp_level, "take_profit"
                else:
                    # Optimistic: assume the target is reached before the stop.
                    if tp_pct > 0 and highs[i] >= tp_level:
                        exit_price, exit_side = tp_level, "take_profit"
                    elif sl_pct > 0 and lows[i] <= sl_level:
                        exit_price, exit_side = sl_level, "stop_loss"

                if exit_price is not None:
                    size     = position
                    proceeds = size * exit_price
                    fee      = proceeds * cfg.fee_rate
                    pnl      = (exit_price - avg_entry) * size - fee
                    cash    += proceeds - fee
                    total_fees += fee
                    trades.append({"bar_index": i, "timestamp": int(ts[i]),
                                   "side": exit_side, "price": exit_price,
                                   "size": size, "fee": fee, "pnl": pnl})
                    closed_pnls.append(pnl)
                    position, avg_entry = 0.0, 0.0
                    pending = 0  # cancel any queued sell

            # ── Portfolio snapshot ─────────────────────────────────────────
            pv[i] = cash + position * closes[i]
            cv[i] = cash
            ps[i] = position

            if signals[i] != 0:
                pending = int(signals[i])

        # ── Liquidate end-of-period position ──────────────────────────────
        if position > 0.0:
            price    = closes[-1]
            proceeds = position * price
            fee      = proceeds * cfg.fee_rate
            pnl      = (price - avg_entry) * position - fee
            cash    += proceeds - fee
            total_fees += fee
            trades.append({"bar_index": n - 1, "timestamp": int(ts[-1]),
                           "side": "liquidate", "price": price,
                           "size": position, "fee": fee, "pnl": pnl})
            pv[-1] = cash

        return {
            "portfolio_values": pv,
            "cash_values":      cv,
            "position_sizes":   ps,
            "trades":           trades,
            "final_value":      float(pv[-1]) if n else cfg.initial_capital,
            "total_fees":       total_fees,
            "total_trades":     len(trades),
        }
