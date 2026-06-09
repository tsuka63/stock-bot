"""
Classic technical-indicator strategies, each computing their own indicators
from raw OHLCV data using only pandas/numpy (no external TA library required).

Signal generation contract:
  - All computations are on close prices unless stated otherwise.
  - Signals at bar i use only data available at bar i's close.
  - The first `warmup_period` signals are always 0.
"""

import numpy as np
import pandas as pd

from bot.strategy.base import BaseStrategy


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _bollinger(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = _sma(series, period)
    std = series.rolling(period, min_periods=period).std()
    return mid - std_mult * std, mid, mid + std_mult * std


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class SmaCrossover(BaseStrategy):
    """
    Golden-cross / death-cross on two SMAs.
    Long when fast SMA crosses above slow SMA; exit when it crosses below.
    """

    def __init__(self, fast: int = 10, slow: int = 50):
        if fast >= slow:
            raise ValueError("fast must be less than slow")
        self.fast = fast
        self.slow = slow

    @property
    def name(self) -> str:
        return f"sma_crossover({self.fast},{self.slow})"

    @property
    def warmup_period(self) -> int:
        return self.slow

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        close = df["close"]
        fast  = _sma(close, self.fast)
        slow  = _sma(close, self.slow)

        above      = (fast > slow).astype(int)
        prev_above = above.shift(1)

        signals = np.zeros(len(df), dtype=np.int32)
        signals[(above == 1) & (prev_above == 0)] = 1
        signals[(above == 0) & (prev_above == 1)] = -1
        signals[: self.warmup_period] = 0
        return signals


class EmaCrossover(BaseStrategy):
    """EMA crossover — same logic as SmaCrossover but using EMAs."""

    def __init__(self, fast: int = 12, slow: int = 26):
        if fast >= slow:
            raise ValueError("fast must be less than slow")
        self.fast = fast
        self.slow = slow

    @property
    def name(self) -> str:
        return f"ema_crossover({self.fast},{self.slow})"

    @property
    def warmup_period(self) -> int:
        return self.slow * 3

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        close = df["close"]
        fast  = _ema(close, self.fast)
        slow  = _ema(close, self.slow)

        above      = (fast > slow).astype(int)
        prev_above = above.shift(1)

        signals = np.zeros(len(df), dtype=np.int32)
        signals[(above == 1) & (prev_above == 0)] = 1
        signals[(above == 0) & (prev_above == 1)] = -1
        signals[: self.warmup_period] = 0
        return signals


class EmaRsi(BaseStrategy):
    """
    Trend-filtered RSI mean-reversion.
    Conditions for long entry:
      - Price is above the long-period EMA (trend filter)
      - RSI crosses up from below oversold level
    Exit: RSI reaches overbought level.
    """

    def __init__(
        self,
        ema_period: int = 200,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ):
        self.ema_period  = ema_period
        self.rsi_period  = rsi_period
        self.oversold    = oversold
        self.overbought  = overbought

    @property
    def name(self) -> str:
        return (
            f"ema_rsi({self.ema_period},{self.rsi_period},"
            f"{self.oversold},{self.overbought})"
        )

    @property
    def warmup_period(self) -> int:
        return self.ema_period

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        close     = df["close"]
        trend_ema = _ema(close, self.ema_period)
        rsi       = _rsi(close, self.rsi_period)

        in_uptrend  = close > trend_ema
        was_oversold = rsi.shift(1) <= self.oversold
        now_above    = rsi > self.oversold

        signals = np.zeros(len(df), dtype=np.int32)
        signals[in_uptrend & was_oversold & now_above] = 1
        signals[rsi >= self.overbought] = -1
        signals[: self.warmup_period] = 0
        return signals


class BollingerBand(BaseStrategy):
    """
    Bollinger Band mean-reversion.
    Buy when close crosses back above the lower band; sell at middle band.
    """

    def __init__(self, period: int = 20, std_mult: float = 2.0):
        self.period   = period
        self.std_mult = std_mult

    @property
    def name(self) -> str:
        return f"bollinger({self.period},{self.std_mult})"

    @property
    def warmup_period(self) -> int:
        return self.period

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        close      = df["close"]
        lower, mid, _ = _bollinger(close, self.period, self.std_mult)

        prev_close = close.shift(1)
        prev_lower = lower.shift(1)

        signals = np.zeros(len(df), dtype=np.int32)
        signals[(prev_close <= prev_lower) & (close > lower)] = 1
        signals[close >= mid] = -1
        signals[: self.warmup_period] = 0
        return signals


class Macd(BaseStrategy):
    """
    MACD histogram sign-change strategy.
    Long when MACD histogram crosses above zero; exit when it crosses below.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast          = fast
        self.slow          = slow
        self.signal_period = signal

    @property
    def name(self) -> str:
        return f"macd({self.fast},{self.slow},{self.signal_period})"

    @property
    def warmup_period(self) -> int:
        return self.slow + self.signal_period

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        close       = df["close"]
        macd_line   = _ema(close, self.fast) - _ema(close, self.slow)
        signal_line = _ema(macd_line, self.signal_period)
        histogram   = macd_line - signal_line

        positive      = (histogram > 0).astype(int)
        prev_positive = positive.shift(1)

        signals = np.zeros(len(df), dtype=np.int32)
        signals[(positive == 1) & (prev_positive == 0)] = 1
        signals[(positive == 0) & (prev_positive == 1)] = -1
        signals[: self.warmup_period] = 0
        return signals


class DonchianBreakout(BaseStrategy):
    """
    Donchian channel breakout (turtle-style momentum).

    Long when close breaks above the highest high of the last `entry`
    bars; exit when it breaks below the lowest low of the last `exit`
    bars.  Channels are computed on bars strictly before the current one
    (shifted by 1) so the signal uses no look-ahead.
    """

    def __init__(self, entry: int = 20, exit: int = 10):
        if entry < 2 or exit < 2:
            raise ValueError("entry and exit periods must be >= 2")
        self.entry = entry
        self.exit  = exit

    @property
    def name(self) -> str:
        return f"donchian({self.entry},{self.exit})"

    @property
    def warmup_period(self) -> int:
        return max(self.entry, self.exit) + 1

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        close = df["close"]
        upper = close.shift(1).rolling(self.entry, min_periods=self.entry).max()
        lower = close.shift(1).rolling(self.exit,  min_periods=self.exit).min()

        breakout_up   = close > upper
        breakdown_down = close < lower

        signals = np.zeros(len(df), dtype=np.int32)
        signals[breakout_up]    = 1
        signals[breakdown_down] = -1
        signals[: self.warmup_period] = 0
        return signals


class Supertrend(BaseStrategy):
    """
    Supertrend — an ATR-banded trend-following strategy.

    A band is placed `multiplier × ATR` away from the HL2 midpoint.  The
    trend flips long when price closes above the upper band and short when
    it closes below the lower band; the band then trails behind price.
    Adapts its stop distance to volatility, unlike fixed-percentage stops.
    """

    def __init__(self, period: int = 10, multiplier: float = 3.0):
        self.period     = period
        self.multiplier = multiplier

    @property
    def name(self) -> str:
        return f"supertrend({self.period},{self.multiplier})"

    @property
    def warmup_period(self) -> int:
        return self.period + 1

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        close = df["close"].to_numpy(dtype=float)
        hl2   = ((df["high"] + df["low"]) / 2).to_numpy(dtype=float)
        atr   = _atr(df, self.period).to_numpy(dtype=float)
        n     = len(close)

        upper = hl2 + self.multiplier * atr
        lower = hl2 - self.multiplier * atr

        # Trailing bands: tighten only in the direction of the trend.
        final_upper = upper.copy()
        final_lower = lower.copy()
        for i in range(1, n):
            final_upper[i] = (
                min(upper[i], final_upper[i - 1])
                if close[i - 1] <= final_upper[i - 1] else upper[i]
            )
            final_lower[i] = (
                max(lower[i], final_lower[i - 1])
                if close[i - 1] >= final_lower[i - 1] else lower[i]
            )

        # Direction: +1 uptrend, -1 downtrend; emit a signal on each flip.
        direction = np.ones(n, dtype=np.int32)
        signals   = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if close[i] > final_upper[i - 1]:
                direction[i] = 1
            elif close[i] < final_lower[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]

            if direction[i] == 1 and direction[i - 1] == -1:
                signals[i] = 1
            elif direction[i] == -1 and direction[i - 1] == 1:
                signals[i] = -1

        signals[: self.warmup_period] = 0
        return signals
