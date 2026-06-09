"""
Shared feature engineering for supervised learning and RL.

All features are computed from OHLCV data using only information
available at each bar's close — no look-ahead.

Returns a float32 numpy array of shape (n_bars, N_FEATURES).
NaN values in the warmup period are forward-filled then zero-filled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    # Price / trend
    "ret_1", "ret_5", "ret_10", "ret_20",
    "close_sma10", "close_sma20", "close_sma50",
    "close_ema12", "close_ema26",
    # Momentum
    "rsi_14",
    "macd_hist",
    "stoch_k", "stoch_d",
    # Volatility
    "bb_width", "bb_pos",
    "vol_10",
    # Volume
    "vol_ratio",
    "vol_force",
]

N_FEATURES = len(FEATURE_NAMES)   # 18


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)) / 100   # normalised 0–1


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]:
    ll    = low.rolling(k, min_periods=k).min()
    hh    = high.rolling(k, min_periods=k).max()
    denom = (hh - ll).replace(0, np.nan)
    pct_k = (close - ll) / denom
    pct_d = pct_k.rolling(d, min_periods=d).mean()
    return pct_k, pct_d


def _bollinger(s: pd.Series, n: int = 20, mult: float = 2.0):
    mid   = _sma(s, n)
    std   = s.rolling(n, min_periods=n).std()
    upper = mid + mult * std
    lower = mid - mult * std
    width = (upper - lower) / mid.replace(0, np.nan)
    pos   = (s - lower) / (upper - lower).replace(0, np.nan)
    return width, pos


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Args:
        df: OHLCV DataFrame with columns [timestamp, open, high, low, close, volume]

    Returns:
        float32 array of shape (len(df), N_FEATURES)
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    log_close = np.log(close.replace(0, np.nan))

    ret_1  = log_close.diff(1)
    ret_5  = log_close.diff(5)
    ret_10 = log_close.diff(10)
    ret_20 = log_close.diff(20)

    close_sma10 = close / _sma(close, 10).replace(0, np.nan) - 1
    close_sma20 = close / _sma(close, 20).replace(0, np.nan) - 1
    close_sma50 = close / _sma(close, 50).replace(0, np.nan) - 1
    close_ema12 = close / _ema(close, 12).replace(0, np.nan) - 1
    close_ema26 = close / _ema(close, 26).replace(0, np.nan) - 1

    rsi_14    = _rsi(close, 14)
    macd_line = _ema(close, 12) - _ema(close, 26)
    macd_sig  = _ema(macd_line, 9)
    macd_hist = (macd_line - macd_sig) / close.replace(0, np.nan)
    stoch_k, stoch_d = _stochastic(high, low, close)

    bb_width, bb_pos = _bollinger(close)
    vol_10 = ret_1.rolling(10, min_periods=10).std()

    vol_sma20 = _sma(volume, 20).replace(0, np.nan)
    vol_ratio = volume / vol_sma20
    sign_ret  = np.sign(close - df["open"])
    vol_force = sign_ret * vol_ratio

    feat_df = pd.DataFrame({
        "ret_1":       ret_1,
        "ret_5":       ret_5,
        "ret_10":      ret_10,
        "ret_20":      ret_20,
        "close_sma10": close_sma10,
        "close_sma20": close_sma20,
        "close_sma50": close_sma50,
        "close_ema12": close_ema12,
        "close_ema26": close_ema26,
        "rsi_14":      rsi_14,
        "macd_hist":   macd_hist,
        "stoch_k":     stoch_k,
        "stoch_d":     stoch_d,
        "bb_width":    bb_width,
        "bb_pos":      bb_pos,
        "vol_10":      vol_10,
        "vol_ratio":   vol_ratio,
        "vol_force":   vol_force,
    })

    feat_df = feat_df.ffill().fillna(0.0)

    # Clip extreme values (±5σ per feature) to reduce outlier impact.
    # Bounds are computed with an EXPANDING window (past data only), so the
    # clip applied at bar i never depends on future bars — no look-ahead.
    # Bars before `min_periods` get NaN bounds, which pandas treats as no-clip.
    clip_min_periods = 50
    for col in feat_df.columns:
        s     = feat_df[col]
        mu    = s.expanding(min_periods=clip_min_periods).mean()
        sigma = s.expanding(min_periods=clip_min_periods).std()
        feat_df[col] = s.clip(lower=mu - 5 * sigma, upper=mu + 5 * sigma)

    return feat_df.to_numpy(dtype=np.float32)
