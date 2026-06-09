"""
Gymnasium trading environment for the RL agent.

State:   18 market features + 4 portfolio state features = 22-dim float32 vector
Actions: Discrete(3) — 0=hold, 1=buy, 2=sell
Reward:  shaped reward = log return - drawdown penalty - idle penalty - fee penalty
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError("Install gymnasium:  pip install gymnasium")

from bot.ml.features import extract_features, N_FEATURES


class TradingEnv(gym.Env):
    """
    Parameters
    ----------
    df:              OHLCV DataFrame (training slice only)
    features:        Pre-computed feature matrix (len(df) × N_FEATURES).
                     Pass None to compute on the fly.
    initial_capital: Starting cash
    fee_rate:        Fee fraction (e.g. 0.001 = 0.1%; use 0.0 for Alpaca)
    slippage_pct:    One-way price slippage fraction
    """

    metadata = {"render_modes": []}
    OBS_DIM  = N_FEATURES + 4   # market features + position/pnl/cash/drawdown

    def __init__(
        self,
        df: pd.DataFrame,
        features: np.ndarray | None = None,
        initial_capital: float = 10_000.0,
        fee_rate: float = 0.0,
        slippage_pct: float = 0.0005,
        drawdown_penalty_coef: float = 1.0,
        idle_penalty: float = 2e-3,
    ):
        super().__init__()
        self.df                    = df.reset_index(drop=True)
        self._base_features        = features if features is not None else extract_features(df)
        self.initial_capital       = initial_capital
        self.fee_rate              = fee_rate
        self.slippage_pct          = slippage_pct
        self.drawdown_penalty_coef = drawdown_penalty_coef
        self.idle_penalty          = idle_penalty

        self._opens  = df["open"].to_numpy(dtype=float)
        self._closes = df["close"].to_numpy(dtype=float)
        self._n      = len(df)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)   # 0=hold, 1=buy, 2=sell

        self._step      = 0
        self._cash      = initial_capital
        self._position  = 0.0
        self._avg_entry = 0.0
        self._pv        = initial_capital
        self._max_pv    = initial_capital
        self._pending   = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step      = 0
        self._cash      = self.initial_capital
        self._position  = 0.0
        self._avg_entry = 0.0
        self._pv        = self.initial_capital
        self._max_pv    = self.initial_capital
        self._pending   = 0
        return self._obs(), {}

    def step(self, action: int):
        i        = self._step
        fee_paid = 0.0

        if self._pending != 0:
            slip = 1.0 + self._pending * self.slippage_pct
            fill = self._opens[i] * slip

            if self._pending == 1 and self._position == 0.0:
                trade_val = self._cash / (1.0 + self.fee_rate)
                fee       = trade_val * self.fee_rate
                if fill > 0 and self._cash >= trade_val + fee:
                    self._position  = trade_val / fill
                    self._avg_entry = fill
                    self._cash     -= trade_val + fee
                    fee_paid        = fee

            elif self._pending == -1 and self._position > 0.0:
                proceeds        = self._position * fill
                fee             = proceeds * self.fee_rate
                self._cash     += proceeds - fee
                self._position  = 0.0
                self._avg_entry = 0.0
                fee_paid        = fee

            self._pending = 0

        prev_pv      = self._pv
        self._pv     = self._cash + self._position * self._closes[i]
        self._max_pv = max(self._max_pv, self._pv)

        reward   = math.log(max(self._pv, 1e-6) / max(prev_pv, 1e-6))
        drawdown = max(0.0, 1.0 - self._pv / self._max_pv)
        reward  -= self.drawdown_penalty_coef * drawdown * 0.01

        if self._position == 0.0:
            reward -= self.idle_penalty

        reward -= fee_paid / self.initial_capital

        if action == 1:
            self._pending = 1
        elif action == 2:
            self._pending = -1

        self._step += 1
        terminated  = self._step >= self._n - 1

        if terminated and self._position > 0.0:
            last_price      = self._closes[-1]
            proceeds        = self._position * last_price
            fee             = proceeds * self.fee_rate
            self._cash     += proceeds - fee
            self._position  = 0.0
            self._pv        = self._cash

        return self._obs(), float(reward), terminated, False, {"portfolio_value": self._pv}

    def render(self):
        pass

    def _obs(self) -> np.ndarray:
        idx        = min(self._step, self._n - 1)
        mkt_feats  = self._base_features[idx]
        position   = np.float32(1.0 if self._position > 0 else 0.0)
        unrealised = np.float32(
            (self._closes[idx] / self._avg_entry - 1.0)
            if (self._position > 0 and self._avg_entry > 0) else 0.0
        )
        cash_frac = np.float32(self._cash / self.initial_capital)
        drawdown  = np.float32(max(0.0, 1.0 - self._pv / max(self._max_pv, 1e-6)))
        return np.append(mkt_feats, [position, unrealised, cash_frac, drawdown]).astype(np.float32)
