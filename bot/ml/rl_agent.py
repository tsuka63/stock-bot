"""
Reinforcement Learning strategy using PPO (Proximal Policy Optimisation).

The agent is trained on the first `train_frac` of the data, then generates
signals for the full dataset (trading only in the out-of-sample window).

Dependencies:
    pip install stable-baselines3 gymnasium torch

Usage:
    strat = RLStrategy(total_timesteps=50_000, train_frac=0.7)
    result = engine.run(strat, "AAPL", "1d", since="2023-01-01")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console

from bot.ml.features import extract_features
from bot.ml.rl_env import TradingEnv
from bot.strategy.base import BaseStrategy

console = Console()


class RLStrategy(BaseStrategy):
    """
    PPO agent trained to maximise portfolio log-return.

    Parameters
    ----------
    total_timesteps: Total environment steps for training
    train_frac:      Fraction of data used as training window
    model_path:      If set, save/load model weights here (.zip)
    net_arch:        MLP hidden layer sizes for actor + critic
    """

    def __init__(
        self,
        total_timesteps: int = 200_000,
        train_frac: float = 0.70,
        model_path: Optional[str] = None,
        net_arch: list[int] | None = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 256,
        ent_coef: float = 0.01,
        initial_capital: float = 10_000.0,
        fee_rate: float = 0.0,
        slippage_pct: float = 0.0005,
        drawdown_penalty_coef: float = 1.0,
        idle_penalty: float = 2e-3,
    ):
        self.total_timesteps       = total_timesteps
        self.train_frac            = train_frac
        self.model_path            = model_path
        self.net_arch              = net_arch or [256, 128, 64]
        self.learning_rate         = learning_rate
        self.n_steps               = n_steps
        self.batch_size            = batch_size
        self.ent_coef              = ent_coef
        self.initial_capital       = initial_capital
        self.fee_rate              = fee_rate
        self.slippage_pct          = slippage_pct
        self.drawdown_penalty_coef = drawdown_penalty_coef
        self.idle_penalty          = idle_penalty

        self._model    = None
        self._features: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return f"rl_ppo(steps={self.total_timesteps},train={self.train_frac:.0%})"

    @property
    def warmup_period(self) -> int:
        return 50

    def fit(self, df: pd.DataFrame) -> "RLStrategy":
        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise ImportError(
                "stable-baselines3 not installed.\n"
                "Run: pip install stable-baselines3 torch gymnasium"
            )

        features         = extract_features(df)
        self._features   = features
        n_train          = int(len(df) * self.train_frac)
        train_df         = df.iloc[:n_train].reset_index(drop=True)
        train_ft         = features[:n_train]

        env = TradingEnv(
            df=train_df, features=train_ft,
            initial_capital=self.initial_capital,
            fee_rate=self.fee_rate, slippage_pct=self.slippage_pct,
            drawdown_penalty_coef=self.drawdown_penalty_coef,
            idle_penalty=self.idle_penalty,
        )

        self._model = PPO(
            "MlpPolicy", env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            ent_coef=self.ent_coef,
            policy_kwargs=dict(net_arch=self.net_arch),
            verbose=0,
        )

        console.print(
            f"[bold cyan]Training PPO agent[/] — "
            f"{n_train:,} bars, {self.total_timesteps:,} timesteps …"
        )
        self._model.learn(total_timesteps=self.total_timesteps, progress_bar=True)
        console.print("[green]Training complete.[/]")

        if self.model_path:
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            self._model.save(self.model_path)
            console.print(f"Model saved → {self.model_path}")

        return self

    def load(self, path: str) -> "RLStrategy":
        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise ImportError("pip install stable-baselines3 torch gymnasium")
        self._model = PPO.load(path)
        console.print(f"Model loaded ← {path}")
        return self

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            self.fit(df)

        features = self._features if self._features is not None else extract_features(df)
        n        = len(df)
        n_train  = int(n * self.train_frac)
        signals  = np.zeros(n, dtype=np.int32)

        oos_df  = df.iloc[n_train:].reset_index(drop=True)
        oos_ft  = features[n_train:]
        env     = TradingEnv(
            df=oos_df, features=oos_ft,
            initial_capital=self.initial_capital,
            fee_rate=self.fee_rate, slippage_pct=self.slippage_pct,
            drawdown_penalty_coef=self.drawdown_penalty_coef,
            idle_penalty=self.idle_penalty,
        )
        obs, _      = env.reset()
        in_position = False

        for j in range(len(oos_df)):
            action, _ = self._model.predict(obs, deterministic=True)
            action    = int(action)

            if action == 1 and not in_position:
                signals[n_train + j] = 1
                in_position = True
            elif action == 2 and in_position:
                signals[n_train + j] = -1
                in_position = False

            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                break

        buy_count  = (signals == 1).sum()
        sell_count = (signals == -1).sum()
        console.print(
            f"[dim]RLStrategy OOS signals: {buy_count} buys, {sell_count} sells "
            f"({len(oos_df):,} bars)[/]"
        )
        return signals
