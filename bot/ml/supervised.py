"""
Supervised learning strategy using XGBoost.

Approach
--------
1. Feature engineering: 18 technical indicators from OHLCV (bot.ml.features)
2. Target: binary classification — will the next bar's close be higher?
3. Walk-forward training: train on first `train_frac` of bars, predict on the rest
4. Signal generation: buy when P(up) > buy_threshold, sell when P(up) < sell_threshold

Usage:
    strat = SupervisedStrategy(train_frac=0.7, buy_threshold=0.55)
    result = engine.run(strat, "AAPL", "1d", since="2023-01-01")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rich.console import Console

from bot.ml.features import extract_features, FEATURE_NAMES
from bot.strategy.base import BaseStrategy

console = Console()


class SupervisedStrategy(BaseStrategy):
    """
    XGBoost next-bar direction classifier.

    Parameters
    ----------
    train_frac:      Fraction of data used for training (e.g. 0.70)
    buy_threshold:   P(up) must exceed this to generate a buy signal
    sell_threshold:  P(up) must fall below this to generate a sell signal
    min_hold_bars:   Minimum bars to hold before selling (reduces fee churn)
    n_estimators:    XGBoost trees
    max_depth:       XGBoost tree depth
    """

    def __init__(
        self,
        train_frac: float = 0.70,
        buy_threshold: float = 0.53,
        sell_threshold: float | None = None,
        min_hold_bars: int = 3,
        pred_horizon: int = 5,
        label_deadband: float = 0.01,
        weight_by_magnitude: bool = True,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        gamma: float = 0.1,
        early_stopping_rounds: int = 20,
        random_state: int = 42,
    ):
        self.train_frac            = train_frac
        self.buy_threshold         = buy_threshold
        self.sell_threshold        = sell_threshold if sell_threshold is not None else 1 - buy_threshold
        self.min_hold_bars         = min_hold_bars
        self.pred_horizon          = pred_horizon
        self.label_deadband        = label_deadband
        self.weight_by_magnitude   = weight_by_magnitude
        self.n_estimators          = n_estimators
        self.max_depth             = max_depth
        self.learning_rate         = learning_rate
        self.subsample             = subsample
        self.colsample_bytree      = colsample_bytree
        self.min_child_weight      = min_child_weight
        self.gamma                 = gamma
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state          = random_state
        self._model = None

    @property
    def name(self) -> str:
        return (
            f"supervised_xgb(train={self.train_frac:.0%},"
            f"h={self.pred_horizon},band={self.label_deadband:.1%},"
            f"thresh={self.buy_threshold:.2f})"
        )

    @property
    def warmup_period(self) -> int:
        return 50

    def _build_model(self):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        return XGBClassifier(
            n_estimators          = self.n_estimators,
            max_depth             = self.max_depth,
            learning_rate         = self.learning_rate,
            subsample             = self.subsample,
            colsample_bytree      = self.colsample_bytree,
            min_child_weight      = self.min_child_weight,
            gamma                 = self.gamma,
            early_stopping_rounds = self.early_stopping_rounds,
            eval_metric           = "logloss",
            random_state          = self.random_state,
            verbosity             = 0,
        )

    def fit(self, df: pd.DataFrame) -> "SupervisedStrategy":
        """Train on the first `train_frac` rows of df.

        Target is the sign of the `pred_horizon`-bar forward return.  Bars
        whose forward move stays inside ±`label_deadband` are treated as
        ambiguous noise and excluded from training, which sharpens the
        decision boundary versus a raw next-bar up/down label.
        """
        X = extract_features(df)

        close   = df["close"].to_numpy(dtype=float)
        n_total = len(df)
        k       = self.pred_horizon

        # k-bar forward return (last k bars have no future → NaN)
        fwd_ret = np.full(n_total, np.nan)
        if n_total > k:
            fwd_ret[: n_total - k] = close[k:] / close[: n_total - k] - 1.0

        target = (fwd_ret > 0).astype(int)             # full-array label for OOS scoring
        # "Confident" bars: a clear move beyond the deadband, with a known future
        confident = (np.abs(fwd_ret) > self.label_deadband) & ~np.isnan(fwd_ret)

        n_train = int(n_total * self.train_frac)
        n_val   = max(50, int(n_train * 0.10))
        n_fit   = n_train - n_val

        def _slice(lo: int, hi: int):
            idx = np.where(confident[lo:hi])[0] + lo
            return X[idx], target[idx], fwd_ret[idx]

        X_fit, y_fit, r_fit = _slice(0, n_fit)
        X_val, y_val, _     = _slice(n_fit, n_train)
        X_oos, y_oos        = X[n_train:], target[n_train:]

        if len(X_fit) < 30 or len(np.unique(y_fit)) < 2:
            raise ValueError(
                f"Too few confident training samples ({len(X_fit)}). "
                "Lower label_deadband or pred_horizon."
            )

        # Weight clear, large moves more heavily than marginal ones
        sample_weight = None
        if self.weight_by_magnitude:
            sample_weight = 1.0 + np.abs(r_fit) / max(self.label_deadband, 1e-9)

        self._model = self._build_model()
        fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if len(X_val) == 0:
            fit_kwargs.pop("eval_set")
        self._model.fit(X_fit, y_fit, **fit_kwargs)

        acc_is  = (self._model.predict(X_fit) == y_fit).mean() if len(X_fit) else float("nan")
        acc_val = (self._model.predict(X_val) == y_val).mean() if len(X_val) else float("nan")
        # OOS accuracy is measured only on confident bars for a fair comparison
        oos_conf = confident[n_train:]
        if oos_conf.any():
            acc_oos = (self._model.predict(X_oos[oos_conf]) == y_oos[oos_conf]).mean()
        else:
            acc_oos = float("nan")
        best_it = getattr(self._model, "best_iteration", self.n_estimators)
        console.print(
            f"[dim]SupervisedStrategy: {len(X_fit):,} fit / {len(X_val):,} val / "
            f"{len(X_oos):,} OOS | acc IS={acc_is:.2%} val={acc_val:.2%} OOS={acc_oos:.2%} | "
            f"h={k} band={self.label_deadband:.1%} best_iter={best_it}[/]"
        )
        return self

    def feature_importance(self) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Call fit() first")
        return pd.Series(
            self._model.feature_importances_, index=FEATURE_NAMES
        ).sort_values(ascending=False)

    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            self.fit(df)

        X       = extract_features(df)
        prob_up = self._model.predict_proba(X)[:, 1]

        n       = len(df)
        n_train = int(n * self.train_frac)
        signals = np.zeros(n, dtype=np.int32)

        in_position = False
        bars_held   = 0
        for i in range(n_train, n):
            p = prob_up[i]
            if in_position:
                bars_held += 1
                if bars_held >= self.min_hold_bars and p < self.sell_threshold:
                    signals[i]  = -1
                    in_position = False
                    bars_held   = 0
            else:
                if p > self.buy_threshold:
                    signals[i]  = 1
                    in_position = True
                    bars_held   = 0

        return signals
