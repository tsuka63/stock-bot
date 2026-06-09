from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseStrategy(ABC):
    """
    All strategies implement `generate_signals`.

    Signals are generated on close prices (no look-ahead) and executed at
    the *next* bar's open, which the simulation engine enforces.

    Signal values:
        1  → go long (or flip from short to long)
       -1  → close long (or go short, depending on engine mode)
        0  → hold current position
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy identifier."""

    @property
    def warmup_period(self) -> int:
        """Number of bars required before the first valid signal."""
        return 0

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> np.ndarray:
        """
        Args:
            df: OHLCV DataFrame with columns
                [timestamp, open, high, low, close, volume]

        Returns:
            int32 numpy array of shape (len(df),) with values in {-1, 0, 1}.
            The first `warmup_period` entries should be 0.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
