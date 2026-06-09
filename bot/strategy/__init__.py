from bot.strategy.base import BaseStrategy
from bot.strategy.technical import (
    SmaCrossover,
    EmaCrossover,
    EmaRsi,
    BollingerBand,
    Macd,
    DonchianBreakout,
    Supertrend,
)

REGISTRY: dict[str, type[BaseStrategy]] = {
    "sma_crossover": SmaCrossover,
    "ema_crossover": EmaCrossover,
    "ema_rsi":       EmaRsi,
    "bollinger":     BollingerBand,
    "macd":          Macd,
    "donchian":      DonchianBreakout,
    "supertrend":    Supertrend,
}

__all__ = [
    "BaseStrategy",
    "SmaCrossover", "EmaCrossover", "EmaRsi", "BollingerBand", "Macd",
    "DonchianBreakout", "Supertrend",
    "REGISTRY",
]
