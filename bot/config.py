from dataclasses import dataclass


@dataclass
class SimConfig:
    initial_capital: float = 10_000.0
    fee_rate: float = 0.0            # Alpaca is commission-free; set ~0.001 for JP brokers
    slippage_pct: float = 0.0005     # base one-way slippage (fraction of fill price)
    position_size_pct: float = 1.0   # fraction of cash per trade ("full" mode)
    annual_bars: int = 252           # fallback; overridden per timeframe + market
    market: str = "us"               # "us" (Alpaca) or "jp" (J-Quants)

    # Position sizing
    position_sizing: str = "full"    # "full" | "fixed_frac" | "kelly"
    risk_per_trade: float = 0.02     # fixed_frac: 2% of equity; kelly: max-cap
    kelly_lookback: int = 20         # min closed trades before Kelly activates

    # Risk management
    stop_loss_pct: float = 0.0       # 0 = disabled; 0.05 = 5% stop below entry
    take_profit_pct: float = 0.0     # 0 = disabled; 0.10 = 10% profit target

    # Execution realism
    slippage_atr_mult: float = 0.0       # extra slippage = mult × (ATR/price); 0 = off
    max_volume_participation: float = 0.0  # cap fill size to this × bar volume; 0 = off
    conservative_intrabar: bool = True   # gap-aware fills + assume SL before TP when both touchable
    atr_period: int = 14                 # lookback for the ATR used by the slippage model


# US equities: ~252 trading days/year, 390 min/day (09:30–16:00 ET)
ANNUAL_BARS = {
    "1m":  252 * 390,
    "5m":  252 * 78,
    "15m": 252 * 26,
    "30m": 252 * 13,
    "1h":  252 * 7,    # ~6.5 trading hours; Alpaca emits ~7 hourly bars/day
    "4h":  252 * 2,
    "1d":  252,
    "1w":  52,
    "1M":  12,
}

# Japanese equities: ~245 trading days/year, 330 min/day (09:00–11:30, 12:30–15:30 JST)
ANNUAL_BARS_JP = {
    "1m":  245 * 330,
    "5m":  245 * 66,
    "15m": 245 * 22,
    "30m": 245 * 11,
    "1h":  245 * 6,
    "4h":  245 * 2,
    "1d":  245,
    "1w":  52,
    "1M":  12,
}

TIMEFRAME_MS = {
    "1m":  60_000,
    "5m":  300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
    "1w":  604_800_000,
    "1M":  2_592_000_000,
}
