"""
Detect whether a stock closed at its daily price limit (ストップ高 / ストップ安).

Japanese exchanges cap how far a stock may move in one session — the 値幅制限.
A stock locked at the upper limit (ストップ高) usually has far more buyers than
sellers, so you often *can't actually buy it*; locked at the lower limit
(ストップ安) you often can't sell. Flagging this next to the ranking tells you
which high-ranked names are realistically tradable.

Detection: take the previous close as the base price (基準値段), look up the
limit from the table below, and check whether today's close sits at that
limit. Uses raw (unadjusted) closes so the comparison is exact.
"""

from __future__ import annotations

# (upper bound of 基準値段, 制限値幅) — TSE standard table.
_LIMIT_TABLE = [
    (100,        30),
    (200,        50),
    (500,        80),
    (700,        100),
    (1_000,      150),
    (1_500,      300),
    (2_000,      400),
    (3_000,      500),
    (5_000,      700),
    (7_000,      1_000),
    (10_000,     1_500),
    (15_000,     3_000),
    (20_000,     4_000),
    (30_000,     5_000),
    (50_000,     7_000),
    (70_000,     10_000),
    (100_000,    15_000),
    (150_000,    30_000),
    (200_000,    40_000),
    (300_000,    50_000),
    (500_000,    70_000),
    (700_000,    100_000),
    (1_000_000,  150_000),
    (1_500_000,  300_000),
    (3_000_000,  400_000),
    (5_000_000,  700_000),
    (10_000_000, 1_000_000),
]


def price_limit(base_price: float) -> float:
    """日本株の制限値幅（±円）for a given previous-close base price."""
    for upper, limit in _LIMIT_TABLE:
        if base_price < upper:
            return float(limit)
    return float(_LIMIT_TABLE[-1][1])


def classify(prev_close: float, close: float) -> str:
    """Return 'high' (ストップ高), 'low' (ストップ安) or 'normal'."""
    if prev_close <= 0:
        return "normal"
    lim  = price_limit(prev_close)
    move = close - prev_close
    eps  = max(1.0, lim * 0.01)        # tolerate tick / float rounding
    if move >= lim - eps:
        return "high"
    if move <= -lim + eps:
        return "low"
    return "normal"


def fetch_limit_status(symbols: list[str]) -> dict[str, str]:
    """
    {symbol: 'high'|'low'|'normal'} for each symbol, from the latest two
    daily closes (Yahoo Finance, raw/unadjusted). Symbols with too little
    data are omitted.
    """
    import yfinance as yf

    out: dict[str, str] = {}
    for sym in symbols:
        ticker = f"{sym}.T" if (sym.isdigit() and len(sym) == 4) else sym
        try:
            h = yf.Ticker(ticker).history(period="5d", auto_adjust=False)
            if len(h) >= 2:
                out[str(sym)] = classify(
                    float(h["Close"].iloc[-2]), float(h["Close"].iloc[-1])
                )
        except Exception:
            continue
    return out
