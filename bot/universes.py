"""
Predefined symbol universes for multi-symbol scanning.

Usage:
    from bot.universes import UNIVERSES
    symbols = UNIVERSES["tech"]
"""

UNIVERSES: dict[str, list[str]] = {
    # Major US tech / mega-cap
    "tech": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META",
        "AMZN", "TSLA", "AMD",  "AVGO",  "INTC",
    ],
    # Broad S&P 500 sample (top ~20 by market cap)
    "sp500": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "LLY",  "AVGO",
        "JPM",  "TSLA", "UNH",  "V",    "XOM",   "WMT",  "MA",   "JNJ",
        "PG",   "COST", "HD",   "BAC",
    ],
    # Nikkei 225 top-20 (J-Quants 4-/5-digit codes)
    "n225": [
        "7203", "6758", "9984", "6861", "8306", "9433", "7974", "8035",
        "4502", "6098", "6367", "4063", "7751", "6902", "7267", "4661",
        "8316", "7011", "5108", "4519",
    ],
    # US-listed ETFs
    "etf": ["SPY", "QQQ", "IWM", "GLD", "TLT", "EEM", "VTI"],
    # Finance sector
    "finance": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP"],
    # Energy sector
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO"],
}
