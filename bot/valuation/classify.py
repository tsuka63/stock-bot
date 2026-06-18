"""
Classify stocks into the "4 types" value-investing matrix.

Two axes:
  • valuation (高PBR・高PER vs 低PBR・低PER) — how expensive the stock is
  • growth    (低成長 vs 高成長)            — how fast earnings/revenue grow

Four quadrants → investment style:

                    低成長            高成長
    高PBR/PER   後退期            天井（グロース株投資）
    低PBR/PER   どん底（資産     回復期（割安成長＝妙味）
                バリュー投資）

Fundamentals (PER, PBR, growth) come from Yahoo Finance via yfinance.
"""

from __future__ import annotations

# Thresholds (tunable). A stock is "expensive" if either multiple is elevated;
# "high growth" if earnings (or, failing that, revenue) grow past the cut-off.
PBR_HIGH    = 1.5
PER_HIGH    = 20.0
GROWTH_HIGH = 0.10   # +10%

# type key → (label, investment style, quadrant emoji)
TYPES = {
    "ceiling":  ("天井",   "グロース株投資",   "🚀"),
    "recovery": ("回復期", "割安成長（妙味）", "🌱"),
    "decline":  ("後退期", "割高・停滞（注意）", "⚠️"),
    "bottom":   ("どん底", "資産バリュー投資", "💎"),
    "unknown":  ("判定不可", "データ不足",       "❓"),
}


def classify(per: float | None, pbr: float | None, growth: float | None) -> str:
    """Return a type key: ceiling | recovery | decline | bottom | unknown."""
    has_val = (per is not None) or (pbr is not None)
    if not has_val:
        return "unknown"

    expensive = (
        (pbr is not None and pbr >= PBR_HIGH) or
        (per is not None and per >= PER_HIGH)
    )
    high_growth = (growth is not None and growth >= GROWTH_HIGH)

    if expensive and high_growth:
        return "ceiling"
    if expensive and not high_growth:
        return "decline"
    if (not expensive) and high_growth:
        return "recovery"
    return "bottom"


def fetch_fundamentals(symbol: str) -> dict:
    """
    Return {symbol, name, price, per, pbr, growth, type} for one stock.
    `growth` prefers earnings growth, falling back to revenue growth.
    """
    import yfinance as yf

    ticker = f"{symbol}.T" if (symbol.isdigit() and len(symbol) == 4) else symbol
    info = {}
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}

    per    = info.get("trailingPE")
    pbr    = info.get("priceToBook")
    egrow  = info.get("earningsGrowth")
    rgrow  = info.get("revenueGrowth")
    growth = egrow if egrow is not None else rgrow

    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    per, pbr, growth = _f(per), _f(pbr), _f(growth)
    return {
        "symbol": str(symbol),
        "name":   info.get("shortName") or info.get("longName") or str(symbol),
        "price":  _f(info.get("currentPrice")),
        "per":    per,
        "pbr":    pbr,
        "growth": growth,
        "type":   classify(per, pbr, growth),
    }


def analyze(symbols: list[str]) -> list[dict]:
    """Classify a list of symbols. Skips ones with no usable data only at display time."""
    return [fetch_fundamentals(s) for s in symbols]
