"""
Japanese mutual-fund NAV (基準価額) fetcher.

Investment trusts aren't on the stock price feed, so their value is read from
Yahoo Finance Japan's fund pages by association code (協会コード). NAV is
quoted per 10,000 units (口), the standard Japanese convention.

A holding's current value is therefore:  units(口) × NAV / 10,000
"""

from __future__ import annotations

import re
import requests

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
_NAV_RE = re.compile(
    r'PriceBoard__price__\w+">.*?StyledNumber__value__\w+">([\d,]+)<',
    re.DOTALL,
)

# Convenience registry for the funds we hold (code → display name).
# Codes are Yahoo Finance JP 協会コード (8 chars).
KNOWN_FUNDS: dict[str, str] = {
    "03311247": "eMAXIS 日経半導体株インデックス",
    "03311187": "eMAXIS Slim 米国株式(S&P500)",
    "0331418A": "eMAXIS Slim 全世界株式(オルカン)",
    "03317172": "eMAXIS Slim 国内株式(TOPIX)",
    "0331C177": "eMAXIS Slim 新興国株式インデックス",
    "04311181": "iFreeNEXT FANG+インデックス",
}

# A Japanese fund code on Yahoo is 8 alphanumerics; stock tickers are 4 digits
# (optionally with a market suffix). This lets the report route each holding.
_FUND_CODE_RE = re.compile(r"^[0-9A-Z]{8}$")


def is_fund_code(symbol: str) -> bool:
    return bool(_FUND_CODE_RE.match(symbol))


def fetch_nav(code: str, timeout: int = 15) -> float | None:
    """Return the latest NAV (基準価額, per 10,000 units) for a fund code."""
    resp = requests.get(
        f"https://finance.yahoo.co.jp/quote/{code}",
        headers={"User-Agent": _UA},
        timeout=timeout,
    )
    resp.raise_for_status()
    m = _NAV_RE.search(resp.text)
    if not m:
        return None
    return float(m.group(1).replace(",", ""))


def units_from_valuation(valuation: float, nav: float) -> float:
    """Back-calculate units (口) so that units × NAV / 10,000 == valuation."""
    if nav <= 0:
        return 0.0
    return valuation * 10_000 / nav
