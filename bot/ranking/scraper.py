"""
Scrape Yahoo Finance Japan stock rankings.

Supported ranking types:
  gainers  — 値上がり率  (/ranking/up)
  volume   — 出来高      (/ranking/volume)
  hot      — 注目株      (/ranking/hot)

Each returns a list of dicts: {rank, symbol, name, value}
"""

from __future__ import annotations

import re
import time
import requests

_UA     = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
_BASE   = "https://finance.yahoo.co.jp/stocks/ranking"
_PARAMS = {"market": "tokyo", "term": "daily", "category": "stock"}


def _get(path: str, retries: int = 3) -> str:
    import time
    last = None
    for i in range(retries):
        try:
            resp = requests.get(
                f"{_BASE}/{path}",
                params=_PARAMS,
                headers={"User-Agent": _UA},
                timeout=15,
            )
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            last = exc
            if i < retries - 1:
                time.sleep(2 * (i + 1))
    raise last


def _parse_standard(html: str) -> list[dict]:
    """Parse standard table-based ranking pages (gainers, volume)."""
    ranks  = re.findall(r'RankingTable__rank__\w+">(\d+)<', html)
    codes  = re.findall(r'quote/(\d{4})\.T[^"]*" data-cl-params="_cl_link:name', html)
    names  = re.findall(r'data-cl-params="_cl_link:name[^"]+">([^<]+)</a>', html)
    # All numeric values in the table (price, abs-change, pct-change, volume…)
    values = re.findall(r'StyledNumber__value__\w+">([^<]+)</span>', html)

    results: list[dict] = []
    for i, (rank, code, name) in enumerate(zip(ranks, codes, names)):
        # ~4 StyledNumber values per row; index 2 is the % change for gainers
        # For volume ranking it appears at index 3 — just take index 2 as primary
        val = values[i * 4 + 2] if i * 4 + 2 < len(values) else ""
        results.append({
            "rank":   int(rank),
            "symbol": code,
            "name":   name.strip(),
            "value":  val,
        })
    return results


def _parse_hot(html: str) -> list[dict]:
    """
    Parse the 注目株 ranking page.

    The page uses Next.js streaming and embeds ranking data as an escaped
    JSON blob.  Each entry looks like:
        \"rank\":5,\"stockCode\":\"6752.T\"
    """
    entries = re.findall(
        r'\\+"rank\\+":\s*(\d+),\\+"stockCode\\+":\\+"(\d{4})\.T\\+"',
        html,
    )
    # Also try the lighter escape variant
    if not entries:
        entries = re.findall(
            r'\"rank\":(\d+),\"stockCode\":\"(\d{4})\.T\"',
            html,
        )

    seen: dict[str, int] = {}
    for rank_str, code in entries:
        rank = int(rank_str)
        # Keep the best (lowest) rank seen per symbol
        if code not in seen or rank < seen[code]:
            seen[code] = rank

    return [
        {"rank": rank, "symbol": code, "name": "", "value": ""}
        for code, rank in sorted(seen.items(), key=lambda x: x[1])
    ]


def fetch_gainers() -> list[dict]:
    """値上がり率 top 50."""
    return _parse_standard(_get("up"))


def fetch_volume() -> list[dict]:
    """出来高 top 50."""
    return _parse_standard(_get("volume"))


def fetch_hot() -> list[dict]:
    """注目株ランキング (Yahoo Finance 閲覧急上昇)."""
    return _parse_hot(_get("hot"))


def fetch_all(delay: float = 1.5) -> dict[str, list[dict]]:
    """
    Fetch all three rankings with a small delay between requests to avoid
    rate-limiting by Yahoo Finance.

    Returns dict keyed by rank type: 'gainers', 'volume', 'hot'.
    """
    result: dict[str, list[dict]] = {}
    for name, fn in [("gainers", fetch_gainers), ("volume", fetch_volume), ("hot", fetch_hot)]:
        result[name] = fn()
        time.sleep(delay)
    return result
