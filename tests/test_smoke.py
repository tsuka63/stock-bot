"""Smoke tests (no network) — run: python tests/test_smoke.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.ranking.limit   import classify, price_limit
from bot.ranking.scraper import _parse_standard


def test_price_limit_table():
    assert price_limit(450) == 80      # 200–500円 → ±80
    assert price_limit(3000) == 700    # 3000–5000円 → ±700


def test_stop_high_low():
    # 前日500 → 当日600 は +100（500円帯の制限値幅）= ストップ高
    assert classify(500, 600) == "high"
    assert classify(500, 400) == "low"
    assert classify(500, 510) == "normal"


def test_ranking_parser():
    # Mirrors Yahoo Finance JP ranking table item structure
    sample = (
        '<th scope="row" class="RankingTable__rank__2fAZ">1</th>'
        '<td class="RankingTable__detail__P452">'
        '<a href="https://finance.yahoo.co.jp/quote/8254.T" '
        'data-cl-params="_cl_link:name;_cl_position:0">(株)テスト</a></td>'
        '<span class="StyledNumber__value__3rXW">290</span>'
        '<span class="StyledNumber__value__3rXW">+80</span>'
        '<span class="StyledNumber__value__3rXW">+38.10</span>'
        '<span class="StyledNumber__value__3rXW">1000</span>'
    )
    rows = _parse_standard(sample)
    assert rows and rows[0]["symbol"] == "8254", "ranking parser broke — Yahoo structure?"
    assert rows[0]["rank"] == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"OK — {len(fns)} tests passed")
