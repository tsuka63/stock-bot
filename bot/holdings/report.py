"""
Encrypted portfolio report.

Computes current value / P&L for your holdings (live prices via Yahoo
Finance), then encrypts the whole portfolio payload with your password using
AES-256-GCM. Only the ciphertext is written into the HTML; the page asks for
the password and decrypts entirely in the browser via the Web Crypto API.

Without the password, viewing the page source shows only encrypted bytes —
this is real protection, not a JavaScript "hide".

Crypto parameters (must match the in-page JS):
  - PBKDF2-HMAC-SHA256, 200,000 iterations, 16-byte salt → 256-bit key
  - AES-GCM, 12-byte IV, tag appended to ciphertext
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime

import pandas as pd
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_PBKDF2_ITERS = 200_000


# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------

def compute_portfolio(holdings: pd.DataFrame, source: str = "yfinance") -> dict:
    """Return a payload dict: {generated, positions[], totals{}}.

    Stocks are priced via the live equity feed; mutual funds (asset_type
    'fund') are priced from their NAV (基準価額, per 10,000 units), so a fund's
    market value is units × NAV / 10,000 and `price` shown is NAV-per-unit.
    """
    from datetime          import date as _date
    from bot.data.fetcher  import DataFetcher
    from bot.holdings.funds import fetch_nav
    from bot.ranking.screen import consensus_signal

    fetcher = DataFetcher(source)
    positions = []
    tot_cost = tot_val = 0.0

    for _, h in holdings.iterrows():
        symbol = str(h["symbol"])
        shares = float(h["shares"])
        cost   = float(h["avg_cost"])
        name   = str(h["name"]) or symbol
        atype  = str(h["asset_type"]) if "asset_type" in h and h["asset_type"] else "stock"
        edate  = str(h["entry_date"]) if "entry_date" in h and h["entry_date"] else ""

        # `price` and `cost` are per-share for stocks, per-10,000-口 (基準価額)
        # for funds; `shares` is the share count for stocks, 口数 for funds.
        price   = float("nan")
        mkt_val = float("nan")
        cons    = None            # consensus signal — stocks only
        if atype == "fund":
            cost_basis = shares / 10_000 * cost
            try:
                nav = fetch_nav(symbol)          # 基準価額, per 10,000 口
                if nav:
                    price   = nav
                    mkt_val = shares / 10_000 * nav
            except Exception:
                pass
        else:
            cost_basis = cost * shares
            try:
                # 2-year history: latest price + today's strategy consensus
                df = fetcher.fetch(symbol, "1d", since=_two_years_ago())
                if not df.empty:
                    price   = float(df["close"].iloc[-1])
                    mkt_val = price * shares
                    cons    = consensus_signal(df)
            except Exception:
                pass

        # Holding period in days, if an entry date was recorded
        holding_days = None
        if edate:
            try:
                holding_days = (_date.today() - _date.fromisoformat(edate)).days
            except ValueError:
                pass

        pnl        = (mkt_val - cost_basis) if mkt_val == mkt_val else float("nan")
        pnl_pct    = (pnl / cost_basis * 100) if (cost_basis and pnl == pnl) else float("nan")

        tot_cost += cost_basis
        if mkt_val == mkt_val:
            tot_val += mkt_val

        positions.append({
            "symbol":     symbol,
            "name":       name,
            "type":       atype,
            "entry_date": edate,
            "holding_days": holding_days,
            "buy_today":  cons["buy_today"]  if cons else None,
            "sell_today": cons["sell_today"] if cons else None,
            "long_count": cons["long_count"] if cons else None,
            "strat_total": cons["total"]     if cons else None,
            "shares":     round(shares, 0) if atype == "fund" else shares,
            "avg_cost":   round(cost, 0),
            "price":      None if price != price else round(price, 0 if atype == "fund" else 2),
            "cost_basis": round(cost_basis, 0),
            "mkt_val":    None if mkt_val != mkt_val else round(mkt_val, 0),
            "pnl":        None if pnl != pnl else round(pnl, 0),
            "pnl_pct":    None if pnl_pct != pnl_pct else round(pnl_pct, 2),
        })

    tot_pnl = tot_val - tot_cost
    # Allocation % per position (of current value)
    for p in positions:
        p["alloc"] = round(p["mkt_val"] / tot_val * 100, 1) if (tot_val and p["mkt_val"]) else None

    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "positions": positions,
        "totals": {
            "cost":    round(tot_cost, 0),
            "value":   round(tot_val, 0),
            "pnl":     round(tot_pnl, 0),
            "pnl_pct": round(tot_pnl / tot_cost * 100, 2) if tot_cost else 0.0,
        },
    }


def _recent_since() -> str:
    from datetime import date, timedelta
    return (date.today() - timedelta(days=10)).isoformat()


def _two_years_ago() -> str:
    from datetime import date, timedelta
    return (date.today() - timedelta(days=730)).isoformat()


# ---------------------------------------------------------------------------
# Encryption
# ---------------------------------------------------------------------------

def encrypt_payload(payload: dict, password: str) -> dict:
    salt = os.urandom(16)
    iv   = os.urandom(12)
    key  = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERS, 32)
    ct   = AESGCM(key).encrypt(iv, json.dumps(payload).encode(), None)  # ct||tag
    b64  = lambda b: base64.b64encode(b).decode()
    return {"salt": b64(salt), "iv": b64(iv), "ct": b64(ct), "iter": _PBKDF2_ITERS}


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def render_portfolio_html(enc: dict, n_positions: int) -> str:
    enc_json = json.dumps(enc)
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>マイポートフォリオ 🔒</title>
<style>
  body {{ font-family:-apple-system,"Hiragino Sans",sans-serif; background:#0f1115;
         color:#e6e6e6; padding:24px; }}
  h1 {{ font-size:20px; margin:0 0 16px; }}
  .gate {{ background:#161a22; border:1px solid #232733; border-radius:10px;
           padding:24px; max-width:360px; }}
  .gate input {{ width:100%; padding:10px; border-radius:8px; border:1px solid #2c3340;
                 background:#0f1115; color:#e6e6e6; font-size:15px; box-sizing:border-box; }}
  .gate button {{ margin-top:12px; width:100%; padding:10px; border:none; border-radius:8px;
                  background:#2563eb; color:#fff; font-size:15px; font-weight:700; cursor:pointer; }}
  .gate button:hover {{ background:#1d4ed8; }}
  .err {{ color:#f87171; font-size:13px; margin-top:10px; min-height:18px; }}
  .cards {{ display:flex; gap:14px; margin-bottom:20px; flex-wrap:wrap; }}
  .card {{ background:#161a22; border:1px solid #232733; border-radius:10px;
           padding:14px 20px; min-width:150px; }}
  .card .lbl {{ color:#8a93a2; font-size:12px; margin-bottom:4px; }}
  .card .val {{ font-size:22px; font-weight:700; }}
  table {{ border-collapse:collapse; width:100%; font-size:14px; }}
  th, td {{ padding:9px 12px; border-bottom:1px solid #232733; text-align:right; }}
  th {{ color:#b8c0cc; background:#161a22; }}
  td.code {{ text-align:left; font-weight:700; color:#6db3f2; }}
  td.name {{ text-align:left; }}
  tr:hover td {{ background:#161a22; }}
  .up {{ color:#34d399; font-weight:700; }}
  .down {{ color:#f87171; font-weight:700; }}
  .meta {{ color:#8a93a2; font-size:12px; margin-top:14px; }}
  .hidden {{ display:none; }}
</style>
</head>
<body>
  <h1>💼 マイポートフォリオ <span style="font-size:14px;color:#8a93a2">🔒 暗号化</span></h1>

  <div id="gate" class="gate">
    <div style="margin-bottom:10px;color:#b8c0cc">パスワードを入力してください</div>
    <input id="pw" type="password" placeholder="パスワード" autofocus
           onkeydown="if(event.key==='Enter')unlock()">
    <button onclick="unlock()">🔓 表示する</button>
    <div id="err" class="err"></div>
  </div>

  <div id="content" class="hidden"></div>

<script>
  const ENC = {enc_json};
  const N_POS = {n_positions};

  const b64d = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
  const yen  = v => v === null ? '—' : '¥' + Math.round(v).toLocaleString('ja-JP');
  const cls  = v => v >= 0 ? 'up' : 'down';
  const sign = v => (v >= 0 ? '+' : '') + yen(v);

  async function decrypt(password) {{
    const salt = b64d(ENC.salt), iv = b64d(ENC.iv), ct = b64d(ENC.ct);
    const base = await crypto.subtle.importKey(
      'raw', new TextEncoder().encode(password), 'PBKDF2', false, ['deriveKey']);
    const key = await crypto.subtle.deriveKey(
      {{ name:'PBKDF2', salt, iterations: ENC.iter, hash:'SHA-256' }},
      base, {{ name:'AES-GCM', length:256 }}, false, ['decrypt']);
    const pt = await crypto.subtle.decrypt({{ name:'AES-GCM', iv }}, key, ct);
    return JSON.parse(new TextDecoder().decode(pt));
  }}

  async function unlock() {{
    const pw = document.getElementById('pw').value;
    const err = document.getElementById('err');
    err.textContent = '';
    try {{
      const data = await decrypt(pw);
      render(data);
      document.getElementById('gate').classList.add('hidden');
      document.getElementById('content').classList.remove('hidden');
    }} catch (e) {{
      err.textContent = 'パスワードが違います';
    }}
  }}

  function render(d) {{
    const t = d.totals;
    let rows = '';
    for (const p of d.positions) {{
      const pc = p.pnl === null ? '' : cls(p.pnl);
      const unit = p.type === 'fund' ? '口' : '株';
      // Holding period + today's strategy consensus (stocks only)
      const hold = p.holding_days === null || p.holding_days === undefined
                   ? '—' : p.holding_days + '日';
      let sigCell = '—';
      if (p.type !== 'fund' && p.strat_total) {{
        const tot = p.strat_total;
        let today = '<span class="flat">—</span>';
        if (p.buy_today > 0)      today = '<span class="up">🟢買い' + p.buy_today + '</span>';
        else if (p.sell_today > 0) today = '<span class="down">🔴売り' + p.sell_today + '</span>';
        const gcls = (p.long_count / tot) >= 0.5 ? 'up' : 'flat';
        sigCell = today + ' <span class="' + gcls + '">' + p.long_count + '/' + tot + '</span>';
      }}
      rows += `<tr>
        <td class="code">${{p.symbol}}</td><td class="name">${{p.name}}</td>
        <td>${{p.shares.toLocaleString('ja-JP')}}${{unit}}</td>
        <td>${{yen(p.avg_cost)}}</td><td>${{yen(p.price)}}</td>
        <td>${{yen(p.mkt_val)}}</td>
        <td class="${{pc}}">${{p.pnl === null ? '—' : sign(p.pnl)}}</td>
        <td class="${{pc}}">${{p.pnl_pct === null ? '—' : (p.pnl_pct>=0?'+':'')+p.pnl_pct.toFixed(1)+'%'}}</td>
        <td>${{hold}}</td><td>${{sigCell}}</td></tr>`;
    }}
    document.getElementById('content').innerHTML = `
      <div class="cards">
        <div class="card"><div class="lbl">取得総額</div><div class="val">${{yen(t.cost)}}</div></div>
        <div class="card"><div class="lbl">現在評価額</div><div class="val">${{yen(t.value)}}</div></div>
        <div class="card"><div class="lbl">評価損益</div><div class="val ${{cls(t.pnl)}}">${{sign(t.pnl)}}</div></div>
        <div class="card"><div class="lbl">損益率</div><div class="val ${{cls(t.pnl)}}">${{(t.pnl_pct>=0?'+':'')+t.pnl_pct.toFixed(1)}}%</div></div>
      </div>
      <table><thead><tr>
        <th>コード</th><th>銘柄</th><th>株数</th><th>取得単価</th><th>現在値</th>
        <th>評価額</th><th>損益</th><th>損益率</th><th>保有</th><th>今日/勢い</th>
      </tr></thead><tbody>${{rows}}</tbody></table>
      <div class="meta">更新: ${{d.generated}} ／ ${{N_POS}}銘柄 ／ 価格はYahoo Finance<br>
        「今日/勢い」= 7戦略中の本日<span class="up">買い</span>/<span class="down">売り</span>シグナル数 と
        現在ロング状態の戦略数（株のみ）。判断材料としてどうぞ。</div>`;
  }}
</script>
</body>
</html>"""


def save_portfolio_html(
    holdings: pd.DataFrame,
    password: str,
    out_path: str = "data/portfolio.html",
    source: str = "yfinance",
) -> str:
    payload = compute_portfolio(holdings, source=source)
    enc     = encrypt_payload(payload, password)
    html    = render_portfolio_html(enc, len(payload["positions"]))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
