"""
HTML report for the 4-type valuation matrix: a 2×2 quadrant grid plus a
sortable detail table (PER / PBR / growth / type).
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from html import escape

from bot.valuation.classify import TYPES

# Quadrant placement + accent colour for each type
_QUADRANT = {
    "decline":  {"pos": "top-left",     "color": "#a78bfa"},
    "ceiling":  {"pos": "top-right",    "color": "#f472b6"},
    "bottom":   {"pos": "bottom-left",  "color": "#34d399"},
    "recovery": {"pos": "bottom-right", "color": "#fbbf24"},
}


def _fmt(v, suffix="", pct=False):
    if v is None:
        return "—"
    return f"{v*100:+.0f}%" if pct else f"{v:.2f}{suffix}"


def _quadrant_cell(key: str, rows: list[dict]) -> str:
    label, style, emoji = TYPES[key]
    color = _QUADRANT[key]["color"]
    chips = "".join(
        f'<div class="chip"><b>{escape(str(r["symbol"]))}</b> '
        f'{escape(str(r["name"])[:14])}</div>'
        for r in rows
    ) or '<div class="chip empty">—</div>'
    return (
        f'<div class="quad" style="border-color:{color}">'
        f'<div class="quad-h" style="color:{color}">{emoji} {label}'
        f'<span class="quad-style">{escape(style)}</span></div>'
        f'<div class="chips">{chips}</div></div>'
    )


def render_valuation_html(results: list[dict], title: str = "株の4タイプ分析") -> str:
    jst       = timezone(timedelta(hours=9))
    generated = datetime.now(jst).strftime("%Y-%m-%d %H:%M")

    by_type: dict[str, list[dict]] = {k: [] for k in TYPES}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)

    # Detail table (sorted: recovery first — the value+growth sweet spot)
    order = {"recovery": 0, "bottom": 1, "ceiling": 2, "decline": 3, "unknown": 4}
    rows_sorted = sorted(results, key=lambda r: order.get(r["type"], 9))
    trows = []
    for r in rows_sorted:
        label, style, emoji = TYPES[r["type"]]
        color = _QUADRANT.get(r["type"], {}).get("color", "#9aa3b2")
        trows.append(
            "<tr>"
            f'<td class="code">{escape(str(r["symbol"]))}</td>'
            f'<td class="name">{escape(str(r["name"])[:22])}</td>'
            f'<td data-sort="{r["per"] if r["per"] is not None else 9999}">{_fmt(r["per"])}</td>'
            f'<td data-sort="{r["pbr"] if r["pbr"] is not None else 9999}">{_fmt(r["pbr"])}</td>'
            f'<td data-sort="{r["growth"] if r["growth"] is not None else -9}">{_fmt(r["growth"], pct=True)}</td>'
            f'<td style="color:{color};font-weight:700">{emoji} {label}</td>'
            "</tr>"
        )
    tbody = "\n".join(trows)

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — {generated}</title>
<style>
  body {{ font-family:-apple-system,"Hiragino Sans",sans-serif; background:#0f1115;
         color:#e6e6e6; padding:24px; }}
  a {{ color:#6db3f2; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  .meta {{ color:#8a93a2; font-size:13px; margin-bottom:18px; }}
  .matrix {{ display:grid; grid-template-columns:64px 1fr 1fr; grid-template-rows:auto 1fr 1fr;
             gap:8px; max-width:900px; margin-bottom:8px; }}
  .axis {{ display:flex; align-items:center; justify-content:center; color:#8a93a2;
           font-size:12px; text-align:center; }}
  .axis.v {{ writing-mode:vertical-rl; }}
  .quad {{ background:#161a22; border:1px solid; border-radius:10px; padding:10px 12px; min-height:120px; }}
  .quad-h {{ font-weight:700; font-size:14px; margin-bottom:8px; }}
  .quad-style {{ color:#8a93a2; font-weight:400; font-size:11px; margin-left:6px; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:5px; }}
  .chip {{ background:#0f1115; border:1px solid #2c3340; border-radius:6px;
           padding:2px 7px; font-size:12px; }}
  .chip b {{ color:#6db3f2; }}
  .chip.empty {{ color:#555; border-style:dashed; }}
  table {{ border-collapse:collapse; width:100%; font-size:14px; margin-top:18px; max-width:900px; }}
  th,td {{ padding:8px 12px; border-bottom:1px solid #232733; text-align:right; }}
  th {{ color:#b8c0cc; background:#161a22; cursor:pointer; user-select:none; }}
  td.code {{ text-align:left; font-weight:700; color:#6db3f2; }}
  td.name {{ text-align:left; }}
  .legend {{ color:#8a93a2; font-size:12px; margin-top:14px; line-height:1.7; max-width:900px; }}
</style>
</head>
<body>
  <h1>📊 {title}</h1>
  <div class="meta">🕒 最終更新: {generated}（JST） ／ {len(results)}銘柄 ／
    <a href="index.html">← 株ランキングへ</a></div>

  <div class="matrix">
    <div class="axis"></div>
    <div class="axis">低成長 ←</div>
    <div class="axis">→ 高成長</div>
    <div class="axis v">高PBR・高PER</div>
    {_quadrant_cell("decline",  by_type["decline"])}
    {_quadrant_cell("ceiling",  by_type["ceiling"])}
    <div class="axis v">低PBR・低PER</div>
    {_quadrant_cell("bottom",   by_type["bottom"])}
    {_quadrant_cell("recovery", by_type["recovery"])}
  </div>

  <table id="t">
    <thead><tr>
      <th data-i="0">コード</th><th data-i="1">銘柄</th>
      <th data-i="2">PER</th><th data-i="3">PBR</th>
      <th data-i="4">成長率</th><th data-i="5">タイプ</th>
    </tr></thead>
    <tbody>
{tbody}
    </tbody>
  </table>

  <div class="legend">
    <b>4タイプ</b>: 🌱回復期=割安なのに成長中（妙味） ／ 💎どん底=資産バリュー（割安・低成長） ／
    🚀天井=グロース（割高・高成長） ／ ⚠️後退期=割高なのに停滞（注意）<br>
    判定: 割高=PBR≥{1.5}またはPER≥{20} ／ 高成長=利益(or売上)成長≥{10}%。
    成長データが無い銘柄は低成長扱い。列ヘッダーで並べ替え。
  </div>
<script>
  const table=document.getElementById('t'), tb=table.tBodies[0];
  let sc=-1, asc=false;
  table.querySelectorAll('th').forEach(th=>th.addEventListener('click',()=>{{
    const i=+th.dataset.i; asc=(i===sc)?!asc:false; sc=i;
    [...tb.rows].sort((a,b)=>{{
      const av=a.cells[i].dataset.sort??a.cells[i].textContent;
      const bv=b.cells[i].dataset.sort??b.cells[i].textContent;
      const an=parseFloat(av),bn=parseFloat(bv);
      const c=(!isNaN(an)&&!isNaN(bn))?an-bn:String(av).localeCompare(String(bv));
      return asc?c:-c;
    }}).forEach(r=>tb.appendChild(r));
  }}));
</script>
</body>
</html>"""


def save_valuation_html(results: list[dict], out_path: str, title: str = "株の4タイプ分析") -> str:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(render_valuation_html(results, title))
    return out_path
