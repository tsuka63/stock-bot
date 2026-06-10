"""
Self-contained HTML report for ranking candidates.

Renders the composite-score candidate table as a single static HTML file
(embedded CSS + a little JS for click-to-sort columns). No external
dependencies, so it opens straight in any browser.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from html import escape

import pandas as pd

_TYPE_JP = {"gainers": "値上がり", "volume": "出来高", "hot": "注目株"}


def _score_class(score: float) -> str:
    if score >= 45:
        return "score-hi"
    if score >= 35:
        return "score-mid"
    return "score-lo"


def _traj_cell(traj: float) -> str:
    if traj > 0.3:
        return f'<span class="up">▲ {traj:.1f}</span>'
    if traj < -0.3:
        return f'<span class="down">▼ {abs(traj):.1f}</span>'
    return '<span class="flat">—</span>'


def _types_badges(types: str) -> str:
    out = []
    for t in types.split(","):
        t = t.strip()
        label = _TYPE_JP.get(t, t)
        out.append(f'<span class="badge badge-{t}">{escape(label)}</span>')
    return " ".join(out)


def _fmt_sharpe(v: float) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return '<span class="flat">—</span>'
    cls = "up" if v > 0 else "down"
    return f'<span class="{cls}">{v:+.2f}</span>'


_HOLD_COLOR = {"✓": "#34d399", "△": "#fbbf24", "✗": "#f87171", "·": "#8a93a2", "—": "#8a93a2"}


def _hold_badge(flag: str) -> str:
    mapping = {
        "✓": ('<span class="hold-ok">✓ 汎化</span>'),
        "△": ('<span class="hold-weak">△ 劣化</span>'),
        "✗": ('<span class="hold-bad">✗ 過学習</span>'),
        "·": ('<span class="flat">·</span>'),
        "—": ('<span class="flat">—</span>'),
    }
    return mapping.get(flag, f'<span class="flat">{escape(str(flag))}</span>')


def _sparkline(equity, split: int, color: str, w: int = 180, h: int = 44, pad: int = 3) -> str:
    """Inline-SVG equity curve: IS portion grey, OOS portion `color`, split dotted."""
    if equity is None or len(equity) < 2:
        return '<span class="flat">—</span>'
    pv = [float(v) for v in equity]
    n  = len(pv)
    lo, hi = min(pv), max(pv)
    rng = (hi - lo) or 1.0
    split = max(1, min(split, n - 1))

    def x(i: int) -> float:
        return pad + (w - 2 * pad) * i / (n - 1)

    def y(v: float) -> float:
        return pad + (h - 2 * pad) * (1 - (v - lo) / rng)

    is_pts  = " ".join(f"{x(i):.1f},{y(pv[i]):.1f}" for i in range(0, split + 1))
    oos_pts = " ".join(f"{x(i):.1f},{y(pv[i]):.1f}" for i in range(split, n))
    sx = x(split)
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f'<line x1="{sx:.1f}" y1="0" x2="{sx:.1f}" y2="{h}" '
        f'stroke="#444" stroke-dasharray="3,3" stroke-width="1"/>'
        f'<polyline points="{is_pts}" fill="none" stroke="#8a93a2" stroke-width="1.5"/>'
        f'<polyline points="{oos_pts}" fill="none" stroke="{color}" stroke-width="2"/>'
        f'</svg>'
    )


def _render_screen_section(screen_df: pd.DataFrame) -> str:
    """Strategy-screening table: per candidate, the strategy that holds up OOS."""
    if screen_df is None or screen_df.empty:
        return ""

    has_equity = "equity" in screen_df.columns

    rows = []
    for _, r in screen_df.iterrows():
        hold = str(r["hold"])
        chart_cell = ""
        if has_equity:
            color = _HOLD_COLOR.get(hold, "#8a93a2")
            spark = _sparkline(r.get("equity"), int(r.get("split", 0) or 0), color)
            chart_cell = f"<td>{spark}</td>"
        rows.append(
            "<tr>"
            f'<td class="code">{escape(str(r["symbol"]))}</td>'
            f'<td class="name">{escape(str(r["best_strategy"]))}</td>'
            f'<td data-sort="{r["is_sharpe"]}">{_fmt_sharpe(r["is_sharpe"])}</td>'
            f'<td data-sort="{r["oos_sharpe"] if not pd.isna(r["oos_sharpe"]) else -99}">'
            f'{_fmt_sharpe(r["oos_sharpe"])}</td>'
            f'<td>{_hold_badge(hold)}</td>'
            f'<td data-sort="{r["return_pct"]}">{_fmt_sharpe(r["return_pct"])}%</td>'
            f'<td data-sort="{int(r["trades"])}">{int(r["trades"])}</td>'
            f'{chart_cell}'
            "</tr>"
        )
    body = "\n".join(rows)
    chart_header = '<th>資産曲線 (IS／OOS)</th>' if has_equity else ""
    return f"""
  <h2>🧪 戦略スクリーニング（OOS検証付き）</h2>
  <div class="meta">各候補で<b>最もアウトオブサンプル(OOS)で通用した</b>戦略を表示。
    OOS Sharpe順。過学習(✗)は額面通り受け取らないこと。<br>
    資産曲線: <span style="color:#8a93a2">グレー=IS(過去70%)</span> ／
    <span style="color:#34d399">色=OOS(直近30%)</span> ／ 点線=分割点。
    分割点の後ろが上向きなら本物。</div>
  <table id="s">
    <thead><tr>
      <th data-i="0">コード</th>
      <th data-i="1">最良戦略</th>
      <th data-i="2">Sharpe(IS)</th>
      <th data-i="3">Sharpe(OOS) ▼</th>
      <th data-i="4">判定</th>
      <th data-i="5">リターン(IS)</th>
      <th data-i="6">取引</th>
      {chart_header}
    </tr></thead>
    <tbody>
{body}
    </tbody>
  </table>
"""


def render_candidates_html(
    df: pd.DataFrame,
    history_dates: list[str],
    top_n: int,
    min_days: int,
    screen_df: pd.DataFrame | None = None,
) -> str:
    """Build the full HTML document string for the candidate table."""
    # Always report in JST — GitHub's runners are UTC, which would otherwise
    # show the wrong day/time to a Japanese user.
    jst       = timezone(timedelta(hours=9))
    generated = datetime.now(jst).strftime("%Y-%m-%d %H:%M")
    latest    = history_dates[0] if history_dates else "—"
    span = (
        f"{history_dates[-1]} 〜 {latest}（{len(history_dates)}日分）"
        if history_dates else "データなし"
    )

    rows_html = []
    for _, r in df.iterrows():
        rows_html.append(
            "<tr>"
            f'<td class="code">{escape(str(r["symbol"]))}</td>'
            f'<td class="name">{escape(str(r["name"]) or "—")}</td>'
            f'<td class="{_score_class(r["score"])}" data-sort="{r["score"]}">'
            f'{r["score"]:.0f}</td>'
            f'<td data-sort="{int(r["streak"])}">{int(r["streak"])}日</td>'
            f'<td data-sort="{int(r["appearances"])}">{int(r["appearances"])}</td>'
            f'<td data-sort="{r["avg_rank"]}">{r["avg_rank"]:.1f}</td>'
            f'<td data-sort="{int(r["best_rank"])}">{int(r["best_rank"])}位</td>'
            f'<td data-sort="{r["trajectory"]}">{_traj_cell(r["trajectory"])}</td>'
            f'<td class="types">{_types_badges(str(r["rank_types"]))}</td>'
            "</tr>"
        )
    tbody = "\n".join(rows_html) if rows_html else (
        '<tr><td colspan="9" class="empty">'
        f'候補なし（上位{top_n}位・{min_days}日以上）。'
        'データが溜まると表示されます。</td></tr>'
    )

    screen_section = _render_screen_section(screen_df)

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>株ランキング候補 — {generated}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, "Hiragino Sans", sans-serif;
          margin: 0; padding: 24px; background: #0f1115; color: #e6e6e6; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  h2 {{ font-size: 17px; margin: 28px 0 4px; }}
  .meta {{ color: #8a93a2; font-size: 13px; margin-bottom: 16px; }}
  .hold-ok   {{ color: #34d399; font-weight: 700; }}
  .hold-weak {{ color: #fbbf24; font-weight: 700; }}
  .hold-bad  {{ color: #f87171; font-weight: 700; }}
  .legend {{ font-size: 12px; color: #8a93a2; margin: 12px 0 18px; line-height: 1.6; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ padding: 8px 10px; text-align: right; border-bottom: 1px solid #232733; }}
  th {{ position: sticky; top: 0; background: #161a22; cursor: pointer;
        user-select: none; color: #b8c0cc; font-weight: 600; white-space: nowrap; }}
  th:hover {{ color: #fff; }}
  td.code {{ text-align: left; font-weight: 700; color: #6db3f2; font-variant-numeric: tabular-nums; }}
  td.name {{ text-align: left; max-width: 220px; overflow: hidden;
             text-overflow: ellipsis; white-space: nowrap; }}
  td.types {{ text-align: left; }}
  tr:hover td {{ background: #161a22; }}
  .score-hi  {{ color: #34d399; font-weight: 700; }}
  .score-mid {{ color: #fbbf24; font-weight: 700; }}
  .score-lo  {{ color: #9aa3b2; }}
  .up   {{ color: #34d399; }}
  .down {{ color: #f87171; }}
  .flat {{ color: #6b7280; }}
  .badge {{ display: inline-block; padding: 1px 7px; border-radius: 10px;
            font-size: 11px; margin-right: 2px; }}
  .badge-gainers {{ background: #3b2a1a; color: #f0a868; }}
  .badge-volume  {{ background: #1a2a3b; color: #68b0f0; }}
  .badge-hot     {{ background: #3b1a2a; color: #f068a8; }}
  .empty {{ text-align: center; color: #8a93a2; padding: 40px; }}
</style>
</head>
<body>
  <h1>📈 株ランキング候補</h1>
  <div class="meta">🕒 最終更新: {generated}（JST） ／ データ対象: {span} ／ 条件: 上位{top_n}位・{min_days}日以上</div>
  <table id="t">
    <thead><tr>
      <th data-i="0">コード</th>
      <th data-i="1">銘柄</th>
      <th data-i="2">スコア ▼</th>
      <th data-i="3">連続</th>
      <th data-i="4">登場</th>
      <th data-i="5">平均順位</th>
      <th data-i="6">最高</th>
      <th data-i="7">推移</th>
      <th data-i="8">種別</th>
    </tr></thead>
    <tbody>
{tbody}
    </tbody>
  </table>
  <div class="legend">
    <b>スコア</b> = 継続性(30) + 複数ランキング登場の幅(25) + 順位の高さ(25) + 順位上昇トレンド(20)<br>
    <b>推移</b> ▲ = ランキング順位が上昇中（注目度が高まっている） ／ ▼ = 下降中<br>
    列ヘッダーをクリックすると並べ替えできます。
  </div>
{screen_section}
<script>
  document.querySelectorAll('table').forEach(table => {{
    const tbody = table.tBodies[0];
    let sortCol = -1, asc = false;
    table.querySelectorAll('th').forEach(th => th.addEventListener('click', () => {{
      const i = +th.dataset.i;
      asc = (i === sortCol) ? !asc : false;
      sortCol = i;
      const rows = [...tbody.rows];
      rows.sort((a, b) => {{
        const av = a.cells[i].dataset.sort ?? a.cells[i].textContent;
        const bv = b.cells[i].dataset.sort ?? b.cells[i].textContent;
        const an = parseFloat(av), bn = parseFloat(bv);
        const cmp = (!isNaN(an) && !isNaN(bn)) ? an - bn : String(av).localeCompare(String(bv));
        return asc ? cmp : -cmp;
      }});
      rows.forEach(r => tbody.appendChild(r));
    }}));
  }});
</script>
</body>
</html>"""


def save_candidates_html(
    df: pd.DataFrame,
    history_dates: list[str],
    out_path: str,
    top_n: int = 50,
    min_days: int = 3,
    screen_df: pd.DataFrame | None = None,
) -> str:
    html = render_candidates_html(df, history_dates, top_n, min_days, screen_df)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
