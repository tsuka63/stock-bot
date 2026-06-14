"""
CLI entry point.

Usage:
  # US equities (Alpaca)
  python -m bot fetch    --symbol AAPL --timeframe 1d --since 2023-01-01
  python -m bot fetch    --symbol TSLA --timeframe 1h --since 2023-01-01 --source alpaca

  # Japanese equities (J-Quants)
  python -m bot fetch    --symbol 7203 --timeframe 1d --since 2023-01-01 --source jquants

  # Backtest
  python -m bot backtest --symbol AAPL --timeframe 1d --strategy sma_crossover --since 2023-01-01
  python -m bot backtest --symbol 7203 --timeframe 1d --strategy macd --since 2023-01-01 --market jp
  python -m bot compare  --symbol AAPL --timeframe 1d --since 2023-01-01
  python -m bot pairs

  # Portfolio
  python -m bot portfolio --strategy sma_crossover --universe tech --since 2023-01-01
  python -m bot portfolio --strategy macd --symbols AAPL,MSFT,NVDA --since 2023-01-01 --allocation risk_parity

  # Multi-symbol scan
  python -m bot scan --universe tech --strategy sma_crossover --since 2023-01-01
"""

from __future__ import annotations

from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()   # read .env regardless of VS Code's python.terminal.useEnvFile setting

console = Console()
app = typer.Typer(help="Stock trading bot — backtesting engine (Alpaca + J-Quants)")

DB_PATH = "data/market.duckdb"


@app.command()
def fetch(
    symbol:    str = typer.Option("AAPL",    help="Ticker symbol (e.g. AAPL, 7203)"),
    timeframe: str = typer.Option("1d",      help="OHLCV timeframe"),
    since:     str = typer.Option(...,       help="Start date ISO-8601, e.g. 2023-01-01"),
    until: Optional[str] = typer.Option(None, help="End date ISO-8601 (default: now)"),
    source:    str = typer.Option("alpaca",  help="Data source: 'alpaca' or 'jquants'"),
    db_path:   str = typer.Option(DB_PATH,   help="DuckDB database path"),
):
    """Fetch historical OHLCV data and store locally."""
    from bot.data.fetcher import DataFetcher
    from bot.data.store   import DataStore

    console.print(f"Fetching [cyan]{symbol}[/] [{timeframe}] from [bold]{source}[/] …")
    fetcher = DataFetcher(source)
    df      = fetcher.fetch(symbol, timeframe, since=since, until=until)

    if df.empty:
        console.print("[red]No data returned — check symbol / timeframe / dates.[/]")
        raise typer.Exit(1)

    store   = DataStore(db_path)
    written = store.save(df, symbol, timeframe)
    console.print(f"[green]Stored {written} bars[/] ({df['timestamp'].min()} … {df['timestamp'].max()})")


@app.command()
def backtest(
    symbol:    str  = typer.Option("AAPL",          help="Ticker symbol"),
    timeframe: str  = typer.Option("1d",             help="OHLCV timeframe"),
    strategy:  str  = typer.Option("sma_crossover",  help="Strategy name"),
    since:     str  = typer.Option(...,              help="Start date"),
    until: Optional[str] = typer.Option(None,        help="End date"),
    capital:   float = typer.Option(10_000.0,        help="Initial capital"),
    fee:       float = typer.Option(0.0,             help="Fee rate (0 = commission-free)"),
    market:    str  = typer.Option("us",             help="Market: 'us' or 'jp' (affects Sharpe annualisation)"),
    sizing:    str  = typer.Option("full",           help="Position sizing: full | fixed_frac | kelly"),
    risk_per_trade: float = typer.Option(0.02,       help="Fraction of equity per trade (fixed_frac/kelly)"),
    stop_loss: float = typer.Option(0.0,             help="Stop-loss % below entry (0=off, e.g. 0.05=5%)"),
    take_profit: float = typer.Option(0.0,           help="Take-profit % above entry (0=off, e.g. 0.10=10%)"),
    walkforward: bool = typer.Option(False, "--walkforward", help="Run walk-forward validation after backtest"),
    wf_windows:  int  = typer.Option(6,   help="Number of walk-forward windows"),
    report:    bool  = typer.Option(False, "--report", help="Save HTML report"),
    db_path:   str  = typer.Option(DB_PATH,          help="DuckDB database path"),
):
    """Run a single strategy backtest."""
    import bot.ml  # noqa: F401 — registers ML strategies
    from bot.backtest.engine      import BacktestEngine
    from bot.backtest.report      import print_report, save_html_report
    from bot.backtest.walkforward import WalkForwardValidator
    from bot.config               import SimConfig
    from bot.strategy             import REGISTRY

    if strategy not in REGISTRY:
        console.print(f"[red]Unknown strategy '{strategy}'. Available: {list(REGISTRY)}[/]")
        raise typer.Exit(1)

    cfg    = SimConfig(
        initial_capital=capital, fee_rate=fee, market=market,
        position_sizing=sizing, risk_per_trade=risk_per_trade,
        stop_loss_pct=stop_loss, take_profit_pct=take_profit,
    )
    engine = BacktestEngine(config=cfg, db_path=db_path)
    strat  = REGISTRY[strategy]()
    result = engine.run(strat, symbol, timeframe, since, until)
    print_report(result)

    if report:
        out = f"data/report_{strategy}_{symbol}_{timeframe}.html"
        save_html_report(result, out)

    if walkforward:
        console.rule("[bold cyan]Walk-Forward Validation[/]")
        validator = WalkForwardValidator(engine, n_windows=wf_windows)
        wf = validator.run(
            strategy_factory=lambda: REGISTRY[strategy](),
            symbol=symbol, timeframe=timeframe,
            since=since, until=until or "2099-01-01",
        )
        validator.print_report(wf)
        if report:
            validator.save_html_report(wf, f"data/wf_{strategy}_{symbol}.html")


@app.command()
def compare(
    symbol:    str  = typer.Option("AAPL", help="Ticker symbol"),
    timeframe: str  = typer.Option("1d",   help="OHLCV timeframe"),
    since:     str  = typer.Option(...,   help="Start date"),
    until: Optional[str] = typer.Option(None, help="End date"),
    capital:  float = typer.Option(10_000.0,  help="Initial capital"),
    market:   str   = typer.Option("us",      help="Market: 'us' or 'jp'"),
    report:   bool  = typer.Option(False, "--report", help="Save comparison HTML"),
    db_path:  str   = typer.Option(DB_PATH,   help="DuckDB database path"),
):
    """Run ALL built-in strategies on the same data and compare."""
    from bot.backtest.engine import BacktestEngine
    from bot.backtest.report import print_comparison, save_comparison_html
    from bot.config          import SimConfig
    from bot.strategy        import REGISTRY

    cfg        = SimConfig(initial_capital=capital, market=market)
    engine     = BacktestEngine(config=cfg, db_path=db_path)
    strategies = [cls() for cls in REGISTRY.values()]

    with console.status("Running strategies …"):
        results = engine.compare(strategies, symbol, timeframe, since, until)

    print_comparison(results)

    if report:
        out = f"data/comparison_{symbol}_{timeframe}.html"
        save_comparison_html(results, out)


@app.command()
def pairs(db_path: str = typer.Option(DB_PATH, help="DuckDB database path")):
    """List all symbol/timeframe pairs stored in the local database."""
    from bot.data.store import DataStore
    from rich import print as rprint

    store = DataStore(db_path)
    df    = store.list_pairs()
    if df.empty:
        console.print("[yellow]No data in database. Run `fetch` first.[/]")
    else:
        rprint(df.to_string(index=False))


@app.command()
def scan(
    symbols:   Optional[str] = typer.Option(None,           help="Comma-separated symbols"),
    universe:  Optional[str] = typer.Option(None,           help="Predefined universe: tech, sp500, n225, etf, finance, energy"),
    strategy:  str           = typer.Option("sma_crossover", help="Strategy name or 'all'"),
    timeframe: str           = typer.Option("1d",            help="OHLCV timeframe"),
    since:     str           = typer.Option(...,             help="Start date"),
    until: Optional[str]     = typer.Option(None,            help="End date"),
    market:    str           = typer.Option("us",            help="Market: 'us' or 'jp'"),
    top_n:     int           = typer.Option(0,               help="Show top N (0=all)"),
    sort_by:   str           = typer.Option("sharpe",        help="Metric to sort by"),
    oos:       bool          = typer.Option(True, "--oos/--no-oos", help="Show out-of-sample (recent holdout) Sharpe to catch overfitting"),
    oos_frac:  float         = typer.Option(0.30,            help="Fraction of the period reserved as the OOS holdout tail"),
    auto_fetch: bool         = typer.Option(False, "--fetch", help="Auto-fetch missing symbols"),
    source:    str           = typer.Option("alpaca",        help="Data source for --fetch"),
    db_path:   str           = typer.Option(DB_PATH,         help="DuckDB database path"),
):
    """Scan a universe of symbols and rank by strategy performance."""
    import bot.ml  # noqa: F401
    from bot.backtest.engine import BacktestEngine
    from bot.config          import SimConfig
    from bot.strategy        import REGISTRY
    from bot.universes       import UNIVERSES

    import pandas as pd
    from rich import box
    from rich.table import Table

    if symbols:
        sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    elif universe:
        if universe not in UNIVERSES:
            console.print(f"[red]Unknown universe '{universe}'. Available: {list(UNIVERSES)}[/]")
            raise typer.Exit(1)
        sym_list = UNIVERSES[universe]
    else:
        console.print("[red]Provide --symbols or --universe.[/]")
        raise typer.Exit(1)

    if strategy == "all":
        strat_names = list(REGISTRY.keys())
    elif strategy not in REGISTRY:
        console.print(f"[red]Unknown strategy '{strategy}'. Available: {list(REGISTRY)}[/]")
        raise typer.Exit(1)
    else:
        strat_names = [strategy]

    if auto_fetch:
        from bot.data.fetcher import DataFetcher
        from bot.data.store   import DataStore
        fetcher = DataFetcher(source)
        store   = DataStore(db_path)
        for sym in sym_list:
            console.print(f"  Fetching [cyan]{sym}[/] …", end=" ")
            try:
                df_f = fetcher.fetch(sym, timeframe, since=since, until=until)
                if df_f.empty:
                    console.print("[yellow]no data[/]")
                else:
                    written = store.save(df_f, sym, timeframe)
                    console.print(f"[green]{written} bars[/]")
            except Exception as exc:
                console.print(f"[red]{exc}[/]")

    cfg    = SimConfig(market=market)
    engine = BacktestEngine(config=cfg, db_path=db_path)
    rows: list[dict] = []
    skipped = 0

    from bot.backtest.report import compute_metrics

    def _oos_metrics(result):
        """Out-of-sample Sharpe/return from the tail of the equity curve.

        Bar-to-bar returns of the equity-curve slice are valid regardless of the
        equity level carried in, so this needs no second backtest."""
        pv = result.portfolio_values
        n  = len(pv)
        split = int(n * (1 - oos_frac))
        if n - split < 5:
            return float("nan"), float("nan")
        tail_trades = [t for t in result.trades if t.get("bar_index", 0) >= split]
        om = compute_metrics(pv[split:], tail_trades, result.df.iloc[split:], result.annual_bars)
        return om.get("sharpe", float("nan")), om.get("total_return_pct", float("nan"))

    with console.status(f"Running {len(strat_names)} strategy × {len(sym_list)} symbols …"):
        for sym in sym_list:
            for sn in strat_names:
                try:
                    result = engine.run(REGISTRY[sn](), sym, timeframe, since, until)
                    m = result.metrics
                    row = {
                        "symbol":     sym,
                        "strategy":   sn,
                        "return_pct": m["total_return_pct"],
                        "cagr_pct":   m["cagr_pct"],
                        "sharpe":     m["sharpe"],
                        "calmar":     m["calmar"],
                        "max_dd_pct": m["max_drawdown_pct"],
                        "win_rate":   m["win_rate_pct"],
                        "trades":     m["total_trades"],
                        "bh_return":  m["bh_return_pct"],
                    }
                    if oos:
                        row["oos_sharpe"], row["oos_return"] = _oos_metrics(result)
                    rows.append(row)
                except ValueError as exc:
                    if "No data" in str(exc):
                        skipped += 1
                    else:
                        console.print(f"[yellow]  {sym}/{sn}: {exc}[/]")
                except Exception as exc:
                    console.print(f"[yellow]  {sym}/{sn}: {exc}[/]")

    if skipped:
        console.print(f"[yellow]{skipped} symbol(s) skipped (no data — run with --fetch)[/]")

    if not rows:
        console.print("[red]No results.[/]")
        raise typer.Exit(1)

    sort_col  = sort_by if sort_by in rows[0] else "sharpe"
    ascending = sort_col == "max_dd_pct"
    df_res = pd.DataFrame(rows).sort_values(sort_col, ascending=ascending)
    if top_n > 0:
        df_res = df_res.head(top_n)

    multi    = len(strat_names) > 1
    show_oos = oos and "oos_sharpe" in df_res.columns
    tbl   = Table(title=f"Scan Results — sorted by {sort_col}", box=box.SIMPLE_HEAVY)
    tbl.add_column("Symbol",  style="bold cyan")
    if multi:
        tbl.add_column("Strategy")
    tbl.add_column("Return",  justify="right")
    tbl.add_column("CAGR",    justify="right")
    tbl.add_column("Sharpe",  justify="right")
    if show_oos:
        tbl.add_column("OOS Sh", justify="right")
        tbl.add_column("Hold?",  justify="center")
    tbl.add_column("Calmar",  justify="right")
    tbl.add_column("MaxDD",   justify="right")
    tbl.add_column("WinRate", justify="right")
    tbl.add_column("Trades",  justify="right")
    tbl.add_column("B&H",     justify="right")

    def _c(v: float, sfx: str = "", pos_good: bool = True) -> str:
        color = "green" if (v > 0) == pos_good else "red"
        return f"[{color}]{v:+.2f}{sfx}[/]"

    import math as _math
    def _hold_flag(is_sh: float, oos_sh: float) -> tuple[str, str]:
        """(coloured OOS Sharpe, verdict mark) comparing in-sample vs OOS."""
        if oos_sh is None or (isinstance(oos_sh, float) and _math.isnan(oos_sh)):
            return "[dim]—[/]", "[dim]—[/]"
        oos_str = _c(oos_sh)
        if is_sh > 0 and oos_sh <= 0:
            return oos_str, "[bold red]✗[/]"      # sign flip → overfit
        if is_sh > 0 and oos_sh < is_sh * 0.5:
            return oos_str, "[yellow]△[/]"         # decayed a lot
        if is_sh > 0:
            return oos_str, "[green]✓[/]"          # holds up
        return oos_str, "[dim]·[/]"

    for _, row in df_res.iterrows():
        cells = [row["symbol"]]
        if multi:
            cells.append(row["strategy"])
        cells += [
            _c(row["return_pct"], "%"),
            _c(row["cagr_pct"],   "%"),
            _c(row["sharpe"]),
        ]
        if show_oos:
            oos_cell, flag = _hold_flag(row["sharpe"], row.get("oos_sharpe"))
            cells += [oos_cell, flag]
        cells += [
            _c(row["calmar"]),
            f"[red]{row['max_dd_pct']:.2f}%[/]",
            f"{row['win_rate']:.1f}%",
            str(int(row["trades"])),
            _c(row["bh_return"],  "%"),
        ]
        tbl.add_row(*cells)

    console.print("\n")
    console.print(tbl)
    shown = len(df_res)
    total = len(rows)
    console.print(f"[dim]{shown} result(s) shown" + (f" of {total}" if total > shown else "") + "[/]")
    if show_oos:
        console.print(
            f"[dim]OOS Sh = Sharpe on the most recent {oos_frac:.0%} (held out). "
            "Hold? ✓ generalises · △ decayed >50% · ✗ sign-flipped (overfit). "
            "Prefer ✓ over a high in-sample Sharpe.[/]"
        )


@app.command()
def rank(
    action:    str = typer.Argument("scan", help="Action: scan | show | history"),
    symbol:    Optional[str] = typer.Option(None,  help="Symbol for 'history' action"),
    top_n:     int           = typer.Option(50,    help="Top N rank threshold"),
    min_days:  int           = typer.Option(3,     help="Min consecutive days in ranking"),
    lookback:  int           = typer.Option(30,    help="Lookback window in days"),
    limit:     int           = typer.Option(20,    help="Max candidates to display (compact view)"),
    full:      bool          = typer.Option(False, "--full", help="Full view: all candidates, all columns (needs wide terminal)"),
    report:    bool          = typer.Option(False, "--report", help="Also write an HTML report to data/candidates.html"),
    backtest:  bool          = typer.Option(False, "--backtest", help="Auto-backtest candidates after scan"),
    db_path:   str           = typer.Option(DB_PATH, help="DuckDB database path"),
):
    """Yahoo Finance Japan ranking commands: scan (scrape + store), show (candidates), history."""
    from bot.ranking.scraper import fetch_all
    from bot.ranking.tracker import RankingTracker
    from rich.table import Table
    from rich       import box
    from datetime   import date

    tracker = RankingTracker(db_path)

    if action == "scan":
        import time
        console.rule("[bold cyan]Scraping Yahoo Finance Japan[/]")
        with console.status("Fetching rankings …"):
            rankings = fetch_all(delay=1.5)
        totals = {k: len(v) for k, v in rankings.items()}
        console.print(
            f"  [green]gainers[/]: {totals['gainers']}  "
            f"[green]volume[/]: {totals['volume']}  "
            f"[green]hot[/]: {totals['hot']}"
        )
        today   = date.today().isoformat()
        written = tracker.store(rankings, as_of=today)
        console.print(f"  Stored [bold]{written}[/] rows for {today}")

        # Fall through to show candidates
        action = "show"

    if action == "show":
        console.rule("[bold cyan]Ranking Candidates[/]")
        df = tracker.get_candidates(top_n=top_n, min_days=min_days, lookback_days=lookback)
        dates = tracker.available_dates()
        console.print(f"  History: [dim]{len(dates)} day(s) stored — "
                      f"{dates[-1] if dates else '?'} … {dates[0] if dates else '?'}[/]")

        if df.empty:
            console.print(
                f"[yellow]No candidates (top {top_n}, ≥{min_days} days).[/] "
                "Run `rank scan` daily to build history."
            )
            return

        def _score_color(s: float) -> str:
            return "bold green" if s >= 45 else "yellow" if s >= 35 else "white"

        def _traj(traj: float, arrow_only: bool = False) -> str:
            mark = "↑" if traj > 0.3 else "↓" if traj < -0.3 else "·"
            if traj > 0.3:   return f"[green]↑{traj:.1f}[/]"
            if traj < -0.3:  return f"[red]↓{abs(traj):.1f}[/]"
            return "·"

        if full:
            # ── Full view: every candidate, all columns ──────────────────
            tbl = Table(title=f"Candidates — top {top_n}, ≥{min_days} days (by score)",
                        box=box.SIMPLE_HEAVY)
            tbl.add_column("Code",   style="bold cyan", no_wrap=True)
            tbl.add_column("Name",   width=20, no_wrap=True)
            tbl.add_column("Score",  justify="right")
            tbl.add_column("Streak", justify="right")
            tbl.add_column("Days",   justify="right")
            tbl.add_column("AvgRank",justify="right")
            tbl.add_column("Best",   justify="right")
            tbl.add_column("Traj",   justify="right")
            tbl.add_column("Types",  width=24, no_wrap=True)
            for _, row in df.iterrows():
                tbl.add_row(
                    str(row["symbol"]), str(row["name"])[:19] or "—",
                    f"[{_score_color(row['score'])}]{row['score']:.0f}[/]",
                    f"[bold green]{int(row['streak'])}d[/]",
                    str(int(row["appearances"])),
                    f"{row['avg_rank']:.1f}", str(int(row["best_rank"])),
                    _traj(row["trajectory"]), str(row["rank_types"]),
                )
            console.print(tbl)
            console.print(f"[dim]{len(df)} candidate(s) shown[/]")
        else:
            # ── Compact view: top `limit`, condensed columns ─────────────
            _TYPE_JP = {"gainers": "値", "volume": "出", "hot": "注"}
            def _types_short(s: str) -> str:
                return "".join(_TYPE_JP.get(x.strip(), "?") for x in s.split(","))

            tbl = Table(title=f"Candidates — top {len(df) if len(df) < limit else limit} by score",
                        box=box.SIMPLE_HEAVY)
            tbl.add_column("Code",  style="bold cyan", no_wrap=True)
            tbl.add_column("Name",  width=14, no_wrap=True)
            tbl.add_column("Score", justify="right")
            tbl.add_column("連続",  justify="right")
            tbl.add_column("平均",  justify="right")
            tbl.add_column("最高",  justify="right")
            tbl.add_column("推移",  justify="right")
            tbl.add_column("種別",  justify="center")
            for _, row in df.head(limit).iterrows():
                tbl.add_row(
                    str(row["symbol"]), str(row["name"])[:13] or "—",
                    f"[{_score_color(row['score'])}]{row['score']:.0f}[/]",
                    f"{int(row['streak'])}d",
                    f"{row['avg_rank']:.0f}", str(int(row["best_rank"])),
                    _traj(row["trajectory"]), _types_short(str(row["rank_types"])),
                )
            console.print(tbl)
            if len(df) > limit:
                console.print(f"[dim]Showing {limit} of {len(df)} — use --full for all, "
                              f"--limit N to change[/]")
            console.print("[dim]種別: 値=値上がり 出=出来高 注=注目株 ／ "
                          "推移↑=順位上昇中 ／ Score=継続30+幅25+質25+上昇20[/]")

        if report:
            from bot.ranking.report import save_candidates_html
            out = save_candidates_html(
                df, dates, "data/candidates.html", top_n=top_n, min_days=min_days
            )
            console.print(f"[green]HTML report →[/] {out}")

        if backtest:
            import bot.ml  # noqa: F401
            from datetime              import date, timedelta
            from bot.backtest.engine   import BacktestEngine
            from bot.backtest.report   import print_comparison
            from bot.config            import SimConfig
            from bot.data.fetcher      import DataFetcher
            from bot.data.store        import DataStore
            from bot.strategy          import REGISTRY

            cfg     = SimConfig(market="jp", fee_rate=0.001)
            engine  = BacktestEngine(config=cfg, db_path=db_path)
            strats  = list(REGISTRY.values())
            # Pull up-to-date prices from Yahoo Finance so freshly-trending
            # candidates can be tested on recent data (no J-Quants 3-month lag).
            fetcher = DataFetcher("yfinance")
            store   = DataStore(db_path)
            since   = (date.today() - timedelta(days=730)).isoformat()  # ~2y window

            for sym in df["symbol"].astype(str):
                console.rule(f"[cyan]{sym}[/]")
                try:
                    fresh = fetcher.fetch(sym, "1d", since=since)
                    if not fresh.empty:
                        store.save(fresh, sym, "1d")
                except Exception as exc:
                    console.print(f"[yellow]  fetch failed: {exc}[/]")

                results = []
                for cls in strats:
                    try:
                        results.append(engine.run(cls(), sym, "1d", since, None))
                    except Exception:
                        pass
                if results:
                    print_comparison(results)
                else:
                    console.print("[yellow]  Not enough data for backtest[/]")

    elif action == "history":
        if not symbol:
            console.print("[red]Provide --symbol for 'history' action.[/]")
            raise typer.Exit(1)
        df = tracker.get_history(symbol, lookback_days=lookback)
        if df.empty:
            console.print(f"[yellow]No ranking history for {symbol}.[/]")
        else:
            from rich import print as rprint
            rprint(df.to_string(index=False))


@app.command()
def holdings(
    action:   str = typer.Argument("list", help="Action: list | set | set-fund | remove | report"),
    symbol:   Optional[str]   = typer.Option(None,  help="Ticker (e.g. 7203) or fund code (e.g. 0331418A)"),
    name:     str             = typer.Option("",    help="Display name"),
    shares:   Optional[float] = typer.Option(None,  help="Share count (for 'set')"),
    cost:     Optional[float] = typer.Option(None,  help="Average cost per share (for 'set')"),
    date:     str             = typer.Option("",    help="Entry date YYYY-MM-DD (for 'set')"),
    currency: str             = typer.Option("JPY", help="Price currency for 'set' (e.g. USD for US stocks)"),
    value:    Optional[float] = typer.Option(None,  help="Current valuation in yen (for 'set-fund')"),
    gain:     Optional[float] = typer.Option(None,  help="Unrealized gain in yen (for 'set-fund')"),
    out:      str             = typer.Option("data/portfolio.html", help="Output HTML path (for 'report')"),
    db_path:  str             = typer.Option(DB_PATH, help="DuckDB database path"),
):
    """Manage personal holdings and build the encrypted portfolio report.

    Examples:
      python -m bot holdings set --symbol 7203 --name トヨタ --shares 100 --cost 2800
      python -m bot holdings set-fund --symbol 0331418A --name オルカン --value 873280 --gain 188264
      python -m bot holdings remove --symbol 7203
      python -m bot holdings list
      python -m bot holdings report          # writes encrypted data/portfolio.html
    """
    from bot.holdings.store import HoldingsStore
    store = HoldingsStore(db_path)

    if action == "set":
        if not symbol or shares is None or cost is None:
            console.print("[red]set requires --symbol, --shares and --cost.[/]")
            raise typer.Exit(1)
        store.set(symbol, shares, cost, name=name, asset_type="stock",
                  entry_date=date, currency=currency)
        dstr = f" (取得日 {date})" if date else ""
        cstr = f" [{currency}建て]" if currency != "JPY" else ""
        console.print(f"[green]Saved[/] {symbol}: {shares:g} shares @ ¥{cost:,.0f}{dstr}{cstr}")
        action = "list"   # fall through to show the book

    elif action == "set-fund":
        # Register a mutual fund from its current valuation + unrealized gain.
        # Units (口) are back-calculated from today's NAV so the value matches.
        from bot.holdings.funds import fetch_nav, KNOWN_FUNDS
        if not symbol or value is None:
            console.print("[red]set-fund requires --symbol (fund code) and --value.[/]")
            raise typer.Exit(1)
        g = gain or 0.0
        nav = fetch_nav(symbol)
        if not nav:
            console.print(f"[red]Could not fetch NAV for fund {symbol}.[/]")
            raise typer.Exit(1)
        units      = value * 10_000 / nav            # 口数
        cost_basis = value - g
        avg_nav    = cost_basis * 10_000 / units if units else 0.0   # 取得時の平均基準価額
        fund_name  = name or KNOWN_FUNDS.get(symbol, symbol)
        store.set(symbol, units, avg_nav, name=fund_name, asset_type="fund")
        console.print(
            f"[green]Saved fund[/] {fund_name}: {units:,.0f}口 "
            f"(NAV {nav:,.0f} / 取得単価 {avg_nav:,.0f} / 評価 ¥{value:,.0f})"
        )
        action = "list"

    elif action == "set-manual":
        # Manually-valued asset (e.g. private/pre-IPO shares like SpaceX) that
        # has no public price feed. Value is stored as-is; update it by hand.
        if not symbol or value is None:
            console.print("[red]set-manual requires --symbol and --value (current valuation).[/]")
            raise typer.Exit(1)
        g = gain or 0.0
        cost_basis = value - g                       # gain = value - cost
        n_sh       = shares if shares is not None else 1.0
        per_share  = cost_basis / n_sh if n_sh else cost_basis
        store.set(symbol, n_sh, per_share, name=name or symbol,
                  asset_type="manual", entry_date=date, manual_value=value)
        console.print(
            f"[green]Saved manual[/] {name or symbol}: {n_sh:g}株 "
            f"(取得 ¥{cost_basis:,.0f} / 評価 ¥{value:,.0f} / 損益 ¥{g:+,.0f})"
        )
        console.print("[dim]※ 自動更新されません。評価額が変わったら再度 set-manual で更新してください。[/]")
        action = "list"

    elif action == "remove":
        if not symbol:
            console.print("[red]remove requires --symbol.[/]")
            raise typer.Exit(1)
        n = store.remove(symbol)
        console.print(f"[green]Removed {symbol}[/]" if n else f"[yellow]{symbol} not found[/]")
        action = "list"

    if action == "list":
        from bot.holdings.report import compute_portfolio
        from rich.table import Table
        from rich       import box

        df = store.list_all()
        if df.empty:
            console.print("[yellow]No holdings yet. Add one with `holdings set`.[/]")
            return

        with console.status("Fetching current prices …"):
            pf = compute_portfolio(df)

        tbl = Table(title="My Holdings", box=box.SIMPLE_HEAVY)
        for c in ("Code", "Name", "Shares", "Cost", "Price", "Value", "P&L", "P&L%"):
            tbl.add_column(c, justify="right" if c not in ("Code", "Name") else "left")
        for p in pf["positions"]:
            pnl = p["pnl"]
            color = "green" if (pnl or 0) >= 0 else "red"
            pl_s  = "—" if pnl is None else f"[{color}]{'+' if pnl>=0 else ''}¥{pnl:,.0f}[/]"
            pp_s  = "—" if p["pnl_pct"] is None else f"[{color}]{p['pnl_pct']:+.1f}%[/]"
            tbl.add_row(
                p["symbol"], p["name"], f"{p['shares']:g}",
                f"¥{p['avg_cost']:,.0f}",
                "—" if p["price"] is None else f"¥{p['price']:,.0f}",
                "—" if p["mkt_val"] is None else f"¥{p['mkt_val']:,.0f}",
                pl_s, pp_s,
            )
        console.print(tbl)
        t = pf["totals"]
        tcolor = "green" if t["pnl"] >= 0 else "red"
        console.print(
            f"  取得 [bold]¥{t['cost']:,.0f}[/] → 評価 [bold]¥{t['value']:,.0f}[/]  "
            f"損益 [{tcolor}]{'+' if t['pnl']>=0 else ''}¥{t['pnl']:,.0f} "
            f"({t['pnl_pct']:+.1f}%)[/]"
        )

    elif action == "report":
        import os
        from bot.holdings.report import save_portfolio_html
        password = os.getenv("PORTFOLIO_PASSWORD", "")
        if not password:
            console.print("[red]Set PORTFOLIO_PASSWORD in .env first (used to encrypt the report).[/]")
            raise typer.Exit(1)
        df = store.list_all()
        if df.empty:
            console.print("[yellow]No holdings to report.[/]")
            raise typer.Exit(1)
        path = save_portfolio_html(df, password, out_path=out)
        console.print(f"[green]Encrypted portfolio →[/] {path} ({len(df)} holdings)")


@app.command()
def portfolio(
    symbols:    Optional[str] = typer.Option(None,            help="Comma-separated symbols"),
    universe:   Optional[str] = typer.Option(None,            help="Predefined universe: tech, sp500, n225, …"),
    strategy:   str           = typer.Option("sma_crossover", help="Strategy name"),
    timeframe:  str           = typer.Option("1d",             help="OHLCV timeframe"),
    since:      str           = typer.Option(...,              help="Start date"),
    until:  Optional[str]     = typer.Option(None,             help="End date"),
    capital:    float         = typer.Option(10_000.0,         help="Total portfolio capital"),
    market:     str           = typer.Option("us",             help="Market: 'us' or 'jp'"),
    allocation: str           = typer.Option("equal_weight",   help="Allocation: equal_weight | risk_parity"),
    report:     bool          = typer.Option(False, "--report", help="Save HTML report"),
    db_path:    str           = typer.Option(DB_PATH,          help="DuckDB database path"),
):
    """Run a strategy across multiple symbols as a portfolio."""
    import bot.ml  # noqa: F401
    from bot.backtest.engine    import BacktestEngine
    from bot.backtest.portfolio import PortfolioEngine, print_portfolio_report, save_portfolio_html
    from bot.config             import SimConfig
    from bot.strategy           import REGISTRY
    from bot.universes          import UNIVERSES

    if symbols:
        sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    elif universe:
        if universe not in UNIVERSES:
            console.print(f"[red]Unknown universe '{universe}'. Available: {list(UNIVERSES)}[/]")
            raise typer.Exit(1)
        sym_list = UNIVERSES[universe]
    else:
        console.print("[red]Provide --symbols or --universe.[/]")
        raise typer.Exit(1)

    if strategy not in REGISTRY:
        console.print(f"[red]Unknown strategy '{strategy}'. Available: {list(REGISTRY)}[/]")
        raise typer.Exit(1)

    cfg         = SimConfig(initial_capital=capital, market=market)
    bt_engine   = BacktestEngine(config=cfg, db_path=db_path)
    port_engine = PortfolioEngine(bt_engine)

    console.print(
        f"Portfolio: [cyan]{strategy}[/] · {len(sym_list)} symbols "
        f"· allocation=[bold]{allocation}[/]"
    )
    with console.status("Running portfolio backtest …"):
        result = port_engine.run(
            strategy_factory = lambda: REGISTRY[strategy](),
            symbols          = sym_list,
            timeframe        = timeframe,
            since            = since,
            until            = until,
            allocation       = allocation,
        )

    print_portfolio_report(result)

    if report:
        syms_tag = universe or "custom"
        out = f"data/portfolio_{strategy}_{syms_tag}_{timeframe}.html"
        save_portfolio_html(result, out)
