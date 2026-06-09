"""
Ranking history storage and candidate detection.

Stores daily ranking snapshots in DuckDB and identifies candidate
stocks that have consistently appeared in the top N positions
for a minimum number of days.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib  import Path
from typing   import Optional

import duckdb
import numpy as np
import pandas as pd


_SCHEMA = """
CREATE TABLE IF NOT EXISTS rankings (
    date      TEXT    NOT NULL,
    rank_type TEXT    NOT NULL,
    rank      INTEGER NOT NULL,
    symbol    TEXT    NOT NULL,
    name      TEXT    DEFAULT '',
    value     TEXT    DEFAULT '',
    PRIMARY KEY (date, rank_type, symbol)
);
"""


class RankingTracker:

    def __init__(self, db_path: str = "data/market.duckdb"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self.db_path)

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(
        self,
        rankings: dict[str, list[dict]],
        as_of: Optional[str] = None,
    ) -> int:
        """
        Persist a full set of rankings for one day.

        Args:
            rankings: {rank_type: [{rank, symbol, name, value}, …]}
            as_of:    ISO date string, defaults to today

        Returns:
            Total number of rows written.
        """
        day = as_of or date.today().isoformat()
        rows: list[tuple] = []
        for rank_type, items in rankings.items():
            for item in items:
                rows.append((
                    day,
                    rank_type,
                    item["rank"],
                    item["symbol"],
                    item.get("name", ""),
                    item.get("value", ""),
                ))

        if not rows:
            return 0

        with self._connect() as conn:
            conn.execute(
                "DELETE FROM rankings WHERE date = ? AND rank_type IN ("
                + ", ".join("?" * len(rankings))
                + ")",
                [day, *list(rankings.keys())],
            )
            conn.executemany(
                "INSERT INTO rankings (date, rank_type, rank, symbol, name, value) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    # Composite-score weights (sum = 100). Tunable.
    _W_STREAK     = 30   # persistence: consecutive days in top N
    _W_BREADTH    = 25   # cross-ranking confirmation: distinct ranking types
    _W_QUALITY    = 25   # rank quality: how near the top
    _W_TRAJECTORY = 20   # momentum: is the rank improving over time?
    _STREAK_TARGET = 5   # streak that saturates the persistence component

    def get_candidates(
        self,
        top_n:        int = 50,
        min_days:     int = 3,
        lookback_days: int = 30,
        rank_types:   Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Rank candidate stocks by a composite momentum score (0–100) that rewards:

          • persistence   — consecutive days in the top N           (streak)
          • breadth       — appearing across multiple ranking types (gainers/volume/hot)
          • quality       — how close to rank #1 on average
          • trajectory    — an *improving* rank over the window (climbing the board)

        Only stocks with ≥ min_days distinct days in the top N qualify.
        Sorted by score descending.
        """
        since = (date.today() - timedelta(days=lookback_days)).isoformat()

        conditions = ["date >= ?", "rank <= ?"]
        params: list = [since, top_n]
        if rank_types:
            placeholders = ", ".join("?" * len(rank_types))
            conditions.append(f"rank_type IN ({placeholders})")
            params.extend(rank_types)

        sql = (
            "SELECT date, rank_type, rank, symbol, name FROM rankings "
            f"WHERE {' AND '.join(conditions)} ORDER BY symbol, date"
        )
        with self._connect() as conn:
            raw = conn.execute(sql, params).df()

        if raw.empty:
            return pd.DataFrame(columns=[
                "symbol", "name", "score", "streak", "appearances",
                "avg_rank", "best_rank", "trajectory", "rank_types", "last_date",
            ])

        rows: list[dict] = []
        for symbol, g in raw.groupby("symbol"):
            n_days = g["date"].nunique()
            if n_days < min_days:
                continue

            name        = g["name"].replace("", pd.NA).dropna()
            name        = name.iloc[-1] if len(name) else ""
            appearances = len(g)
            avg_rank    = g["rank"].mean()
            best_rank   = int(g["rank"].min())
            breadth     = g["rank_type"].nunique()          # 1..3
            types       = ", ".join(sorted(g["rank_type"].unique()))
            last_date   = g["date"].max()
            streak      = self._streak(str(symbol), top_n)

            # Daily best rank → trajectory slope (rank improving = slope < 0)
            daily_best = g.groupby("date")["rank"].min().sort_index()
            trajectory = self._rank_slope(daily_best)        # +ve = climbing the board

            # ── Normalised components (each 0..1) ──────────────────────────
            streak_n  = min(streak / self._STREAK_TARGET, 1.0)
            breadth_n = (breadth - 1) / 2.0                  # 1→0, 3→1
            quality_n = (top_n - avg_rank + 1) / top_n       # rank 1 → ~1
            # trajectory: improving by ~half the board over the window → 1.0
            traj_n    = float(np.clip(trajectory / (top_n / 2), 0.0, 1.0))

            score = (
                self._W_STREAK     * streak_n  +
                self._W_BREADTH    * breadth_n +
                self._W_QUALITY    * quality_n +
                self._W_TRAJECTORY * traj_n
            )

            rows.append({
                "symbol":      str(symbol),
                "name":        name,
                "score":       round(score, 1),
                "streak":      streak,
                "appearances": appearances,
                "avg_rank":    round(float(avg_rank), 1),
                "best_rank":   best_rank,
                "trajectory":  round(trajectory, 1),
                "rank_types":  types,
                "last_date":   last_date,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("score", ascending=False).reset_index(drop=True)

    @staticmethod
    def _rank_slope(daily_best: pd.Series) -> float:
        """
        Slope of *improvement* in rank over time.

        daily_best is indexed by ISO date (sorted).  A least-squares fit of
        rank vs. day-ordinal gives a slope; we negate it so that a rank that
        falls over time (1st → better) yields a positive trajectory.
        Units: rank positions gained per day.
        """
        if len(daily_best) < 2:
            return 0.0
        days = np.array([date.fromisoformat(d).toordinal() for d in daily_best.index], dtype=float)
        days -= days.min()
        ranks = daily_best.to_numpy(dtype=float)
        # slope of rank vs day; negative slope = rank number shrinking = improving
        slope = np.polyfit(days, ranks, 1)[0]
        return float(-slope)

    def _streak(self, symbol: str, top_n: int) -> int:
        """Count consecutive days (ending today or yesterday) in top N."""
        sql = """
            SELECT DISTINCT date FROM rankings
            WHERE symbol = ? AND rank <= ?
            ORDER BY date DESC
        """
        with self._connect() as conn:
            dates = [r[0] for r in conn.execute(sql, [symbol, top_n]).fetchall()]

        if not dates:
            return 0

        streak   = 0
        prev     = date.today()
        for d_str in dates:
            d = date.fromisoformat(d_str)
            # Allow one day gap (weekend/holiday)
            if (prev - d).days <= 3:
                streak += 1
                prev    = d
            else:
                break
        return streak

    def get_history(
        self,
        symbol:       str,
        rank_type:    Optional[str] = None,
        lookback_days: int = 30,
    ) -> pd.DataFrame:
        """Return daily rank history for a symbol."""
        since = (date.today() - timedelta(days=lookback_days)).isoformat()
        conditions = ["symbol = ?", "date >= ?"]
        params: list = [symbol, since]
        if rank_type:
            conditions.append("rank_type = ?")
            params.append(rank_type)

        sql = (
            "SELECT date, rank_type, rank, value FROM rankings "
            f"WHERE {' AND '.join(conditions)} ORDER BY date, rank_type"
        )
        with self._connect() as conn:
            return conn.execute(sql, params).df()

    def available_dates(self) -> list[str]:
        """Return all dates for which ranking data is stored."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT date FROM rankings ORDER BY date DESC"
            ).fetchall()
        return [r[0] for r in rows]

    def matches_last_snapshot(
        self,
        rankings: dict[str, list[dict]],
        exclude_date: Optional[str] = None,
    ) -> Optional[str]:
        """
        Detect a stale (market-closed) scrape.

        On Japanese market holidays Yahoo Finance keeps serving the previous
        session's ranking, which would otherwise be recorded as a fresh day.
        This compares the scraped 'gainers' ordering (the most volatile list)
        against the most recent stored day.

        Returns the matching date if the snapshot is identical, else None.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM rankings WHERE date <> ?",
                [exclude_date or ""],
            ).fetchone()
            last_date = row[0] if row else None
            if not last_date:
                return None
            stored = conn.execute(
                "SELECT rank, symbol FROM rankings "
                "WHERE date = ? AND rank_type = 'gainers' ORDER BY rank",
                [last_date],
            ).fetchall()

        if not stored:
            return None

        scraped = [(it["rank"], it["symbol"]) for it in rankings.get("gainers", [])]
        stored_pairs = [(int(r), str(s)) for r, s in stored]
        return last_date if scraped == stored_pairs else None
