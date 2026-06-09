import os
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


class DataStore:
    """Persists OHLCV data in a local DuckDB file for fast analytical queries."""

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS ohlcv (
            timestamp BIGINT,
            symbol    VARCHAR,
            timeframe VARCHAR,
            open      DOUBLE,
            high      DOUBLE,
            low       DOUBLE,
            close     DOUBLE,
            volume    DOUBLE
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ohlcv_pk
            ON ohlcv (timestamp, symbol, timeframe);
    """

    def __init__(self, db_path: str = "data/market.duckdb"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_schema()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(self._SCHEMA)

    def save(self, df: pd.DataFrame, symbol: str, timeframe: str) -> int:
        """Upsert OHLCV rows. Returns number of new rows written."""
        if df.empty:
            return 0

        work = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        work["symbol"]    = symbol
        work["timeframe"] = timeframe
        work = work[["timestamp", "symbol", "timeframe", "open", "high", "low", "close", "volume"]]

        with self._connect() as conn:
            min_ts = int(work["timestamp"].min())
            max_ts = int(work["timestamp"].max())
            conn.execute(
                "DELETE FROM ohlcv WHERE symbol=? AND timeframe=? "
                "AND timestamp>=? AND timestamp<=?",
                [symbol, timeframe, min_ts, max_ts],
            )
            conn.register("_save_df", work)
            conn.execute("INSERT INTO ohlcv SELECT * FROM _save_df")
            conn.unregister("_save_df")
            return len(work)

    def load(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load OHLCV rows, optionally filtered by ISO date strings."""
        from bot.data.fetcher import _to_ms

        conditions = ["symbol=? AND timeframe=?"]
        params: list = [symbol, timeframe]

        if since:
            conditions.append("timestamp >= ?")
            params.append(_to_ms(since))
        if until:
            conditions.append("timestamp <= ?")
            params.append(_to_ms(until))

        sql = f"SELECT * FROM ohlcv WHERE {' AND '.join(conditions)} ORDER BY timestamp"
        with self._connect() as conn:
            df = conn.execute(sql, params).df()

        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def available(self, symbol: str, timeframe: str) -> tuple[Optional[int], Optional[int]]:
        """Return (min_timestamp_ms, max_timestamp_ms) stored for this pair, or (None, None)."""
        sql = "SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv WHERE symbol=? AND timeframe=?"
        with self._connect() as conn:
            row = conn.execute(sql, [symbol, timeframe]).fetchone()
        if row is None or row[0] is None:
            return None, None
        return int(row[0]), int(row[1])

    def list_pairs(self) -> pd.DataFrame:
        """Return a DataFrame summarising all stored symbol/timeframe pairs."""
        sql = """
            SELECT symbol, timeframe,
                   COUNT(*)            AS bars,
                   MIN(timestamp)      AS first_ts,
                   MAX(timestamp)      AS last_ts
            FROM ohlcv
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """
        with self._connect() as conn:
            return conn.execute(sql).df()
