"""
Personal holdings store (snapshot model).

Records what you currently own — symbol, display name, share count and
average cost — in a DuckDB table. This is intentionally a *snapshot* (your
positions right now), not a transaction ledger.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


_SCHEMA = """
CREATE TABLE IF NOT EXISTS holdings (
    symbol       VARCHAR PRIMARY KEY,
    name         VARCHAR DEFAULT '',
    shares       DOUBLE  NOT NULL,   -- stocks: share count; funds: units (口)
    avg_cost     DOUBLE  NOT NULL,   -- cost per share / per 1 口
    asset_type   VARCHAR DEFAULT 'stock',  -- 'stock' | 'fund' | 'manual'
    entry_date   VARCHAR DEFAULT '',        -- ISO date the position was opened
    manual_value DOUBLE  DEFAULT 0,          -- current valuation for 'manual' assets
    currency     VARCHAR DEFAULT 'JPY'       -- price currency; non-JPY is FX-converted
);
"""


class HoldingsStore:

    def __init__(self, db_path: str = "data/market.duckdb"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        with self._connect() as conn:
            conn.execute(_SCHEMA)
            # Migrate older tables that predate newer columns
            cols = [r[1] for r in conn.execute("PRAGMA table_info('holdings')").fetchall()]
            if "asset_type" not in cols:
                conn.execute("ALTER TABLE holdings ADD COLUMN asset_type VARCHAR DEFAULT 'stock'")
            if "entry_date" not in cols:
                conn.execute("ALTER TABLE holdings ADD COLUMN entry_date VARCHAR DEFAULT ''")
            if "manual_value" not in cols:
                conn.execute("ALTER TABLE holdings ADD COLUMN manual_value DOUBLE DEFAULT 0")
            if "currency" not in cols:
                conn.execute("ALTER TABLE holdings ADD COLUMN currency VARCHAR DEFAULT 'JPY'")

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self.db_path)

    def set(
        self,
        symbol: str,
        shares: float,
        avg_cost: float,
        name: str = "",
        asset_type: str = "stock",
        entry_date: str = "",
        manual_value: float = 0.0,
        currency: str = "JPY",
    ) -> None:
        """Insert or update a holding (full overwrite of that symbol's row)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM holdings WHERE symbol = ?", [symbol])
            conn.execute(
                "INSERT INTO holdings (symbol, name, shares, avg_cost, asset_type, "
                "entry_date, manual_value, currency) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [symbol, name, float(shares), float(avg_cost), asset_type,
                 entry_date, float(manual_value), currency],
            )

    def remove(self, symbol: str) -> int:
        with self._connect() as conn:
            before = conn.execute("SELECT COUNT(*) FROM holdings WHERE symbol = ?", [symbol]).fetchone()[0]
            conn.execute("DELETE FROM holdings WHERE symbol = ?", [symbol])
            return int(before)

    def list_all(self) -> pd.DataFrame:
        with self._connect() as conn:
            return conn.execute(
                "SELECT symbol, name, shares, avg_cost, asset_type, entry_date, "
                "manual_value, currency FROM holdings ORDER BY symbol"
            ).df()

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM holdings")
