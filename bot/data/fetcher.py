import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def _to_ms(date_str: str) -> int:
    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


class BaseFetcher:
    COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
    # Keep _to_ms as a static method so store.py can call DataFetcher._to_ms()
    _to_ms = staticmethod(_to_ms)

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        since: str,
        until: Optional[str] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Alpaca (US equities)
# ---------------------------------------------------------------------------

class AlpacaFetcher(BaseFetcher):
    """Fetches OHLCV data from Alpaca for US equities."""

    _TF_MAP: dict  # populated lazily

    def __init__(self):
        self._client = None

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        since: str,
        until: Optional[str] = None,
    ) -> pd.DataFrame:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests   import StockBarsRequest
        from alpaca.data.timeframe  import TimeFrame, TimeFrameUnit

        _tf_map = {
            "1m":  TimeFrame(1,  TimeFrameUnit.Minute),
            "5m":  TimeFrame(5,  TimeFrameUnit.Minute),
            "15m": TimeFrame(15, TimeFrameUnit.Minute),
            "30m": TimeFrame(30, TimeFrameUnit.Minute),
            "1h":  TimeFrame(1,  TimeFrameUnit.Hour),
            "4h":  TimeFrame(4,  TimeFrameUnit.Hour),
            "1d":  TimeFrame(1,  TimeFrameUnit.Day),
            "1w":  TimeFrame(1,  TimeFrameUnit.Week),
            "1M":  TimeFrame(1,  TimeFrameUnit.Month),
        }
        tf = _tf_map.get(timeframe)
        if tf is None:
            raise ValueError(f"Alpaca: unknown timeframe '{timeframe}'. Supported: {list(_tf_map)}")

        if self._client is None:
            self._client = StockHistoricalDataClient(
                api_key=os.getenv("APCA_API_KEY_ID"),
                secret_key=os.getenv("APCA_API_SECRET_KEY"),
            )

        start_dt = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
        end_dt   = datetime.fromisoformat(until).replace(tzinfo=timezone.utc) if until else None

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start_dt,
            end=end_dt,
            feed="iex",       # free tier; swap to "sip" with paid subscription
            adjustment="all", # split + dividend adjusted
        )
        bars = self._client.get_stock_bars(request)
        raw  = bars.df

        if raw is None or raw.empty:
            return pd.DataFrame(columns=self.COLUMNS)

        # bars.df uses MultiIndex (symbol, timestamp) for single-symbol requests
        if isinstance(raw.index, pd.MultiIndex):
            df = raw.xs(symbol, level="symbol").reset_index()
        else:
            df = raw.reset_index()

        df["timestamp"] = df["timestamp"].apply(
            lambda t: int(pd.Timestamp(t).timestamp() * 1000)
        )
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df["timestamp"] = df["timestamp"].astype("int64")
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)

        df.drop_duplicates("timestamp", inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


# ---------------------------------------------------------------------------
# J-Quants (Japanese equities)
# ---------------------------------------------------------------------------

class JQuantsFetcher(BaseFetcher):
    """
    Fetches OHLCV data from J-Quants V2 API for Japanese equities.

    Auth: JQUANTS_API_KEY in .env (x-api-key header)
    Free plan: daily bars only (1d), rate limit 5 req/min.

    Symbol format: 4-digit (e.g. "7203") or 5-digit (e.g. "72030").
    4-digit codes are automatically padded to 5-digit (appends "0").
    """

    _API_BASE        = "https://api.jquants.com/v2"
    _FREE_TIMEFRAMES = {"1d"}

    def __init__(self):
        self._api_key = os.getenv("JQUANTS_API_KEY", "")

    def _to_code5(self, symbol: str) -> str:
        return symbol if len(symbol) == 5 else symbol + "0"

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        since: str,
        until: Optional[str] = None,
    ) -> pd.DataFrame:
        if timeframe not in self._FREE_TIMEFRAMES:
            raise ValueError(
                f"J-Quants free plan supports only {self._FREE_TIMEFRAMES}. "
                f"Got: '{timeframe}'."
            )
        if not self._api_key:
            raise ValueError("JQUANTS_API_KEY not set in .env")

        import re as _re
        import time as _time
        import requests as _req

        code   = self._to_code5(symbol)
        from_d = datetime.fromisoformat(since).strftime("%Y%m%d")
        to_d   = datetime.fromisoformat(until).strftime("%Y%m%d") if until else datetime.now().strftime("%Y%m%d")

        headers = {"x-api-key": self._api_key}
        params  = {"code": code, "from": from_d, "to": to_d}
        records: list[dict] = []

        while True:
            resp = _req.get(
                f"{self._API_BASE}/equities/bars/daily",
                params=params,
                headers=headers,
                timeout=30,
            )
            if resp.status_code == 429:
                # Rate limit: free plan allows 5 req/min — wait and retry
                _time.sleep(13)
                continue
            if resp.status_code == 400:
                # Free plan uses a rolling 2-year window.
                # 400 when `from` or `to` is outside the allowed range.
                # Parse the allowed range and clamp both dates, then retry.
                msg = resp.json().get("message", "")
                m   = _re.search(r"(\d{4}-\d{2}-\d{2}) ~ (\d{4}-\d{2}-\d{2})", msg)
                if m:
                    allowed_from = m.group(1).replace("-", "")
                    allowed_to   = m.group(2).replace("-", "")
                    new_from = max(params["from"], allowed_from)
                    new_to   = min(params["to"],   allowed_to)
                    if new_from != params["from"] or new_to != params["to"]:
                        params["from"] = new_from
                        params["to"]   = new_to
                        continue
                resp.raise_for_status()
            else:
                resp.raise_for_status()
            body = resp.json()
            records.extend(body.get("data", []))
            pagination_key = body.get("pagination_key")
            if not pagination_key:
                break
            params["pagination_key"] = pagination_key

        if not records:
            return pd.DataFrame(columns=self.COLUMNS)

        raw = pd.DataFrame(records)
        df  = pd.DataFrame()
        df["timestamp"] = pd.to_datetime(raw["Date"]).apply(
            lambda d: int(d.replace(tzinfo=timezone.utc).timestamp() * 1000)
        )
        df["open"]   = raw["AdjO"].astype(float)
        df["high"]   = raw["AdjH"].astype(float)
        df["low"]    = raw["AdjL"].astype(float)
        df["close"]  = raw["AdjC"].astype(float)
        df["volume"] = raw["AdjVo"].astype(float)

        df = df[(df["open"] > 0) & df["close"].notna()]
        df.drop_duplicates("timestamp", inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


# ---------------------------------------------------------------------------
# Yahoo Finance (free, real-time-ish — US & JP equities)
# ---------------------------------------------------------------------------

class YFinanceFetcher(BaseFetcher):
    """
    Fetches OHLCV from Yahoo Finance via the `yfinance` library.

    Free and up to date (no subscription window), so it closes the gap left
    by J-Quants' free plan. Works for both US tickers ("AAPL") and Japanese
    ones — a bare 4-digit code like "7203" is auto-suffixed with ".T".

    Only daily and weekly/monthly bars are reliable on the free endpoint;
    intraday history is limited to recent weeks.
    """

    _TF_MAP = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "1d": "1d", "1w": "1wk", "1M": "1mo",
    }

    def _to_yahoo_symbol(self, symbol: str) -> str:
        # Bare Japanese codes → append the Tokyo ".T". J-Quants 5-digit codes
        # are the 4-digit code plus a single trailing "0"; strip exactly that.
        if symbol.isdigit() and len(symbol) == 5 and symbol.endswith("0"):
            return f"{symbol[:-1]}.T"
        if symbol.isdigit() and len(symbol) == 4:
            return f"{symbol}.T"
        return symbol

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        since: str,
        until: Optional[str] = None,
    ) -> pd.DataFrame:
        import yfinance as yf

        interval = self._TF_MAP.get(timeframe)
        if interval is None:
            raise ValueError(
                f"yfinance: unsupported timeframe '{timeframe}'. "
                f"Supported: {list(self._TF_MAP)}"
            )

        yf_symbol = self._to_yahoo_symbol(symbol)
        raw = yf.Ticker(yf_symbol).history(
            start=since,
            end=until,
            interval=interval,
            auto_adjust=True,   # split + dividend adjusted
        )

        if raw is None or raw.empty:
            return pd.DataFrame(columns=self.COLUMNS)

        raw = raw.reset_index()
        ts_col = "Datetime" if "Datetime" in raw.columns else "Date"

        df = pd.DataFrame()
        df["timestamp"] = raw[ts_col].apply(
            lambda t: int(pd.Timestamp(t).timestamp() * 1000)
        )
        df["open"]   = raw["Open"].astype(float)
        df["high"]   = raw["High"].astype(float)
        df["low"]    = raw["Low"].astype(float)
        df["close"]  = raw["Close"].astype(float)
        df["volume"] = raw["Volume"].astype(float)

        df = df[(df["open"] > 0) & df["close"].notna()]
        df.drop_duplicates("timestamp", inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def DataFetcher(source: str = "alpaca") -> BaseFetcher:
    """
    Return a fetcher for the given data source.

    Args:
        source: "alpaca"   — US equities (Alpaca)
                "jquants"  — Japanese equities (J-Quants; free plan lags ~3 months)
                "yfinance" — US & JP equities (Yahoo Finance, free, up to date)
    """
    if source == "alpaca":
        return AlpacaFetcher()
    if source == "jquants":
        return JQuantsFetcher()
    if source == "yfinance":
        return YFinanceFetcher()
    raise ValueError(
        f"Unknown source '{source}'. Choose 'alpaca', 'jquants', or 'yfinance'."
    )
