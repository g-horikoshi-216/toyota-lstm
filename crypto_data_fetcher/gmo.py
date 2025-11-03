"""GMO Coin data fetch utilities used by the DL4US notebooks.

The original course environment ships a ``GmoFetcher`` helper.  This file
implements the subset of that behaviour that the notebooks rely on so that
``from crypto_data_fetcher.gmo import GmoFetcher`` runs without additional
infrastructure.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Optional

import ccxt  # type: ignore
import pandas as pd


_INTERVAL_TO_TIMEFRAME = {
    60: "1m",
    180: "3m",
    300: "5m",
    600: "10m",
    900: "15m",
    1800: "30m",
    3600: "1h",
    7200: "2h",
    14400: "4h",
    21600: "6h",
    43200: "12h",
    86400: "1d",
}


class GmoFetcher:
    """Minimal GMO Coin OHLCV fetcher with optional joblib caching."""

    def __init__(self, *, memory: Optional[Any] = None, rate_limit_sleep: float = 1.0) -> None:
        self._memory = memory
        self._rate_limit_sleep = rate_limit_sleep
        self._exchange = ccxt.gmo({"enableRateLimit": True})

        base_callable = self._fetch_ohlcv_impl
        if memory is not None:
            # ``joblib.Memory.cache`` returns a wrapper that stores results on disk.
            base_callable = memory.cache(base_callable)
        self._fetch_callable = base_callable

    def fetch_ohlcv(
        self,
        *,
        market: str = "BTC_JPY",
        interval_sec: int = 900,
        since: Optional[Any] = None,
        until: Optional[Any] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for the requested market.

        Parameters mirror the original helper:
        - ``market``: GMO symbol (e.g. ``BTC_JPY``).
        - ``interval_sec``: candle width in seconds (supports common CCXT frames).
        - ``since``/``until``: optional start/end bounds (datetime-like).
        - ``limit``: chunk size per request.
        """
        timeframe = self._timeframe(interval_sec)
        symbol = market.replace("_", "/")
        interval_ms = interval_sec * 1000
        since_ms = self._to_milliseconds(since)
        until_ms = self._to_milliseconds(until)

        raw_rows = self._fetch_callable(symbol, timeframe, interval_ms, since_ms, until_ms, limit)
        if not raw_rows:
            columns = ["timestamp", "open", "high", "low", "close", "volume"]
            return pd.DataFrame(columns=columns).set_index("timestamp")

        df = pd.DataFrame(raw_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
        return df

    def _fetch_ohlcv_impl(
        self,
        symbol: str,
        timeframe: str,
        interval_ms: int,
        since_ms: Optional[int],
        until_ms: Optional[int],
        limit: int,
    ) -> Iterable[Iterable[float]]:
        rows = []
        next_since = since_ms

        while True:
            chunk = self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=next_since, limit=limit)
            if not chunk:
                break

            for item in chunk:
                ts = item[0]
                if until_ms is not None and ts >= until_ms:
                    return rows
                rows.append(item)

            if len(chunk) < limit:
                break

            last_ts = chunk[-1][0]
            candidate = last_ts + interval_ms
            if next_since is not None and candidate <= next_since:
                break

            next_since = candidate
            if until_ms is not None and next_since >= until_ms:
                break

            time.sleep(self._rate_limit_sleep)

        return rows

    @staticmethod
    def _timeframe(interval_sec: int) -> str:
        try:
            return _INTERVAL_TO_TIMEFRAME[interval_sec]
        except KeyError as exc:  # pragma: no cover - defensive path
            raise ValueError(f"Unsupported interval (seconds): {interval_sec}") from exc

    @staticmethod
    def _to_milliseconds(value: Optional[Any]) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            number = int(value)
            return number if number > 10**11 else number * 1000
        timestamp = pd.Timestamp(value, tz="UTC")
        return int(timestamp.timestamp() * 1000)

