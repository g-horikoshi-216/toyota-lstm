"""Lightweight helpers for fetching GMO Coin market data.

This module provides ``GmoFetcher`` which mirrors the interface used in the
DL4US tutorials.  Only the pieces required by the notebooks in this repository
are implemented (``fetch_ohlcv`` with optional joblib caching).
"""

from .gmo import GmoFetcher

__all__ = ["GmoFetcher"]

