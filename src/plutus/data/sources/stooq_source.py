"""Stooq adapter — free, anonymous daily CSV bars (no key, no registration).

Two jobs, both attacking the survivorship problem that yfinance can't:
  1. INDEPENDENT CROSS-CHECK of the yfinance price backbone (a second opinion on a suspicious
     bar / adjustment), in the spirit of cross-checking a backtest against a second engine.
  2. DELISTED coverage: Stooq retains some delisted US tickers that Yahoo drops — partial, not
     complete, but better than nothing for survivorship-free work.

US symbols carry a ".US" suffix on Stooq (e.g. AAPL -> aapl.us); this adapter adds it. Stooq
daily US bars are split-adjusted.

CAVEATS (degrade gracefully -> EMPTY frame, never raise, so a partial pull is fine):
  - Rate limit: the free endpoint throttles ("Exceeded the daily hits limit").
  - !! Verified 2026-06-21: the CSV endpoint currently serves a JavaScript browser-verification
    CHALLENGE page (not CSV) to plain HTTP clients, so a direct `requests` pull returns empty.
    Treat Stooq as BEST-EFFORT, not a reliable backbone. When it is challenged you need a
    browser-driven fetch or a different source; we do NOT attempt to evade the bot check. The
    symbol mapping + CSV parsing here are correct and unit-tested for when the endpoint serves
    data (e.g. from an un-challenged network).
The panel pull paces requests to stay polite.

    from plutus.data.sources import stooq_source as st
    df = st.daily_bars("AAPL", "2015-01-01", "2025-12-31")
"""
from __future__ import annotations

import time
from io import StringIO

import pandas as pd
import requests

_URL = "https://stooq.com/q/d/l/"
_OHLCV = ["open", "high", "low", "close", "volume"]


def stooq_symbol(ticker: str) -> str:
    """Map a US ticker to its Stooq symbol: lower-cased, with a '.us' suffix if none present.
    A ticker that already carries an exchange suffix (contains '.') is passed through lowered."""
    t = ticker.strip().lower()
    return t if "." in t else f"{t}.us"


def _parse_daily_csv(text: str) -> pd.DataFrame:
    """Parse a Stooq daily CSV into a typed, date-indexed OHLCV frame. Returns an EMPTY frame
    for a rate-limit / no-data response (which is plain text, not a CSV header)."""
    head = text.lstrip()[:4].lower()
    if not head.startswith("date"):
        return pd.DataFrame()                       # "Exceeded the daily hits limit" / "No data"
    df = pd.read_csv(StringIO(text))
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()
    df = df.rename(columns=str.lower)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    cols = [c for c in _OHLCV if c in df.columns]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.index.name = "date"
    return df[cols].sort_index()


def daily_bars(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Daily OHLCV for one ticker. Empty frame on a rate-limit / no-data response."""
    params = {"s": stooq_symbol(ticker), "i": "d",
              "d1": pd.Timestamp(start).strftime("%Y%m%d"),
              "d2": pd.Timestamp(end).strftime("%Y%m%d")}
    resp = requests.get(_URL, params=params, headers={"User-Agent": "plutus-quant"}, timeout=30)
    resp.raise_for_status()
    return _parse_daily_csv(resp.text)


def close_panel(tickers: list[str], start: str, end: str, pause: float = 0.25) -> pd.DataFrame:
    """Wide (date x ticker) close panel. Stooq serves one symbol per request, so this paces
    them (`pause` seconds) to stay under the free rate limit; tickers with no data are simply
    absent from the columns."""
    out: dict[str, pd.Series] = {}
    for t in tickers:
        df = daily_bars(t, start, end)
        if not df.empty and "close" in df.columns:
            out[t.upper()] = df["close"]
        time.sleep(pause)
    return pd.DataFrame(out).sort_index() if out else pd.DataFrame()
