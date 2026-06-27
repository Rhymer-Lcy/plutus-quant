"""Yahoo Finance adapter via `yfinance` — free, anonymous (no key, no registration).

The free daily backbone to START with (the US analog of hermes-quant's BaoStock). Caveats,
spelled out because they bite a backtest:
  - UNOFFICIAL (scrapes Yahoo); the API can break or rate-limit without notice.
  - SURVIVORSHIP BIAS: Yahoo drops most delisted tickers, so a universe built from
    *currently* listed tickers is survivorship-contaminated. For a survivorship-free study
    you must feed a point-in-time ticker list (incl. delisted) from elsewhere and accept that
    yfinance may not have the dead names -- see docs/data_sources.md.
  - Use auto_adjust=True so OHLC are split- AND dividend-adjusted (the right series for a
    total-return backtest); `close` is then the adjusted close.

    from plutus.data.sources import yfinance_source as yf
    panel = yf.adjusted_close_panel(["AAPL", "MSFT"], "2015-01-01", "2025-12-31")
"""
from __future__ import annotations

import os
import time
from functools import lru_cache

import pandas as pd
import yfinance as yf


@lru_cache(maxsize=1)
def _pinned_session():
    """A Yahoo-DNS-pinned yfinance session, or None when no pin is configured.

    yfinance fetches over curl/libcurl, which resolves DNS itself -- so a corporate VPN that breaks
    the SYSTEM resolver for *.finance.yahoo.com makes every pull fail (curl "(6) Could not resolve
    host"), even though the IPs are reachable. Set ``PLUTUS_YF_RESOLVE`` to a comma-separated list of
    curl ``host:port:ip`` entries and each request connects to those IPs while keeping the hostname
    for TLS/SNI, surviving the broken resolver. The scheduler wrapper (scripts/paper_forward.ps1)
    fills the variable by resolving Yahoo via a PUBLIC DNS server. Unset (e.g. a manual run) returns
    None, so yfinance uses its default session unchanged."""
    entries = [e for e in os.environ.get("PLUTUS_YF_RESOLVE", "").split(",") if e]
    if not entries:
        return None
    from curl_cffi import CurlOpt
    from curl_cffi import requests as creq
    return creq.Session(impersonate="chrome", curl_options={CurlOpt.RESOLVE: entries})


def daily_bars(ticker: str, start: str, end: str, auto_adjust: bool = True) -> pd.DataFrame:
    """Daily bars for one ticker as a typed, date-indexed DataFrame.

    Columns: open, high, low, close, volume (close is ADJUSTED when auto_adjust=True).
    Returns an empty frame if Yahoo has no data for the ticker/window."""
    df = yf.Ticker(ticker, session=_pinned_session()).history(
        start=start, end=end, auto_adjust=auto_adjust, actions=False, raw=False)
    if df.empty:
        return df
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)   # drop tz for a clean DatetimeIndex
    df.index.name = "date"
    return df


# Pull the price panel in small SERIAL batches with a brief pause between them, instead of one
# threads=True burst of the whole book. yfinance/Yahoo rate-limits on requests-per-second per IP,
# and firing all of a 50-name book at once tripped "YFRateLimitError: Too Many Requests" on the
# scheduled forward pulls. Serial batches (threads=False) plus a short inter-batch sleep keep the
# request rate under Yahoo's threshold; the macro retry in scripts/paper_forward.ps1 still covers a
# residual block. The defaults are conservative; PLUTUS_YF_BATCH / PLUTUS_YF_SLEEP_SEC override them
# for tuning without a code change. This is a pure I/O-pacing change: the returned prices are
# byte-for-byte the same as a single download, so no backtest/paper number moves.
_YF_BATCH = int(os.environ.get("PLUTUS_YF_BATCH", "8"))
_YF_SLEEP_SEC = float(os.environ.get("PLUTUS_YF_SLEEP_SEC", "1.0"))


def _download_close(tickers: list[str], start: str, end: str, auto_adjust: bool) -> pd.DataFrame:
    """Wide (date x ticker) close panel, pulled in serial batches to stay under Yahoo's rate limit.

    Splits `tickers` into `_YF_BATCH`-sized chunks, downloads each with threads=False (no request
    burst) and a `_YF_SLEEP_SEC` pause between chunks, and concatenates the Close columns. A chunk
    Yahoo has no data for contributes nothing (not zero-filled). Returns an empty frame when every
    chunk is empty. Common engine for the adjusted (auto_adjust=True) and raw (False) panels below."""
    frames = []
    for i in range(0, len(tickers), _YF_BATCH):
        chunk = tickers[i:i + _YF_BATCH]
        raw = yf.download(chunk, start=start, end=end, auto_adjust=auto_adjust, progress=False,
                          group_by="column", threads=False, session=_pinned_session())
        if not raw.empty:
            # Single ticker -> flat columns; multiple -> MultiIndex (field, ticker).
            if isinstance(raw.columns, pd.MultiIndex):
                frames.append(raw["Close"].copy())
            else:
                frames.append(raw[["Close"]].rename(columns={"Close": chunk[0]}))
        if i + _YF_BATCH < len(tickers):
            time.sleep(_YF_SLEEP_SEC)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def _clean_close_panel(close: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw close panel: tz-naive 'date' index, sorted, with all-NaN rows AND columns
    dropped (a column with no data = a ticker Yahoo lacks, e.g. delisted; keeping it as a NaN
    column would overstate universe coverage)."""
    if close.empty:
        return pd.DataFrame()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "date"
    return close.sort_index().dropna(how="all").dropna(how="all", axis=1)


def adjusted_close_panel(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Wide (date x ticker) ADJUSTED-close panel — the working set for the backtest engine.

    Pulled in serial batches (see _download_close). Tickers Yahoo has no data for are simply absent
    from the columns (not zero-filled). Forward/total-return adjusted via auto_adjust=True."""
    if not tickers:
        return pd.DataFrame()
    return _clean_close_panel(_download_close(tickers, start, end, auto_adjust=True))


def raw_close_panel(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Wide (date x ticker) UNADJUSTED close panel — the actual traded price each day.

    Use this (NOT the adjusted panel) for MARKET CAP = raw_close * shares_outstanding: shares
    outstanding from SEC are as-reported (not split-adjusted), so they must be paired with the
    unadjusted price to stay on the same basis. The adjusted panel is for returns/backtesting;
    mixing adjusted price with as-reported shares would jump market cap at every split."""
    if not tickers:
        return pd.DataFrame()
    return _clean_close_panel(_download_close(tickers, start, end, auto_adjust=False))
