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

import pandas as pd
import yfinance as yf


def daily_bars(ticker: str, start: str, end: str, auto_adjust: bool = True) -> pd.DataFrame:
    """Daily bars for one ticker as a typed, date-indexed DataFrame.

    Columns: open, high, low, close, volume (close is ADJUSTED when auto_adjust=True).
    Returns an empty frame if Yahoo has no data for the ticker/window."""
    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=auto_adjust,
                                   actions=False, raw=False)
    if df.empty:
        return df
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)   # drop tz for a clean DatetimeIndex
    df.index.name = "date"
    return df


def adjusted_close_panel(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Wide (date x ticker) ADJUSTED-close panel — the working set for the backtest engine.

    Pulls all tickers in one batched request. Tickers Yahoo has no data for are simply absent
    from the columns (not zero-filled). Forward/total-return adjusted via auto_adjust=True."""
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False,
                      group_by="column", threads=True)
    if raw.empty:
        return pd.DataFrame()
    # Single ticker -> flat columns; multiple -> MultiIndex (field, ticker).
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "date"
    # drop all-NaN rows AND all-NaN columns (a column with no data = a ticker Yahoo lacks,
    # e.g. delisted; keeping it as a NaN column would overstate universe coverage).
    return close.sort_index().dropna(how="all").dropna(how="all", axis=1)


def raw_close_panel(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Wide (date x ticker) UNADJUSTED close panel — the actual traded price each day.

    Use this (NOT the adjusted panel) for MARKET CAP = raw_close * shares_outstanding: shares
    outstanding from SEC are as-reported (not split-adjusted), so they must be paired with the
    unadjusted price to stay on the same basis. The adjusted panel is for returns/backtesting;
    mixing adjusted price with as-reported shares would jump market cap at every split."""
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False,
                      group_by="column", threads=True)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "date"
    # drop all-NaN rows AND all-NaN columns (a column with no data = a ticker Yahoo lacks,
    # e.g. delisted; keeping it as a NaN column would overstate universe coverage).
    return close.sort_index().dropna(how="all").dropna(how="all", axis=1)
