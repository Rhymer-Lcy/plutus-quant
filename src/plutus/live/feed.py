"""End-of-day data feed for paper trading.

A monthly-rebalance strategy needs only an END-OF-DAY price feed, so this is intentionally
lightweight: pull the latest daily closes for the current universe and return marks the
ledger can value the book against. The same adjusted-close convention as the research lake
(yfinance_source.adjusted_close_panel) is reused so paper marks match backtest valuation.

DEFERRED: a real-time intraday feed (Alpaca) is only needed if/when the cadence moves below
daily. Not built yet.
"""
from __future__ import annotations

import pandas as pd

from ..data.sources import yfinance_source as yfs


def latest_marks(tickers: list[str], asof: str, lookback_days: int = 7) -> dict[str, float]:
    """Return {ticker: latest adjusted close on/just before `asof`} for valuation.

    Pulls a short window and takes each ticker's last available close, so a single missing
    bar (holiday/halt) does not blank the mark. `asof` is 'YYYY-MM-DD'."""
    start = (pd.Timestamp(asof) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(asof) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    panel = yfs.adjusted_close_panel(tickers, start, end)
    if panel.empty:
        return {}
    last = panel.ffill().iloc[-1]
    return {t: float(v) for t, v in last.items() if pd.notna(v)}
