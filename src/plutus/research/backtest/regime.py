"""Market-regime exposure overlay: scale gross exposure by a trend filter to control drawdown.

The survivorship-free study (docs/survivorship_study.md) showed a value+reversal book losing
-88% in 2008 — it bought value-trap financials straight into the crash. A simple, well-
documented fix is a TREND filter: hold full exposure only while the broad market is at/above
its long moving average, otherwise cut to a floor (e.g. cash). This plugs straight into the
backtest engine's `exposure_asof` hook (portfolio.signal_portfolio_backtest), so the strategy
object is unchanged — only the gross-invested fraction is scaled.

No-look-ahead: the market index is built from prior-day-cap-weighted name returns, and the
exposure at a signal date uses only data through that date.
"""
from __future__ import annotations

import pandas as pd


def cap_weighted_index(adj_close: pd.DataFrame, mktcap: pd.DataFrame) -> pd.Series:
    """Daily cap-weighted TOTAL-RETURN index from the lake (a broad-market / S&P 500 proxy),
    compounded to a level series starting at 1.0. Each day's return is the mean of name returns
    weighted by PRIOR-day market cap (so the weight is known before the return)."""
    rets = adj_close.pct_change(fill_method=None)
    w = mktcap.shift(1).where(rets.notna())
    wsum = w.sum(axis=1)
    port_ret = (rets * w).sum(axis=1) / wsum.where(wsum > 0)
    return (1.0 + port_ret.fillna(0.0)).cumprod()


def trend_exposure(market_index: pd.Series, window: int = 200, floor: float = 0.0):
    """Build `exposure_asof(date) -> exposure in [floor, 1]` for the engine: full (1.0) exposure
    when `market_index` is at/above its `window`-day moving average on that date, else `floor`
    (0.0 = go to cash). Uses the latest observation on/before the query date (no look-ahead)."""
    ma = market_index.rolling(window).mean()
    above = (market_index >= ma)

    def exposure_asof(date) -> float:
        d = pd.Timestamp(date)
        s = above.loc[:d]
        return 1.0 if (len(s) and bool(s.iloc[-1])) else floor

    return exposure_asof
