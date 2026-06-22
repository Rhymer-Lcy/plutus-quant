"""Event-driven signal: Post-Earnings-Announcement Drift (PEAD).

Stocks tend to drift in the direction of their earnings SURPRISE for weeks/months after the
announcement — an anomaly that has held up better than the valuation factors (which the CRSP
long-short study found arbitraged away). With no analyst estimates, the standard academic proxy
for the surprise is **SUE** (Standardized Unexpected Earnings) under a seasonal random walk:
expected earnings_q = earnings_{q-4} (same quarter last year), and

    SUE_q = (E_q − E_{q−4}) / σ(E − E_{−4})        [σ = rolling std of the YoY change]

The surprise becomes known at the SEC FILING date (point-in-time), and PEAD DECAYS — a stale
surprise is not a signal — so the daily signal carries the latest SUE only within a freshness
window (≈ one quarter) after the filing.

Pure/testable; fed discrete quarterly earnings from sec_edgar.discrete_quarters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def standardized_unexpected_earnings(quarters: pd.DataFrame, lookback: int = 8,
                                     min_periods: int = 4) -> pd.DataFrame:
    """SUE per quarter from discrete quarterly earnings (cols end, filed, val). Returns
    (filed, sue): the year-over-year earnings change standardized by the rolling std of that
    change. Needs ≥4 prior YoY observations; sparse names yield few/no rows."""
    if quarters.empty:
        return pd.DataFrame(columns=["filed", "sue"])
    q = quarters.sort_values("end").reset_index(drop=True).copy()
    yoy = q["val"] - q["val"].shift(4)
    sigma = yoy.rolling(lookback, min_periods=min_periods).std()
    q["sue"] = yoy / sigma.where(sigma > 0)
    return q.dropna(subset=["sue", "filed"])[["filed", "sue"]].reset_index(drop=True)


def pit_event_signal(sue_by_name: dict[str, pd.DataFrame], dates,
                     freshness_days: int = 63) -> pd.DataFrame:
    """Daily (date x name) PEAD signal: the most recent SUE FILED on/before each date, but only
    while it is FRESH (filing within `freshness_days`, ~one quarter); otherwise NaN, because the
    drift has decayed. NaN before a name's first filing. `sue_by_name[name]` is a (filed, sue)
    frame from `standardized_unexpected_earnings`."""
    dates = pd.DatetimeIndex(dates)
    cols: dict[str, pd.Series] = {}
    for name, s in sue_by_name.items():
        if s is None or s.empty:
            continue
        ss = s.dropna(subset=["filed", "sue"]).copy()
        ss["filed"] = pd.to_datetime(ss["filed"])
        ss = ss.sort_values("filed")
        ser = pd.Series(ss["sue"].to_numpy(), index=pd.DatetimeIndex(ss["filed"]))
        ser = ser[~ser.index.duplicated(keep="last")]
        allidx = ser.index.union(dates)
        sue_ff = ser.reindex(allidx).ffill().reindex(dates)
        filed_ff = pd.Series(ser.index, index=ser.index).reindex(allidx).ffill().reindex(dates)
        age = (pd.Series(dates, index=dates) - pd.to_datetime(filed_ff.to_numpy())).dt.days
        fresh = age.le(freshness_days)                      # NaT age -> False
        cols[name] = sue_ff.where(fresh.to_numpy() & np.isfinite(sue_ff.to_numpy()))
    return pd.DataFrame(cols, index=dates) if cols else pd.DataFrame(index=dates)
