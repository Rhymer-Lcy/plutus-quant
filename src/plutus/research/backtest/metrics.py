"""Shared backtest helpers reused across the research study scripts.

Centralizes small functions that were otherwise copy-pasted (or imported script-to-script) in a
dozen scripts, so there is one source of truth and no inter-script imports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def month_ends(dates: pd.DatetimeIndex) -> list:
    """The last trading date of each calendar month in `dates` (the monthly rebalance/eval grid)."""
    s = pd.Series(dates, index=dates)
    return s.groupby(dates.to_period("M")).max().tolist()


def tstat(x: pd.Series) -> float:
    """Plain t-stat of the mean against zero. NaN when there is nothing to test.

    The zero-dispersion guard uses a TOLERANCE, not `== 0`: the sample standard deviation of a
    constant series is not exactly zero in floating point (ten copies of 0.05 give 7e-18), and an
    exact comparison would sail past that and report a t-stat of 1e16 instead of NaN. In return
    space a standard deviation below 1e-12 is numerical noise, never signal."""
    x = pd.Series(x, dtype=float).dropna()
    if len(x) < 2:
        return float("nan")
    sd = float(x.std(ddof=1))
    if not np.isfinite(sd) or sd < 1e-12:
        return float("nan")
    return float(x.mean() / (sd / np.sqrt(len(x))))


def clustered_tstat(x: pd.Series, dates: pd.Series, freq: str = "M") -> float:
    """Clustering-robust t-stat: average the observations inside each calendar period first, then
    take the t-stat across periods.

    Event studies here have events that pile up in time -- biotech catalysts cluster around
    conferences and PDUFA dates (`freq='M'`), and 13F filings cluster on the 45-day deadline
    (`freq='Q'`). An event-level t treats such clustered events as independent draws and badly
    overstates significance, so the pre-registered verdicts use THIS statistic."""
    g = pd.DataFrame({"x": np.asarray(x, dtype=float),
                      "p": pd.DatetimeIndex(dates).to_period(freq)}).dropna()
    return tstat(g.groupby("p")["x"].mean())
