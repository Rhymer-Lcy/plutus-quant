"""Shared backtest helpers reused across the research study scripts.

Centralizes small functions that were otherwise copy-pasted (or imported script-to-script) in a
dozen scripts, so there is one source of truth and no inter-script imports.
"""
from __future__ import annotations

import pandas as pd


def month_ends(dates: pd.DatetimeIndex) -> list:
    """The last trading date of each calendar month in `dates` (the monthly rebalance/eval grid)."""
    s = pd.Series(dates, index=dates)
    return s.groupby(dates.to_period("M")).max().tolist()
