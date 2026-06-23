"""Rich price/size feature library (Alpha158-inspired) for the ML model zoo.

The 6 classic factors had no edge. The one untried lever is a much LARGER feature space fed to
stronger models. These are ~40 cross-sectional features derived from the adjusted (total-return)
close panel and market cap — returns over many horizons, volatility, moving-average ratios,
price-in-range position, return distribution shape, up-day frequency, market beta, size.

Feature DEFINITIONS are reimplemented from public Qlib Alpha158 formulas (no fork). Every
feature is strictly backward-looking (rolling windows on past data only), so a panel read at a
signal date carries no look-ahead. Returns a dict {name -> wide (date x ticker) panel} to feed
research.model.walk_forward.build_dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_RET_WINDOWS = (1, 5, 10, 20, 40, 60, 120)
_VOL_WINDOWS = (5, 10, 20, 60)
_MA_WINDOWS = (5, 10, 20, 60, 120)
_RANGE_WINDOWS = (5, 10, 20, 60)
_DIST_WINDOWS = (20, 60)


def build_features(close: pd.DataFrame, mktcap: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Build the feature dict from a (date x ticker) adjusted-close panel (+ optional mktcap).
    All features use only past data as of each row."""
    feats: dict[str, pd.DataFrame] = {}
    rets = close.pct_change(fill_method=None)

    for w in _RET_WINDOWS:                         # momentum / rate-of-change over many horizons
        feats[f"roc{w}"] = close / close.shift(w) - 1.0
    feats["mom_12_1"] = close.shift(21) / close.shift(252) - 1.0     # 12-1 momentum

    for w in _VOL_WINDOWS:                          # realized volatility (low-vol style)
        feats[f"vol{w}"] = rets.rolling(w).std()

    for w in _MA_WINDOWS:                           # price vs moving average (trend)
        feats[f"ma{w}"] = close / close.rolling(w).mean() - 1.0

    for w in _RANGE_WINDOWS:                        # position within the high-low range (RSV)
        lo = close.rolling(w).min()
        hi = close.rolling(w).max()
        feats[f"rsv{w}"] = (close - lo) / (hi - lo).where((hi - lo) > 0)

    for w in _DIST_WINDOWS:                         # return distribution shape + tails
        feats[f"skew{w}"] = rets.rolling(w).skew()
        feats[f"kurt{w}"] = rets.rolling(w).kurt()
        feats[f"maxret{w}"] = rets.rolling(w).max()
        feats[f"minret{w}"] = rets.rolling(w).min()
        feats[f"updays{w}"] = (rets > 0).rolling(w).mean()

    # market beta (rolling) vs the equal-weight market return
    mkt = rets.mean(axis=1)
    for w in (60, 120):
        cov = rets.rolling(w).cov(mkt)
        var = mkt.rolling(w).var()
        feats[f"beta{w}"] = cov.div(var.where(var > 0), axis=0)

    if mktcap is not None:
        feats["size"] = np.log(mktcap.where(mktcap > 0))     # log market cap (size factor)

    return feats
