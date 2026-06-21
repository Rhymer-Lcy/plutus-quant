"""Position sizing: inverse-volatility weights for the rebalance basket.

Equal weight ignores that names differ in risk -- a single high-vol holding can drive most
of the portfolio's drawdown. Inverse-vol weighting tilts capital toward the calmer names so
each contributes more equally to portfolio variance.

PIT: the weight at a signal date uses only daily returns through that date; execution is the
next trading day (same no-look-ahead discipline as the selection score). Market-agnostic --
carried over from hermes-quant.
"""
from __future__ import annotations

import pandas as pd


def _apply_cap(w: pd.Series, cap: float | None) -> pd.Series:
    """Cap each weight at `cap`, spilling the excess onto the uncapped names pro-rata,
    iterating until no weight exceeds the cap (or the cap can no longer bind). Requires
    cap * n >= 1 to be feasible; otherwise returns the input unchanged."""
    if cap is None or cap * len(w) < 1.0 - 1e-12:
        return w
    w = w.copy()
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = ~over
        pool = float(w[under].sum())
        if not under.any() or pool <= 0:
            break
        w[under] = w[under] + excess * w[under] / pool
    return w


def inverse_vol_weighter(close: pd.DataFrame, lookback: int = 60, cap: float | None = 0.25,
                         vol_floor: float = 1e-4):
    """Build a `weight_asof(signal_date, tickers) -> {ticker: weight}` callable (weights of
    the passed `tickers` sum to 1) for the portfolio engine.

    w_i is proportional to 1 / sigma_i, where sigma_i is the trailing `lookback`-day
    daily-return std as of the signal date (point-in-time). Names with too little history
    fall back to the basket-median sigma so they are still sized, not dropped. Each weight is
    capped at `cap` and the basket renormalized, so no single ultra-low-vol name dominates;
    pass cap=None to disable. Returns equal weights if no name has a usable sigma.
    """
    rets = close.pct_change(fill_method=None)   # no pad across gaps

    def asof(when, tickers) -> dict[str, float]:
        tickers = list(tickers)
        if not tickers:
            return {}
        window = rets.loc[:pd.Timestamp(when)].tail(lookback)
        vol = window.reindex(columns=tickers).std()
        vol = vol.where(vol >= vol_floor)             # guard near-zero / degenerate sigma
        if not vol.notna().any():
            return {c: 1.0 / len(tickers) for c in tickers}
        vol = vol.fillna(vol.median())                # thin-history names -> median sigma
        inv = 1.0 / vol
        w = inv / inv.sum()
        w = _apply_cap(w, cap)
        return {c: float(w.get(c, 0.0)) for c in tickers}

    return asof
