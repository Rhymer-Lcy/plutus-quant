"""Cross-sectional factor definitions and processing for US equities.

Convention: every factor is oriented so that HIGHER = more attractive (expected higher
forward return), which makes IC signs and quantile spreads comparable across factors.
All take/return wide daily panels (date x ticker).

Two families:
  - PRICE-BASED (momentum, reversal, low-vol): use only past adjusted closes, so they are
    point-in-time clean by construction.
  - FUNDAMENTAL (value, quality): take ratio panels built from SEC EDGAR data
    (plutus.data.sources.sec_edgar) that MUST already be aligned to the FILING date, not the
    fiscal-period-end date — otherwise look-ahead leaks (you would "know" Q4 earnings on
    Dec 31 when they were actually filed in February). The factor functions here assume that
    point-in-time alignment was done upstream.

The cross-sectional processing functions (restrict_to_universe / winsorize / z-score /
standardize / blend) are market-agnostic and carried over verbatim from hermes-quant; the
survivorship discipline they enforce matters even MORE for US, where free data sources drop
delisted tickers (see docs/data_sources.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def restrict_to_universe(panel: pd.DataFrame, members_asof) -> pd.DataFrame:
    """NaN out, per date, every name NOT in the point-in-time universe `members_asof(date)`.

    CRITICAL for survivorship-free studies: a cross-sectional op (winsorize/z-score/blend)
    computed over the survivorship-defined UNION (names ever in the index) leaks future
    membership into the normalization -- a member's standardized score then depends on the
    presence of names that are in the panel only because they JOIN the index LATER. Always
    restrict to the PIT universe BEFORE standardize()/blend() in a PIT study.
    `members_asof`: callable(date)->set[ticker]."""
    mask = pd.DataFrame(False, index=panel.index, columns=panel.columns)
    for d in panel.index:
        present = panel.columns[panel.columns.isin(members_asof(d))]
        mask.loc[d, present] = True
    return panel.where(mask)


def winsorize_xs(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip each row (date) to its [lower, upper] cross-sectional quantiles.

    NOTE: the cross-section is whatever columns are non-NaN that date. In a PIT study,
    restrict_to_universe() the panel FIRST, or the union leaks (see that function)."""
    lo = df.quantile(lower, axis=1)
    hi = df.quantile(upper, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def zscore_xs(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score each row (mean 0, std 1 across names that date)."""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorize then z-score, cross-sectionally. (Rank IC is invariant to this; it matters
    for the ML combiner and for quantile-cut stability.) In a PIT study, the input must
    already be restrict_to_universe()'d -- the union cross-section leaks otherwise."""
    return zscore_xs(winsorize_xs(df))


def blend(panels: list[pd.DataFrame], weights: list[float] | None = None) -> pd.DataFrame:
    """Combine factor panels into one score: standardize each cross-sectionally (so different
    scales are comparable), then take the weighted mean across factors per (date, ticker),
    skipping factors missing for that name. All inputs must already be oriented higher = more
    attractive; the result is too.

    SURVIVORSHIP: because each panel is standardized cross-sectionally, the inputs must be
    restrict_to_universe()'d to the PIT members in a survivorship-free study (see
    restrict_to_universe)."""
    weights = weights if weights is not None else [1.0] * len(panels)
    if len(weights) != len(panels):
        raise ValueError("weights must match panels")
    zsum = wsum = None
    for panel, wt in zip(panels, weights):
        z = standardize(panel)
        contrib = (z * wt).fillna(0.0)
        present = z.notna() * float(wt)
        zsum = contrib if zsum is None else zsum.add(contrib, fill_value=0.0)
        wsum = present if wsum is None else wsum.add(present, fill_value=0.0)
    return zsum / wsum.where(wsum > 0)


# --- price-based factors (higher = more attractive) ---

def trailing_return(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return close / close.shift(lookback) - 1.0


def momentum(close: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """12-1 style: return over [t-lookback, t-skip], skipping the recent `skip` days (the
    short-horizon reversal window). US daily convention: ~252 trading days/year, ~21/month."""
    return close.shift(skip) / close.shift(lookback) - 1.0


def reversal(close: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """Short-term reversal: NEGATIVE trailing return (recent losers tend to bounce). ~21
    trading days = 1 month. Higher = more attractive."""
    return -trailing_return(close, lookback)


def low_vol(close: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Negative rolling std of daily returns (low volatility = attractive)."""
    # fill_method=None: do NOT pad across gaps (padding would inject spurious 0 returns).
    return -close.pct_change(fill_method=None).rolling(window).std()


# --- fundamental factors (higher = more attractive) ---
# Inputs are wide daily panels already aligned to FILING dates (point-in-time). Build them
# from SEC EDGAR company facts (plutus.data.sources.sec_edgar) joined to market cap.

def earnings_yield(net_income_ttm: pd.DataFrame, market_cap: pd.DataFrame) -> pd.DataFrame:
    """E/P: trailing-twelve-month net income / market cap. Non-positive earnings -> NaN
    (earnings yield undefined for losses). Higher = cheaper relative to earnings."""
    ey = net_income_ttm / market_cap
    return ey.where(net_income_ttm > 0)


def book_yield(book_equity: pd.DataFrame, market_cap: pd.DataFrame) -> pd.DataFrame:
    """B/P: common shareholders' equity / market cap. Non-positive book -> NaN."""
    by = book_equity / market_cap
    return by.where(book_equity > 0)


def roe(net_income_ttm: pd.DataFrame, book_equity: pd.DataFrame) -> pd.DataFrame:
    """Quality factor: return on equity = TTM net income / common equity. Non-positive book
    -> NaN. Higher = more profitable per unit of equity. Pairs with value to screen out the
    'cheap for a reason' value traps that pure E/P piles into (low-quality distressed names)."""
    return (net_income_ttm / book_equity).where(book_equity > 0)


def small_size(market_cap: pd.DataFrame) -> pd.DataFrame:
    """Negative log market cap (the small-size premium). NOT assumed to be deployed: a
    size tilt WITHIN a large-cap index can be distress beta rather than the SMB premium --
    validate before using (cf. the rejected within-HS300 size tilt in hermes-quant)."""
    return -np.log(market_cap.where(market_cap > 0))
