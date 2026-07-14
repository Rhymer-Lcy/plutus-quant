"""Distance-method statistical-arbitrage pairs trading (Gatev-Goetzmann-Rouwenhorst 2006).

A structurally DIFFERENT bet from everything else in plutus: not a monthly cross-sectional alpha
(zero-sum vs pros, which our OOS test showed is arbitraged out of US equities) but a DAILY
time-series mean-reversion between cointegrated-by-comovement pairs. It is capacity-LIMITED, which
is precisely where a retail-size book has an advantage rather than a handicap (big money can't fit,
so the spread is less competed away).

Method (the canonical GGR distance rule, no stats-library dependency):
  FORMATION (lookback L days ending at f): over names with COMPLETE prices in the window, build a
    normalized total-return index (price / price[0], so the ratio IS cumulative TR because the lake
    is already DlyRet-adjusted). For every pair compute SSD = sum((norm_i - norm_j)^2). Select the
    top-K most-similar pairs (smallest SSD) and record each pair's formation spread std.
  TRADING (next T days): carry the SAME formation normalization forward (NO re-normalization with
    trading data — that is the classic look-ahead trap). Open when |spread| > entry_z * formation_std
    (long the underperformer, short the outperformer, dollar-neutral); close when the spread reverts
    through zero, at window end, or if a leg delists. Costs: slippage per leg on open+close, borrow
    on the short leg per day held.

No-look-ahead by construction: pair selection, the normalization base, and the trigger scale use
ONLY formation data; each day's position is decided at close t and realized t -> t+1; survivorship
is handled by the CRSP lake (a delisted leg carries its DlyRet delisting return on its last day).

Trading windows are non-overlapping (step = T) and walked forward across the whole sample, so the
aggregate daily-return series is a clean walk-forward backtest, sliceable by year for an OOS read.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


@dataclass
class PairsResult:
    returns: pd.Series        # portfolio daily net return (equal capital across K slots)
    equity: pd.Series
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    n_days: int
    n_windows: int
    avg_pairs_traded: float   # mean # pairs that opened at least once per window
    trades_per_pair: float    # mean round-trips per selected pair-window
    avg_days_in_market: float # fraction of trading days a slot holds a position


def select_pairs(form_prices: pd.DataFrame, top_k: int) -> list[tuple[str, str, float, float, float]]:
    """Top-K most-similar pairs by SSD over the formation window. `form_prices` is a [L x N] price
    slice. Returns (name_i, name_j, ssd, base_i, base_j, spread_std) for each pair — base_* are the
    first-row prices (the normalization denominators carried into trading)."""
    cols = form_prices.columns[form_prices.notna().all(axis=0)]   # complete-data (alive) names only
    if len(cols) < 2:
        return []
    p = form_prices[cols]
    base = p.iloc[0]
    norm = (p / base).to_numpy().T                                 # [N x L], each row a TR index
    d = squareform(pdist(norm, metric="sqeuclidean"))             # SSD between every pair
    iu = np.triu_indices(len(cols), k=1)
    ssd = d[iu]
    order = np.argsort(ssd)[:top_k]
    out = []
    for o in order:
        i, j = iu[0][o], iu[1][o]
        ni, nj = cols[i], cols[j]
        spread_std = float((norm[i] - norm[j]).std())
        out.append((ni, nj, float(ssd[o]), float(base[ni]), float(base[nj]), spread_std))
    return out


def _trade_signal(pi: pd.Series, pj: pd.Series, signal: pd.Series, threshold: float,
                  slip: float, borrow_pp: float) -> tuple[pd.Series, int, int]:
    """Trade one pair from a precomputed mean-reversion `signal` (formation-scaled): open when
    |signal| > threshold (signal>0 => i rich => short i/long j), close when it reverts through 0.
    Dollar-neutral 50/50 legs. Position decided at close d, realized d -> d+1. Returns (daily net
    return on the pair's $1 slot, n_round_trips, n_days_in_market)."""
    dates = pi.index
    rets = pd.Series(0.0, index=dates)
    pos = 0                                                       # +1: long j/short i; -1: long i/short j
    trips, days_in = 0, 0
    for d in range(len(dates) - 1):
        s = signal.iloc[d]
        if pos != 0:                                             # realize carried position d -> d+1
            ri = pi.iloc[d + 1] / pi.iloc[d] - 1.0
            rj = pj.iloc[d + 1] / pj.iloc[d] - 1.0
            if np.isnan(ri) or np.isnan(rj):                     # a leg delisted -> force close
                trips += 1
                pos = 0
                continue
            leg = 0.5 * (rj - ri) if pos > 0 else 0.5 * (ri - rj)
            rets.iloc[d + 1] += leg - 0.5 * borrow_pp            # borrow on the $0.5 short leg
            days_in += 1
        if np.isnan(s):
            if pos != 0:
                rets.iloc[d + 1] -= 2 * slip
                trips += 1
                pos = 0
            continue
        if pos == 0:
            if s > threshold:                                    # i rich -> short i, long j
                pos = +1
                rets.iloc[d + 1] -= 2 * slip
            elif s < -threshold:                                 # i cheap -> long i, short j
                pos = -1
                rets.iloc[d + 1] -= 2 * slip
        else:
            if (pos > 0 and s <= 0) or (pos < 0 and s >= 0):     # reverted through zero -> close
                rets.iloc[d + 1] -= 2 * slip
                trips += 1
                pos = 0
    if pos != 0:                                                 # close at window end
        rets.iloc[-1] -= 2 * slip
        trips += 1
    return rets, trips, days_in


def distance_pairs_backtest(price: pd.DataFrame, *, formation: int = 252, trading: int = 126,
                            step: int | None = None, top_k: int = 20, entry_z: float = 2.0,
                            slippage_bps: float = 5.0, borrow_bps_annual: float = 50.0,
                            min_names: int = 50) -> PairsResult:
    """Walk-forward distance-method pairs backtest. Non-overlapping trading windows (step defaults to
    `trading`). Portfolio return each day = equal-weight mean over the K pair slots (a flat slot
    earns 0). Costs: `slippage_bps` one-way per leg on open/close; `borrow_bps_annual` on the short
    leg, pro-rated daily."""
    step = step or trading
    slip = slippage_bps * 1e-4
    borrow_pp = (borrow_bps_annual * 1e-4) / 252.0
    idx = price.index
    all_rets, n_windows, pairs_traded, all_trips, days_in_frac = [], 0, [], [], []

    f = formation
    while f + trading <= len(idx):
        form = price.iloc[f - formation:f]
        trade = price.iloc[f:f + trading]
        if form.notna().all(axis=0).sum() < min_names:
            f += step
            continue
        pairs = select_pairs(form, top_k)
        if not pairs:
            f += step
            continue
        slot_rets = []
        opened = 0
        for ni, nj, _ssd, bi, bj, sstd in pairs:
            if sstd <= 0:
                continue
            signal = trade[ni] / bi - trade[nj] / bj            # formation-scaled price spread
            r, trips, din = _trade_signal(trade[ni], trade[nj], signal, entry_z * sstd,
                                          slip, borrow_pp)
            slot_rets.append(r)
            all_trips.append(trips)
            days_in_frac.append(din / max(len(trade) - 1, 1))
            if trips > 0 or din > 0:
                opened += 1
        if slot_rets:
            window_ret = pd.concat(slot_rets, axis=1).mean(axis=1)   # equal capital across slots
            all_rets.append(window_ret)
            n_windows += 1
            pairs_traded.append(opened)
        f += step

    if not all_rets:
        empty = pd.Series(dtype=float)
        return PairsResult(empty, empty, float("nan"), float("nan"), float("nan"), float("nan"),
                           0, 0, float("nan"), float("nan"), float("nan"))
    returns = pd.concat(all_rets).sort_index()
    returns = returns[~returns.index.duplicated(keep="first")]
    equity = (1.0 + returns).cumprod()
    ppy = 252.0
    years = max(len(returns) / ppy, 1e-9)
    ann_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    ann_vol = float(returns.std() * np.sqrt(ppy))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else float("nan")
    mdd = float((equity / equity.cummax() - 1.0).min())
    return PairsResult(returns, equity, ann_return, ann_vol, sharpe, mdd, len(returns), n_windows,
                       float(np.mean(pairs_traded)), float(np.mean(all_trips)),
                       float(np.mean(days_in_frac)))


# --------------------------------------------------------------------------------------------------
# Cointegration method (Engle-Granger): select pairs whose residual spread is statistically
# stationary (mean-reverting), not merely co-moving in level. Practitioners' default; a distinct
# selection mechanism, so it is a real second opinion on whether US large-cap stat-arb is alive.
# --------------------------------------------------------------------------------------------------

def _engle_granger(pi: np.ndarray, pj: np.ndarray):
    """OLS hedge ratio pi = alpha + beta*pj + resid, then ADF on resid. Returns
    (beta, resid_mean, resid_std, adf_stat) or None if degenerate/failed."""
    from statsmodels.tsa.stattools import adfuller
    X = np.column_stack([np.ones_like(pj), pj])
    try:
        coef, *_ = np.linalg.lstsq(X, pi, rcond=None)
    except Exception:
        return None
    alpha, beta = float(coef[0]), float(coef[1])
    resid = pi - (alpha + beta * pj)
    sd = float(resid.std())
    if sd <= 0 or not np.isfinite(beta):
        return None
    try:
        adf = float(adfuller(resid, maxlag=1, regression="c", autolag=None)[0])
    except Exception:
        return None
    return beta, float(resid.mean()), sd, adf


def select_cointegrated_pairs(form_prices: pd.DataFrame, top_k: int, ssd_prescreen: int = 5,
                              adf_max: float = -2.86) -> list[tuple[str, str, float, float, float, float]]:
    """Pre-screen by SSD to top (top_k*ssd_prescreen) candidates (keeps ADF count tractable), run
    Engle-Granger, keep pairs with ADF < adf_max (~5% stationarity), return the top_k most stationary
    as (name_i, name_j, beta, resid_mean, resid_std, adf)."""
    cand = select_pairs(form_prices, top_k * ssd_prescreen)
    out = []
    for ni, nj, *_ in cand:
        eg = _engle_granger(form_prices[ni].to_numpy(float), form_prices[nj].to_numpy(float))
        if eg is None:
            continue
        beta, mu, sd, adf = eg
        if adf < adf_max:
            out.append((ni, nj, beta, mu, sd, adf))
    out.sort(key=lambda x: x[5])                                 # most-negative ADF = most stationary
    return out[:top_k]


def cointegration_pairs_backtest(price: pd.DataFrame, *, formation: int = 252, trading: int = 126,
                                 step: int | None = None, top_k: int = 20, entry_z: float = 2.0,
                                 adf_max: float = -2.86, slippage_bps: float = 5.0,
                                 borrow_bps_annual: float = 50.0, min_names: int = 50) -> PairsResult:
    """Walk-forward cointegration-method pairs backtest. Same harness and dollar-neutral 50/50
    execution as the distance method, but pairs are selected by ADF-stationarity of the OLS-residual
    spread and the trade trigger is the residual z-score (formation mean/std), |z| > entry_z."""
    step = step or trading
    slip = slippage_bps * 1e-4
    borrow_pp = (borrow_bps_annual * 1e-4) / 252.0
    idx = price.index
    all_rets, n_windows, pairs_traded, all_trips, days_in_frac = [], 0, [], [], []

    f = formation
    while f + trading <= len(idx):
        form = price.iloc[f - formation:f]
        trade = price.iloc[f:f + trading]
        if form.notna().all(axis=0).sum() < min_names:
            f += step
            continue
        pairs = select_cointegrated_pairs(form, top_k, adf_max=adf_max)
        if not pairs:
            f += step
            continue
        slot_rets, opened = [], 0
        for ni, nj, beta, mu, sd, _adf in pairs:
            signal = (trade[ni] - beta * trade[nj] - mu) / sd     # residual z-score (formation scale)
            r, trips, din = _trade_signal(trade[ni], trade[nj], signal, entry_z, slip, borrow_pp)
            slot_rets.append(r)
            all_trips.append(trips)
            days_in_frac.append(din / max(len(trade) - 1, 1))
            if trips > 0 or din > 0:
                opened += 1
        if slot_rets:
            all_rets.append(pd.concat(slot_rets, axis=1).mean(axis=1))
            n_windows += 1
            pairs_traded.append(opened)
        f += step

    if not all_rets:
        empty = pd.Series(dtype=float)
        return PairsResult(empty, empty, float("nan"), float("nan"), float("nan"), float("nan"),
                           0, 0, float("nan"), float("nan"), float("nan"))
    returns = pd.concat(all_rets).sort_index()
    returns = returns[~returns.index.duplicated(keep="first")]
    equity = (1.0 + returns).cumprod()
    ppy = 252.0
    years = max(len(returns) / ppy, 1e-9)
    ann_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    ann_vol = float(returns.std() * np.sqrt(ppy))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else float("nan")
    mdd = float((equity / equity.cummax() - 1.0).min())
    return PairsResult(returns, equity, ann_return, ann_vol, sharpe, mdd, len(returns), n_windows,
                       float(np.mean(pairs_traded)), float(np.mean(all_trips)),
                       float(np.mean(days_in_frac)))
