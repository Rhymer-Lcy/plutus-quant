"""Turnover-aware dollar-neutral long-short portfolio construction.

The GRU edge (IC t≈2.6) is real but eaten by ~270%/mo turnover in the naive quintile book. An
optimizer that explicitly trades off expected return against turnover COST keeps more of it:

    maximize  alpha·w  −  gamma · slip · ||w − w_prev||_1
    s.t.      sum(w) = 0 (dollar-neutral),  ||w||_1 ≤ gross,  |w_i| ≤ name_cap

gamma=0 ≈ full rebalance (the naive book); larger gamma trades less. `alpha` is the signal in
return units (the GRU predicts demeaned forward returns), so the penalty `slip·||Δw||_1` is the
real trading cost and gamma is a turnover-aversion multiplier on top of it.

>>> NOTE: when gamma is swept on the whole sample and the best picked, the resulting net Sharpe
>>> is IN-SAMPLE-optimistic; a deployable version selects gamma out-of-sample. The study reports
>>> the full sweep so the lift (if any) over the naive book is visible, with that caveat.
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass
class OptResult:
    returns: pd.Series
    equity: pd.Series
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    avg_turnover: float
    avg_gross: float
    n_periods: int


def turnover_aware_weights(alpha: pd.Series, w_prev: pd.Series, gamma: float, slip: float,
                           name_cap: float = 0.02, gross: float = 2.0) -> pd.Series:
    """Solve the dollar-neutral L1-turnover-penalized long-short for one period. Returns weights
    aligned to `alpha.index` (falls back to carrying `w_prev` if the solve fails)."""
    a = alpha.to_numpy(dtype=float)
    wp = w_prev.reindex(alpha.index).fillna(0.0).to_numpy(dtype=float)
    w = cp.Variable(len(a))
    prob = cp.Problem(cp.Maximize(a @ w - gamma * slip * cp.norm1(w - wp)),
                      [cp.sum(w) == 0, cp.norm1(w) <= gross, cp.abs(w) <= name_cap])
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        return pd.Series(wp, index=alpha.index)
    if w.value is None:
        return pd.Series(wp, index=alpha.index)
    return pd.Series(np.where(np.abs(w.value) < 1e-6, 0.0, w.value), index=alpha.index)


def turnover_aware_backtest(price: pd.DataFrame, signal: pd.DataFrame, eval_dates: list,
                            members_asof=None, *, gamma: float = 5.0, slippage_bps: float = 5.0,
                            borrow_bps_annual: float = 50.0, name_cap: float = 0.02,
                            gross: float = 2.0, cand_frac: float = 0.3,
                            aum: float | None = None, adv: pd.DataFrame | None = None,
                            impact_coef: float = 0.01) -> OptResult:
    """Walk the optimizer month to month, carrying weights (so turnover is real), net of slippage
    (on turnover, incl. liquidating names that leave the universe) and borrow (on the short leg).
    `cand_frac`: restrict each period's candidates to the top+bottom fraction by signal.

    Optional MARKET IMPACT (capacity study): if `aum` (book $) and `adv` (a date×name dollar-volume
    panel reindexed to eval_dates) are given, per-name trade cost rate = slip + impact_coef·
    sqrt(participation), participation = (|Δw|·aum)/ADV (Almgren square-root law) — so cost rises
    with AUM relative to each name's liquidity, and the result depends on book size."""
    fwd = price.reindex(eval_dates)
    ppy = 365.25 / np.median(np.diff(pd.DatetimeIndex(eval_dates).asi8) / 8.64e13)
    slip = slippage_bps * 1e-4
    borrow_pp = (borrow_bps_annual * 1e-4) / ppy
    w_prev = pd.Series(dtype=float)
    dates, rets, turns, grosses = [], [], [], []

    for i in range(len(eval_dates) - 1):
        t, t1 = eval_dates[i], eval_dates[i + 1]
        if t not in signal.index:
            continue
        s = signal.loc[t].dropna()
        r = fwd.loc[t1] / fwd.loc[t] - 1.0
        s = s[r.reindex(s.index).notna()]
        if members_asof is not None:
            s = s[s.index.isin(members_asof(t))]
        if len(s) < 20:
            continue
        k = max(int(len(s) * cand_frac), 1)
        cand = s.nlargest(k).index.union(s.nsmallest(k).index)
        w = turnover_aware_weights(s[cand], w_prev, gamma, slip, name_cap, gross)
        allnames = w.index.union(w_prev.index)
        wn = w.reindex(allnames).fillna(0.0)
        wpn = w_prev.reindex(allnames).fillna(0.0)
        dw = (wn - wpn).abs()
        turn = float(dw.sum())
        if aum is not None and adv is not None and t in adv.index:
            advt = adv.loc[t].reindex(allnames)
            part = (dw * aum) / advt.where(advt > 0)            # participation = traded$ / ADV$
            rate = slip + impact_coef * np.sqrt(part.clip(lower=0).fillna(0.0))   # sqrt-impact
            trade_cost = float((dw * rate).sum())
        else:
            trade_cost = turn * slip
        gross_ret = float((w * r.reindex(w.index).fillna(0.0)).sum())
        short_notional = float(w[w < 0].abs().sum())
        dates.append(t1)
        rets.append(gross_ret - trade_cost - short_notional * borrow_pp)
        turns.append(turn)
        grosses.append(float(w.abs().sum()))
        w_prev = w[w.abs() > 0]

    returns = pd.Series(rets, index=pd.DatetimeIndex(dates))
    if returns.empty:
        return OptResult(returns, returns, float("nan"), float("nan"), float("nan"),
                         float("nan"), float("nan"), float("nan"), 0)
    equity = (1.0 + returns).cumprod()
    years = max(len(returns) / ppy, 1e-9)
    ann_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    ann_vol = float(returns.std() * np.sqrt(ppy))
    return OptResult(returns, equity, ann_return, ann_vol,
                     float(ann_return / ann_vol) if ann_vol > 0 else float("nan"),
                     float((equity / equity.cummax() - 1.0).min()),
                     float(np.mean(turns)), float(np.mean(grosses)), len(returns))
