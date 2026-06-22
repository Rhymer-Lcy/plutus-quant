"""Dollar-neutral quantile long-short factor portfolio — how factors are actually traded.

The long-only studies showed factors can't beat the index, because a long-only book is ~90%
just market beta. To measure a factor's ACTUAL edge you strip the beta: each period go LONG the
top-quantile names and SHORT the bottom-quantile names by the signal, equal-weight, equal gross
on each side (dollar-neutral). The return is then the factor SPREAD (top − bottom), with the
market largely removed — the quantity that matters for "is there alpha here".

This is a returns-based portfolio (the academic factor-portfolio convention), not the
share-level long-only engine: it is the right, clean tool for cross-sectional factor research,
and it models the real frictions a market-neutral book pays — turnover cost on BOTH legs plus a
borrow fee on the short leg.

No-look-ahead: the signal is read at each eval date; the return is realized to the NEXT eval
date; weights/turnover use only past baskets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LongShortResult:
    returns: pd.Series        # net long-short return per period (indexed by realization date)
    equity: pd.Series         # cumulative (1+r).cumprod()
    ann_return: float         # geometric, annualized
    ann_vol: float
    sharpe: float             # ann_return / ann_vol
    max_drawdown: float
    market_beta: float        # OLS beta of LS returns on market returns (≈0 if neutral)
    avg_turnover: float       # mean per-period two-sided one-way turnover (1.0 = 100%/side)
    n_periods: int


def _max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def quantile_long_short(price: pd.DataFrame, signal: pd.DataFrame, eval_dates: list,
                        members_asof=None, quantile: float = 0.2, slippage_bps: float = 5.0,
                        borrow_bps_annual: float = 50.0,
                        market_index: pd.Series | None = None) -> LongShortResult:
    """Long top-`quantile`, short bottom-`quantile` by `signal`, equal-weight, dollar-neutral,
    rebalanced on `eval_dates` (read signal at t, realize return t→t+1 from `price`).

    Costs: `slippage_bps` one-way per unit of turnover on each leg; `borrow_bps_annual` charged
    on the short notional, pro-rated per period. `market_index` (a level series) is used only to
    report the realized market beta. Names without a tradable price at both t and t+1, or
    outside `members_asof(t)`, are dropped that period."""
    fwd = price.reindex(eval_dates)
    ppy = 365.25 / np.median(np.diff(pd.DatetimeIndex(eval_dates).asi8) / 8.64e13)  # periods/yr
    borrow_per_period = (borrow_bps_annual * 1e-4) / ppy

    prev_long: dict[str, float] = {}
    prev_short: dict[str, float] = {}
    rets: dict[pd.Timestamp, float] = {}
    turnovers: list[float] = []

    for i in range(len(eval_dates) - 1):
        t, t1 = eval_dates[i], eval_dates[i + 1]
        if t not in signal.index:
            continue
        s = signal.loc[t].dropna()
        r = fwd.loc[t1] / fwd.loc[t] - 1.0
        s = s[r.reindex(s.index).notna()]                       # tradable both dates
        if members_asof is not None:
            s = s[s.index.isin(members_asof(t))]
        k = int(len(s) * quantile)
        if k < 1:
            continue
        longs = s.nlargest(k).index
        shorts = s.nsmallest(k).index
        gross = float(r[longs].mean() - r[shorts].mean())       # dollar-neutral spread
        wl = {c: 1.0 / k for c in longs}
        ws = {c: 1.0 / k for c in shorts}
        turn = (sum(abs(wl.get(c, 0.0) - prev_long.get(c, 0.0)) for c in set(wl) | set(prev_long))
                + sum(abs(ws.get(c, 0.0) - prev_short.get(c, 0.0)) for c in set(ws) | set(prev_short)))
        cost = turn * (slippage_bps * 1e-4)                     # one-way turnover * one-way slip
        rets[t1] = gross - cost - borrow_per_period
        turnovers.append(turn)
        prev_long, prev_short = wl, ws

    returns = pd.Series(rets).sort_index()
    if returns.empty:
        return LongShortResult(returns, returns, float("nan"), float("nan"), float("nan"),
                               float("nan"), float("nan"), float("nan"), 0)
    equity = (1.0 + returns).cumprod()
    years = max(len(returns) / ppy, 1e-9)
    ann_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    ann_vol = float(returns.std() * np.sqrt(ppy))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else float("nan")

    beta = float("nan")
    if market_index is not None:
        m = market_index.reindex(fwd.index)
        mret = (m / m.shift(1) - 1.0).reindex(returns.index)
        df = pd.concat([returns, mret], axis=1, keys=["ls", "m"]).dropna()
        if len(df) > 2 and df["m"].var() > 0:
            beta = float(df["ls"].cov(df["m"]) / df["m"].var())

    return LongShortResult(returns, equity, ann_return, ann_vol, sharpe,
                           _max_drawdown(equity), beta, float(np.mean(turnovers)), len(returns))
