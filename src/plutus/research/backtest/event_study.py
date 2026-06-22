"""Event-time backtest for drift signals (PEAD): trade on the EVENT clock, not the calendar.

The calendar-monthly PEAD test missed the drift because the drift is front-loaded in the days
right after an earnings announcement. Here positions are opened the day AFTER each event and
held for a fixed number of trading days, so the book is an OVERLAPPING set of every still-young
event — exactly how the drift would be harvested.

Two tools:
  - `event_caar`: cumulative average ABNORMAL return (vs the cross-sectional mean) by surprise
    group, in event time — the diagnostic that shows the drift's SHAPE (is it front-loaded? how
    big? does the top−bottom spread widen?).
  - `event_time_portfolio`: a dollar-neutral long-short of fresh positive- vs negative-surprise
    events, daily, net of turnover + borrow costs — the tradeable test.

No look-ahead: an event filed on day d is entered at the next trading day's close (strictly
after d).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _entry_positions(idx: pd.DatetimeIndex, entry_dates) -> np.ndarray:
    """First trading-day index strictly AFTER each entry/filing date (avoids same-day look-ahead)."""
    return idx.searchsorted(pd.DatetimeIndex(entry_dates), side="right")


def event_caar(events: pd.DataFrame, ret_panel: pd.DataFrame, hold_days: int = 60,
               n_groups: int = 3) -> pd.DataFrame:
    """Mean cumulative ABNORMAL return (name return minus that day's cross-sectional mean) over
    event days 1..hold_days, averaged across events within each surprise group (n_groups tiles
    of `sue`). Returns a DataFrame indexed by event day, one column per group (group 0 = lowest
    SUE ... n_groups-1 = highest). The top-minus-bottom path is the PEAD drift."""
    idx = ret_panel.index
    abn = ret_panel.sub(ret_panel.mean(axis=1), axis=0)        # market-neutral abnormal return
    ev = events.dropna(subset=["permno", "entry_date", "sue"]).copy()
    ev["grp"] = pd.qcut(ev["sue"].rank(method="first"), n_groups, labels=False)
    ev["epos"] = _entry_positions(idx, ev["entry_date"])

    sums = {g: np.zeros(hold_days) for g in range(n_groups)}
    counts = {g: np.zeros(hold_days) for g in range(n_groups)}
    for permno, epos, grp in zip(ev["permno"], ev["epos"], ev["grp"]):
        if permno not in abn.columns or epos >= len(idx):
            continue
        seg = abn[permno].to_numpy()[epos:epos + hold_days]
        car = np.nancumsum(np.where(np.isnan(seg), 0.0, seg))
        m = min(len(car), hold_days)
        sums[grp][:m] += car[:m]
        counts[grp][:m] += 1
    out = {f"q{g}": np.where(counts[g] > 0, sums[g] / np.where(counts[g] == 0, 1, counts[g]), np.nan)
           for g in range(n_groups)}
    df = pd.DataFrame(out, index=pd.RangeIndex(1, hold_days + 1, name="event_day"))
    df["top_minus_bottom"] = df[f"q{n_groups - 1}"] - df["q0"]
    return df


@dataclass
class EventPortfolioResult:
    returns: pd.Series
    equity: pd.Series
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    avg_long: float        # mean number of long positions held/day
    avg_short: float
    n_periods: int


def event_time_portfolio(events: pd.DataFrame, ret_panel: pd.DataFrame, hold_days: int = 40,
                         sue_threshold: float = 0.5, slippage_bps: float = 5.0,
                         borrow_bps_annual: float = 50.0) -> EventPortfolioResult:
    """Dollar-neutral overlapping long-short: each event with sue >= +threshold is held LONG for
    `hold_days` trading days from the day after filing; sue <= -threshold held SHORT. Equal-
    weight within each leg each day. Net of per-turnover slippage (both legs) and a daily borrow
    fee on the short leg."""
    idx = ret_panel.index
    n = len(idx)
    ev = events.dropna(subset=["permno", "entry_date", "sue"]).copy()
    ev["epos"] = _entry_positions(idx, ev["entry_date"])

    long_on: dict[int, list] = defaultdict(list)
    short_on: dict[int, list] = defaultdict(list)
    for permno, epos, sue in zip(ev["permno"], ev["epos"], ev["sue"]):
        if epos >= n or permno not in ret_panel.columns:
            continue
        book = long_on if sue >= sue_threshold else short_on if sue <= -sue_threshold else None
        if book is None:
            continue
        for d in range(epos, min(epos + hold_days, n)):
            book[d].append(permno)

    borrow_per_day = (borrow_bps_annual * 1e-4) / 252.0
    slip = slippage_bps * 1e-4
    prev_l: dict[str, float] = {}
    prev_s: dict[str, float] = {}
    rets, n_long, n_short = [], [], []
    for i in range(n):
        row = ret_panel.iloc[i]
        L, S = long_on.get(i, []), short_on.get(i, [])
        wl = {c: 1.0 / len(L) for c in L} if L else {}
        ws = {c: 1.0 / len(S) for c in S} if S else {}
        lr = float(np.nanmean([row.get(c, np.nan) for c in L])) if L else 0.0
        sr = float(np.nanmean([row.get(c, np.nan) for c in S])) if S else 0.0
        lr = 0.0 if np.isnan(lr) else lr
        sr = 0.0 if np.isnan(sr) else sr
        turn = (sum(abs(wl.get(c, 0.0) - prev_l.get(c, 0.0)) for c in set(wl) | set(prev_l))
                + sum(abs(ws.get(c, 0.0) - prev_s.get(c, 0.0)) for c in set(ws) | set(prev_s)))
        rets.append(lr - sr - turn * slip - (borrow_per_day if S else 0.0))
        n_long.append(len(L))
        n_short.append(len(S))
        prev_l, prev_s = wl, ws

    returns = pd.Series(rets, index=idx)
    equity = (1.0 + returns).cumprod()
    ann_return = float(equity.iloc[-1] ** (252.0 / n) - 1.0) if n else float("nan")
    ann_vol = float(returns.std() * np.sqrt(252.0))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else float("nan")
    maxdd = float((equity / equity.cummax() - 1.0).min())
    return EventPortfolioResult(returns, equity, ann_return, ann_vol, sharpe, maxdd,
                                float(np.mean(n_long)), float(np.mean(n_short)), n)
