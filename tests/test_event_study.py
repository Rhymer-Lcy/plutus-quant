"""Event-time backtest: CAAR shows the post-event drift shape (top-minus-bottom widens), and the
overlapping long-short portfolio captures it (and costs reduce it)."""
import numpy as np
import pandas as pd

from plutus.research.backtest.event_study import event_caar, event_time_portfolio

DRIFT_DAYS = 30
AMT = 0.004


def _panels():
    dates = pd.bdate_range("2020-01-01", periods=120)
    names = [f"N{i}" for i in range(12)]
    ret = pd.DataFrame(0.0, index=dates, columns=names)
    entries = [40, 45, 50, 55]
    rows = []
    groups = [("pos", [f"N{i}" for i in range(0, 4)], +2.0, +AMT),
              ("neg", [f"N{i}" for i in range(4, 8)], -2.0, -AMT),
              ("neu", [f"N{i}" for i in range(8, 12)], 0.0, 0.0)]
    for _, nms, sue, amt in groups:
        for j, nm in enumerate(nms):
            e = entries[j]
            rows.append({"permno": nm, "entry_date": dates[e], "sue": sue})
            for d in range(e + 1, min(e + 1 + DRIFT_DAYS, 120)):
                ret.loc[dates[d], nm] += amt          # post-event drift in the sign of the surprise
    return ret, pd.DataFrame(rows)


def test_caar_top_minus_bottom_widens():
    ret, events = _panels()
    caar = event_caar(events, ret, hold_days=DRIFT_DAYS, n_groups=3)
    tmb = caar["top_minus_bottom"]
    assert tmb.iloc[-1] > tmb.iloc[0] > 0              # drift accumulates: positive & widening
    assert caar["q2"].iloc[-1] > 0 and caar["q0"].iloc[-1] < 0   # winners up, losers down


def test_event_portfolio_captures_drift_and_costs_bite():
    ret, events = _panels()
    free = event_time_portfolio(events, ret, hold_days=DRIFT_DAYS, sue_threshold=0.5,
                                slippage_bps=0.0, borrow_bps_annual=0.0)
    assert free.ann_return > 0.0                       # long winners / short losers -> positive
    assert free.avg_long > 0 and free.avg_short > 0
    costly = event_time_portfolio(events, ret, hold_days=DRIFT_DAYS, sue_threshold=0.5,
                                  slippage_bps=200.0, borrow_bps_annual=0.0)
    assert costly.ann_return < free.ann_return         # turnover cost drags it


def test_threshold_excludes_neutral_events():
    ret, events = _panels()
    # threshold above the neutral SUE (0) but below the ±2 surprises -> only 4 long / 4 short names
    res = event_time_portfolio(events, ret, hold_days=DRIFT_DAYS, sue_threshold=0.5)
    assert res.avg_long <= 4.0 and res.avg_short <= 4.0
