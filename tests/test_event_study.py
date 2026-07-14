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


def test_entry_offset_skips_the_first_day():
    # entry_offset delays entry by N trading days (used to skip the announcement-reaction-day
    # gap). Skipping a day of the drift must capture strictly less.
    ret, events = _panels()
    full = event_time_portfolio(events, ret, hold_days=DRIFT_DAYS, sue_threshold=0.5,
                                slippage_bps=0.0, borrow_bps_annual=0.0, entry_offset=0)
    skipped = event_time_portfolio(events, ret, hold_days=DRIFT_DAYS, sue_threshold=0.5,
                                   slippage_bps=0.0, borrow_bps_annual=0.0, entry_offset=2)
    assert skipped.ann_return < full.ann_return


def test_threshold_excludes_neutral_events():
    ret, events = _panels()
    # threshold above the neutral SUE (0) but below the ±2 surprises -> only 4 long / 4 short names
    res = event_time_portfolio(events, ret, hold_days=DRIFT_DAYS, sue_threshold=0.5)
    assert res.avg_long <= 4.0 and res.avg_short <= 4.0


def test_intraday_entry_replaces_only_the_entry_day():
    # entry day earns the intraday panel's value; all later days the close-to-close panel.
    ret, events = _panels()
    intr = ret + 0.01                                  # intraday = close-to-close +1% every day
    base = event_time_portfolio(events, ret, hold_days=DRIFT_DAYS, sue_threshold=0.5,
                                slippage_bps=0.0, borrow_bps_annual=0.0)
    # longs gain +1% on their entry day, shorts LOSE 1% (short leg is subtracted): with equal
    # long/short books the two entry-day effects cancel in the LS return only if entries align;
    # here they do (same entry dates), so isolate via a long-only event set instead.
    long_events = events[events["sue"] > 0]
    b1 = event_time_portfolio(long_events, ret, hold_days=DRIFT_DAYS, sue_threshold=0.5,
                              slippage_bps=0.0, borrow_bps_annual=0.0)
    f1 = event_time_portfolio(long_events, ret, hold_days=DRIFT_DAYS, sue_threshold=0.5,
                              slippage_bps=0.0, borrow_bps_annual=0.0, intraday_entry=intr)
    assert f1.ann_return > b1.ann_return               # faster entry captured the extra day-0 gain
    # and with an intraday panel IDENTICAL to close-to-close, results are unchanged (parity)
    same = event_time_portfolio(events, ret, hold_days=DRIFT_DAYS, sue_threshold=0.5,
                                slippage_bps=0.0, borrow_bps_annual=0.0, intraday_entry=ret)
    assert np.isclose(same.ann_return, base.ann_return)


def test_halfspread_panel_charges_per_name_and_falls_back():
    ret, events = _panels()
    # a zero-spread panel = free trading; a wide-spread panel must cost more than flat 5 bps
    zero = ret * 0.0
    wide = ret * 0.0 + 0.02                            # 2% half-spread, every name/day
    r_zero = event_time_portfolio(events, ret, hold_days=DRIFT_DAYS, sue_threshold=0.5,
                                  slippage_bps=5.0, borrow_bps_annual=0.0, halfspread_panel=zero)
    r_flat = event_time_portfolio(events, ret, hold_days=DRIFT_DAYS, sue_threshold=0.5,
                                  slippage_bps=5.0, borrow_bps_annual=0.0)
    r_wide = event_time_portfolio(events, ret, hold_days=DRIFT_DAYS, sue_threshold=0.5,
                                  slippage_bps=5.0, borrow_bps_annual=0.0, halfspread_panel=wide)
    assert r_zero.ann_return > r_flat.ann_return > r_wide.ann_return
    # NaN spreads fall back to the flat slippage -> identical to the flat run
    nan_panel = ret * np.nan
    r_nan = event_time_portfolio(events, ret, hold_days=DRIFT_DAYS, sue_threshold=0.5,
                                 slippage_bps=5.0, borrow_bps_annual=0.0, halfspread_panel=nan_panel)
    assert np.isclose(r_nan.ann_return, r_flat.ann_return)


def test_default_path_parity_with_new_arguments_off():
    # the two new arguments default to None; the default path must be numerically unchanged
    ret, events = _panels()
    a = event_time_portfolio(events, ret, hold_days=DRIFT_DAYS, sue_threshold=0.5,
                             slippage_bps=15.0, borrow_bps_annual=300.0, entry_offset=1)
    b = event_time_portfolio(events, ret, hold_days=DRIFT_DAYS, sue_threshold=0.5,
                             slippage_bps=15.0, borrow_bps_annual=300.0, entry_offset=1,
                             intraday_entry=None, halfspread_panel=None)
    assert np.isclose(a.ann_return, b.ann_return) and np.isclose(a.sharpe, b.sharpe)
