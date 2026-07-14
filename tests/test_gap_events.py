"""Catalyst gap events: overnight decomposition, event detection, CARs, clustered inference."""
import math

import numpy as np
import pandas as pd

from plutus.research.backtest.gap_events import (decompose_overnight, event_cars, find_events)

DATES = pd.bdate_range("2024-01-01", periods=40)


def _panel(values, codes=("A",)):
    return pd.DataFrame({c: values for c in codes}, index=DATES[:len(values)], dtype=float)


def test_flat_open_means_no_gap():
    close_raw = _panel([100.0, 110.0])
    open_raw = _panel([np.nan, 100.0])            # opened exactly at the prior close
    dlyret = _panel([np.nan, 0.10])
    overnight, intraday = decompose_overnight(dlyret, open_raw, close_raw)
    assert math.isclose(float(intraday.iloc[1, 0]), 0.10)
    assert abs(float(overnight.iloc[1, 0])) < 1e-12


def test_gap_up_then_fade_decomposes_exactly():
    # prior close 100 -> opens 120 (+20% gap) -> closes 110 (total day +10%)
    close_raw = _panel([100.0, 110.0])
    open_raw = _panel([np.nan, 120.0])
    dlyret = _panel([np.nan, 0.10])
    overnight, intraday = decompose_overnight(dlyret, open_raw, close_raw)
    assert math.isclose(float(intraday.iloc[1, 0]), 110 / 120 - 1)
    assert math.isclose(float(overnight.iloc[1, 0]), 0.20, abs_tol=1e-12)


def test_split_does_not_create_a_phantom_gap():
    # 2:1 split overnight: raw price halves, but the total return is +10% and the gap is ZERO.
    close_raw = _panel([100.0, 55.0])
    open_raw = _panel([np.nan, 50.0])
    dlyret = _panel([np.nan, 0.10])
    overnight, _ = decompose_overnight(dlyret, open_raw, close_raw)
    assert abs(float(overnight.iloc[1, 0])) < 1e-12


def test_find_events_thresholds_and_needs_history():
    n = 30
    overnight = _panel([0.0] * n)
    close_raw = _panel([100.0] * n)
    overnight.iloc[5, 0] = 0.25          # a big gap, but only 5 prior days of history
    overnight.iloc[25, 0] = 0.25         # a big gap with enough history
    overnight.iloc[26, 0] = 0.19         # just under the threshold
    ev = find_events(overnight, close_raw, threshold=0.20, min_history=20)
    assert list(ev["date"]) == [DATES[25]]
    assert math.isclose(float(ev["gap"].iloc[0]), 0.25)


def test_eligibility_gates_the_event_day_only():
    n = 30
    overnight = _panel([0.0] * n)
    close_raw = _panel([100.0] * n)
    overnight.iloc[25, 0] = 0.25
    overnight.iloc[27, 0] = 0.25
    eligible = _panel([True] * n).astype(bool)
    eligible.iloc[25, 0] = False          # untradable on the day of the first gap
    ev = find_events(overnight, close_raw, threshold=0.20, min_history=20, eligible=eligible)
    assert list(ev["date"]) == [DATES[27]]


def test_holding_period_is_never_truncated_by_the_price_floor():
    # The gap-up name craters afterwards. The floor gates the EVENT, not the holding period:
    # those losses belong to whoever bought, and must still be counted.
    n = 30
    abn_cc = _panel([0.0] * n)
    abn_cc.iloc[26:31, 0] = -0.10         # -10% abnormal every day after the event
    abn_intra = _panel([0.0] * n)
    hs = _panel([0.0] * n)
    ev = pd.DataFrame({"date": [DATES[25]], "permno": ["A"], "gap": [0.25]})
    out = event_cars(ev, abn_cc, abn_intra, hs, horizons=(4,))
    assert math.isclose(float(out["close_4"].iloc[0]), -0.40)
    assert out["n_days_4"].iloc[0] == 4


def test_event_cars_sums_abnormal_and_charges_both_spreads():
    n = 20
    abn_cc = _panel([0.01] * n)                    # +1% abnormal every day
    abn_intra = _panel([0.02] * n)
    hs = _panel([0.005] * n)                       # 0.5% half-spread
    ev = pd.DataFrame({"date": [DATES[5]], "permno": ["A"], "gap": [0.30]})
    out = event_cars(ev, abn_cc, abn_intra, hs, horizons=(5,), runup_days=3)
    r = out.iloc[0]
    assert math.isclose(r["close_5"], 0.05)                    # 5 days x 1%
    assert math.isclose(r["open_5"], 0.07)                     # + the entry-day intraday 2%
    assert math.isclose(r["close_5_net"], 0.05 - 0.01)         # entry + exit half-spread
    assert math.isclose(r["runup"], 0.03)                      # 3 prior days x 1%
    assert r["n_days_5"] == 5


def test_delisted_name_contributes_only_the_days_it_traded():
    n = 20
    abn_cc = _panel([0.01] * n)
    abn_cc.iloc[9:, 0] = np.nan                    # the name dies after index 8
    abn_intra = _panel([0.0] * n)
    hs = _panel([0.0] * n)
    ev = pd.DataFrame({"date": [DATES[5]], "permno": ["A"], "gap": [0.30]})
    out = event_cars(ev, abn_cc, abn_intra, hs, horizons=(20,))
    r = out.iloc[0]
    assert r["n_days_20"] == 3                     # days 6, 7, 8 only
    assert math.isclose(r["close_20"], 0.03)
