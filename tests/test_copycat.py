"""13F copycat mechanics: new-position detection, filing-date entry, clustered inference."""
import math

import numpy as np
import pandas as pd

from plutus.research.backtest.copycat import (basket_cars, concentration, new_positions,
                                              quarterly_tstat, top_weights)

DATES = pd.bdate_range("2024-01-01", periods=300)


def _filings():
    return pd.DataFrame({
        "accession": ["a1", "a2", "a3", "b1"],
        "cik": ["1", "1", "1", "2"],
        "period": pd.to_datetime(["2023-12-31", "2024-03-31", "2024-06-30", "2024-03-31"]),
        "filing_date": pd.to_datetime(["2024-02-14", "2024-05-15", "2024-08-14", "2024-05-15"]),
    })


def _holdings():
    # manager 1: a1 holds X,Y -> a2 holds Y,Z (Z is NEW) -> a3 holds Z,W (W is NEW)
    # manager 2: b1 holds X (first filing -- NOT new, no predecessor)
    return pd.DataFrame({
        "accession": ["a1", "a1", "a2", "a2", "a3", "a3", "b1"],
        "permno":    ["X",  "Y",  "Y",  "Z",  "Z",  "W",  "X"],
        "value_usd": [100.0, 300.0, 200.0, 200.0, 50.0, 50.0, 999.0],
    })


def test_new_positions_are_absent_from_the_prior_filing():
    fresh = new_positions(_holdings(), _filings())
    got = set(zip(fresh["cik"], fresh["permno"]))
    assert got == {("1", "Z"), ("1", "W")}          # Y was re-weighted, not new


def test_first_filing_is_never_all_new():
    fresh = new_positions(_holdings(), _filings())
    assert "2" not in set(fresh["cik"])             # manager 2's opening book is not conviction
    assert not ((fresh["cik"] == "1") & (fresh["permno"] == "X")).any()


def test_weight_is_share_of_reported_value():
    fresh = new_positions(_holdings(), _filings())
    z = fresh[(fresh["cik"] == "1") & (fresh["permno"] == "Z")].iloc[0]
    assert math.isclose(z["weight"], 200.0 / 400.0)  # a2 total = 200 + 200


def test_top_weights_picks_the_largest():
    top = top_weights(_holdings(), _filings(), n=1)
    a1 = top[top["accession"] == "a1"].iloc[0]
    assert a1["permno"] == "Y"                       # 300 > 100


def test_concentration_is_a_unit_free_ratio():
    c = concentration(_holdings(), top=1)
    assert math.isclose(c["a1"], 300.0 / 400.0)
    scaled = _holdings().assign(value_usd=lambda d: d["value_usd"] * 1000.0)
    assert math.isclose(concentration(scaled, top=1)["a1"], c["a1"])


def _abn(values, permnos=("X",)):
    return pd.DataFrame({p: values for p in permnos}, index=DATES[:len(values)], dtype=float)


def test_entry_is_the_filing_date_and_that_close_earns_nothing():
    abn = _abn([0.0] * 50)
    abn.iloc[10] = 0.05          # the filing-date close itself -- must NOT be earned
    abn.iloc[11] = 0.02          # the first day the copycat actually holds
    ev = pd.DataFrame({"permno": ["X"], "filing_date": [DATES[10]]})
    out = basket_cars(ev, abn, horizons=(1,), cost_per_side=0.0)
    assert math.isclose(float(out["car_1"].iloc[0]), 0.02)


def test_entry_rolls_forward_to_the_next_trading_day():
    abn = _abn([0.0] * 50)
    abn.iloc[12] = 0.03
    saturday = DATES[10] + pd.Timedelta(days=1)      # a non-trading day
    ev = pd.DataFrame({"permno": ["X"], "filing_date": [saturday]})
    out = basket_cars(ev, abn, horizons=(1,), cost_per_side=0.0)
    assert math.isclose(float(out["car_1"].iloc[0]), 0.03)   # entered at DATES[11], earns [12]


def test_round_trip_cost_is_charged_once():
    abn = _abn([0.0] * 50)
    ev = pd.DataFrame({"permno": ["X"], "filing_date": [DATES[10]]})
    out = basket_cars(ev, abn, horizons=(5,), cost_per_side=0.0005)
    assert math.isclose(float(out["car_5"].iloc[0]), -0.001)


def test_horizon_that_runs_off_the_panel_is_nan_not_truncated():
    abn = _abn([0.01] * 50)
    ev = pd.DataFrame({"permno": ["X"], "filing_date": [DATES[45]]})
    out = basket_cars(ev, abn, horizons=(2, 40), cost_per_side=0.0)
    assert not math.isnan(float(out["car_2"].iloc[0]))
    assert math.isnan(float(out["car_40"].iloc[0]))   # a 3-year column must not average stubs


def test_delisted_name_keeps_the_days_it_traded():
    abn = _abn([0.01] * 50)
    abn.iloc[15:] = np.nan                            # the name dies after index 14
    ev = pd.DataFrame({"permno": ["X"], "filing_date": [DATES[10]]})
    out = basket_cars(ev, abn, horizons=(20,), cost_per_side=0.0)
    assert out["n_days_20"].iloc[0] == 4              # days 11,12,13,14
    assert math.isclose(float(out["car_20"].iloc[0]), 0.04)


def test_quarterly_clustering_deflates_a_deadline_driven_t_stat():
    # 30 filings land on one 45-day deadline; two other quarters are flat. The event-level t
    # treats the 30 as independent; the quarterly t must not.
    dates = pd.Series([pd.Timestamp("2024-05-15")] * 30
                      + [pd.Timestamp("2024-08-14"), pd.Timestamp("2024-11-14")])
    x = pd.Series([0.05] * 30 + [0.001, -0.001])
    naive = float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))
    assert naive > quarterly_tstat(x, dates)
    assert quarterly_tstat(x, dates) < 2.0            # fails the frozen bar
