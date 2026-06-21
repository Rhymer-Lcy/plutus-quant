"""Inverse-vol weighting: weights sum to 1, the calmer name gets more capital, and the
per-name cap binds."""
import numpy as np
import pandas as pd

from plutus.research.backtest.sizing import inverse_vol_weighter


def _two_name_panel():
    dates = pd.bdate_range("2020-01-01", periods=120)
    rng = np.random.default_rng(0)
    calm = 100.0 * np.cumprod(1 + rng.normal(0, 0.005, len(dates)))    # low vol
    wild = 100.0 * np.cumprod(1 + rng.normal(0, 0.030, len(dates)))    # high vol
    return pd.DataFrame({"CALM": calm, "WILD": wild}, index=dates), dates


def test_weights_sum_to_one_and_favor_low_vol():
    panel, dates = _two_name_panel()
    w = inverse_vol_weighter(panel, lookback=60, cap=None)(dates[-1], ["CALM", "WILD"])
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["CALM"] > w["WILD"]              # inverse-vol tilts to the calmer name


def test_cap_binds_and_renormalizes():
    panel, dates = _two_name_panel()
    w = inverse_vol_weighter(panel, lookback=60, cap=0.6)(dates[-1], ["CALM", "WILD"])
    assert w["CALM"] <= 0.6 + 1e-9           # capped
    assert abs(sum(w.values()) - 1.0) < 1e-9 # still normalized after the cap spill


def test_empty_tickers_returns_empty():
    panel, dates = _two_name_panel()
    assert inverse_vol_weighter(panel)(dates[-1], []) == {}
