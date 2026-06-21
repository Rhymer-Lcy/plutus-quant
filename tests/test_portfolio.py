"""Backtest engine invariants: delisting force-liquidation, valuation/delisting bookkeeping,
top-N hysteresis band, intra-basket weighting, rebalance cadence, zero-cost sanity."""
import numpy as np
import pandas as pd

from plutus.research.backtest.frictions import ZERO_COSTS
from plutus.research.backtest.portfolio import (_hold_value, _select_top,
                                               signal_portfolio_backtest, valuation_panel)


def test_valuation_panel_ffills_gap_but_stops_after_delisting():
    dates = pd.bdate_range("2020-01-01", periods=6)
    price = pd.DataFrame(index=dates, dtype=float)
    price["halt"] = [10.0, np.nan, np.nan, 13.0, 14.0, 15.0]    # interior gap, resumes
    price["dead"] = [20.0, 21.0, 22.0, np.nan, np.nan, np.nan]  # delists after bar 2
    val, last_valid, last_price = valuation_panel(price)
    assert val.loc[dates[1], "halt"] == 10.0 and val.loc[dates[2], "halt"] == 10.0
    assert np.isnan(val.loc[dates[3], "dead"]) and np.isnan(val.loc[dates[5], "dead"])
    assert last_valid["dead"] == dates[2] and last_price["dead"] == 22.0


def test_hold_value_ignores_nan_prices():
    val = pd.Series({"AAA": 10.0, "BBB": np.nan})
    assert _hold_value({"AAA": 100, "BBB": 200}, val) == 1000.0   # NaN-priced name contributes 0


def test_rebalance_freq_changes_cadence_and_defaults_to_monthly():
    dates = pd.bdate_range("2020-01-01", "2020-12-31")
    price = pd.DataFrame({"AAA": np.linspace(10, 14, len(dates)),
                          "BBB": np.linspace(20, 18, len(dates))}, index=dates)
    signal = pd.DataFrame({"AAA": 2.0, "BBB": 1.0}, index=dates)
    n = {f: signal_portfolio_backtest(price, signal, 1_000_000.0, 1, rebalance_freq=f).n_rebalances
         for f in ("Q", "M", "W")}
    assert n["Q"] < n["M"] < n["W"]
    default = signal_portfolio_backtest(price, signal, 1_000_000.0, 1).n_rebalances
    assert default == n["M"]


def test_delisted_holding_is_liquidated_and_capital_recycled():
    dates = pd.bdate_range("2020-01-02", "2020-03-31")
    price = pd.DataFrame(index=dates, dtype=float)
    price["DIES"] = 10.0
    price.loc[price.index > pd.Timestamp("2020-02-14"), "DIES"] = np.nan   # delists mid-Feb
    price["KEEP"] = 10.0
    price.loc[price.index > pd.Timestamp("2020-03-02"), "KEEP"] = 20.0     # doubles once buyable

    signal = pd.DataFrame(index=dates, dtype=float)
    signal["DIES"] = 2.0    # preferred while tradable -> bought at the first rebalance
    signal["KEEP"] = 1.0

    r = signal_portfolio_backtest(price, signal, capital=1_000_000, n_hold=1, costs=ZERO_COSTS)
    # equity never NaN/phantom; capital force-liquidated out of the dead name and recycled
    # into KEEP (which doubled) -> ~+100%. A stuck delisted holding would freeze return ~0.
    assert r.equity.notna().all()
    assert r.total_return > 0.5


def test_constant_price_zero_cost_is_flat():
    dates = pd.bdate_range("2020-01-02", "2020-04-30")
    price = pd.DataFrame({"AAA": 10.0, "BBB": 10.0}, index=dates)
    signal = pd.DataFrame({"AAA": 1.0, "BBB": 2.0}, index=dates)
    r = signal_portfolio_backtest(price, signal, capital=1_000_000, n_hold=1, costs=ZERO_COSTS)
    assert abs(r.total_return) < 1e-9


def _one_rebalance_then_b_doubles():
    dates = pd.bdate_range("2020-01-02", "2020-02-14")
    price = pd.DataFrame({"AAA": 10.0, "BBB": 10.0}, index=dates)
    price.loc[price.index > pd.Timestamp("2020-02-07"), "BBB"] = 20.0
    signal = pd.DataFrame({"AAA": 1.0, "BBB": 1.0}, index=dates)   # tie -> both held at n_hold=2
    return price, signal


def test_equal_weight_callable_matches_default():
    price, signal = _one_rebalance_then_b_doubles()
    kw = dict(capital=10_000_000, n_hold=2, costs=ZERO_COSTS)
    base = signal_portfolio_backtest(price, signal, **kw)
    explicit = signal_portfolio_backtest(price, signal, weight_asof=lambda d, c: {x: 1.0 for x in c}, **kw)
    assert abs(explicit.total_return - base.total_return) < 1e-9


def test_weighting_shifts_capital_toward_overweighted_name():
    price, signal = _one_rebalance_then_b_doubles()
    kw = dict(capital=10_000_000, n_hold=2, costs=ZERO_COSTS)
    eq = signal_portfolio_backtest(price, signal, **kw)
    fav_b = signal_portfolio_backtest(price, signal, weight_asof=lambda d, c: {"AAA": 0.2, "BBB": 0.8}, **kw)
    assert 0.45 < eq.total_return < 0.55       # BBB doubles; equal weight ~+50%
    assert 0.75 < fav_b.total_return < 0.85    # overweight BBB (0.8) ~+80%
    assert fav_b.total_return > eq.total_return


def test_select_top_band_zero_is_plain_topn():
    assert _select_top(["a", "b", "c", "d", "e"], {"x", "y"}, 3, 0) == ["a", "b", "c"]


def test_select_top_keeps_incumbent_in_buffer_zone():
    out = _select_top(["a", "b", "c", "d", "e"], {"e"}, 3, 2)   # e held, in exit zone (rank 5)
    assert out[0] == "e" and len(out) == 3 and set(out) == {"e", "a", "b"}


def test_select_top_new_name_must_rank_in_strict_topn():
    assert _select_top(["a", "b", "c", "d", "e"], set(), 3, 2) == ["a", "b", "c"]


def test_rebalance_buffer_cuts_turnover_cost():
    dates = pd.bdate_range("2020-01-02", "2020-06-30")
    price = pd.DataFrame(10.0, index=dates, columns=["AAA", "BBB", "CCC"])   # flat prices
    periods = dates.to_period("M")
    uniq = list(dict.fromkeys(periods))
    even = {p for i, p in enumerate(uniq) if i % 2 == 0}
    signal = pd.DataFrame(index=dates)
    signal["AAA"] = 3.0
    signal["BBB"] = [2.0 if p in even else 1.0 for p in periods]
    signal["CCC"] = [1.0 if p in even else 2.0 for p in periods]
    kw = dict(capital=1_000_000, n_hold=2)
    no_buf = signal_portfolio_backtest(price, signal, rebalance_band=0, **kw)
    buf = signal_portfolio_backtest(price, signal, rebalance_band=1, **kw)
    assert buf.total_costs < no_buf.total_costs
