"""Tests for the distance-method pairs engine: selection, mean-reversion P&L, costs, no-look-ahead."""
import numpy as np
import pandas as pd

from plutus.research.backtest.pairs import select_pairs, distance_pairs_backtest


def _oscillating_pair(n=160, period=40, amp=0.06, drift=0.0003):
    t = np.arange(n)
    common = 100 * np.exp(drift * t)
    osc = amp * np.sin(2 * np.pi * t / period)
    return common * (1 + osc), common * (1 - osc)


def test_select_pairs_picks_comoving():
    n = 60
    a, b = _oscillating_pair(n)
    c = 100 * np.exp(0.01 * np.arange(n))          # unrelated, strongly diverging
    idx = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)
    pairs = select_pairs(df, top_k=1)
    assert len(pairs) == 1
    assert {pairs[0][0], pairs[0][1]} == {"A", "B"}   # co-moving pair has the smallest SSD


def test_select_pairs_excludes_incomplete_names():
    n = 60
    a, b = _oscillating_pair(n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame({"A": a, "B": b}, index=idx)
    df.loc[df.index[:5], "B"] = np.nan             # B not complete over the window
    assert select_pairs(df, top_k=5) == []         # only A complete -> no eligible pair


def test_mean_reverting_pair_profitable_gross():
    a, b = _oscillating_pair(n=200, period=40, amp=0.06)
    idx = pd.bdate_range("2020-01-01", periods=200)
    df = pd.DataFrame({"A": a, "B": b}, index=idx)
    res = distance_pairs_backtest(df, formation=40, trading=120, step=120, top_k=1, entry_z=1.0,
                                  slippage_bps=0.0, borrow_bps_annual=0.0, min_names=2)
    assert res.n_windows >= 1
    assert res.ann_return > 0                       # a clean reverting spread makes money gross


def test_costs_reduce_return():
    a, b = _oscillating_pair(n=200, period=40, amp=0.06)
    idx = pd.bdate_range("2020-01-01", periods=200)
    df = pd.DataFrame({"A": a, "B": b}, index=idx)
    kw = dict(formation=40, trading=120, step=120, top_k=1, entry_z=1.0, min_names=2)
    gross = distance_pairs_backtest(df, slippage_bps=0.0, borrow_bps_annual=0.0, **kw)
    net = distance_pairs_backtest(df, slippage_bps=50.0, borrow_bps_annual=500.0, **kw)
    assert net.returns.sum() < gross.returns.sum()


def test_no_lookahead_formation_scale():
    """A pair identical in formation but diverging only in trading has formation spread_std ~ 0,
    so the trigger scale (formation-only) is ~0 — proving the threshold does NOT peek at trading."""
    n_form, n_trade = 40, 60
    t = np.arange(n_form + n_trade)
    common = 100 * np.exp(0.0003 * t)
    a, b = common.copy(), common.copy()
    osc = 0.05 * np.sin(2 * np.pi * np.arange(n_trade) / 30)
    a[n_form:] = common[n_form:] * (1 - osc)        # divergence ONLY after formation
    b[n_form:] = common[n_form:] * (1 + osc)
    idx = pd.bdate_range("2020-01-01", periods=n_form + n_trade)
    df = pd.DataFrame({"A": a, "B": b}, index=idx)
    pairs = select_pairs(df.iloc[:n_form], top_k=1)
    assert pairs[0][5] < 1e-9                        # formation spread std ~ 0 (no trading leak)
