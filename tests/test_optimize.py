"""Turnover-aware optimizer: dollar-neutral + gross/cap constraints, longs the high-alpha names,
and a turnover penalty that demonstrably reduces trading."""
import pandas as pd

from plutus.research.backtest.optimize import turnover_aware_weights


def test_dollar_neutral_gross_and_direction():
    alpha = pd.Series({"a": 2.0, "b": 1.0, "c": -1.0, "d": -2.0})
    w = turnover_aware_weights(alpha, pd.Series(dtype=float), gamma=0.0, slip=5e-4,
                               name_cap=0.6, gross=2.0)
    assert abs(w.sum()) < 1e-5                 # dollar-neutral
    assert w.abs().sum() <= 2.0 + 1e-5         # gross constraint
    assert (w.abs() <= 0.6 + 1e-5).all()       # per-name cap
    assert w["a"] > 0 and w["d"] < 0           # long highest alpha, short lowest


def test_turnover_penalty_reduces_trading():
    alpha = pd.Series({"a": 1.0, "b": -1.0, "c": 0.1, "d": -0.1})
    w_prev = pd.Series({"c": 1.0, "d": -1.0})  # currently positioned in c (long) / d (short)
    names = ["a", "b", "c", "d"]

    def turn(w):
        return (w.reindex(names).fillna(0) - w_prev.reindex(names).fillna(0)).abs().sum()

    w_free = turnover_aware_weights(alpha, w_prev, gamma=0.0, slip=5e-4, name_cap=1.0, gross=2.0)
    w_sticky = turnover_aware_weights(alpha, w_prev, gamma=1e7, slip=5e-4, name_cap=1.0, gross=2.0)
    assert turn(w_sticky) < turn(w_free)        # a big turnover penalty keeps closer to w_prev
