"""Single-factor evaluation: a factor that perfectly orders forward returns must show
rank IC = +1 and a monotone-increasing quantile profile."""
import numpy as np
import pandas as pd

from plutus.research.eval.factor_eval import compute_ic, quantile_returns


def _perfect_factor_panel():
    # 5 eval dates, 10 names. Factor at t exactly equals next-period return rank -> IC = +1.
    eval_dates = list(pd.bdate_range("2020-01-31", periods=5, freq="BME"))
    names = [f"N{i}" for i in range(10)]
    close = pd.DataFrame(1.0, index=eval_dates, columns=names)
    factor = pd.DataFrame(0.0, index=eval_dates, columns=names)
    for i in range(len(eval_dates) - 1):
        # name j grows by (j+1)% next period; factor ranks names by that growth
        growth = {n: 1.0 + 0.01 * (j + 1) for j, n in enumerate(names)}
        close.loc[eval_dates[i + 1]] = close.loc[eval_dates[i]] * pd.Series(growth)
        factor.loc[eval_dates[i]] = pd.Series({n: float(j) for j, n in enumerate(names)})
    return factor, close, eval_dates


def test_perfect_factor_has_ic_one():
    factor, close, eval_dates = _perfect_factor_panel()
    res = compute_ic(factor, close, eval_dates)
    assert res.n_periods == 4
    assert res.mean_ic > 0.999            # perfect rank agreement
    assert res.hit_rate == 1.0


def test_quantile_profile_is_monotone_for_perfect_factor():
    factor, close, eval_dates = _perfect_factor_panel()
    q = quantile_returns(factor, close, eval_dates, n_q=5)
    assert list(q.index) == [0, 1, 2, 3, 4]
    assert q.is_monotonic_increasing      # higher factor quantile -> higher forward return
