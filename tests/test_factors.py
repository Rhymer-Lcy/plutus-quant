"""Factor processing: cross-sectional z-score / standardize / blend, PIT universe
restriction, and factor orientation (higher = more attractive)."""
import numpy as np
import pandas as pd

from plutus.research.factors import library as fl


def test_zscore_xs_is_row_standardized():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
    z = fl.zscore_xs(df)
    assert np.allclose(z.mean(axis=1), 0.0, atol=1e-12)
    assert np.allclose(z.std(axis=1, ddof=1), 1.0, atol=1e-12)


def test_restrict_to_universe_nans_non_members():
    dates = pd.bdate_range("2020-01-01", periods=2)
    panel = pd.DataFrame({"AAA": [1.0, 1.0], "BBB": [2.0, 2.0]}, index=dates)
    members = {dates[0]: {"AAA"}, dates[1]: {"AAA", "BBB"}}
    out = fl.restrict_to_universe(panel, lambda d: members[d])
    assert np.isnan(out.loc[dates[0], "BBB"])           # not a member on day 0 -> NaN
    assert out.loc[dates[1], "BBB"] == 2.0              # member on day 1 -> kept


def test_blend_single_factor_reduces_to_its_zscore():
    df = pd.DataFrame({"a": [1.0, 9.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
    blended = fl.blend([df])
    assert np.allclose(blended.fillna(0), fl.standardize(df).fillna(0))


def test_blend_skips_missing_factor_for_a_name():
    a = pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]})
    b = pd.DataFrame({"x": [np.nan], "y": [2.0], "z": [3.0]})   # x missing in factor b
    out = fl.blend([a, b])
    assert out["x"].notna().all()                       # x still scored on factor a alone


def test_reversal_is_negative_trailing_return():
    close = pd.DataFrame({"AAA": np.linspace(10, 20, 30)})      # rising -> negative reversal
    rev = fl.reversal(close, lookback=5)
    assert (rev.dropna() < 0).all().all()


def test_earnings_yield_excludes_losses():
    ni = pd.DataFrame({"PROF": [100.0], "LOSS": [-50.0]})
    mc = pd.DataFrame({"PROF": [1000.0], "LOSS": [1000.0]})
    ey = fl.earnings_yield(ni, mc)
    assert ey.loc[0, "PROF"] == 0.1
    assert np.isnan(ey.loc[0, "LOSS"])                  # negative earnings -> NaN
