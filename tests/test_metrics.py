"""Shared helpers extracted during the dedup cleanups: month_ends and the clustering-robust
inference (backtest.metrics), and ticker_panel_to_permno (crsp_source). These were previously
copy-pasted / imported script-to-script."""
import math

import pandas as pd

from plutus.data.sources.crsp_source import ticker_panel_to_permno
from plutus.research.backtest.metrics import clustered_tstat, month_ends, tstat


def test_month_ends_returns_last_trading_day_per_month():
    dates = pd.bdate_range("2024-01-01", "2024-03-31")
    me = month_ends(dates)
    assert len(me) == 3                                   # one per calendar month
    for period in ("2024-01", "2024-02", "2024-03"):
        last = dates[dates.to_period("M") == pd.Period(period)].max()
        assert last in me
    assert me == sorted(me)                               # ascending


def test_ticker_panel_to_permno_rekeys_columns():
    dates = pd.bdate_range("2024-01-01", periods=3)
    panel = pd.DataFrame({"AAPL": [1.0, 2.0, 3.0], "MSFT": [4.0, 5.0, 6.0]}, index=dates)
    out = ticker_panel_to_permno(panel, {"14593": "AAPL", "10107": "MSFT", "99999": "GONE"})
    assert set(out.columns) == {"14593", "10107"}        # only mapped+present tickers carried over
    assert list(out["14593"]) == [1.0, 2.0, 3.0]


def test_tstat_of_a_constant_is_undefined():
    assert math.isnan(tstat(pd.Series([0.05] * 10)))      # no dispersion
    assert math.isnan(tstat(pd.Series([0.05])))           # nothing to test


def test_tstat_grows_with_the_sample():
    assert tstat(pd.Series([0.02, 0.04, 0.06] * 12)) > tstat(pd.Series([0.02, 0.04, 0.06])) > 0


def test_clustering_collapses_events_inside_one_period():
    dates = pd.Series([pd.Timestamp("2024-03-05")] * 10)
    x = pd.Series([0.04, 0.05, 0.06] + [0.05] * 7)
    assert math.isnan(clustered_tstat(x, dates, freq="M"))   # one cluster -> undefined
    assert tstat(x) > 10                                     # the naive t looks "significant"


def test_clustering_deflates_a_deadline_driven_t_stat():
    # One month supplies 20 winners; two ordinary months are flat. The event-level t is inflated by
    # treating the 20 clustered events as independent draws; the clustered t must not be.
    dates = pd.Series([pd.Timestamp("2024-06-05")] * 20
                      + [pd.Timestamp("2024-07-05"), pd.Timestamp("2024-08-05")])
    x = pd.Series([0.05] * 20 + [0.001, -0.001])
    assert tstat(x) > clustered_tstat(x, dates, freq="M")
    assert clustered_tstat(x, dates, freq="M") < 2.0         # fails a frozen t > 2 bar


def test_freq_sets_the_cluster_width():
    # Three filing deadlines inside ONE quarter: monthly clustering sees three observations,
    # quarterly clustering sees one (hence undefined). 13F needs the quarterly view, biotech the
    # monthly one -- the same function, told which.
    dates = pd.Series([pd.Timestamp("2024-01-15")] * 5 + [pd.Timestamp("2024-02-15")] * 5
                      + [pd.Timestamp("2024-03-15")] * 5)
    x = pd.Series([0.01] * 5 + [0.05] * 5 + [0.03] * 5)
    assert not math.isnan(clustered_tstat(x, dates, freq="M"))
    assert math.isnan(clustered_tstat(x, dates, freq="Q"))
