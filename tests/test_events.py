"""PEAD signal: standardized unexpected earnings (SUE) and the point-in-time, freshness-windowed
event signal (latest surprise only while it is recent)."""
import numpy as np
import pandas as pd

from plutus.research.factors.events import (pit_event_signal,
                                            standardized_unexpected_earnings)


def test_sue_positive_for_accelerating_earnings():
    # 8 quarters; YoY change is steady at 40 then jumps to 50 -> a positive surprise
    ends = pd.to_datetime(["2018-03-31", "2018-06-30", "2018-09-30", "2018-12-31",
                           "2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31"])
    filed = ends + pd.Timedelta(days=30)
    vals = [100, 110, 120, 130, 140, 150, 160, 180]    # YoY: 40,40,40,50
    q = pd.DataFrame({"end": ends, "filed": filed, "val": vals})
    sue = standardized_unexpected_earnings(q)
    assert len(sue) >= 1
    # last quarter: YoY=50, std of [40,40,40,50]=5 -> SUE=10
    assert abs(sue["sue"].iloc[-1] - 10.0) < 0.5


def test_sue_empty_for_short_history():
    q = pd.DataFrame({"end": pd.to_datetime(["2020-03-31", "2020-06-30"]),
                      "filed": pd.to_datetime(["2020-05-01", "2020-08-01"]), "val": [10, 20]})
    assert standardized_unexpected_earnings(q).empty   # <4 YoY obs -> no SUE


def test_pit_event_signal_freshness_window():
    sue = {"AAA": pd.DataFrame({"filed": pd.to_datetime(["2020-02-15"]), "sue": [2.0]})}
    dates = pd.bdate_range("2020-01-01", "2020-12-31")
    panel = pit_event_signal(sue, dates, freshness_days=63)
    assert np.isnan(panel.loc[pd.Timestamp("2020-01-31"), "AAA"])   # before the filing
    assert panel.loc[pd.Timestamp("2020-02-28"), "AAA"] == 2.0      # fresh (≤63d)
    assert panel.loc[pd.Timestamp("2020-03-31"), "AAA"] == 2.0      # still fresh (45d)
    assert np.isnan(panel.loc[pd.Timestamp("2020-05-29"), "AAA"])   # stale (>63d) -> no signal


def test_pit_event_signal_updates_on_new_filing():
    sue = {"AAA": pd.DataFrame({"filed": pd.to_datetime(["2020-02-15", "2020-05-10"]),
                                "sue": [2.0, -1.5]})}
    dates = pd.bdate_range("2020-01-01", "2020-08-31")
    panel = pit_event_signal(sue, dates, freshness_days=63)
    assert panel.loc[pd.Timestamp("2020-02-28"), "AAA"] == 2.0      # first surprise
    assert panel.loc[pd.Timestamp("2020-05-29"), "AAA"] == -1.5     # refreshed to the new one
