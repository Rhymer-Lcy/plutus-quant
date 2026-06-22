"""Market-regime overlay: cap-weighted index construction and the trend-filter exposure
(full exposure in an uptrend, floor when the market falls below its moving average)."""
import numpy as np
import pandas as pd

from plutus.research.backtest.regime import cap_weighted_index, trend_exposure


def test_cap_weighted_index_tracks_big_name():
    dates = pd.bdate_range("2020-01-01", periods=5)
    # BIG has 10x the cap of SMALL, so the index should track BIG's path closely
    adj = pd.DataFrame({"BIG": [100, 110, 121, 133.1, 146.41],
                        "SMALL": [10, 9, 8, 7, 6]}, index=dates, dtype=float)
    cap = pd.DataFrame({"BIG": 1_000_000.0, "SMALL": 100_000.0}, index=dates)
    idx = cap_weighted_index(adj, cap)
    assert idx.iloc[0] == 1.0                      # starts at 1.0 (day-0 return is 0)
    # BIG compounds +10%/day; cap-weighted index should rise strongly (BIG dominates)
    assert idx.iloc[-1] > 1.30


def test_trend_exposure_full_in_uptrend_floor_in_downtrend():
    dates = pd.bdate_range("2020-01-01", periods=300)
    up = pd.Series(np.linspace(100, 200, 300), index=dates)        # steady uptrend
    exp = trend_exposure(up, window=200, floor=0.0)
    assert exp(dates[-1]) == 1.0                                   # above its MA -> full

    down = pd.Series(np.r_[np.linspace(100, 200, 150), np.linspace(200, 80, 150)], index=dates)
    exp2 = trend_exposure(down, window=100, floor=0.0)
    assert exp2(dates[-1]) == 0.0                                  # crashed below MA -> cash


def test_trend_exposure_respects_floor():
    dates = pd.bdate_range("2020-01-01", periods=250)
    down = pd.Series(np.linspace(200, 80, 250), index=dates)       # falling
    exp = trend_exposure(down, window=100, floor=0.3)
    assert exp(dates[-1]) == 0.3                                   # floor, not zero
