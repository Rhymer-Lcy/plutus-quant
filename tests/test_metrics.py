"""Shared helpers extracted during the dedup cleanup: month_ends (backtest.metrics) and
ticker_panel_to_permno (crsp_source). These were previously copy-pasted / imported script-to-script."""
import pandas as pd

from plutus.data.sources.crsp_source import ticker_panel_to_permno
from plutus.research.backtest.metrics import month_ends


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
