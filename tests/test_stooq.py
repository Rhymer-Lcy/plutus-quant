"""Stooq adapter (network-free): symbol mapping and CSV parsing, including graceful handling
of the rate-limit / no-data response."""
import pandas as pd

from plutus.data.sources import stooq_source as st


def test_stooq_symbol_adds_us_suffix():
    assert st.stooq_symbol("AAPL") == "aapl.us"
    assert st.stooq_symbol("brk-b") == "brk-b.us"     # US class shares use a hyphen on Stooq
    assert st.stooq_symbol("aapl.us") == "aapl.us"    # pass through if a suffix is already present


def test_parse_daily_csv():
    text = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-02,185.6,186.9,185.1,186.3,52000000\n"
        "2024-01-03,184.2,185.0,183.4,184.2,48000000\n"
    )
    df = st._parse_daily_csv(text)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index[0] == pd.Timestamp("2024-01-02")
    assert df.loc[pd.Timestamp("2024-01-03"), "close"] == 184.2
    assert df.index.is_monotonic_increasing


def test_parse_rate_limit_returns_empty():
    assert st._parse_daily_csv("Exceeded the daily hits limit").empty
    assert st._parse_daily_csv("No data").empty
    assert st._parse_daily_csv("").empty
