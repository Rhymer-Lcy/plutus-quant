"""Point-in-time S&P 500 membership: members_asof returns the constituents of the most recent
change date on or before the query date, and the empty set before history begins."""
import pandas as pd

from plutus.data import universe as u


def _history():
    return [
        (pd.Timestamp("2020-01-01"), frozenset({"AAA", "BBB", "CCC"})),
        (pd.Timestamp("2020-06-01"), frozenset({"AAA", "BBB", "DDD"})),
        (pd.Timestamp("2021-01-01"), frozenset({"AAA", "DDD", "EEE"})),
    ]


def test_members_asof_picks_latest_change_on_or_before():
    m = u.members_asof_from_history(_history())
    assert m("2019-12-31") == set()                       # before history -> empty
    assert m("2020-01-01") == {"AAA", "BBB", "CCC"}       # exactly on a change date
    assert m("2020-03-15") == {"AAA", "BBB", "CCC"}       # between changes -> prior set
    assert m("2020-06-01") == {"AAA", "BBB", "DDD"}
    assert m("2020-12-31") == {"AAA", "BBB", "DDD"}
    assert m("2021-06-01") == {"AAA", "DDD", "EEE"}


def test_load_sp500_history_parses_csv(tmp_path):
    csv = tmp_path / "hist.csv"
    csv.write_text(
        "date,tickers\n"
        '1996-01-02,"AAA,BBB,CCC"\n'
        '1996-03-01,"AAA,BBB,DDD"\n',
        encoding="utf-8",
    )
    hist = u.load_sp500_history(csv)
    assert len(hist) == 2
    assert hist[0][0] == pd.Timestamp("1996-01-02")
    assert hist[0][1] == frozenset({"AAA", "BBB", "CCC"})
    assert hist[1][1] == frozenset({"AAA", "BBB", "DDD"})


def test_history_round_trips_through_members_asof(tmp_path):
    csv = tmp_path / "hist.csv"
    csv.write_text('date,tickers\n2000-01-03,"XOM,GE,MSFT"\n2010-01-04,"AAPL,MSFT,XOM"\n',
                   encoding="utf-8")
    m = u.members_asof_from_history(u.load_sp500_history(csv))
    assert m("2005-06-15") == {"XOM", "GE", "MSFT"}
    assert m("2015-06-15") == {"AAPL", "MSFT", "XOM"}
