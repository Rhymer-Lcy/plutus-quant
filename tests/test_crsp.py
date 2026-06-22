"""CRSP adapter (network-free): membership spells -> members_asof, total-return-adjusted
price (anchored, delisting-aware), market cap, and streaming PERMNO/date filter out of a zip."""
import zipfile

import numpy as np
import pandas as pd

from plutus.data.sources import crsp_source as crsp


def _spells():
    return pd.DataFrame({
        "permno": [10, 20, 30],
        "start": pd.to_datetime(["2010-01-01", "2012-01-01", "2015-01-01"]),
        "end": pd.to_datetime(["2014-12-31", "2018-12-31", "2024-12-31"]),
    })


def test_members_asof_and_union():
    m = crsp.members_asof_from_spells(_spells())
    assert m("2011-06-30") == {10}
    assert m("2013-06-30") == {10, 20}
    assert m("2016-06-30") == {20, 30}
    assert m("2009-01-01") == set()
    assert crsp.union_permnos(_spells(), "2013-01-01", "2016-01-01") == {10, 20, 30}
    assert crsp.union_permnos(_spells(), "2011-01-01", "2011-06-30") == {10}


def test_load_membership_from_xlsx(tmp_path):
    p = tmp_path / "con.xlsx"
    pd.DataFrame({
        "CRSP Permanent Number (permno)": [10, 20],
        "start": ["2010-01-01", "2012-01-01"],
        "end": ["2014-12-31", "2018-12-31"],
    }).to_excel(p, index=False)
    sp = crsp.load_membership(p)
    assert list(sp.columns) == ["permno", "start", "end"]
    assert set(sp["permno"]) == {10, 20}
    assert sp["start"].dtype.kind == "M"


def _long():
    # PERMNO 1: 3 days; PERMNO 2: delists after day 2 (no day-3 row)
    d = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    rows = [
        {"PERMNO": 1, "date": d[0], "Ticker": "AAA", "DlyClose": 100.0, "DlyRet": 0.0, "DlyCap": 1000.0},
        {"PERMNO": 1, "date": d[1], "Ticker": "AAA", "DlyClose": 110.0, "DlyRet": 0.10, "DlyCap": 1100.0},
        {"PERMNO": 1, "date": d[2], "Ticker": "AAA", "DlyClose": 104.5, "DlyRet": -0.05, "DlyCap": 1045.0},
        {"PERMNO": 2, "date": d[0], "Ticker": "BBB", "DlyClose": 50.0, "DlyRet": 0.0, "DlyCap": 500.0},
        {"PERMNO": 2, "date": d[1], "Ticker": "BBB", "DlyClose": 60.0, "DlyRet": 0.20, "DlyCap": 600.0},
    ]
    return pd.DataFrame(rows)


def test_build_tr_adjusted_close_anchors_and_compounds():
    adj = crsp.build_tr_adjusted_close(_long())
    d = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    assert adj.loc[d[0], 1] == 100.0                       # anchored to first real close
    assert abs(adj.loc[d[1], 1] - 110.0) < 1e-9            # +10% total return
    assert abs(adj.loc[d[2], 1] - 104.5) < 1e-9            # then -5%
    assert abs(adj.loc[d[0], 2] - 50.0) < 1e-9
    assert abs(adj.loc[d[1], 2] - 60.0) < 1e-9
    assert np.isnan(adj.loc[d[2], 2])                      # delisted -> NaN after last bar


def test_build_mktcap_dollars():
    cap = crsp.build_mktcap(_long())
    assert cap.loc[pd.Timestamp("2020-01-03"), 1] == 1100.0 * 1000.0   # $000s -> $


def test_latest_ticker_map():
    assert crsp.latest_ticker_map(_long()) == {1: "AAA", 2: "BBB"}


def test_size_band_members_asof():
    dates = pd.to_datetime(["2020-01-31"])
    cap = pd.DataFrame({"A": [500.0], "B": [400.0], "C": [300.0], "D": [200.0], "E": [100.0]},
                       index=dates)
    m = crsp.size_band_members_asof(cap, exclude_top=1, band_size=2)
    assert m("2020-02-15") == {"B", "C"}        # drop largest (A), take next 2 by cap
    m_all = crsp.size_band_members_asof(cap, exclude_top=0, band_size=3)
    assert m_all("2020-01-31") == {"A", "B", "C"}


def test_stream_universe_filters_common_major_priced(tmp_path):
    csv = (
        "PERMNO,DlyCalDt,Ticker,DlyRet,SecurityType,SecuritySubType,PrimaryExch,DlyClose,DlyCap\n"
        "10,2020-01-02,KEEP1,0.01,EQTY,COM,N,50.0,200000\n"   # keep: common, NYSE, $50, $200M
        "20,2020-01-02,LOWPX,0.01,EQTY,COM,Q,3.0,200000\n"    # drop: price < $5
        "30,2020-01-02,MICRO,0.01,EQTY,COM,Q,50.0,50000\n"    # drop: cap < $100M
        "40,2020-01-02,PREF,0.01,EQTY,PFD,N,50.0,200000\n"    # drop: not common (preferred)
        "50,2020-01-02,OTC,0.01,EQTY,COM,P,50.0,200000\n"     # drop: not a major exchange
        "60,2020-01-02,KEEP2,0.01,EQTY,COM,A,10.0,150000\n"   # keep: common, AMEX, $10, $150M
    )
    zp = tmp_path / "u.csv.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("u.csv", csv)
    out = crsp.stream_universe(zp, "2020-01-01", "2020-12-31", price_min=5.0, cap_min_000=100000.0)
    assert set(out["PERMNO"]) == {10, 60}
    assert "SecurityType" not in out.columns           # filter cols dropped


def test_stream_filtered_from_zip(tmp_path):
    csv = (
        "PERMNO,DlyCalDt,Ticker,DlyClose,DlyRet,DlyCap,ShrOut\n"
        "10,2010-01-04,AAA,100.25,0.01,1000.5,10\n"
        "10,2010-01-05,AAA,101.50,0.0099,1010.5,10\n"
        "10,2010-01-06,AAA,102.00,0.0099,1020.5,10\n"
        "20,2010-01-04,BBB,50.10,0.02,500.5,10\n"
        "30,2010-01-04,CCC,200.75,0.0,2000.5,10\n"
        "30,2010-01-05,CCC,202.30,0.01,2020.5,10\n"
    )
    zp = tmp_path / "crsp.csv.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("crsp.csv", csv)
    out = crsp.stream_filtered(zp, {10, 30}, "2010-01-04", "2010-01-05")
    assert set(out["PERMNO"]) == {10, 30}                  # PERMNO 20 filtered out
    assert out["date"].max() == pd.Timestamp("2010-01-05") # 2010-01-06 outside window
    assert len(out) == 4
    assert out["DlyClose"].dtype.kind == "f"               # coerced to float
