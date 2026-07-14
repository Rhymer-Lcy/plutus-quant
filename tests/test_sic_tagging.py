"""Point-in-time industry tagging: a reclassified company must be labelled by what it WAS."""
import pandas as pd

from plutus.data.sources.crsp_source import sic_membership_panel, tag_sic_asof

SPELLS = pd.DataFrame({
    "permno": ["1", "1", "2"],
    "siccd":  ["3674", "2836", "2834"],          # name 1: semiconductors -> biotech
    "start":  pd.to_datetime(["2010-01-01", "2018-01-01", "2010-01-01"]),
    "end":    pd.to_datetime(["2017-12-31", "2024-12-31", "2024-12-31"]),
})


def test_a_reclassified_name_is_tagged_by_what_it_was_then():
    events = pd.DataFrame({"permno": ["1", "1"],
                           "date": pd.to_datetime(["2015-06-01", "2020-06-01"])})
    out = tag_sic_asof(events, SPELLS).sort_values("date")
    assert list(out["siccd"]) == ["3674", "2836"]     # NOT biotech in 2015; biotech in 2020


def test_a_stable_name_keeps_its_code():
    events = pd.DataFrame({"permno": ["2"], "date": pd.to_datetime(["2015-06-01"])})
    assert tag_sic_asof(events, SPELLS)["siccd"].iloc[0] == "2834"


def test_a_date_outside_every_spell_is_dropped_not_guessed():
    events = pd.DataFrame({"permno": ["1", "2"],
                           "date": pd.to_datetime(["2005-06-01", "2015-06-01"])})
    out = tag_sic_asof(events, SPELLS)
    assert len(out) == 1 and out["permno"].iloc[0] == "2"


def test_an_unknown_permno_is_dropped():
    events = pd.DataFrame({"permno": ["99"], "date": pd.to_datetime(["2015-06-01"])})
    assert tag_sic_asof(events, SPELLS).empty


def test_membership_panel_follows_the_reclassification():
    idx = pd.bdate_range("2014-12-29", "2018-01-05", freq="YS")
    panel = sic_membership_panel(SPELLS, {"2836", "2834"}, idx, ["1", "2"])
    # name 1 is semiconductors until 2017 and biotech from 2018; name 2 is always pharma
    assert not panel.loc[pd.Timestamp("2015-01-01"), "1"]
    assert panel.loc[pd.Timestamp("2018-01-01"), "1"]
    assert panel["2"].all()


def test_membership_panel_ignores_names_outside_the_columns():
    idx = pd.bdate_range("2015-01-01", periods=3)
    panel = sic_membership_panel(SPELLS, {"2834"}, idx, ["2"])
    assert list(panel.columns) == ["2"] and panel["2"].all()


def test_a_code_that_recurs_must_not_produce_overlapping_spells():
    # A -> B -> A. Built as a min/max envelope per (permno, code), the "A" spell would span B's
    # whole period, so a date inside B matches BOTH spells and the event is counted twice under
    # two industries. Contiguous runs are the only correct construction.
    runs = pd.DataFrame({
        "permno": ["1", "1", "1"],
        "siccd":  ["3674", "2836", "3674"],
        "start":  pd.to_datetime(["2010-01-01", "2015-01-01", "2020-01-01"]),
        "end":    pd.to_datetime(["2014-12-31", "2019-12-31", "2024-12-31"]),
    })
    events = pd.DataFrame({"permno": ["1", "1", "1"],
                           "date": pd.to_datetime(["2012-06-01", "2017-06-01", "2022-06-01"])})
    out = tag_sic_asof(events, runs)
    assert len(out) == len(events)                     # exactly one industry per event, no dupes
    assert list(out.sort_values("date")["siccd"]) == ["3674", "2836", "3674"]
