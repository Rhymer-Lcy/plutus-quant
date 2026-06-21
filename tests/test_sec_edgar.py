"""SEC EDGAR parsing (network-free): concept extraction, point-in-time filing-date alignment
(no value visible before it is filed), and trailing-twelve-month flow aggregation with the
synthesized Q4 (= 10-K annual minus that fiscal year's Q1+Q2+Q3)."""
import numpy as np
import pandas as pd

from plutus.data.sources import sec_edgar as se


def _facts() -> dict:
    def q(start, end, val, filed, fy, fp):
        return {"start": start, "end": end, "val": val, "filed": filed,
                "fy": fy, "fp": fp, "form": "10-Q", "frame": None}

    def fy(start, end, val, filed, year):
        return {"start": start, "end": end, "val": val, "filed": filed,
                "fy": year, "fp": "FY", "form": "10-K", "frame": None}

    net_income = [
        q("2022-01-01", "2022-03-31", 100, "2022-04-30", 2022, "Q1"),
        q("2022-04-01", "2022-06-30", 110, "2022-07-31", 2022, "Q2"),
        q("2022-01-01", "2022-06-30", 210, "2022-07-31", 2022, "Q2"),   # YTD 6-mo -> must be ignored
        q("2022-07-01", "2022-09-30", 120, "2022-10-31", 2022, "Q3"),
        fy("2022-01-01", "2022-12-31", 500, "2023-02-15", 2022),         # -> Q4 = 500-330 = 170
        q("2023-01-01", "2023-03-31", 130, "2023-04-30", 2023, "Q1"),
        q("2023-04-01", "2023-06-30", 140, "2023-07-31", 2023, "Q2"),
        q("2023-07-01", "2023-09-30", 150, "2023-10-31", 2023, "Q3"),
        fy("2023-01-01", "2023-12-31", 600, "2024-02-15", 2023),         # -> Q4 = 600-420 = 180
    ]
    equity = [  # instant concept: no "start"
        {"end": "2023-03-31", "val": 1000, "filed": "2023-05-01", "fy": 2023, "fp": "Q1", "form": "10-Q"},
        {"end": "2023-06-30", "val": 1100, "filed": "2023-08-01", "fy": 2023, "fp": "Q2", "form": "10-Q"},
    ]
    return {"facts": {"us-gaap": {
        "NetIncomeLoss": {"units": {"USD": net_income}},
        "StockholdersEquity": {"units": {"USD": equity}},
    }}}


def test_concept_frame_extracts_and_types():
    cf = se.concept_frame(_facts(), "NetIncomeLoss")
    assert not cf.empty
    assert {"start", "end", "filed", "val"}.issubset(cf.columns)
    assert pd.api.types.is_datetime64_any_dtype(cf["filed"])
    assert cf["filed"].is_monotonic_increasing            # sorted by filed


def test_missing_concept_is_empty():
    assert se.concept_frame(_facts(), "DoesNotExist").empty


def test_point_in_time_instant_not_visible_before_filed():
    eq = se.concept_frame(_facts(), "StockholdersEquity")
    dates = pd.bdate_range("2023-04-01", "2023-09-30")
    pit = se.point_in_time_series(eq, dates)
    assert np.isnan(pit.loc[pd.Timestamp("2023-04-14")])   # before first filing (2023-05-01)
    assert pit.loc[pd.Timestamp("2023-05-15")] == 1000     # Q1 book equity, once filed
    assert pit.loc[pd.Timestamp("2023-08-15")] == 1100     # steps up after the Q2 filing


def test_ttm_values_and_q4_synthesis():
    ttm = se.trailing_twelve_months(se.concept_frame(_facts(), "NetIncomeLoss"))
    by_end = dict(zip(ttm["end"], ttm["val"]))
    # first full TTM is the 4 quarters of FY2022 -> equals the reported annual (sanity)
    assert by_end[pd.Timestamp("2022-12-31")] == 500
    assert by_end[pd.Timestamp("2023-09-30")] == 170 + 130 + 140 + 150   # 590, spans the Q4 synthesis
    assert by_end[pd.Timestamp("2023-12-31")] == 600


def test_ttm_is_dated_by_latest_component_filing():
    ttm = se.trailing_twelve_months(se.concept_frame(_facts(), "NetIncomeLoss"))
    filed_by_end = dict(zip(ttm["end"], ttm["filed"]))
    # the FY2022 TTM only becomes fully known when the 10-K (with the synthesized Q4) is filed
    assert filed_by_end[pd.Timestamp("2022-12-31")] == pd.Timestamp("2023-02-15")


def _facts_with_comparatives() -> dict:
    """The same facts, but with FY2022 Q1/Q2 ALSO reported as prior-year comparatives inside
    the FY2023 10-Qs (a later `filed` date) — exactly what company facts contains in practice."""
    base = _facts()
    usd = base["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"]
    comparatives = [
        {"start": "2022-01-01", "end": "2022-03-31", "val": 100, "filed": "2023-04-30",
         "fy": 2023, "fp": "Q1", "form": "10-Q", "frame": None},
        {"start": "2022-04-01", "end": "2022-06-30", "val": 110, "filed": "2023-07-31",
         "fy": 2023, "fp": "Q2", "form": "10-Q", "frame": None},
    ]
    base["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"] = usd + comparatives
    return base


def test_prior_year_comparatives_do_not_corrupt_ttm():
    # regression: comparatives must be deduped to the FIRST filing of each period, so they
    # change neither the TTM values nor (critically) their filed dates.
    a = se.trailing_twelve_months(se.concept_frame(_facts(), "NetIncomeLoss")).reset_index(drop=True)
    b = se.trailing_twelve_months(
        se.concept_frame(_facts_with_comparatives(), "NetIncomeLoss")).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_build_fundamental_panel_instant_and_flow():
    facts_by_ticker = {"AAA": _facts()}
    dates = pd.bdate_range("2023-01-02", "2024-03-29")
    book = se.build_fundamental_panel(facts_by_ticker, "StockholdersEquity", dates, kind="instant")
    assert book.loc[pd.Timestamp("2023-08-15"), "AAA"] == 1100
    ni = se.build_fundamental_panel(facts_by_ticker, "NetIncomeLoss", dates, kind="flow_ttm")
    # FY2022 TTM (500) visible from the 10-K filing (2023-02-15), before FY2023 supersedes it
    assert ni.loc[pd.Timestamp("2023-03-01"), "AAA"] == 500
