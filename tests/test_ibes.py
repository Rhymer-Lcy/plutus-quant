"""IBES consensus-surprise logic (network-free): latest-estimate-per-analyst before the
announcement, dispersion, and SUE = (actual - consensus)/dispersion; plus the vectorized
event builder matching the pure reference."""
import numpy as np
import pandas as pd

from plutus.data.sources.ibes_source import build_surprise_events, consensus_surprise


def _est_group():
    # analyst A: 1.0 then revised to 1.2; analyst B: 0.8; (one analyst, one revision)
    return pd.DataFrame({
        "analys": ["A", "A", "B"],
        "value": [1.0, 1.2, 0.8],
        "anndats": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-15"]),
    })


def test_consensus_latest_per_analyst_and_sue():
    mean, std, n, sue = consensus_surprise(_est_group(), actual_value=1.4, cutoff="2020-03-01")
    assert n == 2                                   # A's latest (1.2) + B (0.8)
    assert abs(mean - 1.0) < 1e-9                   # mean of 1.2, 0.8
    assert abs(std - np.std([1.2, 0.8], ddof=1)) < 1e-9
    assert abs(sue - (1.4 - 1.0) / np.std([1.2, 0.8], ddof=1)) < 1e-9   # ~1.41


def test_consensus_cutoff_excludes_later_estimates():
    # cutoff before B's estimate and A's revision -> only A's first (1.0); <2 analysts -> no SUE
    mean, std, n, sue = consensus_surprise(_est_group(), actual_value=1.4, cutoff="2020-01-10")
    assert n == 1 and np.isnan(sue)


def test_build_surprise_events_matches_reference():
    actuals = pd.DataFrame({"ticker": ["T"], "cusip": ["12345678"],
                            "pends": pd.to_datetime(["2020-03-31"]),
                            "anndats": pd.to_datetime(["2020-04-20"]), "actual": [1.4]})
    estimates = pd.DataFrame({
        "ticker": ["T", "T", "T", "T"],
        "cusip": ["12345678"] * 4,
        "fpedats": pd.to_datetime(["2020-03-31"] * 4),
        "analys": ["A", "A", "B", "C"],
        "value": [1.0, 1.2, 0.8, 5.0],
        "anndats": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-15", "2020-04-25"]),
    })  # C's 5.0 is filed AFTER the announcement -> must be excluded
    ev = build_surprise_events(actuals, estimates)
    assert len(ev) == 1
    row = ev.iloc[0]
    assert row["n_est"] == 2                                   # A(1.2) + B(0.8); C excluded
    assert abs(row["consensus"] - 1.0) < 1e-9
    assert abs(row["sue"] - (1.4 - 1.0) / np.std([1.2, 0.8], ddof=1)) < 1e-9
    assert row["anndats"] == pd.Timestamp("2020-04-20")       # event date = announcement


def test_build_surprise_events_empty_inputs():
    empty = pd.DataFrame()
    assert build_surprise_events(empty, empty).empty
