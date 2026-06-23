"""IBES (LSEG) adapter — analyst EPS estimates + actuals -> analyst-consensus earnings surprise.

>>> LICENSING: a personal/academic WRDS IBES extract. Reads data the USER supplies under their
>>> own license, kept locally under data/raw/ibes/ (gitignored). Personal research only.

The seasonal-random-walk SUE (research.factors.events) is crude. IBES gives the real thing:
the analyst CONSENSUS just before the announcement, so the surprise = (actual − consensus) /
dispersion separates the quintiles far more sharply — the test of whether PEAD clears the cost
line (docs/smallcap_pead_study.md).

Files (US, EPS, UNADJUSTED): actuals (small) + detail history estimates (4.65GB, streamed with
pyarrow, never extracted). Pure consensus/surprise functions are unit-tested; only the loaders
touch the big files.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

EST_COLS = ["TICKER", "CUSIP", "FPEDATS", "ANNDATS", "REVDATS", "ANALYS", "ESTIMATOR",
            "VALUE", "FPI", "PDF"]
_QUARTERLY_FPI = ["6", "7", "8", "9"]   # IBES FPI: 6-9 = quarterly (Q1-Q4 ahead)


def _inner_csv(z: zipfile.ZipFile) -> str:
    names = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]
    if not names:
        raise ValueError("no CSV member in zip")
    return names[0]


def load_actuals(zip_path: str | Path, periodicity: str = "QTR") -> pd.DataFrame:
    """Reported EPS + announcement date per company-period. Returns (ticker, cusip, pends,
    anndats, actual). `periodicity`: 'QTR' (quarterly, for PEAD) or 'ANN'."""
    with zipfile.ZipFile(zip_path) as z:
        with z.open(_inner_csv(z)) as f:
            df = pd.read_csv(f, dtype=str)
    df = df[(df["MEASURE"] == "EPS") & (df["PDICITY"] == periodicity)].copy()
    df["actual"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df["pends"] = pd.to_datetime(df["PENDS"], errors="coerce")
    df["anndats"] = pd.to_datetime(df["ANNDATS"], errors="coerce")
    df["cusip"] = df["CUSIP"].fillna("").str.strip()
    df = df.dropna(subset=["actual", "pends", "anndats"])
    df = df[df["cusip"].str.len() > 0]
    return (df[["TICKER", "cusip", "pends", "anndats", "actual"]]
            .rename(columns={"TICKER": "ticker"}).reset_index(drop=True))


def stream_estimates(zip_path: str | Path, tickers: set | None = None, start=None,
                     block_size: int = 64 << 20) -> pd.DataFrame:
    """Stream the 4.65GB estimates CSV, keeping only QUARTERLY EPS estimates (and, if given, only
    `tickers`), projected to the needed columns. Returns (ticker, cusip, fpedats, anndats,
    analys, value). Never extracts the full file."""
    conv = pacsv.ConvertOptions(include_columns=EST_COLS,
                                column_types={c: pa.string() for c in EST_COLS})
    fpi_set = pa.array(_QUARTERLY_FPI, type=pa.string())
    want = pa.array(sorted(tickers), type=pa.string()) if tickers else None
    parts: list[pa.Table] = []
    with zipfile.ZipFile(zip_path) as z:
        with z.open(_inner_csv(z)) as f:
            reader = pacsv.open_csv(f, read_options=pacsv.ReadOptions(block_size=block_size),
                                    convert_options=conv)
            for batch in reader:
                t = pa.Table.from_batches([batch])
                mask = pc.is_in(t["FPI"], value_set=fpi_set)
                if want is not None:
                    mask = pc.and_kleene(mask, pc.is_in(t["TICKER"], value_set=want))
                t = t.filter(mask)
                if t.num_rows:
                    parts.append(t.select(["TICKER", "CUSIP", "FPEDATS", "ANNDATS", "ANALYS", "VALUE"]))
    if not parts:
        return pd.DataFrame(columns=["ticker", "cusip", "fpedats", "anndats", "analys", "value"])
    df = pa.concat_tables(parts).to_pandas()
    df["value"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df["fpedats"] = pd.to_datetime(df["FPEDATS"], errors="coerce")
    df["anndats"] = pd.to_datetime(df["ANNDATS"], errors="coerce")
    df = df.dropna(subset=["value", "fpedats", "anndats"])
    if start is not None:
        df = df[df["fpedats"] >= pd.Timestamp(start)]
    return (df.rename(columns={"TICKER": "ticker", "CUSIP": "cusip", "ANALYS": "analys"})
            [["ticker", "cusip", "fpedats", "anndats", "analys", "value"]].reset_index(drop=True))


def consensus_surprise(est_group: pd.DataFrame, actual_value: float, cutoff) -> tuple:
    """For ONE company-quarter (est_group cols: analys, value, anndats), the consensus just
    before `cutoff`: keep estimates announced before cutoff, the LATEST per analyst, then mean.
    Returns (consensus_mean, dispersion_std, n_analysts, sue) where sue = (actual − mean)/std."""
    e = est_group[est_group["anndats"] < pd.Timestamp(cutoff)].sort_values("anndats")
    e = e.drop_duplicates("analys", keep="last")
    if e.empty:
        return (np.nan, np.nan, 0, np.nan)
    vals = e["value"].to_numpy()
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) >= 2 else np.nan
    sue = (actual_value - mean) / std if (std and std > 0) else np.nan
    return mean, std, len(vals), sue


def build_surprise_events(actuals: pd.DataFrame, estimates: pd.DataFrame) -> pd.DataFrame:
    """Join actuals to their pre-announcement consensus and compute the analyst SUE per
    company-quarter (vectorized equivalent of `consensus_surprise`). Returns (ticker, cusip,
    pends, anndats=announcement, actual, consensus, dispersion, n_est, sue). Requires ≥2
    analysts (dispersion defined)."""
    if actuals.empty or estimates.empty:
        return pd.DataFrame(columns=["ticker", "cusip", "pends", "anndats", "actual",
                                     "consensus", "dispersion", "n_est", "sue"])
    a = actuals.rename(columns={"anndats": "ann_act", "cusip": "cusip_act"})
    m = estimates.merge(a[["ticker", "pends", "ann_act", "actual", "cusip_act"]],
                        left_on=["ticker", "fpedats"], right_on=["ticker", "pends"], how="inner")
    m = m[m["anndats"] < m["ann_act"]]                      # estimate made before the announcement
    m = m.sort_values("anndats").drop_duplicates(["ticker", "pends", "analys"], keep="last")
    g = m.groupby(["ticker", "pends"])
    agg = g.agg(consensus=("value", "mean"), dispersion=("value", "std"), n_est=("value", "size"),
                actual=("actual", "first"), anndats=("ann_act", "first"),
                cusip=("cusip_act", "first")).reset_index()
    agg["sue"] = (agg["actual"] - agg["consensus"]) / agg["dispersion"].where(agg["dispersion"] > 0)
    return (agg.dropna(subset=["sue"])
            [["ticker", "cusip", "pends", "anndats", "actual", "consensus", "dispersion", "n_est", "sue"]]
            .reset_index(drop=True))
