"""SEC Form 13F structured data sets — free, official institutional holdings.

Quarterly ZIPs from https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets
(2013Q2 onward), unpacked into data/raw/form13f/. Each ZIP carries SUBMISSION.tsv (accession,
FILING_DATE, type, CIK, period), COVERPAGE.tsv (manager name), SUMMARYPAGE.tsv (reported
portfolio total), and INFOTABLE.tsv (one row per position: CUSIP, VALUE, shares).

Three things bite a study built on this file, all handled here:

  - THE FILING DATE IS THE ONLY LEGAL ENTRY. A 13F is public ~45 days after the period it
    describes (measured: median 40-44 days). A backtest that enters on the period-end date is
    trading on information nobody had. `filing_index` keeps FILING_DATE for exactly this.
  - VALUE CHANGES UNITS MID-SAMPLE. The SEC moved holdings value from THOUSANDS of dollars to
    whole dollars in 2023Q1. Detected per filing from the implied price per share
    (VALUE / SSHPRNAMT; a median below $1 means thousands), never from a hardcoded date. The
    quantities a study needs -- portfolio weights, concentration ratios, within-quarter size
    rank -- are ratios and so unit-invariant anyway; the normalisation is for correctness.
  - MANAGERS MUST BE MATCHED BY CIK. Name substrings collide badly: "BERKSHIRE" also hits
    Berkshire Bank and Berkshire Asset Management; "ARK INVESTMENT" hits Benchmark Investment
    Advisors. Never resolve a manager by name.

Only original 13F-HR reports are read: 13F-NT notices carry no holdings, and 13F-HR/A
amendments are later restatements of a filing the copycat already acted on.
"""
from __future__ import annotations

import zipfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

SUBMISSION_COLS = ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"]
COVER_COLS = ["ACCESSION_NUMBER", "FILINGMANAGER_NAME"]
SUMMARY_COLS = ["ACCESSION_NUMBER", "TABLEVALUETOTAL", "TABLEENTRYTOTAL"]
INFO_COLS = ["ACCESSION_NUMBER", "CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"]
_DATE_FMT = "%d-%b-%Y"


def _read(zip_path: Path, member: str, usecols: list[str]) -> pd.DataFrame:
    """Read one TSV out of a quarterly ZIP. Most archives store the tables at the root, but at
    least one nests them under a folder, so the member is located by file NAME, not by path."""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.rsplit("/", 1)[-1] == member]
        if not names:
            raise KeyError(f"{member} not found in {zip_path.name}")
        with z.open(names[0]) as f:
            return pd.read_csv(f, sep="\t", dtype=str, usecols=usecols)


def filing_index(zips: Iterable[Path]) -> pd.DataFrame:
    """One row per original 13F-HR filing across every ZIP.

    Returns [cik, manager, accession, filing_date, period, table_value, n_positions, source].
    `table_value` is AS REPORTED (units differ before/after 2023Q1); use it only for
    within-quarter comparisons, or normalise with `value_scale`."""
    out = []
    for zf in zips:
        sub = _read(zf, "SUBMISSION.tsv", SUBMISSION_COLS)
        sub = sub[sub["SUBMISSIONTYPE"] == "13F-HR"]
        if sub.empty:
            continue
        cov = _read(zf, "COVERPAGE.tsv", COVER_COLS)
        summ = _read(zf, "SUMMARYPAGE.tsv", SUMMARY_COLS)
        m = sub.merge(cov, on="ACCESSION_NUMBER", how="left") \
               .merge(summ, on="ACCESSION_NUMBER", how="left")
        m["source"] = zf.name
        out.append(m)
    df = pd.concat(out, ignore_index=True)
    df = df.rename(columns={"ACCESSION_NUMBER": "accession", "CIK": "cik",
                            "FILINGMANAGER_NAME": "manager"})
    df["filing_date"] = pd.to_datetime(df["FILING_DATE"], format=_DATE_FMT, errors="coerce")
    df["period"] = pd.to_datetime(df["PERIODOFREPORT"], format=_DATE_FMT, errors="coerce")
    df["table_value"] = pd.to_numeric(df["TABLEVALUETOTAL"], errors="coerce")
    df["n_positions"] = pd.to_numeric(df["TABLEENTRYTOTAL"], errors="coerce")
    df["cik"] = df["cik"].str.lstrip("0")
    df = df.dropna(subset=["filing_date", "period"])
    cols = ["cik", "manager", "accession", "filing_date", "period", "table_value",
            "n_positions", "source"]
    # A manager occasionally files twice for one period; the copycat acts on the FIRST one.
    return (df[cols].sort_values(["cik", "period", "filing_date"])
            .drop_duplicates(["cik", "period"], keep="first").reset_index(drop=True))


def value_scale(holdings: pd.DataFrame) -> pd.Series:
    """Per-accession multiplier turning VALUE into dollars: 1000 where the filing reports in
    thousands, else 1. Detected from the implied price per share (median VALUE/SSHPRNAMT): a
    median below $1 cannot be a real share price, so the filing must be in thousands."""
    px = holdings["value"] / holdings["shares"].where(holdings["shares"] > 0)
    med = px.groupby(holdings["accession"]).median()
    return med.lt(1.0).map({True: 1000.0, False: 1.0})


def read_holdings(zips: Iterable[Path], accessions: set[str]) -> pd.DataFrame:
    """Share positions for the given accessions: [accession, cusip9, shares, value_usd].

    Drops principal-amount lines (SSHPRNAMTTYPE != 'SH') and option lines (PUTCALL set), which
    are not the common-stock positions this study prices, and normalises VALUE to dollars."""
    out = []
    for zf in zips:
        info = _read(zf, "INFOTABLE.tsv", INFO_COLS)
        info = info[info["ACCESSION_NUMBER"].isin(accessions)]
        if info.empty:
            continue
        info = info[(info["SSHPRNAMTTYPE"] == "SH") & info["PUTCALL"].isna()]
        out.append(info)
    if not out:
        return pd.DataFrame(columns=["accession", "cusip9", "shares", "value_usd"])
    df = pd.concat(out, ignore_index=True).rename(
        columns={"ACCESSION_NUMBER": "accession", "CUSIP": "cusip9"})
    df["shares"] = pd.to_numeric(df["SSHPRNAMT"], errors="coerce")
    df["value"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df = df.dropna(subset=["cusip9", "shares", "value"])
    df["value_usd"] = df["value"] * df["accession"].map(value_scale(df)).astype(float)
    # One issuer can appear on several lines (different managers/discretion); sum them.
    return (df.groupby(["accession", "cusip9"], as_index=False)
            .agg(shares=("shares", "sum"), value_usd=("value_usd", "sum")))


def map_to_permno(holdings: pd.DataFrame, spells: pd.DataFrame,
                  filings: pd.DataFrame) -> pd.DataFrame:
    """Attach a PERMNO to each holding, point-in-time: the permno whose CUSIP spell contains
    the FILING date. Unmatched rows (ETFs, foreign issues, junk CUSIPs) are dropped; the caller
    reports the drop rate."""
    h = holdings.merge(filings[["accession", "filing_date"]], on="accession", how="left")
    m = h.merge(spells[["cusip9", "permno", "start", "end"]], on="cusip9", how="inner")
    ok = (m["filing_date"] >= m["start"]) & (m["filing_date"] <= m["end"])
    # filing_date is dropped again: it belongs to the filing index, and leaving a copy on the
    # holdings table collides on every later join back to it.
    return m[ok].drop(columns=["start", "end", "filing_date"]).reset_index(drop=True)
