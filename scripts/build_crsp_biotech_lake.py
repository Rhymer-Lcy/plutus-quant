"""Build the pharma/biotech CRSP lake (run once) for the catalyst study, docs/biotech_catalyst_study.md.

Streams the 28 GB daily zip keeping common stocks on a major exchange whose SIC code is
pharma/biotech. Survivorship-free by construction: a name that delists simply ends, with its
delisting return inside DlyRet.

SIC SET -- and the bug it fixes. CRSP carries the drug industry BOTH as specific 4-digit codes
AND as the GROUP-level code 2830 ("Drugs"), and a name can move between them: Amgen is coded
2830 for 2000-2021 and 2836 only from 2021-12-06. A first version of this lake enumerated only
{2833, 2834, 2835, 2836, 8731} and therefore silently dropped 249 names -- Amgen among them, for
17 of 20 years. The set below adds the group codes 2830 and 2831, which name the same industry.
Codes NOT included, on classification grounds: 8730 (group-level research services, mixed -- it
contains e.g. Energy Conversion Devices, a solar company), 8732 (market research: IQVIA), 8733,
8734 (testing labs). 8731 is kept because the study's frozen design named it.

NO price/cap filter is applied here, deliberately -- see crsp_source.stream_industry. The
tradability gate (price >= $5, cap >= $100M) belongs at STUDY time, applied to EVENT ELIGIBILITY.
Filtering the lake instead would delete a name's rows the moment it fell below the threshold, so
a biotech that gaps up and then craters would have its post-event LOSSES truncated -- an upward
bias in exactly the quantity the study measures. The lake keeps close and cap so the study can gate.

Panels written (the study derives the overnight gap from them, split- and dividend-immune):

  crsp_biotech_adj_close.parquet   TR-adjusted close   (date x PERMNO)
  crsp_biotech_dlyret.parquet      DlyRet              close-to-close TOTAL return
  crsp_biotech_open_raw.parquet    raw DlyOpen         -> intraday[t] = close[t]/open[t]-1
  crsp_biotech_close_raw.parquet   raw DlyClose        -> overnight[t] = (1+ret)/(1+intraday)-1
  crsp_biotech_halfspread.parquet  (ask-bid)/mid/2     the real cost of a catalyst-day fill
  crsp_biotech_mktcap.parquet      market cap ($)      the study's cap gate + cap terciles
  crsp_biotech_dollarvol.parquet   dollar volume
  crsp_biotech_meta.parquet        PERMNO -> latest ticker and SIC (the 283x robustness split)

    conda activate plutus
    python scripts/build_crsp_biotech_lake.py
"""
from __future__ import annotations

import argparse

import pandas as pd

from plutus.data.sources import crsp_source as crsp
from plutus.io import atomic_to_parquet
from plutus.paths import PARQUET_DIR, RAW_DIR, ensure_dirs

ZIP = RAW_DIR / "crsp" / "daily_2000_2025.csv.zip"
# Drug manufacturers (283x, incl. the group codes) + the commercial-research code the study froze.
BIOTECH_SIC = {"2830", "2831", "2833", "2834", "2835", "2836", "8731"}
DRUG_SIC = {"2830", "2831", "2833", "2834", "2835", "2836"}   # the 283x-only robustness split
START, END = "2005-01-01", "2024-12-31"       # match the other lakes / membership spells window
PREFIX = "crsp_biotech"


def _pivot(long: pd.DataFrame, col: str, take_abs: bool = False) -> pd.DataFrame:
    d = long.dropna(subset=[col])[["PERMNO", "date", col]].drop_duplicates(["PERMNO", "date"])
    p = d.pivot(index="date", columns="PERMNO", values=col).sort_index()
    p.columns = p.columns.astype(str)
    return p.abs() if take_abs else p           # legacy CRSP used the sign as a no-trade flag


def build(start: str, end: str) -> None:
    ensure_dirs()
    print(f"streaming pharma/biotech (SIC {sorted(BIOTECH_SIC)}) {start}..{end} "
          f"from the 28GB zip...", flush=True)
    long = crsp.stream_industry(ZIP, BIOTECH_SIC, start, end)
    if long.empty:
        raise SystemExit("no rows matched -- check the zip path and the SIC set")
    print(f"kept {len(long):,} daily rows for {long['PERMNO'].nunique()} names "
          f"({long['date'].min().date()} -> {long['date'].max().date()})")

    adj = crsp.build_tr_adjusted_close(long)
    adj.columns = adj.columns.astype(str)
    cap = crsp.build_mktcap(long)
    cap.columns = cap.columns.astype(str)
    bid, ask = _pivot(long, "DlyBid", take_abs=True), _pivot(long, "DlyAsk", take_abs=True)
    mid = (bid + ask) / 2.0

    panels = {
        "adj_close": adj,
        "dlyret": _pivot(long, "DlyRet"),
        "open_raw": _pivot(long, "DlyOpen", take_abs=True),
        "close_raw": _pivot(long, "DlyClose", take_abs=True),
        "halfspread": ((ask - bid) / mid / 2.0).where(mid > 0),
        "mktcap": cap,
        "dollarvol": _pivot(long, "DlyPrcVol"),
    }
    for name, panel in panels.items():
        atomic_to_parquet(panel, PARQUET_DIR / f"{PREFIX}_{name}.parquet")

    meta = (long.sort_values("date").groupby("PERMNO")
            .agg(ticker=("Ticker", "last"), sic=("SICCD", "last")).reset_index()
            .rename(columns={"PERMNO": "permno"}))
    meta["permno"] = meta["permno"].astype(str)
    atomic_to_parquet(meta, PARQUET_DIR / f"{PREFIX}_meta.parquet")

    print(f"\nlake written: {adj.shape[0]} dates x {adj.shape[1]} pharma/biotech names")
    print(f"  drug manufacturers (283x): {(meta['sic'].isin(DRUG_SIC)).sum()} names; "
          f"commercial research (8731): {(meta['sic'] == '8731').sum()}")
    tradable = (panels["close_raw"] >= 5.0) & (cap >= 100e6)
    print(f"  names alive: {adj.notna().sum(axis=1).max()} peak; "
          f"tradable (>= $5, >= $100M) on the last day: {int(tradable.iloc[-1].sum())}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the pharma/biotech CRSP lake.")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    args = ap.parse_args()
    build(args.start, args.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
