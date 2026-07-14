"""Build the pharma/biotech CRSP lake (run once) for the catalyst-drift study.

Streams the 28 GB daily zip keeping tradable common stocks (major exchange, price >= $5, cap >=
$100M -- the same gates as the other lakes, so the sample is retail-tradeable) whose SIC code is
pharma/biotech. Survivorship-free by construction: a name that delists simply ends, with its
delisting return inside DlyRet.

SIC set (the standard pharma+biotech classification): 2833 medicinal chemicals, 2834
pharmaceutical preparations, 2835 in-vitro/in-vivo diagnostics, 2836 biological products, and
8731 commercial physical/biological research -- where most clinical-stage biotechs file. SIC is
per-row and point-in-time, so a reclassified name enters/leaves on the CRSP date.

Panels written (same conventions as the other lakes; the study derives the overnight gap from
them, split- and dividend-immune):

  crsp_biotech_adj_close.parquet   TR-adjusted close   (date x PERMNO)
  crsp_biotech_dlyret.parquet      DlyRet              close-to-close TOTAL return
  crsp_biotech_open_raw.parquet    raw DlyOpen         -> intraday[t] = close[t]/open[t]-1
  crsp_biotech_close_raw.parquet   raw DlyClose        -> overnight[t] = (1+ret)/(1+intraday)-1
  crsp_biotech_halfspread.parquet  (ask-bid)/mid/2     the real cost of a catalyst-day fill
  crsp_biotech_mktcap.parquet      market cap ($)
  crsp_biotech_dollarvol.parquet   dollar volume
  crsp_biotech_ticker_map.parquet  PERMNO -> latest ticker

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
BIOTECH_SIC = {"2833", "2834", "2835", "2836", "8731"}
START, END = "2005-01-01", "2024-12-31"       # match the other lakes / membership spells window
PREFIX = "crsp_biotech"


def _pivot(long: pd.DataFrame, col: str, take_abs: bool = False) -> pd.DataFrame:
    d = long.dropna(subset=[col])[["PERMNO", "date", col]].drop_duplicates(["PERMNO", "date"])
    p = d.pivot(index="date", columns="PERMNO", values=col).sort_index()
    p.columns = p.columns.astype(str)
    return p.abs() if take_abs else p           # legacy CRSP used the sign as a no-trade flag


def build(start: str, end: str, price_min: float, cap_min_000: float) -> None:
    ensure_dirs()
    print(f"streaming pharma/biotech (SIC {sorted(BIOTECH_SIC)}, price>=${price_min:.0f}, "
          f"cap>=${cap_min_000 / 1000:.0f}M) {start}..{end} from the 28GB zip...", flush=True)
    long = crsp.stream_industry(ZIP, BIOTECH_SIC, start, end,
                                price_min=price_min, cap_min_000=cap_min_000)
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
    half_spread = ((ask - bid) / mid / 2.0).where(mid > 0)
    tmap = crsp.latest_ticker_map(long)

    panels = {
        "adj_close": adj,
        "dlyret": _pivot(long, "DlyRet"),
        "open_raw": _pivot(long, "DlyOpen", take_abs=True),
        "close_raw": _pivot(long, "DlyClose", take_abs=True),
        "halfspread": half_spread,
        "mktcap": cap,
        "dollarvol": _pivot(long, "DlyPrcVol"),
    }
    for name, panel in panels.items():
        atomic_to_parquet(panel, PARQUET_DIR / f"{PREFIX}_{name}.parquet")
    atomic_to_parquet(pd.DataFrame({"permno": list(tmap), "ticker": list(tmap.values())}),
                      PARQUET_DIR / f"{PREFIX}_ticker_map.parquet")

    print(f"\nlake written: {adj.shape[0]} dates x {adj.shape[1]} pharma/biotech names")
    print(f"  median half-spread: {half_spread.stack().median():.4%} "
          f"(the cost of transacting one side)")
    alive = adj.notna().sum(axis=1)
    print(f"  names alive: {alive.iloc[0]:.0f} on {adj.index[0].date()}, "
          f"{alive.iloc[-1]:.0f} on {adj.index[-1].date()}, {alive.max():.0f} peak")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the pharma/biotech CRSP lake.")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--price-min", type=float, default=5.0)
    ap.add_argument("--cap-min-000", type=float, default=100_000.0)
    args = ap.parse_args()
    build(args.start, args.end, args.price_min, args.cap_min_000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
