"""Build the CRSP-based, SURVIVORSHIP-FREE price lake (run once; studies read the parquet).

Streams the ~28GB daily CSV out of the zip (never extracts it), keeping only S&P 500 PERMNOs
over the window, and writes compact parquet panels:
  - crsp_adj_close.parquet : total-return-adjusted price (date x PERMNO)  [for returns/backtest]
  - crsp_mktcap.parquet    : market cap in $ (date x PERMNO)              [for value factors]
  - crsp_ticker_map.parquet: PERMNO -> latest ticker                      [join to SEC EDGAR]
  - crsp_members.parquet   : S&P 500 membership spells (permno,start,end) [PIT universe]

PERMNO columns are stored as strings (parquet needs string column names); the study casts
membership PERMNOs to str to match.

    conda activate plutus
    python scripts/build_crsp_lake.py --start 2005-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse

from plutus.data.sources import crsp_source as crsp
from plutus.io import atomic_to_parquet
from plutus.paths import PARQUET_DIR, RAW_DIR, ensure_dirs

CRSP_DIR = RAW_DIR / "crsp"


def build(start: str, end: str) -> None:
    ensure_dirs()
    zip_path = CRSP_DIR / "daily_2000_2025.csv.zip"
    xlsx_path = CRSP_DIR / "sp500_constituents.xlsx"

    spells = crsp.load_membership(xlsx_path)
    permnos = crsp.union_permnos(spells, start, end)
    print(f"S&P 500 PIT union over {start}..{end}: {len(permnos)} PERMNOs")
    print("streaming the 28GB daily CSV out of the zip (pyarrow, column-projected)…")
    long = crsp.stream_filtered(zip_path, permnos, start, end)
    print(f"kept {len(long):,} daily rows for {long['PERMNO'].nunique()} PERMNOs "
          f"({long['date'].min().date()} -> {long['date'].max().date()})")

    adj = crsp.build_tr_adjusted_close(long)
    cap = crsp.build_mktcap(long)
    tmap = crsp.latest_ticker_map(long)

    adj.columns = adj.columns.astype(str)
    cap.columns = cap.columns.astype(str)
    atomic_to_parquet(adj, PARQUET_DIR / "crsp_adj_close.parquet")
    atomic_to_parquet(cap, PARQUET_DIR / "crsp_mktcap.parquet")
    import pandas as pd
    atomic_to_parquet(pd.DataFrame({"permno": list(tmap), "ticker": list(tmap.values())}),
                      PARQUET_DIR / "crsp_ticker_map.parquet")
    atomic_to_parquet(spells, PARQUET_DIR / "crsp_members.parquet")

    print(f"\nlake written to {PARQUET_DIR}:")
    print(f"  crsp_adj_close : {adj.shape[0]} dates x {adj.shape[1]} PERMNOs")
    print(f"  crsp_mktcap    : {cap.shape[0]} dates x {cap.shape[1]} PERMNOs")
    print(f"  crsp_ticker_map: {len(tmap)} PERMNO->ticker")
    print(f"  crsp_members   : {len(spells)} membership spells")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default="2024-12-31")
    args = ap.parse_args()
    build(args.start, args.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
