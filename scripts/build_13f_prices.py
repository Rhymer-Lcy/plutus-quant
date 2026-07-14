"""Price panel for every stock the 13F managers hold — deliberately UNFILTERED.

The existing CRSP lakes gate rows on price >= $5 and cap >= $100M. Reusing them here would
delete a holding's rows the moment it fell below those floors, truncating the LOSSES of any
name that a manager bought and that then collapsed -- an upward bias in exactly the quantity the
copycat study measures, and the same bug that had to be corrected in the biotech catalyst study
(docs/biotech_catalyst_study.md). So this streams the CRSP zip for the 13F permnos with NO price
or cap filter: a position is followed for as long as CRSP has it, and the delisting return in
DlyRet ends it honestly.

Prereq: python scripts/build_13f_lake.py

    conda activate plutus
    python scripts/build_13f_prices.py
"""
from __future__ import annotations

import pandas as pd

from plutus.data.sources import crsp_source as crsp
from plutus.io import atomic_to_parquet
from plutus.paths import PARQUET_DIR, RAW_DIR, ensure_dirs

ZIP = RAW_DIR / "crsp" / "daily_2000_2025.csv.zip"
START, END = "2005-01-01", "2024-12-31"       # the CRSP lake's span


def main() -> int:
    ensure_dirs()
    holdings = pd.read_parquet(PARQUET_DIR / "form13f_holdings.parquet")
    permnos = {int(p) for p in holdings["permno"].unique()}
    print(f"{len(permnos):,} distinct permnos held by the in-scope 13F managers")
    print(f"streaming their daily returns {START}..{END} from the 28GB zip (no filters)...",
          flush=True)

    long = crsp.stream_filtered(ZIP, permnos, START, END)
    if long.empty:
        raise SystemExit("no rows matched -- check the zip path")

    ret = (long.dropna(subset=["DlyRet"])[["PERMNO", "date", "DlyRet"]]
           .drop_duplicates(["PERMNO", "date"])
           .pivot(index="date", columns="PERMNO", values="DlyRet").sort_index())
    ret.columns = ret.columns.astype(str)
    atomic_to_parquet(ret, PARQUET_DIR / "form13f_dlyret.parquet")

    print(f"\nwrote form13f_dlyret.parquet: {ret.shape[0]:,} dates x {ret.shape[1]:,} permnos")
    print(f"  priceable: {ret.shape[1] / len(permnos):.1%} of the held permnos")
    print(f"  span {ret.index.min().date()} -> {ret.index.max().date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
