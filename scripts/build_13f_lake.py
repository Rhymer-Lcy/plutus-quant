"""Build the 13F lake for the copycat study (issue #4): filings, holdings, PERMNO join.

Reads the quarterly SEC Form 13F ZIPs in data/raw/form13f/ and writes:

  form13f_filings.parquet    one row per original 13F-HR (cik, manager, FILING DATE, period,
                             reported portfolio value, position count)
  form13f_holdings.parquet   share positions of the IN-SCOPE managers, mapped to CRSP PERMNOs
                             point-in-time via the CUSIP spell map

In-scope managers = the frozen ARM A legends (by CIK -- names collide, see form13f docstring)
UNION the ARM B control (the N largest filers by reported portfolio value each quarter, which is
observable at the time). Everything else is skipped, so the 130M-row holdings table never has to
be parsed in full.

Prereq: python scripts/build_crsp_cusip_map.py

    conda activate plutus
    python scripts/build_13f_lake.py
"""
from __future__ import annotations

import argparse

import pandas as pd

from plutus.data.sources import form13f
from plutus.io import atomic_to_parquet
from plutus.paths import PARQUET_DIR, RAW_DIR, ensure_dirs

ZIP_DIR = RAW_DIR / "form13f"
SPELLS = PARQUET_DIR / "crsp_cusip_spells.parquet"

# ARM A -- the hindsight-selected legends, pinned to CIK (issue #4). Appaloosa filed under two
# CIKs after a mid-window reorganisation; both are the same manager.
LEGEND_CIKS: dict[str, str] = {
    "1067983": "Berkshire Hathaway",
    "1061768": "Baupost Group",
    "1336528": "Pershing Square",
    "1079114": "Greenlight Capital",
    "1040273": "Third Point",
    "1006438": "Appaloosa",
    "1656456": "Appaloosa",
    "921669": "Icahn",
    "1350694": "Bridgewater Associates",
    "1037389": "Renaissance Technologies",
    "1536411": "Duquesne Family Office",
    "1029160": "Soros Fund Management",
    "1167483": "Tiger Global Management",
    "1061165": "Lone Pine Capital",
    "1103804": "Viking Global Investors",
    "1135730": "Coatue Management",
    "1697748": "ARK Investment Management",
    "1649339": "Scion Asset Management",
}
ARM_B_TOP_N = 20          # the no-look-ahead control: the largest filers each quarter


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-n", type=int, default=ARM_B_TOP_N)
    args = ap.parse_args()
    ensure_dirs()

    zips = sorted(ZIP_DIR.glob("*.zip"))
    if not zips:
        raise SystemExit(f"no ZIPs under {ZIP_DIR} -- download the SEC Form 13F data sets first")
    print(f"reading {len(zips)} quarterly ZIPs from {ZIP_DIR.name}/ ...", flush=True)

    filings = form13f.filing_index(zips)
    atomic_to_parquet(filings, PARQUET_DIR / "form13f_filings.parquet")
    print(f"filings: {len(filings):,} original 13F-HR from {filings['cik'].nunique():,} managers, "
          f"periods {filings['period'].min().date()} -> {filings['period'].max().date()}")
    lag = (filings["filing_date"] - filings["period"]).dt.days
    print(f"  filing lag (the study's whole point): median {lag.median():.0f}d, "
          f"p95 {lag.quantile(0.95):.0f}d")

    legends = filings[filings["cik"].isin(LEGEND_CIKS)].copy()
    legends["legend"] = legends["cik"].map(LEGEND_CIKS)
    print(f"\nARM A (frozen legends): {legends['legend'].nunique()} managers, "
          f"{len(legends)} filings")
    for name, g in legends.groupby("legend"):
        print(f"   {name:<26} {len(g):>3} filings  "
              f"{g['period'].min().date()} -> {g['period'].max().date()}")

    # ARM B: the largest filers by REPORTED value within each period. Ranking inside a period is
    # unit-safe (every filing in a period reports on the same basis).
    big = (filings.sort_values(["period", "table_value"], ascending=[True, False])
           .groupby("period").head(args.top_n))
    print(f"\nARM B (top {args.top_n} by reported value each quarter): "
          f"{big['cik'].nunique()} distinct CIKs over {big['period'].nunique()} quarters")

    scope = filings[filings["cik"].isin(set(LEGEND_CIKS) | set(big["cik"]))]
    print(f"\nin scope: {len(scope):,} filings / {scope['cik'].nunique()} managers "
          f"-- parsing their holdings...", flush=True)

    holdings = form13f.read_holdings(zips, set(scope["accession"]))
    print(f"  {len(holdings):,} share positions parsed")

    spells = pd.read_parquet(SPELLS)
    mapped = form13f.map_to_permno(holdings, spells, scope)
    hit_pos = len(mapped) / len(holdings)
    hit_val = mapped["value_usd"].sum() / holdings["value_usd"].sum()
    print(f"  mapped to a CRSP permno: {hit_pos:.1%} of positions, {hit_val:.1%} of dollars "
          f"(the rest are ETFs, foreign issues, junk CUSIPs -- this repo does not price them)")

    atomic_to_parquet(mapped, PARQUET_DIR / "form13f_holdings.parquet")
    print(f"\nwrote form13f_filings.parquet ({len(filings):,} rows) and "
          f"form13f_holdings.parquet ({len(mapped):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
