"""Build the point-in-time CUSIP -> PERMNO map (run once) that joins 13F holdings to prices.

13F information tables identify a security ONLY by its 9-digit CUSIP, while every price panel in
this repo is keyed by PERMNO. A CUSIP can be REUSED by a different issuer years after a
delisting, so the join must be point-in-time: a holding matches the PERMNO whose CUSIP SPELL
contains the FILING date, never one that carried the same CUSIP in another decade.

Streams the 28 GB daily zip once, keeping common stock on a major exchange (the part of the 13F
universe this repo can price), and collapses it to one row per (permno, cusip9) with the first
and last date CRSP carried that pairing.

    conda activate plutus
    python scripts/build_crsp_cusip_map.py
"""
from __future__ import annotations

import argparse

from plutus.data.sources import crsp_source as crsp
from plutus.io import atomic_to_parquet
from plutus.paths import PARQUET_DIR, RAW_DIR, ensure_dirs

ZIP = RAW_DIR / "crsp" / "daily_2000_2025.csv.zip"
START, END = "2005-01-01", "2024-12-31"      # match the price lakes
OUT = PARQUET_DIR / "crsp_cusip_spells.parquet"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    args = ap.parse_args()

    ensure_dirs()
    print(f"streaming CUSIP spells {args.start}..{args.end} from the 28GB zip...", flush=True)
    spells = crsp.stream_cusip_spells(ZIP, args.start, args.end)
    if spells.empty:
        raise SystemExit("no rows matched -- check the zip path")
    atomic_to_parquet(spells, OUT)

    reused = spells.groupby("cusip9")["permno"].nunique()
    print(f"\nwrote {len(spells):,} (permno, cusip9) spells for "
          f"{spells['permno'].nunique():,} permnos / {spells['cusip9'].nunique():,} CUSIPs")
    print(f"  CUSIPs carried by MORE THAN ONE permno (why the join must be point-in-time): "
          f"{int((reused > 1).sum())}")
    print(f"  spell span: {spells['start'].min().date()} -> {spells['end'].max().date()}")
    print(f"  saved to {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
