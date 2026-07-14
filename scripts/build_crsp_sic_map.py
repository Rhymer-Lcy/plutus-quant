"""Build the point-in-time SIC map (run once): what industry was a name ON A GIVEN DAY?

Industry has to be point-in-time. CRSP reclassifies companies -- some are coded into pharma only
late in life -- so tagging an event by "was this name EVER a biotech" silently counts gaps that
happened while it was something else. In this window that mistake would add 173 phantom biotech
events to a 1,257-event sample. The spells written here let a study ask what a name WAS on the
day of the event.

Streams the 28 GB daily zip once, keeping common stock on a major exchange (the universe the
price panels cover), and collapses it to one row per (permno, siccd) with the first and last date
CRSP carried that pairing. Small output -- the price panels are reused, not rebuilt.

    conda activate plutus
    python scripts/build_crsp_sic_map.py
"""
from __future__ import annotations

import argparse

from plutus.data.sources import crsp_source as crsp
from plutus.io import atomic_to_parquet
from plutus.paths import PARQUET_DIR, RAW_DIR, ensure_dirs

ZIP = RAW_DIR / "crsp" / "daily_2000_2025.csv.zip"
START, END = "2005-01-01", "2024-12-31"          # match the price lakes
OUT = PARQUET_DIR / "crsp_sic_spells.parquet"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    args = ap.parse_args()

    ensure_dirs()
    print(f"streaming SIC spells {args.start}..{args.end} from the 28GB zip...", flush=True)
    spells = crsp.stream_sic_spells(ZIP, args.start, args.end)
    if spells.empty:
        raise SystemExit("no rows matched -- check the zip path")
    atomic_to_parquet(spells, OUT)

    per_name = spells.groupby("permno")["siccd"].nunique()
    print(f"\nwrote {len(spells):,} (permno, siccd) spells for {spells['permno'].nunique():,} names")
    print(f"  names RECLASSIFIED at least once (why this map exists): "
          f"{int((per_name > 1).sum()):,}")
    print(f"  distinct SIC codes: {spells['siccd'].nunique():,}")
    print(f"  saved to {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
