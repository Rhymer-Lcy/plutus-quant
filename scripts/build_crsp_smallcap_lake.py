"""Build a broad small/mid-cap CRSP lake (run once) to test whether factor premia survive
where arbitrage capital can't fish.

Streams the daily zip keeping ALL tradable common stocks (major exchange, price >= $5, cap >=
$100M), not just the S&P 500. The size BAND (mid/small = exclude the largest N, keep the next
M) is applied at study time via crsp_source.size_band_members_asof on the cap panel here.

    conda activate plutus
    python scripts/build_crsp_smallcap_lake.py --start 2005-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse

from plutus.data.sources import crsp_source as crsp
from plutus.io import atomic_to_parquet
from plutus.paths import PARQUET_DIR, RAW_DIR, ensure_dirs

CRSP_DIR = RAW_DIR / "crsp"


def build(start: str, end: str, price_min: float, cap_min_000: float) -> None:
    ensure_dirs()
    zip_path = CRSP_DIR / "daily_2000_2025.csv.zip"
    print(f"streaming common stocks (price>=${price_min}, cap>=${cap_min_000/1000:.0f}M) "
          f"{start}..{end} from the 28GB zip…")
    long = crsp.stream_universe(zip_path, start, end, price_min=price_min, cap_min_000=cap_min_000)
    print(f"kept {len(long):,} daily rows for {long['PERMNO'].nunique()} common stocks "
          f"({long['date'].min().date()} -> {long['date'].max().date()})")

    import pandas as pd
    adj = crsp.build_tr_adjusted_close(long)
    cap = crsp.build_mktcap(long)
    tmap = crsp.latest_ticker_map(long)

    def _pivot(col):                                  # date x PERMNO panel of a raw column
        d = long.dropna(subset=[col])[["PERMNO", "date", col]].drop_duplicates(["PERMNO", "date"])
        p = d.pivot(index="date", columns="PERMNO", values=col).sort_index()
        p.columns = p.columns.astype(str)
        return p
    vol = _pivot("DlyVol")            # share volume
    dvol = _pivot("DlyPrcVol")        # dollar volume (for liquidity / Amihud)

    adj.columns = adj.columns.astype(str)
    cap.columns = cap.columns.astype(str)
    atomic_to_parquet(adj, PARQUET_DIR / "crsp_smallcap_adj_close.parquet")
    atomic_to_parquet(cap, PARQUET_DIR / "crsp_smallcap_mktcap.parquet")
    atomic_to_parquet(vol, PARQUET_DIR / "crsp_smallcap_volume.parquet")
    atomic_to_parquet(dvol, PARQUET_DIR / "crsp_smallcap_dollarvol.parquet")
    atomic_to_parquet(pd.DataFrame({"permno": list(tmap), "ticker": list(tmap.values())}),
                      PARQUET_DIR / "crsp_smallcap_ticker_map.parquet")
    print(f"\nlake written: {adj.shape[0]} dates x {adj.shape[1]} common stocks")
    # how many names are in the mid/small band on a recent date?
    mb = crsp.size_band_members_asof(cap, exclude_top=500, band_size=2500)
    print(f"  mid/small band (rank 501-3000 by cap) on {adj.index[-1].date()}: "
          f"{len(mb(adj.index[-1]))} names")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--price-min", type=float, default=5.0)
    ap.add_argument("--cap-min-000", type=float, default=100_000.0)
    args = ap.parse_args()
    build(args.start, args.end, args.price_min, args.cap_min_000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
