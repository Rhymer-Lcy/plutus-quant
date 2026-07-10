"""Build the OPEN/CLOSE/quote panels for a chosen universe (large-cap or smallcap).

The main CRSP lakes keep only the TR-adjusted close. The overnight-return decomposition and any
OPEN-entry event study need the raw daily OPEN and CLOSE (same day) and the closing BID/ASK quote.
We stream five fields (DlyOpen, DlyClose, DlyRet, DlyBid, DlyAsk) for the chosen universe's union
PERMNOs (the columns of its adj_close panel), in one pass over the 28 GB zip, into:

  crsp[_smallcap]_open_raw.parquet    raw DlyOpen   (date x PERMNO, abs value)
  crsp[_smallcap]_close_raw.parquet   raw DlyClose  (date x PERMNO, abs value)
  crsp[_smallcap]_dlyret.parquet      DlyRet        (date x PERMNO; close-to-close TOTAL return)
  crsp[_smallcap]_halfspread.parquet  (DlyAsk-DlyBid)/mid/2 = relative HALF-spread (date x PERMNO)

The overnight study derives, split- and dividend-immune, intraday[t] = close[t]/open[t]-1 (same-day
ratio, so any split/dividend that occurs OVERNIGHT cannot contaminate it) and then overnight[t] =
(1+DlyRet[t])/(1+intraday[t])-1. The half-spread is the realistic cost of actually transacting at
the close/open auctions -- the heart of whether an anomaly is retail-tradeable.

    conda activate plutus
    python scripts/build_crsp_open_lake.py                       # large-cap (default)
    python scripts/build_crsp_open_lake.py --universe smallcap   # mid/small band union
"""
from __future__ import annotations

import argparse
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from plutus.data.sources import crsp_source as crsp
from plutus.io import atomic_to_parquet
from plutus.paths import PARQUET_DIR, RAW_DIR, ensure_dirs

ZIP = RAW_DIR / "crsp" / "daily_2000_2025.csv.zip"
COLUMNS = ["PERMNO", "DlyCalDt", "DlyOpen", "DlyClose", "DlyRet", "DlyBid", "DlyAsk"]
NUMERIC = ["DlyOpen", "DlyClose", "DlyRet", "DlyBid", "DlyAsk"]
START, END = "2005-01-01", "2024-12-31"      # match the lakes / membership spells window

# universe -> (adj panel whose columns define the union, output file prefix)
UNIVERSES = {
    "largecap": ("crsp_adj_close.parquet", "crsp"),
    "smallcap": ("crsp_smallcap_adj_close.parquet", "crsp_smallcap"),
}


def stream_open(permnos: set, block_size: int = 64 << 20) -> pd.DataFrame:
    """Stream COLUMNS for the given PERMNOs out of the zip (per-batch PERMNO filter); never
    extracts the full CSV. Mirrors crsp_source.stream_filtered but for the open/quote columns."""
    want = pa.array([str(int(p)) for p in permnos], type=pa.string())
    conv = pacsv.ConvertOptions(include_columns=COLUMNS,
                                column_types={c: pa.string() for c in COLUMNS})
    parts: list[pa.Table] = []
    with zipfile.ZipFile(ZIP) as z:
        inner = crsp._inner_csv_name(z)
        with z.open(inner) as f:
            reader = pacsv.open_csv(f, read_options=pacsv.ReadOptions(block_size=block_size),
                                    convert_options=conv)
            for batch in reader:
                tbl = pa.Table.from_batches([batch])
                tbl = tbl.filter(pc.is_in(tbl["PERMNO"], value_set=want))
                if tbl.num_rows:
                    parts.append(tbl)
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    df = pa.concat_tables(parts).to_pandas()
    df["PERMNO"] = pd.to_numeric(df["PERMNO"], errors="coerce").astype("int64")
    df["date"] = pd.to_datetime(df["DlyCalDt"], errors="coerce")
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= pd.Timestamp(START)) & (df["date"] <= pd.Timestamp(END))]
    return df.drop_duplicates(["PERMNO", "date"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", choices=sorted(UNIVERSES), default="largecap",
                    help="which union to stream: largecap (crsp_*) or smallcap (crsp_smallcap_*)")
    args = ap.parse_args()
    adj_name, prefix = UNIVERSES[args.universe]

    ensure_dirs()
    adj = pd.read_parquet(PARQUET_DIR / adj_name)
    permnos = {int(c) for c in adj.columns}
    print(f"streaming open/close/quote for {len(permnos)} {args.universe} PERMNOs, {START}..{END} "
          f"(one pass over {ZIP.name})...")
    long = stream_open(permnos)
    print(f"  {len(long):,} rows")

    def pivot(col: str, take_abs: bool = False) -> pd.DataFrame:
        d = long[["PERMNO", "date", col]].dropna(subset=[col]).drop_duplicates(["PERMNO", "date"])
        wide = d.pivot(index="date", columns="PERMNO", values=col).sort_index()
        wide.columns = [str(c) for c in wide.columns]      # str(PERMNO) to match the adj panels
        return wide.abs() if take_abs else wide

    open_raw = pivot("DlyOpen", take_abs=True)             # legacy CRSP sign = no-trade flag
    close_raw = pivot("DlyClose", take_abs=True)
    ret = pivot("DlyRet")
    bid, ask = pivot("DlyBid", take_abs=True), pivot("DlyAsk", take_abs=True)
    mid = (bid + ask) / 2.0
    half_spread = ((ask - bid) / mid / 2.0).where(mid > 0)

    atomic_to_parquet(open_raw, PARQUET_DIR / f"{prefix}_open_raw.parquet")
    atomic_to_parquet(close_raw, PARQUET_DIR / f"{prefix}_close_raw.parquet")
    atomic_to_parquet(ret, PARQUET_DIR / f"{prefix}_dlyret.parquet")
    atomic_to_parquet(half_spread, PARQUET_DIR / f"{prefix}_halfspread.parquet")
    print(f"  wrote {prefix}_open/close/dlyret/halfspread panels ({open_raw.shape}) to {PARQUET_DIR}")
    print(f"  median relative half-spread (all names/days): {half_spread.stack().median():.4%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
