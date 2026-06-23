"""CRSP adapter — gold-standard, survivorship-bias-free US equity data.

>>> LICENSING: CRSP is institution-licensed. This module READS a CRSP extract that the USER
>>> supplies under their own valid (e.g. academic/WRDS) license, kept locally under
>>> data/raw/crsp/ (gitignored). It is for PERSONAL research only — do not redistribute the
>>> data or use it in any monetized/redistributed product. For commercial use, license a
>>> commercial source (EODHD/Sharadar/Norgate).

Input is the modern CRSP **CIZ** daily stock file (94 cols) plus the S&P 500 membership
spells. The two things CRSP gives that free sources can't:
  - `DlyRet` — daily TOTAL return (dividends + delisting handled), so a return series is
    survivorship-free by construction; a name simply ends when it delists.
  - PIT index membership keyed by PERMNO (survives ticker changes/reuse).

The 2000–2025 daily CSV is ~28 GB uncompressed, so it is NEVER extracted to disk: it is
streamed straight out of the zip with pyarrow (column-projected, batch-filtered to the S&P 500
PERMNOs) into compact parquet panels. Pure transforms (membership, TR-index, market cap) are
unit-tested; only `stream_filtered` touches the big file.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

# Columns we pull from the daily file (of 94). Read as strings (robust to stray codes), then
# coerced — pyarrow type-inference can trip on a non-numeric value 20 GB into the stream.
DAILY_COLUMNS = ["PERMNO", "DlyCalDt", "Ticker", "DlyClose", "DlyRet", "DlyCap", "ShrOut"]
_NUMERIC = ["DlyClose", "DlyRet", "DlyCap", "ShrOut"]


# --- S&P 500 membership (PERMNO spells) --------------------------------------------

def load_membership(xlsx_path: str | Path) -> pd.DataFrame:
    """Load the S&P 500 constituents spell file -> DataFrame[permno, start, end].

    The xlsx is one row per continuous membership period: (PERMNO, start date, end date)."""
    df = pd.read_excel(xlsx_path)
    df = df.iloc[:, :3]
    df.columns = ["permno", "start", "end"]
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    return df.dropna(subset=["permno", "start", "end"]).reset_index(drop=True)


def members_asof_from_spells(spells: pd.DataFrame):
    """Build `members_asof(date) -> set[permno]` from membership spells (start <= date <= end)."""
    permno = spells["permno"].to_numpy()
    start = spells["start"].to_numpy()
    end = spells["end"].to_numpy()

    def members_asof(date) -> set:
        d = pd.Timestamp(date).to_datetime64()
        return set(int(p) for p in permno[(start <= d) & (end >= d)])

    return members_asof


def union_permnos(spells: pd.DataFrame, start, end) -> set:
    """Every PERMNO that was an S&P 500 member at any point in [start, end] — the set to pull
    prices for so a PIT backtest has data for each name while it was in the index."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    m = spells[(spells["start"] <= end) & (spells["end"] >= start)]
    return set(int(p) for p in m["permno"])


# --- streaming extraction from the big zip -----------------------------------------

def _inner_csv_name(z: zipfile.ZipFile) -> str:
    names = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]
    if not names:
        raise ValueError("no CSV member found in zip")
    return names[0]


def stream_filtered(zip_path: str | Path, permnos: set, start, end,
                    columns: list[str] = DAILY_COLUMNS, block_size: int = 64 << 20) -> pd.DataFrame:
    """Stream the daily CSV out of `zip_path`, keeping only rows whose PERMNO is in `permnos`
    and whose date is in [start, end], for `columns` only. Returns a tidy long DataFrame with
    coerced dtypes (PERMNO int, date datetime, numerics float). Never extracts the full CSV."""
    want = pa.array([str(int(p)) for p in permnos], type=pa.string())
    read_opts = pacsv.ReadOptions(block_size=block_size)
    conv_opts = pacsv.ConvertOptions(include_columns=columns,
                                     column_types={c: pa.string() for c in columns})
    parts: list[pa.Table] = []
    with zipfile.ZipFile(zip_path) as z:
        inner = _inner_csv_name(z)
        with z.open(inner) as f:
            reader = pacsv.open_csv(f, read_options=read_opts, convert_options=conv_opts)
            for batch in reader:
                tbl = pa.Table.from_batches([batch])
                tbl = tbl.filter(pc.is_in(tbl["PERMNO"], value_set=want))
                if tbl.num_rows:
                    parts.append(tbl)
    if not parts:
        return pd.DataFrame(columns=columns)
    df = pa.concat_tables(parts).to_pandas()
    df["PERMNO"] = pd.to_numeric(df["PERMNO"], errors="coerce").astype("int64")
    df["date"] = pd.to_datetime(df["DlyCalDt"], errors="coerce")
    for c in _NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    return df.drop_duplicates(["PERMNO", "date"]).reset_index(drop=True)


def stream_universe(zip_path: str | Path, start, end, price_min: float = 5.0,
                    cap_min_000: float = 100_000.0, block_size: int = 64 << 20) -> pd.DataFrame:
    """Stream the daily CSV keeping ALL tradable COMMON stocks (not just an index) — used to
    build a broad small/mid-cap universe. Per-batch filter: common stock (SecurityType EQTY,
    SecuritySubType COM) on a major exchange (NYSE/AMEX/NASDAQ = N/A/Q), with DlyClose >=
    `price_min` (drops penny stocks) and DlyCap >= `cap_min_000` (in $000s; drops micro-caps
    that can't really be traded/borrowed). Price/cap are read as floats for the filter; the
    rest as strings. Returns a tidy long DataFrame."""
    str_cols = ["PERMNO", "DlyCalDt", "Ticker", "DlyRet", "SecurityType", "SecuritySubType", "PrimaryExch"]
    float_cols = ["DlyClose", "DlyCap", "DlyVol", "DlyPrcVol"]   # incl. share volume + dollar volume
    cols = str_cols + float_cols
    conv = pacsv.ConvertOptions(
        include_columns=cols,
        column_types={**{c: pa.string() for c in str_cols}, **{c: pa.float64() for c in float_cols}})
    exch = pa.array(["N", "A", "Q"], type=pa.string())
    parts: list[pa.Table] = []
    with zipfile.ZipFile(zip_path) as z:
        inner = _inner_csv_name(z)
        with z.open(inner) as f:
            reader = pacsv.open_csv(f, read_options=pacsv.ReadOptions(block_size=block_size),
                                    convert_options=conv)
            for batch in reader:
                t = pa.Table.from_batches([batch])
                mask = pc.and_kleene(
                    pc.and_kleene(pc.equal(t["SecurityType"], "EQTY"),
                                  pc.equal(t["SecuritySubType"], "COM")),
                    pc.and_kleene(pc.is_in(t["PrimaryExch"], value_set=exch),
                                  pc.and_kleene(pc.greater_equal(t["DlyClose"], price_min),
                                                pc.greater_equal(t["DlyCap"], cap_min_000))))
                t = t.filter(mask)
                if t.num_rows:
                    parts.append(t.drop_columns(["SecurityType", "SecuritySubType", "PrimaryExch"]))
    if not parts:
        return pd.DataFrame()
    df = pa.concat_tables(parts).to_pandas()
    df["PERMNO"] = pd.to_numeric(df["PERMNO"], errors="coerce").astype("int64")
    df["date"] = pd.to_datetime(df["DlyCalDt"], errors="coerce")
    df["DlyRet"] = pd.to_numeric(df["DlyRet"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    return df.drop_duplicates(["PERMNO", "date"]).reset_index(drop=True)


def size_band_members_asof(mktcap: pd.DataFrame, exclude_top: int = 500, band_size: int = 2500):
    """Build `members_asof(date) -> set[str PERMNO]` = the cap-rank BAND [exclude_top,
    exclude_top+band_size) of `mktcap` on that date — i.e. drop the largest `exclude_top` names
    (the mega/large caps) and keep the next `band_size` (the mid/small caps). Uses the latest
    cap row on/before the query date."""
    def members_asof(date) -> set:
        s = mktcap.loc[:pd.Timestamp(date)]
        if s.empty:
            return set()
        row = s.iloc[-1].dropna().sort_values(ascending=False)
        return set(row.index[exclude_top:exclude_top + band_size])

    return members_asof


# --- panel builders (pure, unit-tested) --------------------------------------------

def build_tr_adjusted_close(long: pd.DataFrame) -> pd.DataFrame:
    """Wide (date x PERMNO) TOTAL-RETURN-adjusted price: compound DlyRet, anchored to each
    name's first real close so dollar levels are realistic (for lot-sizing). Consecutive
    ratios then equal the total return (dividends included); the series ends when the name
    delists, so the backtest engine force-liquidates it (no survivorship leak)."""
    d = long[["PERMNO", "date", "DlyClose", "DlyRet"]].sort_values(["PERMNO", "date"]).copy()
    d["ret"] = d["DlyRet"].fillna(0.0)
    d["close"] = d["DlyClose"].abs()                          # legacy CRSP used sign for no-trade
    d["cum"] = (1.0 + d["ret"]).groupby(d["PERMNO"]).cumprod()
    first_close = d.groupby("PERMNO")["close"].transform(
        lambda s: s.dropna().iloc[0] if s.notna().any() else float("nan"))
    first_cum = d.groupby("PERMNO")["cum"].transform("first")
    d["adj"] = first_close * d["cum"] / first_cum
    return d.pivot(index="date", columns="PERMNO", values="adj").sort_index()


def build_mktcap(long: pd.DataFrame) -> pd.DataFrame:
    """Wide (date x PERMNO) market cap in DOLLARS (CRSP DlyCap is in $000s)."""
    d = long.dropna(subset=["DlyCap"])[["PERMNO", "date", "DlyCap"]].drop_duplicates(["PERMNO", "date"])
    return d.pivot(index="date", columns="PERMNO", values="DlyCap").sort_index() * 1000.0


def latest_ticker_map(long: pd.DataFrame) -> dict[int, str]:
    """PERMNO -> its most recent ticker (for joining to SEC EDGAR fundamentals by ticker)."""
    d = long.dropna(subset=["Ticker"]).sort_values("date")
    return {int(k): str(v) for k, v in d.groupby("PERMNO")["Ticker"].last().items()}
