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


def stream_industry(zip_path: str | Path, sic_codes: set[str], start, end,
                    block_size: int = 64 << 20) -> pd.DataFrame:
    """Stream the daily CSV keeping COMMON stocks on a major exchange whose SIC code is in
    `sic_codes` -- an INDUSTRY lake (pharma/biotech for the catalyst study, docs/biotech_catalyst_study.md).

    DELIBERATELY NO price/cap filter, unlike `stream_universe`. A tradability gate belongs at
    STUDY time (was this event tradable when it happened?), not in the lake: filtering rows out
    of the lake would delete a name's history the moment it fell below the threshold, so a
    biotech that gaps up and then craters past $5 would have its post-event LOSSES truncated --
    an upward bias in exactly the quantity the study measures. The lake therefore keeps every
    row and carries DlyClose and DlyCap so the study can gate on them.

    Keeps the bar's OPEN and the quoted BID/ASK: the overnight gap needs the open, and a
    catalyst-day spread in a small biotech is the dominant cost. SIC is the security's code AS
    OF each row, so a reclassified name enters/leaves the industry on the CRSP date, not
    retroactively (point-in-time by construction). Returns a tidy long frame."""
    str_cols = ["PERMNO", "DlyCalDt", "Ticker", "SICCD", "DlyRet", "SecurityType",
                "SecuritySubType", "PrimaryExch"]
    float_cols = ["DlyClose", "DlyOpen", "DlyCap", "DlyPrcVol", "DlyBid", "DlyAsk"]
    cols = str_cols + float_cols
    conv = pacsv.ConvertOptions(
        include_columns=cols,
        column_types={**{c: pa.string() for c in str_cols}, **{c: pa.float64() for c in float_cols}})
    exch = pa.array(["N", "A", "Q"], type=pa.string())
    want_sic = pa.array(sorted(sic_codes), type=pa.string())
    parts: list[pa.Table] = []
    with zipfile.ZipFile(zip_path) as z:
        inner = _inner_csv_name(z)
        with z.open(inner) as f:
            reader = pacsv.open_csv(f, read_options=pacsv.ReadOptions(block_size=block_size),
                                    convert_options=conv)
            for batch in reader:
                t = pa.Table.from_batches([batch])
                common = pc.and_kleene(
                    pc.and_kleene(pc.equal(t["SecurityType"], "EQTY"),
                                  pc.equal(t["SecuritySubType"], "COM")),
                    pc.is_in(t["PrimaryExch"], value_set=exch))
                t = t.filter(pc.and_kleene(common, pc.is_in(t["SICCD"], value_set=want_sic)))
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


def stream_sic_spells(zip_path: str | Path, start, end,
                      block_size: int = 64 << 20) -> pd.DataFrame:
    """Stream the daily CSV and collapse it to SIC SPELLS: one row per (permno, siccd) with the
    first and last date CRSP carried that industry code for that name.

    Industry must be POINT IN TIME. A company can be reclassified: CRSP codes some names into
    pharma only late in life, and tagging an event by "was this name EVER a biotech" would count
    gaps that happened while it was something else entirely -- 173 of them, in this window. The
    spells let a study ask what the name WAS on the day of the event. Keeps common stock on a
    major exchange (the universe the price panels cover).

    Returns [permno, siccd, start, end]."""
    cols = ["PERMNO", "DlyCalDt", "SICCD", "SecurityType", "SecuritySubType", "PrimaryExch"]
    conv = pacsv.ConvertOptions(include_columns=cols,
                                column_types={c: pa.string() for c in cols})
    exch = pa.array(["N", "A", "Q"], type=pa.string())
    parts: list[pa.Table] = []
    with zipfile.ZipFile(zip_path) as z:
        inner = _inner_csv_name(z)
        with z.open(inner) as f:
            reader = pacsv.open_csv(f, read_options=pacsv.ReadOptions(block_size=block_size),
                                    convert_options=conv)
            for batch in reader:
                t = pa.Table.from_batches([batch])
                t = t.filter(pc.and_kleene(
                    pc.and_kleene(pc.equal(t["SecurityType"], "EQTY"),
                                  pc.equal(t["SecuritySubType"], "COM")),
                    pc.is_in(t["PrimaryExch"], value_set=exch)))
                if t.num_rows:
                    parts.append(t.drop_columns(["SecurityType", "SecuritySubType", "PrimaryExch"]))
    if not parts:
        return pd.DataFrame()
    df = pa.concat_tables(parts).to_pandas()
    df["date"] = pd.to_datetime(df["DlyCalDt"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    df = df[df["SICCD"].notna() & (df["SICCD"] != "")]
    # CONTIGUOUS runs, not a min/max envelope per (permno, siccd): a company whose code goes
    # A -> B -> A would otherwise get an "A" spell spanning B's whole period, so a date inside B
    # would match BOTH spells and the event would be counted twice under two industries.
    df = df.sort_values(["PERMNO", "date"], kind="stable")
    change = (df["PERMNO"] != df["PERMNO"].shift()) | (df["SICCD"] != df["SICCD"].shift())
    df["run"] = change.cumsum()
    spells = (df.groupby("run")
              .agg(permno=("PERMNO", "first"), siccd=("SICCD", "first"),
                   start=("date", "min"), end=("date", "max"))
              .reset_index(drop=True))
    spells["permno"] = spells["permno"].astype(str)
    return spells.sort_values(["permno", "start"]).reset_index(drop=True)


def stream_cusip_spells(zip_path: str | Path, start, end,
                        block_size: int = 64 << 20) -> pd.DataFrame:
    """Stream the daily CSV and collapse it to CUSIP -> PERMNO SPELLS: one row per
    (permno, cusip9) with the first and last date CRSP carried that pairing.

    13F holdings identify a security only by its 9-digit CUSIP, while every price panel here
    is keyed by PERMNO. A CUSIP can be REUSED by a different issuer after a delisting, so the
    join must be point-in-time: match a holding's CUSIP to the permno whose spell CONTAINS the
    filing date, never to a permno that carried that CUSIP in some other decade. Keeps common
    stock on a major exchange (the 13F universe that this repo can price); the caller reports
    the share of holdings that fail to match.

    Returns [permno, cusip9, cusip8, ticker, start, end]."""
    cols = ["PERMNO", "DlyCalDt", "CUSIP9", "CUSIP", "Ticker",
            "SecurityType", "SecuritySubType", "PrimaryExch"]
    conv = pacsv.ConvertOptions(include_columns=cols,
                                column_types={c: pa.string() for c in cols})
    exch = pa.array(["N", "A", "Q"], type=pa.string())
    parts: list[pa.Table] = []
    with zipfile.ZipFile(zip_path) as z:
        inner = _inner_csv_name(z)
        with z.open(inner) as f:
            reader = pacsv.open_csv(f, read_options=pacsv.ReadOptions(block_size=block_size),
                                    convert_options=conv)
            for batch in reader:
                t = pa.Table.from_batches([batch])
                t = t.filter(pc.and_kleene(
                    pc.and_kleene(pc.equal(t["SecurityType"], "EQTY"),
                                  pc.equal(t["SecuritySubType"], "COM")),
                    pc.is_in(t["PrimaryExch"], value_set=exch)))
                if t.num_rows:
                    parts.append(t.drop_columns(["SecurityType", "SecuritySubType", "PrimaryExch"]))
    if not parts:
        return pd.DataFrame()
    df = pa.concat_tables(parts).to_pandas()
    df["date"] = pd.to_datetime(df["DlyCalDt"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    df = df[df["CUSIP9"].notna() & (df["CUSIP9"] != "")]
    spells = (df.groupby(["PERMNO", "CUSIP9"])
              .agg(cusip8=("CUSIP", "last"), ticker=("Ticker", "last"),
                   start=("date", "min"), end=("date", "max"))
              .reset_index().rename(columns={"PERMNO": "permno", "CUSIP9": "cusip9"}))
    spells["permno"] = spells["permno"].astype(str)
    return spells.sort_values(["cusip9", "start"]).reset_index(drop=True)


def tag_sic_asof(events: pd.DataFrame, spells: pd.DataFrame,
                 date_col: str = "date") -> pd.DataFrame:
    """Attach to each row the SIC code CRSP carried for that permno ON THAT DATE.

    Never tag an event by "what the company became": 4,698 of the 13,159 names in this window are
    reclassified at least once, so a whole-life label would count a gap that happened while the
    company was in another industry entirely. Rows whose (permno, date) falls in no spell are
    DROPPED -- the caller reports the loss rather than guessing an industry."""
    m = events.merge(spells[["permno", "siccd", "start", "end"]], on="permno", how="inner")
    hit = (m[date_col] >= m["start"]) & (m[date_col] <= m["end"])
    return m[hit].drop(columns=["start", "end"]).reset_index(drop=True)


def sic_membership_panel(spells: pd.DataFrame, sic_codes: set[str], index: pd.DatetimeIndex,
                         columns) -> pd.DataFrame:
    """Boolean (date x permno) panel: was this name in `sic_codes` ON THAT DAY?

    The peer group of an industry study is every name CLASSIFIED into the industry that day --
    not the names that happen to have an event. Using event names as their own benchmark would
    net the effect against itself."""
    out = pd.DataFrame(False, index=index, columns=list(columns))
    cols = set(out.columns)
    sp = spells[spells["siccd"].isin(sic_codes)]
    for permno, start, end in zip(sp["permno"], sp["start"], sp["end"]):
        if permno in cols:
            out.loc[start:end, permno] = True
    return out


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


def ticker_panel_to_permno(panel_tkr: pd.DataFrame, permno_to_ticker: dict) -> pd.DataFrame:
    """Re-key a ticker-columned panel to PERMNO columns via the PERMNO->ticker map (the inverse of
    joining SEC fundamentals, which arrive keyed by ticker, back onto the PERMNO-keyed CRSP lake)."""
    cols = {permno: panel_tkr[tkr] for permno, tkr in permno_to_ticker.items()
            if tkr in panel_tkr.columns}
    return pd.DataFrame(cols)
