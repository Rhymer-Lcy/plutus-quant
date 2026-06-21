"""Point-in-time index membership — the survivorship-free universe.

A backtest universe built from *currently* listed names is survivorship-contaminated (it only
ever holds companies that survived to today), which inflates returns. This module builds a
`members_asof(date) -> set[ticker]` callable from a point-in-time membership history, which is
exactly what the backtest engine, factor eval, and `restrict_to_universe` consume.

FREE source (verified 2026-06-21): fja05680/sp500, "S&P 500 Historical Components & Changes"
— S&P 500 membership since 1996 as (date, comma-separated tickers) rows, each row the FULL
constituent list as of that change date. https://github.com/fja05680/sp500

LIMITATIONS (be honest about them — see docs/data_sources.md):
  - It gives PIT *membership*, but you still need delisted *price* series separately; free
    price sources (yfinance) drop most delisted tickers, so early backtests carry a residual
    survivorship caveat until delisted prices are sourced.
  - Tickers are symbols as-of-then; symbol reuse/changes over decades are not reconciled here.
  - Coverage starts 1996; the gold standard (CRSP/Norgate) is paid.
"""
from __future__ import annotations

import bisect
from pathlib import Path

import pandas as pd
import requests

from .. import config
from ..io import atomic_write_text
from ..paths import RAW_DIR

SP500_HISTORY_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes.csv"
)
_DEFAULT_PATH = RAW_DIR / "sp500_history.csv"


def fetch_sp500_history(dest: str | Path | None = None, refresh: bool = False) -> Path:
    """Download the fja05680 S&P 500 history CSV into the local lake (gitignored). Returns the
    path. Uses SEC_EDGAR_USER_AGENT as a courtesy UA if set, else a generic one (GitHub raw
    needs no key)."""
    dest = Path(dest) if dest else _DEFAULT_PATH
    if refresh or not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        ua = config.get("SEC_EDGAR_USER_AGENT") or "plutus-quant data fetch"
        resp = requests.get(SP500_HISTORY_URL, headers={"User-Agent": ua}, timeout=30)
        resp.raise_for_status()
        atomic_write_text(resp.text, dest)
    return dest


def load_sp500_history(path: str | Path | None = None) -> list[tuple[pd.Timestamp, frozenset]]:
    """Parse the membership CSV into a chronologically sorted list of (date, members) where
    members is the full constituent set as of that date. Robust to the tickers field being one
    quoted comma-separated string."""
    path = Path(path) if path else _DEFAULT_PATH
    df = pd.read_csv(path)
    # tolerate column-name casing/spelling ("date","tickers")
    cols = {c.lower().strip(): c for c in df.columns}
    date_col, tick_col = cols.get("date"), cols.get("tickers")
    if date_col is None or tick_col is None:
        raise ValueError(f"expected 'date' and 'tickers' columns, got {list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    history = []
    for _, row in df.iterrows():
        raw = str(row[tick_col]) if pd.notna(row[tick_col]) else ""
        members = frozenset(t.strip().upper() for t in raw.split(",") if t.strip())
        history.append((pd.Timestamp(row[date_col]), members))
    return history


def members_asof_from_history(history: list[tuple[pd.Timestamp, frozenset]]):
    """Build `members_asof(date) -> set[ticker]` from a (date, members) history: the membership
    of the most recent change date on or before `date`. Before the first date -> empty set."""
    dates = [d for d, _ in history]
    sets = [m for _, m in history]

    def members_asof(date) -> set:
        i = bisect.bisect_right(dates, pd.Timestamp(date)) - 1
        return set(sets[i]) if i >= 0 else set()

    return members_asof


def sp500_members_asof(path: str | Path | None = None, refresh: bool = False):
    """Convenience: ensure the history CSV is present, then return a `members_asof` callable."""
    p = fetch_sp500_history(path, refresh=refresh)
    return members_asof_from_history(load_sp500_history(p))
