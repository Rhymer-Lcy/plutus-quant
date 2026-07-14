"""Build the local price lake (adjusted + unadjusted daily close) for a universe over a window.

Pulls yfinance ONCE and caches to data/parquet, so studies read from disk and re-runs are
cheap (the hermes build_* -> *_study pattern). Default universe is the point-in-time S&P 500
union over the window (every name that was a member at any point), which is what a
survivorship-aware backtest needs — names yfinance lacks (mostly delisted) just come back as
absent columns, and the coverage report says how many were found.

    conda activate plutus
    python scripts/build_price_lake.py --start 2016-01-01 --end 2025-12-31
    python scripts/build_price_lake.py --tickers AAPL MSFT NVDA --start 2018-01-01 --end 2025-12-31
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

# This MUST run before the plutus imports below, which pull in yfinance: the library installs its
# own handler at import time, and silencing it afterwards no longer suppresses the per-ticker
# "delisted" noise. Hence the deliberate import-after-statement (E402).
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

from plutus.data import universe as uni                      # noqa: E402
from plutus.data.sources import yfinance_source as yfs       # noqa: E402
from plutus.io import atomic_to_parquet                      # noqa: E402
from plutus.paths import PARQUET_DIR, ensure_dirs            # noqa: E402


def build(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull adjusted + unadjusted close panels for `tickers` and cache to data/parquet."""
    ensure_dirs()
    print(f"pulling {len(tickers)} tickers {start} -> {end} (yfinance)…")
    adj = yfs.adjusted_close_panel(tickers, start, end)
    raw = yfs.raw_close_panel(tickers, start, end)
    atomic_to_parquet(adj, PARQUET_DIR / "adj_close.parquet")
    atomic_to_parquet(raw, PARQUET_DIR / "raw_close.parquet")
    got = sorted(set(adj.columns))
    missing = sorted(set(t.upper() for t in tickers) - set(c.upper() for c in got))
    print(f"adjusted: {adj.shape[0]} dates x {adj.shape[1]} tickers  "
          f"({adj.index.min().date()} -> {adj.index.max().date()})")
    print(f"coverage: {len(got)}/{len(tickers)} tickers have data; {len(missing)} missing "
          f"(mostly delisted — survivorship gap)")
    if missing[:15]:
        print("  missing sample:", ", ".join(missing[:15]))
    return adj, raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--tickers", nargs="*", help="explicit tickers; default = PIT S&P 500 union")
    ap.add_argument("--cap", type=int, default=0, help="cap universe size (0 = no cap; for a quick run)")
    args = ap.parse_args()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        hist = uni.load_sp500_history(uni.fetch_sp500_history())
        tickers = sorted(uni.union_members(hist, args.start, args.end))
        print(f"PIT S&P 500 union over window: {len(tickers)} unique tickers")
    if args.cap and len(tickers) > args.cap:
        tickers = tickers[:args.cap]
        print(f"capped to {len(tickers)} tickers")
    build(tickers, args.start, args.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
