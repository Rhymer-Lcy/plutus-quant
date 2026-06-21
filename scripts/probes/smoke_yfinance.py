"""Smoke test the free yfinance data link: pull a few liquid names and print a panel summary.

    conda activate plutus
    python scripts/probes/smoke_yfinance.py

Exits non-zero if nothing came back (so it can gate a setup check).
"""
from __future__ import annotations

import sys

from plutus.data.sources import yfinance_source as yfs


def main() -> int:
    tickers = ["AAPL", "MSFT", "SPY"]
    panel = yfs.adjusted_close_panel(tickers, "2024-01-01", "2024-03-31")
    if panel.empty:
        print("FAILED: yfinance returned no data (network? rate limit?).")
        return 1
    print(f"OK: pulled {panel.shape[0]} rows x {panel.shape[1]} tickers "
          f"({panel.index.min().date()} → {panel.index.max().date()})")
    print(panel.tail(3).round(2).to_string())
    missing = [t for t in tickers if t not in panel.columns]
    if missing:
        print(f"NOTE: no data for {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
