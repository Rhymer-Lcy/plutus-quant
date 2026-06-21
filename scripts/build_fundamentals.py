"""Build point-in-time fundamental panels for a US universe from SEC EDGAR.

Maps each ticker -> CIK (company_tickers.json), pulls cached company facts, and assembles
filing-date PIT panels aligned to a daily date index:
  - net_income_ttm  (us-gaap:NetIncomeLoss, a FLOW -> trailing twelve months)
  - book_equity     (us-gaap:StockholdersEquity, an INSTANT)
  - shares          (dei:EntityCommonStockSharesOutstanding, an INSTANT, unit "shares")

Pair `shares` with the UNADJUSTED close (yfinance_source.raw_close_panel) to get market cap.
Tickers without a CIK, or without a given concept, are simply absent from that panel's columns
(not zero-filled). company_facts is cached per-CIK by the adapter, so re-runs are cheap.

    conda activate plutus
    python scripts/build_fundamentals.py AAPL MSFT --start 2018-01-01 --end 2025-12-31
"""
from __future__ import annotations

import argparse

import pandas as pd

from plutus.data.sources import sec_edgar as se


def build_panels(tickers: list[str], dates: pd.DatetimeIndex,
                 verbose: bool = False) -> dict[str, pd.DataFrame]:
    """Return {'net_income_ttm', 'book_equity', 'shares'} wide PIT panels over `dates`."""
    cikmap = se.load_ticker_cik_map()
    facts_by_ticker: dict[str, dict] = {}
    for t in tickers:
        cik = cikmap.get(t.upper())
        if cik is None:
            if verbose:
                print(f"  no CIK for {t}")
            continue
        try:
            facts_by_ticker[t] = se.company_facts(cik)
        except Exception as exc:                      # network / missing filer
            if verbose:
                print(f"  skip {t}: {exc}")
    panels = {
        "net_income_ttm": se.build_fundamental_panel(
            facts_by_ticker, "NetIncomeLoss", dates, kind="flow_ttm"),
        "book_equity": se.build_fundamental_panel(
            facts_by_ticker, "StockholdersEquity", dates, kind="instant"),
        "shares": se.build_fundamental_panel(
            facts_by_ticker, "EntityCommonStockSharesOutstanding", dates,
            kind="instant", unit="shares", taxonomy="dei"),
    }
    return panels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2025-12-31")
    args = ap.parse_args()
    dates = pd.bdate_range(args.start, args.end)
    panels = build_panels(args.tickers, dates, verbose=True)
    for name, p in panels.items():
        cover = int(p.notna().any().sum())
        print(f"{name:16s}: {p.shape[0]} dates x {cover} tickers with data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
