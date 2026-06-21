"""First end-to-end study: value + short-term reversal on a US universe.

Ties the whole stack together on live free data:
  prices (yfinance, adjusted for returns + unadjusted for market cap)
  + fundamentals (SEC EDGAR: TTM net income, book equity, shares)
  + point-in-time universe (optional members_asof)
  -> factors (earnings yield, book yield, reversal)
  -> single-factor rank IC (the honest gate)
  -> the candidate value+reversal blend through the US-friction backtest.

This is a CAPABILITY + a sanity harness, NOT a validated result. Run it on a handful of names
first (scripts/probes/smoke_pipeline.py), then on a real PIT S&P 500 universe.

    conda activate plutus
    python scripts/value_reversal_study.py AAPL MSFT NVDA XOM JPM PG --start 2018-01-01 --end 2025-12-31
"""
from __future__ import annotations

import argparse

import pandas as pd

from plutus.data.sources import yfinance_source as yfs
from plutus.research.backtest.frictions import USEquityCosts
from plutus.research.backtest.portfolio import signal_portfolio_backtest
from plutus.research.eval.factor_eval import compute_ic
from plutus.research.factors import library as fl

from build_fundamentals import build_panels


def _month_end_dates(dates: pd.DatetimeIndex) -> list:
    """Last trading day of each month present in `dates` (the non-overlapping eval/rebalance grid)."""
    s = pd.Series(dates, index=dates)
    return s.groupby(dates.to_period("M")).max().tolist()


def run_study(tickers: list[str], start: str, end: str, members_asof=None,
              capital: float = 100_000.0, n_hold: int = 10) -> dict:
    """Build factors from live data, report IC, and run the frictioned candidate backtest.
    Returns a dict with the IC table and the PortfolioResult. `members_asof` (optional) makes
    it survivorship-free; omit to use the passed tickers as the whole universe."""
    adj = yfs.adjusted_close_panel(tickers, start, end)
    raw = yfs.raw_close_panel(tickers, start, end)
    if adj.empty:
        raise RuntimeError("no price data returned (network? tickers?)")
    dates = adj.index
    cols = adj.columns

    funds = build_panels(list(cols), dates, verbose=True)
    shares = funds["shares"].reindex(index=dates, columns=cols)
    ni = funds["net_income_ttm"].reindex(index=dates, columns=cols)
    book = funds["book_equity"].reindex(index=dates, columns=cols)
    market_cap = raw.reindex(index=dates, columns=cols) * shares    # unadjusted price * as-reported shares

    factors = {
        "earnings_yield": fl.earnings_yield(ni, market_cap),
        "book_yield": fl.book_yield(book, market_cap),
        "reversal": fl.reversal(adj, 21),
    }
    eval_dates = _month_end_dates(dates)

    print(f"\nUniverse: {len(cols)} tickers, {dates.min().date()} -> {dates.max().date()}, "
          f"{len(eval_dates)} monthly eval dates\n")
    print(f"{'factor':16s} {'mean IC':>9s} {'IC-IR':>7s} {'t-stat':>7s} {'hit':>6s} {'n':>4s}")
    ic_table = {}
    for name, fac in factors.items():
        r = compute_ic(fac, adj, eval_dates, members_asof)
        ic_table[name] = r
        print(f"{name:16s} {r.mean_ic:9.4f} {r.ic_ir:7.3f} {r.t_stat:7.2f} "
              f"{r.hit_rate:6.2f} {r.n_periods:4d}")

    # candidate value + light reversal blend (a PRIOR to test, not a result)
    ey = fl.restrict_to_universe(factors["earnings_yield"], members_asof) if members_asof else factors["earnings_yield"]
    rev = fl.restrict_to_universe(factors["reversal"], members_asof) if members_asof else factors["reversal"]
    signal = fl.blend([ey, rev], [5.0, 1.0])
    res = signal_portfolio_backtest(adj, signal, capital=capital, n_hold=n_hold,
                                    costs=USEquityCosts(), members_asof=members_asof)
    print(f"\nCandidate value+reversal backtest (top-{n_hold}, monthly, US frictions):")
    print(f"  total return : {res.total_return:8.2%}")
    print(f"  CAGR         : {res.cagr:8.2%}")
    print(f"  max drawdown : {res.max_drawdown:8.2%}")
    print(f"  avg names    : {res.avg_names_held:8.2f} / {n_hold}")
    print(f"  total costs  : ${res.total_costs:,.0f}")
    return {"ic": ic_table, "backtest": res}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--n-hold", type=int, default=10)
    args = ap.parse_args()
    run_study(args.tickers, args.start, args.end, capital=args.capital, n_hold=args.n_hold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
