"""First real PIT S&P 500 factor study — reads the cached price lake, joins SEC fundamentals,
reports single-factor rank IC for the factor battery, and runs the candidate value+reversal
backtest through the US friction model. All point-in-time (members_asof + filing-date facts).

>>> SURVIVORSHIP CAVEAT: the free price backbone (yfinance) is missing ~16% of PIT S&P 500
>>> members over the window (delisted/acquired names it drops; see build_price_lake coverage).
>>> So these numbers lean optimistic and are a CAPABILITY READOUT, not a tradeable result.
>>> Closing the delisted-price gap is the top open data problem (docs/data_sources.md).

Prereq: scripts/build_price_lake.py (prices) and SEC_EDGAR_USER_AGENT set (fundamentals).

    conda activate plutus
    python scripts/build_price_lake.py --start 2018-01-01 --end 2025-12-31
    python scripts/yfinance_factor_study.py
"""
from __future__ import annotations

import argparse

import pandas as pd

from plutus.data import universe as uni
from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from plutus.research.backtest.frictions import USEquityCosts
from plutus.research.backtest.metrics import month_ends
from plutus.research.backtest.portfolio import signal_portfolio_backtest
from plutus.research.eval.factor_eval import compute_ic
from plutus.research.factors import library as fl

from build_fundamentals import build_panels


def run(capital: float = 100_000.0, n_hold: int = 20) -> dict:
    ensure_dirs()
    adj = pd.read_parquet(PARQUET_DIR / "adj_close.parquet")
    raw = pd.read_parquet(PARQUET_DIR / "raw_close.parquet")
    dates = adj.index
    tickers = list(adj.columns)
    members_asof = uni.sp500_members_asof()

    print(f"price lake: {len(tickers)} tickers, {dates.min().date()} -> {dates.max().date()}")
    funds = build_panels(tickers, dates)
    shares = funds["shares"].reindex(index=dates, columns=tickers)
    ni = funds["net_income_ttm"].reindex(index=dates, columns=tickers)
    book = funds["book_equity"].reindex(index=dates, columns=tickers)
    market_cap = raw.reindex(index=dates, columns=tickers) * shares
    for name, p in [("net_income_ttm", ni), ("book_equity", book), ("shares", shares)]:
        print(f"  fundamentals {name:16s}: {int(p.notna().any().sum())}/{len(tickers)} tickers")

    factors = {
        "earnings_yield": fl.earnings_yield(ni, market_cap),
        "book_yield": fl.book_yield(book, market_cap),
        "reversal_1m": fl.reversal(adj, 21),
        "momentum_12_1": fl.momentum(adj, 252, 21),
        "low_vol": fl.low_vol(adj, 252),
    }
    eval_dates = month_ends(dates)
    print(f"\n{len(eval_dates)} monthly eval dates over the PIT S&P 500 universe\n")
    print(f"{'factor':16s} {'mean IC':>9s} {'IC-IR':>7s} {'t-stat':>7s} {'hit':>6s} {'n':>4s}")
    ic_rows = []
    for name, fac in factors.items():
        r = compute_ic(fac, adj, eval_dates, members_asof)
        ic_rows.append({"factor": name, "mean_ic": r.mean_ic, "ic_ir": r.ic_ir,
                        "t_stat": r.t_stat, "hit_rate": r.hit_rate, "n": r.n_periods})
        print(f"{name:16s} {r.mean_ic:9.4f} {r.ic_ir:7.3f} {r.t_stat:7.2f} "
              f"{r.hit_rate:6.2f} {r.n_periods:4d}")
    ic_table = pd.DataFrame(ic_rows).set_index("factor")
    atomic_to_parquet(ic_table.reset_index(), BACKTESTS_DIR / "first_study_ic.parquet")

    # candidate value + light reversal blend, PIT-restricted before the cross-sectional blend
    ey = fl.restrict_to_universe(factors["earnings_yield"], members_asof)
    rev = fl.restrict_to_universe(factors["reversal_1m"], members_asof)
    signal = fl.blend([ey, rev], [5.0, 1.0])
    res = signal_portfolio_backtest(adj, signal, capital=capital, n_hold=n_hold,
                                    costs=USEquityCosts(), members_asof=members_asof)
    res.equity.to_frame("equity").to_parquet(BACKTESTS_DIR / "first_study_equity.parquet")
    print(f"\nCandidate value+reversal backtest (top-{n_hold}, monthly, US frictions, PIT):")
    print(f"  total return : {res.total_return:8.2%}")
    print(f"  CAGR         : {res.cagr:8.2%}")
    print(f"  max drawdown : {res.max_drawdown:8.2%}")
    print(f"  avg names    : {res.avg_names_held:8.2f} / {n_hold}")
    print(f"  total costs  : ${res.total_costs:,.0f}  ({res.n_rebalances} rebalances)")
    print("\n[!] survivorship caveat: free data lacks ~16% of PIT members (delisted) — capability"
          " readout, NOT a tradeable result.")
    return {"ic": ic_table, "backtest": res}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--n-hold", type=int, default=20)
    args = ap.parse_args()
    run(capital=args.capital, n_hold=args.n_hold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
