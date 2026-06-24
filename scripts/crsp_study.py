"""TRUE survivorship-free PIT S&P 500 factor study, on CRSP data.

Reads the CRSP lake (scripts/build_crsp_lake.py) — total-return-adjusted prices, market cap,
and PIT membership are all survivorship-free (delisted names included while they were members,
then their series ends and the engine force-liquidates). Fundamentals come from SEC EDGAR via
PERMNO->ticker->CIK.

Contrast with scripts/crsp_factor_study.py (yfinance), which is missing ~16% of PIT members
(delisted) and so leans optimistic. This script is the honest version.

HONEST SCOPE: prices/returns/universe are fully survivorship-free here. The SEC-fundamentals
JOIN is still survivor-skewed (delisted tickers don't resolve in SEC's current ticker map), so
the VALUE factors' coverage tilts to survivors; PRICE factors (reversal/momentum/low-vol) and
the backtest RETURNS are clean. That is already the dominant fix.

    conda activate plutus
    python scripts/build_crsp_lake.py            # once
    python scripts/crsp_study.py                 # needs SEC_EDGAR_USER_AGENT for fundamentals
"""
from __future__ import annotations

import argparse

import pandas as pd

from plutus.data.sources import crsp_source as crsp
from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from plutus.research.backtest.frictions import USEquityCosts
from plutus.research.backtest.portfolio import signal_portfolio_backtest
from plutus.research.eval.factor_eval import compute_ic
from plutus.research.factors import library as fl

from build_fundamentals import build_panels


def _month_ends(dates: pd.DatetimeIndex) -> list:
    s = pd.Series(dates, index=dates)
    return s.groupby(dates.to_period("M")).max().tolist()


def _ticker_panel_to_permno(panel_tkr: pd.DataFrame, permno_to_ticker: dict) -> pd.DataFrame:
    """Re-key a ticker-columned panel to PERMNO columns via the PERMNO->ticker map."""
    cols = {permno: panel_tkr[tkr] for permno, tkr in permno_to_ticker.items()
            if tkr in panel_tkr.columns}
    return pd.DataFrame(cols)


def run(capital: float = 100_000.0, n_hold: int = 20,
        start: str | None = None, end: str | None = None) -> dict:
    ensure_dirs()
    adj = pd.read_parquet(PARQUET_DIR / "crsp_adj_close.parquet")
    cap = pd.read_parquet(PARQUET_DIR / "crsp_mktcap.parquet")
    tmap_df = pd.read_parquet(PARQUET_DIR / "crsp_ticker_map.parquet")
    spells = pd.read_parquet(PARQUET_DIR / "crsp_members.parquet")

    if start or end:                                    # slice to a sub-window (e.g. match yfinance)
        adj = adj.loc[(start or adj.index.min()):(end or adj.index.max())]
        cap = cap.loc[(start or cap.index.min()):(end or cap.index.max())]
        adj = adj.dropna(how="all", axis=1)
        cap = cap.reindex(columns=adj.columns)
    dates = adj.index
    permno_to_ticker = {str(int(p)): t for p, t in zip(tmap_df["permno"], tmap_df["ticker"])}
    _members_int = crsp.members_asof_from_spells(spells)
    def members_asof(d):                       # CRSP panels are keyed by str(PERMNO)
        return {str(p) for p in _members_int(d)}

    print(f"CRSP lake: {adj.shape[1]} PERMNOs, {dates.min().date()} -> {dates.max().date()} "
          f"(survivorship-free)")

    # fundamentals via SEC EDGAR, keyed by the names' tickers, then re-keyed to PERMNO
    tickers = sorted(set(permno_to_ticker.values()))
    funds = build_panels(tickers, dates)
    ni_t, book_t = funds["net_income_ttm"], funds["book_equity"]
    ni = _ticker_panel_to_permno(ni_t, permno_to_ticker).reindex(index=dates, columns=adj.columns)
    book = _ticker_panel_to_permno(book_t, permno_to_ticker).reindex(index=dates, columns=adj.columns)
    cap = cap.reindex(index=dates, columns=adj.columns)
    print(f"  fundamentals coverage: net_income {int(ni.notna().any().sum())}, "
          f"book {int(book.notna().any().sum())} / {adj.shape[1]} PERMNOs "
          f"(survivor-skewed — SEC join only)")

    factors = {
        "earnings_yield": fl.earnings_yield(ni, cap),
        "book_yield": fl.book_yield(book, cap),
        "reversal_1m": fl.reversal(adj, 21),
        "momentum_12_1": fl.momentum(adj, 252, 21),
        "low_vol": fl.low_vol(adj, 252),
    }
    eval_dates = _month_ends(dates)
    print(f"\n{len(eval_dates)} monthly eval dates\n")
    print(f"{'factor':16s} {'mean IC':>9s} {'IC-IR':>7s} {'t-stat':>7s} {'hit':>6s} {'n':>4s}")
    rows = []
    for name, fac in factors.items():
        r = compute_ic(fac, adj, eval_dates, members_asof)
        rows.append({"factor": name, "mean_ic": r.mean_ic, "ic_ir": r.ic_ir,
                     "t_stat": r.t_stat, "hit_rate": r.hit_rate, "n": r.n_periods})
        print(f"{name:16s} {r.mean_ic:9.4f} {r.ic_ir:7.3f} {r.t_stat:7.2f} "
              f"{r.hit_rate:6.2f} {r.n_periods:4d}")
    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "crsp_study_ic.parquet")

    ey = fl.restrict_to_universe(factors["earnings_yield"], members_asof)
    rev = fl.restrict_to_universe(factors["reversal_1m"], members_asof)
    signal = fl.blend([ey, rev], [5.0, 1.0])
    res = signal_portfolio_backtest(adj, signal, capital=capital, n_hold=n_hold,
                                    costs=USEquityCosts(), members_asof=members_asof)
    res.equity.to_frame("equity").to_parquet(BACKTESTS_DIR / "crsp_study_equity.parquet")
    print(f"\nCandidate value+reversal backtest (top-{n_hold}, monthly, US frictions, PIT, CRSP):")
    print(f"  total return : {res.total_return:8.2%}")
    print(f"  CAGR         : {res.cagr:8.2%}")
    print(f"  max drawdown : {res.max_drawdown:8.2%}")
    print(f"  avg names    : {res.avg_names_held:8.2f} / {n_hold}")
    print(f"  total costs  : ${res.total_costs:,.0f}  ({res.n_rebalances} rebalances)")
    print("\n[OK] survivorship-FREE prices/universe (CRSP). Compare vs the yfinance run "
          "(factor_study.py) to see the bias.")
    return {"ic": pd.DataFrame(rows), "backtest": res}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--n-hold", type=int, default=20)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    run(capital=args.capital, n_hold=args.n_hold, start=args.start, end=args.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
