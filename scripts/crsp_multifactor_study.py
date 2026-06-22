"""Multi-factor walk-forward + risk overlay, on survivorship-free CRSP data.

Builds on docs/survivorship_study.md, which showed the naive value+reversal candidate is
non-viable (-88% in 2008). Two fixes, measured here on clean CRSP data (2005-2024, PIT S&P 500):
  1. RISK OVERLAY — a market trend filter (regime.trend_exposure) scales gross exposure to
     cash when the broad market is below its 200-day MA. Should cut the 2008 drawdown.
  2. MULTI-FACTOR ML — a walk-forward LightGBM combiner (research.model.walk_forward) blends
     the factor battery out-of-sample, instead of a fixed value+reversal tilt.

Reports CAGR / max-DD / Calmar / Sharpe for: candidate, candidate+regime, ML, ML+regime.

    conda activate plutus
    python scripts/build_crsp_lake.py            # once
    python scripts/crsp_multifactor_study.py     # needs SEC_EDGAR_USER_AGENT
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from plutus.data.sources import crsp_source as crsp
from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from plutus.research.backtest.frictions import USEquityCosts
from plutus.research.backtest.portfolio import signal_portfolio_backtest
from plutus.research.backtest.regime import cap_weighted_index, trend_exposure
from plutus.research.factors import library as fl
from plutus.research.model.walk_forward import build_dataset, walk_forward_predict

from build_fundamentals import build_panels
from crsp_study import _month_ends, _ticker_panel_to_permno


def _eq_metrics(eq: pd.Series, capital: float) -> dict:
    """Performance metrics from an equity curve: CAGR, max drawdown, Calmar, annualized
    Sharpe (of daily returns), total return."""
    eq = eq.dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    r = eq.pct_change().dropna()
    maxdd = float((eq / eq.cummax() - 1.0).min())
    cagr = float((eq.iloc[-1] / capital) ** (1.0 / years) - 1.0)
    return {
        "cagr": cagr, "max_dd": maxdd,
        "calmar": float(cagr / abs(maxdd)) if maxdd < 0 else float("nan"),
        "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan"),
        "total_return": float(eq.iloc[-1] / capital - 1.0),
    }


def run(capital: float = 100_000.0, n_hold: int = 20, ma_window: int = 200,
        eval_start: str | None = None) -> dict:
    """`eval_start`: optionally measure performance only from this date (signals are still
    computed on FULL history — so factor warm-up and ML training are intact — but the backtest
    is sliced, giving an apples-to-apples window for the ML-vs-candidate comparison)."""
    ensure_dirs()
    adj = pd.read_parquet(PARQUET_DIR / "crsp_adj_close.parquet")
    cap = pd.read_parquet(PARQUET_DIR / "crsp_mktcap.parquet")
    tmap_df = pd.read_parquet(PARQUET_DIR / "crsp_ticker_map.parquet")
    spells = pd.read_parquet(PARQUET_DIR / "crsp_members.parquet")
    dates = adj.index
    permno_to_ticker = {str(int(p)): t for p, t in zip(tmap_df["permno"], tmap_df["ticker"])}
    _m = crsp.members_asof_from_spells(spells)
    members_asof = lambda d: {str(p) for p in _m(d)}

    funds = build_panels(sorted(set(permno_to_ticker.values())), dates)
    ni = _ticker_panel_to_permno(funds["net_income_ttm"], permno_to_ticker).reindex(index=dates, columns=adj.columns)
    book = _ticker_panel_to_permno(funds["book_equity"], permno_to_ticker).reindex(index=dates, columns=adj.columns)
    cap = cap.reindex(index=dates, columns=adj.columns)

    factors = {
        "earnings_yield": fl.earnings_yield(ni, cap),
        "book_yield": fl.book_yield(book, cap),
        "roe": fl.roe(ni, book),
        "reversal_1m": fl.reversal(adj, 21),
        "momentum_12_1": fl.momentum(adj, 252, 21),
        "low_vol": fl.low_vol(adj, 252),
    }
    eval_dates = _month_ends(dates)

    # --- risk overlay: market trend filter -> exposure_asof
    market = cap_weighted_index(adj, cap)
    regime = trend_exposure(market, window=ma_window, floor=0.0)

    # --- signals: fixed candidate vs walk-forward ML combiner
    candidate = fl.blend([fl.restrict_to_universe(factors["earnings_yield"], members_asof),
                          fl.restrict_to_universe(factors["reversal_1m"], members_asof)], [5.0, 1.0])
    print(f"building ML dataset over {len(eval_dates)} months x {adj.shape[1]} PERMNOs…")
    data, cols = build_dataset(factors, adj, eval_dates, members_asof)
    print(f"  {len(data):,} samples, features={cols}; walk-forward LightGBM…")
    ml = walk_forward_predict(data, cols, min_train=24, window=36)
    print(f"  OOS signal: {ml.shape[0]} months x {ml.shape[1]} PERMNOs "
          f"(from {ml.index.min().date()})")

    costs = USEquityCosts()
    adj_bt = adj.loc[pd.Timestamp(eval_start):] if eval_start else adj   # backtest window only
    win = f"{adj_bt.index.min().date()} -> {adj_bt.index.max().date()}"
    print(f"backtest window: {win}")
    def bt(sig, exposure):
        return signal_portfolio_backtest(adj_bt, sig, capital=capital, n_hold=n_hold, costs=costs,
                                         members_asof=members_asof, exposure_asof=exposure)
    runs = {
        "candidate (val+rev)": bt(candidate, None).equity,
        "candidate + regime": bt(candidate, regime).equity,
        "ML multi-factor": bt(ml, None).equity,
        "ML + regime": bt(ml, regime).equity,
    }
    # passive benchmark: buy & hold the cap-weighted market proxy over the same window
    bench = market.reindex(adj_bt.index).ffill()
    runs["S&P500 proxy (B&H)"] = (capital * bench / bench.dropna().iloc[0])

    print(f"\n{'strategy':22s} {'CAGR':>8s} {'maxDD':>8s} {'Calmar':>7s} {'Sharpe':>7s} {'totRet':>9s}")
    rows = []
    for name, eq in runs.items():
        m = _eq_metrics(eq, capital)
        rows.append({"strategy": name, **m})
        print(f"{name:22s} {m['cagr']:8.2%} {m['max_dd']:8.2%} {m['calmar']:7.2f} "
              f"{m['sharpe']:7.2f} {m['total_return']:9.1%}")
    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "crsp_multifactor_summary.parquet")
    print("\n[OK] all survivorship-free (CRSP). Regime overlay targets the -88% 2008 drawdown; "
          "ML combiner targets return. NOT yet a validated strategy — see docs/multifactor_study.md.")
    return {"summary": pd.DataFrame(rows), "runs": runs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--n-hold", type=int, default=20)
    ap.add_argument("--ma-window", type=int, default=200)
    ap.add_argument("--eval-start", default=None, help="measure performance only from this date")
    args = ap.parse_args()
    run(capital=args.capital, n_hold=args.n_hold, ma_window=args.ma_window,
        eval_start=args.eval_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
