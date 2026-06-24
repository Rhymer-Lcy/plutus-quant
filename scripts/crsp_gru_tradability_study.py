"""Is the GRU edge actually deployable for a retail trader? Three rigorous checks on the cached
5-seed ensemble GRU signal (scripts/crsp_gru_pipeline.py):

  A. LONG-ONLY top-decile (no shorting/borrow — retail can't short small caps) vs the equal-weight
     same-universe benchmark, net of slippage. Does the tilt add alpha you can actually keep?
  B. COST-SENSITIVITY curve — market-neutral q0.10 net Sharpe across (slippage x borrow); at what
     cost does it cross zero?
  C. LIQUIDITY TIERS — same signal evaluated on tighter cap-rank bands (more-liquid mid-caps =
     lower real cost); is a liquid tier clearly net-positive?

No training here (signal is cached) — pure, fast, survivorship-free, cost-aware evaluation.

    conda activate plutus
    python scripts/crsp_gru_pipeline.py --universe smallcap --ensemble 5   # produces the signal
    python scripts/crsp_gru_tradability_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from plutus.data.sources import crsp_source as crsp
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR
from plutus.research.backtest.frictions import USEquityCosts
from plutus.research.backtest.long_short import quantile_long_short
from plutus.research.backtest.portfolio import signal_portfolio_backtest

from crsp_study import _month_ends


def _metrics(monthly_ret: pd.Series) -> dict:
    r = monthly_ret.dropna()
    if r.empty:
        return {"cagr": float("nan"), "sharpe": float("nan"), "maxdd": float("nan")}
    eq = (1 + r).cumprod()
    yrs = len(r) / 12.0
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "sharpe": float(r.mean() / r.std() * np.sqrt(12)),
            "maxdd": float((eq / eq.cummax() - 1).min())}


def _eq_weight_benchmark(adj, eval_dates, members_asof) -> pd.Series:
    fwd = adj.reindex(eval_dates)
    out = {}
    for i in range(len(eval_dates) - 1):
        t, t1 = eval_dates[i], eval_dates[i + 1]
        m = members_asof(t)
        r = (fwd.loc[t1] / fwd.loc[t] - 1.0)[[c for c in adj.columns if c in m]].dropna()
        out[t1] = float(r.mean()) if len(r) else 0.0
    return pd.Series(out)


def main() -> int:
    adj = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_adj_close.parquet")
    cap = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_mktcap.parquet")
    signal = pd.read_parquet(BACKTESTS_DIR / "crsp_dl_smallcap_gru_signal.parquet")
    eval_dates = _month_ends(adj.index)
    signal = signal.reindex(eval_dates)
    band = crsp.size_band_members_asof(cap, exclude_top=500, band_size=2500)
    print(f"GRU signal: {signal.notna().any(axis=1).sum()} active months, {adj.shape[1]} names\n")

    # --- A. long-only top-decile vs equal-weight benchmark (no borrow) ---
    print("A. LONG-ONLY top-decile (~250 names) vs equal-weight band benchmark, net of slippage:")
    bench = _eq_weight_benchmark(adj, eval_dates, band)
    bm = _metrics(bench)
    print(f"   {'benchmark (EW band)':28s} CAGR {bm['cagr']:7.2%}  Sharpe {bm['sharpe']:6.2f}  maxDD {bm['maxdd']:7.2%}")
    for slp in (5.0, 10.0, 15.0, 20.0):
        res = signal_portfolio_backtest(adj, signal, capital=1_000_000.0, n_hold=250,
                                        costs=USEquityCosts(slippage_bps=slp), members_asof=band)
        eqm = res.equity.reindex(eval_dates).ffill()
        sret = eqm.pct_change()
        m = _metrics(sret)
        active = (sret - bench.reindex(sret.index)).dropna()
        ir = float(active.mean() / active.std() * np.sqrt(12)) if active.std() > 0 else float("nan")
        print(f"   long-only @ slip {slp:4.0f}bps        CAGR {m['cagr']:7.2%}  Sharpe {m['sharpe']:6.2f}  "
              f"maxDD {m['maxdd']:7.2%}  active-IR {ir:5.2f}")

    # --- B. market-neutral q0.10 cost-sensitivity ---
    print("\nB. MARKET-NEUTRAL q0.10 net Sharpe across (slippage x borrow/yr) — breakeven map:")
    borrows = [0, 50, 150, 300]
    print("   slip\\borrow " + "".join(f"{b:>8d}" for b in borrows))
    for slp in (3.0, 5.0, 10.0, 15.0, 20.0):
        cells = []
        for brw in borrows:
            r = quantile_long_short(adj, signal, eval_dates, band, quantile=0.10,
                                    slippage_bps=slp, borrow_bps_annual=float(brw))
            cells.append(r.sharpe)
        print(f"   {slp:4.0f}bps     " + "".join(f"{c:8.2f}" for c in cells))

    # --- C. liquidity tiers (tighter, more-liquid cap-rank bands) ---
    print("\nC. LIQUIDITY TIERS — same signal, market-neutral q0.10:")
    print(f"   {'tier (cap rank)':22s} {'low 5/50':>10s} {'mid 10/150':>11s} {'real 15/300':>12s}")
    tiers = [("501-1000", 500, 500), ("501-1500", 500, 1000), ("501-2500", 500, 2000),
             ("1501-3000", 1500, 1500)]
    for label, ex, bs in tiers:
        m = crsp.size_band_members_asof(cap, exclude_top=ex, band_size=bs)
        s = []
        for slp, brw in [(5.0, 50.0), (10.0, 150.0), (15.0, 300.0)]:
            r = quantile_long_short(adj, signal, eval_dates, m, quantile=0.10,
                                    slippage_bps=slp, borrow_bps_annual=brw)
            s.append(r.sharpe)
        print(f"   {label:22s} {s[0]:10.2f} {s[1]:11.2f} {s[2]:12.2f}")

    print("\n[OK] deployability check on the cached GRU ensemble signal (survivorship-free, "
          "cost-aware). See docs/ml_zoo_study.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
