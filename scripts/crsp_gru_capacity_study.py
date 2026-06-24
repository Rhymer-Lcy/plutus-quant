"""Capacity study: at what AUM does the optimized GRU market-neutral edge survive market impact?

The Phase-6 Sharpe (~0.46 net of flat costs) ignores market impact. Here the per-name trade cost
grows with participation = traded$ / ADV$ (Almgren square-root law, research.backtest.optimize),
so running the same optimized book at each capital tier (live.strategy.CAPITAL_TIERS) and reading
the net Sharpe gives the CAPACITY curve — the AUM where moving the mid/small book erodes the edge.
ADV = 21-day mean CRSP dollar volume. (impact_coef is an assumption; the curve's SHAPE is the point.)

    conda activate plutus
    python scripts/crsp_gru_capacity_study.py
"""
from __future__ import annotations

import pandas as pd

from plutus.data.sources import crsp_source as crsp
from plutus.io import atomic_to_parquet
from plutus.live.strategy import CAPITAL_TIERS, TIER_LABEL
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR
from plutus.research.backtest.optimize import turnover_aware_backtest
from plutus.research.backtest.metrics import month_ends


def main() -> int:
    adj = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_adj_close.parquet")
    cap = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_mktcap.parquet")
    dvol = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_dollarvol.parquet")
    signal = pd.read_parquet(BACKTESTS_DIR / "crsp_dl_smallcap_gru_signal.parquet")
    eval_dates = month_ends(adj.index)
    signal = signal.reindex(eval_dates)
    band = crsp.size_band_members_asof(cap)
    adv = dvol.rolling(21, min_periods=5).mean().reindex(eval_dates)   # ADV ($) at each month-end

    aums = sorted({v for tier in CAPITAL_TIERS.values() for v in tier})
    print("Optimized GRU market-neutral book vs AUM, with square-root market impact")
    print("(base slip 5bps + borrow 300bps/yr + size impact; gamma=2, name_cap=0.02):\n")
    print(f"{'AUM':>14s} {'tier':>7s} {'annRet':>8s} {'Sharpe':>7s} {'turn':>6s}")
    rows = []
    for aum in aums:
        r = turnover_aware_backtest(adj, signal, eval_dates, band, gamma=2.0, slippage_bps=5.0,
                                    borrow_bps_annual=300.0, name_cap=0.02, gross=2.0,
                                    cand_frac=0.3, aum=float(aum), adv=adv, impact_coef=0.01)
        rows.append({"aum": aum, "tier": TIER_LABEL.get(aum, "?"), "ann_return": r.ann_return,
                     "sharpe": r.sharpe, "turnover": r.avg_turnover})
        print(f"{aum:14,.0f} {TIER_LABEL.get(aum,'?'):>7s} {r.ann_return:8.2%} {r.sharpe:7.2f} "
              f"{r.avg_turnover:6.2f}")
    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "crsp_dl_capacity.parquet")
    print("\n[OK] capacity curve (Sharpe vs AUM under market impact). See docs/ml_zoo_study.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
