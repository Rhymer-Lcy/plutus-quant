"""Does turnover-aware optimization make the GRU edge more tradeable?

The naive quintile long-short on the GRU signal turns over ~270%/mo, which eats the (real) edge
at realistic cost. Run the same cached GRU signal through the turnover-aware optimizer
(research.backtest.optimize) sweeping the turnover-aversion gamma, at low and realistic cost, and
compare net Sharpe to the naive quintile baseline. (gamma swept in-sample -> the peak is
optimistic; shows whether the lever helps at all.)

    conda activate plutus
    python scripts/crsp_gru_optimize_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from plutus.data.sources import crsp_source as crsp
from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR
from plutus.research.backtest.long_short import quantile_long_short
from plutus.research.backtest.optimize import turnover_aware_backtest

from crsp_study import _month_ends


def main() -> int:
    adj = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_adj_close.parquet")
    cap = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_mktcap.parquet")
    signal = pd.read_parquet(BACKTESTS_DIR / "crsp_dl_smallcap_gru_signal.parquet")
    eval_dates = _month_ends(adj.index)
    signal = signal.reindex(eval_dates)
    band = crsp.size_band_members_asof(cap, exclude_top=500, band_size=2500)

    costs = [("low 5/50", 5.0, 50.0), ("realistic 15/300", 15.0, 300.0)]
    print("Baseline: naive quintile (q0.10) long-short on the GRU signal")
    print(f"{'costs':>16s} {'annRet':>8s} {'Sharpe':>7s} {'turn':>6s}")
    for label, slp, brw in costs:
        r = quantile_long_short(adj, signal, eval_dates, band, quantile=0.10,
                                slippage_bps=slp, borrow_bps_annual=brw)
        print(f"{label:>16s} {r.ann_return:8.2%} {r.sharpe:7.2f} {r.avg_turnover:6.2f}")

    print("\nTurnover-aware optimizer (gamma = turnover-aversion; gamma=0 ~ naive):")
    print(f"{'costs':>16s} {'gamma':>6s} {'annRet':>8s} {'Sharpe':>7s} {'turn':>6s} {'gross':>6s}")
    rows = []
    for label, slp, brw in costs:
        for gamma in (0.0, 2.0, 5.0, 15.0, 40.0):
            r = turnover_aware_backtest(adj, signal, eval_dates, band, gamma=gamma,
                                        slippage_bps=slp, borrow_bps_annual=brw,
                                        name_cap=0.02, gross=2.0, cand_frac=0.3)
            rows.append({"costs": label, "gamma": gamma, "ann_return": r.ann_return,
                         "sharpe": r.sharpe, "turnover": r.avg_turnover, "gross": r.avg_gross})
            print(f"{label:>16s} {gamma:6.0f} {r.ann_return:8.2%} {r.sharpe:7.2f} "
                  f"{r.avg_turnover:6.2f} {r.avg_gross:6.2f}")
    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "crsp_dl_optimize.parquet")
    print("\n[OK] turnover-aware optimization on the GRU signal (in-sample gamma sweep). "
          "See docs/ml_zoo_study.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
