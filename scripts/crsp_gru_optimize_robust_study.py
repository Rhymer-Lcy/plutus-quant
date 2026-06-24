"""Robustness of the turnover-aware-optimizer result: is the realistic-cost Sharpe ~0.46 a
knife-edge on the per-name cap, or stable? Sweep name_cap (and a couple gammas) at realistic
small-cap cost on the cached GRU signal. If Sharpe stays clearly positive across caps, the
concentration gain is real, not a fitted artifact."""
from __future__ import annotations

import pandas as pd

from plutus.data.sources import crsp_source as crsp
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR
from plutus.research.backtest.optimize import turnover_aware_backtest
from plutus.research.backtest.metrics import month_ends


def main() -> int:
    adj = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_adj_close.parquet")
    cap = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_mktcap.parquet")
    signal = pd.read_parquet(BACKTESTS_DIR / "crsp_dl_smallcap_gru_signal.parquet")
    eval_dates = month_ends(adj.index)
    signal = signal.reindex(eval_dates)
    band = crsp.size_band_members_asof(cap)

    print("Realistic cost (15bps slip + 300bps borrow) — name_cap x gamma robustness:")
    print(f"{'name_cap':>9s} {'gamma':>6s} {'annRet':>8s} {'Sharpe':>7s} {'turn':>6s} {'gross':>6s} {'~names':>7s}")
    for cap_i in (0.01, 0.02, 0.03, 0.05, 0.10):
        for gamma in (0.0, 2.0):
            r = turnover_aware_backtest(adj, signal, eval_dates, band, gamma=gamma,
                                        slippage_bps=15.0, borrow_bps_annual=300.0,
                                        name_cap=cap_i, gross=2.0, cand_frac=0.3)
            print(f"{cap_i:9.2f} {gamma:6.0f} {r.ann_return:8.2%} {r.sharpe:7.2f} "
                  f"{r.avg_turnover:6.2f} {r.avg_gross:6.2f} {2.0/cap_i:7.0f}")
    print("\n[OK] name_cap robustness (~names = gross/cap = book breadth at the cap).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
