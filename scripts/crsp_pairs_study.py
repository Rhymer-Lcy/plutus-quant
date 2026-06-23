"""Distance-method pairs trading on the survivorship-free large-cap CRSP lake — a structurally
different bet from the (OOS-failed) monthly cross-sectional alpha: daily mean-reversion between
co-moving liquid pairs, capacity-limited (the retail-advantage zone).

Walk-forward (formation 252d -> trading 126d, non-overlapping), top-K pairs by SSD, open at
entry_z formation-std, close on reversion. Net of per-leg slippage + short borrow. Reported with a
PER-YEAR breakdown up front (the OOS lesson: see decay if it is there) and a small robustness sweep.

    conda activate plutus
    python scripts/crsp_pairs_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR
from plutus.research.backtest.pairs import distance_pairs_backtest, cointegration_pairs_backtest

PPY = 252.0


def _year_stats(returns: pd.Series) -> pd.DataFrame:
    rows = []
    for y, r in returns.groupby(returns.index.year):
        if len(r) < 5:
            continue
        ann = (1 + r).prod() ** (PPY / len(r)) - 1
        shp = r.mean() / r.std() * np.sqrt(PPY) if r.std() > 0 else float("nan")
        rows.append({"year": int(y), "days": len(r), "ann_return": float(ann), "sharpe": float(shp)})
    return pd.DataFrame(rows)


def main() -> int:
    price = pd.read_parquet(PARQUET_DIR / "crsp_adj_close.parquet")
    print(f"large-cap lake: {price.shape[1]} names, {price.shape[0]} days, "
          f"{price.index.min().date()} -> {price.index.max().date()}")

    costs = [("low 2/30", 2.0, 30.0), ("realistic 5/50", 5.0, 50.0), ("high 10/100", 10.0, 100.0)]
    print("\n===== headline: formation 252d, trading 126d, top_k 20, entry_z 2.0 =====")
    print(f"{'costs':>16s} {'annRet':>8s} {'Sharpe':>7s} {'MDD':>7s} {'pairs':>6s} "
          f"{'trips':>6s} {'in-mkt':>7s}")
    base = None
    for label, slp, brw in costs:
        r = distance_pairs_backtest(price, formation=252, trading=126, step=126, top_k=20,
                                    entry_z=2.0, slippage_bps=slp, borrow_bps_annual=brw)
        print(f"{label:>16s} {r.ann_return:8.2%} {r.sharpe:7.2f} {r.max_drawdown:7.1%} "
              f"{r.avg_pairs_traded:6.1f} {r.trades_per_pair:6.2f} {r.avg_days_in_market:7.1%}")
        if label == "realistic 5/50":
            base = r

    print("\n===== per-year (realistic 5/50) — is there decay? =====")
    ys = _year_stats(base.returns)
    print(f"{'year':>6s} {'days':>5s} {'annRet':>8s} {'Sharpe':>7s}")
    for _, row in ys.iterrows():
        print(f"{int(row.year):>6d} {int(row.days):>5d} {row.ann_return:>8.2%} {row.sharpe:>7.2f}")
    pre, post = ys[ys.year <= 2014], ys[ys.year >= 2015]
    print(f"\n  2005-2014 mean yearly Sharpe {pre.sharpe.mean():.2f}  |  "
          f"2015-2024 mean yearly Sharpe {post.sharpe.mean():.2f}")

    print("\n===== robustness sweep (realistic 5/50) =====")
    print(f"{'top_k':>6s} {'entry_z':>8s} {'annRet':>8s} {'Sharpe':>7s} {'in-mkt':>7s}")
    rows = []
    for top_k in (10, 20, 50):
        for ez in (1.5, 2.0, 2.5):
            r = distance_pairs_backtest(price, formation=252, trading=126, step=126, top_k=top_k,
                                        entry_z=ez, slippage_bps=5.0, borrow_bps_annual=50.0)
            rows.append({"top_k": top_k, "entry_z": ez, "ann_return": r.ann_return,
                         "sharpe": r.sharpe, "in_mkt": r.avg_days_in_market})
            print(f"{top_k:>6d} {ez:>8.1f} {r.ann_return:8.2%} {r.sharpe:7.2f} "
                  f"{r.avg_days_in_market:7.1%}")
    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "crsp_pairs_sweep.parquet")
    atomic_to_parquet(base.returns.rename("ret").to_frame(),
                      BACKTESTS_DIR / "crsp_pairs_returns.parquet")

    # ---- cointegration method (Engle-Granger) — a distinct selection, second opinion ----------
    print("\n===== COINTEGRATION method (Engle-Granger, ADF<-2.86), formation 252d, trading 126d, "
          "top_k 20, entry_z 2.0 =====")
    print(f"{'costs':>16s} {'annRet':>8s} {'Sharpe':>7s} {'MDD':>7s} {'pairs':>6s} "
          f"{'trips':>6s} {'in-mkt':>7s}")
    coint_base = None
    for label, slp, brw in costs:
        r = cointegration_pairs_backtest(price, formation=252, trading=126, step=126, top_k=20,
                                         entry_z=2.0, slippage_bps=slp, borrow_bps_annual=brw)
        print(f"{label:>16s} {r.ann_return:8.2%} {r.sharpe:7.2f} {r.max_drawdown:7.1%} "
              f"{r.avg_pairs_traded:6.1f} {r.trades_per_pair:6.2f} {r.avg_days_in_market:7.1%}")
        if label == "realistic 5/50":
            coint_base = r
    print("\n===== cointegration per-year (realistic 5/50) =====")
    cys = _year_stats(coint_base.returns)
    print(f"{'year':>6s} {'days':>5s} {'annRet':>8s} {'Sharpe':>7s}")
    for _, row in cys.iterrows():
        print(f"{int(row.year):>6d} {int(row.days):>5d} {row.ann_return:>8.2%} {row.sharpe:>7.2f}")
    cpre, cpost = cys[cys.year <= 2014], cys[cys.year >= 2015]
    print(f"\n  2005-2014 mean yearly Sharpe {cpre.sharpe.mean():.2f}  |  "
          f"2015-2024 mean yearly Sharpe {cpost.sharpe.mean():.2f}")
    atomic_to_parquet(coint_base.returns.rename("ret").to_frame(),
                      BACKTESTS_DIR / "crsp_pairs_coint_returns.parquet")

    print("\n[OK] distance + cointegration pairs, survivorship-free, net of costs. "
          "See docs/pairs_study.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
