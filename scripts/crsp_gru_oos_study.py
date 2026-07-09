"""Out-of-sample capstone: does the GRU market-neutral edge hold in 2025 — a year that did NOT
exist when the whole program (features, architecture, optimizer config, capital tiers) was designed?

The entire design was chosen on 2005-2024. The honest test of program-level overfitting is to
extend the lake to 2025, re-run the LOCKED pipeline (the GRU walk-forward naturally trains on
<=t and predicts t, so 2025 predictions are made by models that never saw 2025), and then look
*only* at 2025 — both the rank IC and the optimized book's net Sharpe.

A single year is ~11-12 monthly observations: noisy. So 2025 is not judged against a point
estimate but placed in the DISTRIBUTION of every prior year's IC / Sharpe — if 2025 sits inside
the historical spread (not a negative outlier), the edge survives the only genuinely unseen data
we have. The design is frozen here: gamma=2, name_cap=0.02, gross=2.0 — the exact capacity config.

    conda activate plutus
    python scripts/crsp_gru_oos_study.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from plutus.data.sources import crsp_source as crsp
from plutus.io import atomic_to_parquet
from plutus.live.strategy import CAPITAL_TIERS, TIER_LABEL
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR
from plutus.research.backtest.optimize import turnover_aware_backtest
from plutus.research.eval.factor_eval import compute_ic

from plutus.research.backtest.metrics import month_ends

HOLDOUT = 2025          # the year added AFTER the whole design was frozen
PPY = 12                # monthly eval


def _slice_stats(returns: pd.Series, ppy: int = PPY) -> dict:
    """Annualized stats for a monthly net-return slice (Sharpe = mean/std * sqrt(12))."""
    r = returns.dropna()
    if len(r) < 2:
        return {"n": len(r), "ann_return": float("nan"), "sharpe": float("nan"),
                "cum": float((1 + r).prod() - 1) if len(r) else float("nan")}
    ann_ret = float((1 + r).prod() ** (ppy / len(r)) - 1)
    sharpe = float(r.mean() / r.std() * np.sqrt(ppy)) if r.std() > 0 else float("nan")
    return {"n": len(r), "ann_return": ann_ret, "sharpe": sharpe, "cum": float((1 + r).prod() - 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--signal", default="crsp_dl_smallcap_gru_signal.parquet",
                    help="signal parquet under BACKTESTS_DIR (default: the locked revision-augmented GRU)")
    args = ap.parse_args()
    adj = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_adj_close.parquet")
    cap = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_mktcap.parquet")
    dvol = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_dollarvol.parquet")
    signal = pd.read_parquet(BACKTESTS_DIR / args.signal)
    print(f"signal file: {args.signal}")
    eval_dates = month_ends(adj.index)
    band = crsp.size_band_members_asof(cap)
    sig = signal.reindex(eval_dates)
    adv = dvol.rolling(21, min_periods=5).mean().reindex(eval_dates)

    last = pd.DatetimeIndex(eval_dates).max().date()
    n_2025 = int((pd.DatetimeIndex(signal.index).year == HOLDOUT).sum())
    print(f"lake/eval span ends {last}; GRU signal has {n_2025} month-ends in {HOLDOUT} "
          f"(the holdout year).")
    if n_2025 == 0:
        print("No 2025 signal months — rebuild the lake to 2025 first. Aborting.")
        return 1

    # ---- (1) rank IC by year; 2025 is the genuine out-of-sample year -------------------------
    icr = compute_ic(sig, adj, eval_dates, band)
    ic = icr.ic
    by_year = ic.groupby(ic.index.year)
    print("\n===== rank IC by year (2025 = out-of-sample) =====")
    print(f"{'year':>6s} {'n':>4s} {'meanIC':>8s} {'IC-IR':>7s} {'t':>6s} {'hit':>5s}")
    yearly_ic = {}
    for yr, s in by_year:
        ir = s.mean() / s.std() if s.std() > 0 else float("nan")
        t = ir * np.sqrt(len(s)) if np.isfinite(ir) else float("nan")
        yearly_ic[int(yr)] = {"n": len(s), "mean_ic": float(s.mean()), "ic_ir": float(ir),
                              "t": float(t), "hit": float((s > 0).mean())}
        flag = "  <== OOS" if int(yr) == HOLDOUT else ""
        print(f"{int(yr):>6d} {len(s):>4d} {s.mean():>8.4f} {ir:>7.3f} {t:>6.2f} "
              f"{(s > 0).mean():>5.2f}{flag}")

    prior = ic[ic.index.year < HOLDOUT]
    oos = ic[ic.index.year == HOLDOUT]
    yr_means = pd.Series({y: v["mean_ic"] for y, v in yearly_ic.items() if y < HOLDOUT})
    pct = float((yr_means < yearly_ic[HOLDOUT]["mean_ic"]).mean()) if len(yr_means) else float("nan")
    print(f"\ndesign period (<{HOLDOUT}): meanIC {prior.mean():.4f}  t {prior.mean()/prior.std()*np.sqrt(len(prior)):.2f}  n {len(prior)}")
    print(f"holdout  ({HOLDOUT})    : meanIC {oos.mean():.4f}  t {oos.mean()/oos.std()*np.sqrt(len(oos)):.2f}  n {len(oos)}")
    print(f"  2025 mean IC sits at the {pct:.0%} percentile of prior yearly mean-ICs "
          f"(prior range {yr_means.min():.4f}..{yr_means.max():.4f}).")

    # ---- (2) optimized book net Sharpe by year (LOCKED config, warmed over full history) -----
    print("\n===== optimized book (gamma=2, name_cap=0.02) net Sharpe by year =====")
    print("(full-history walk so weights are warm by 2025; returns then sliced by year)")
    for label, slp, brw in [("low 5/50", 5.0, 50.0), ("realistic 15/300", 15.0, 300.0)]:
        res = turnover_aware_backtest(adj, sig, eval_dates, band, gamma=2.0, slippage_bps=slp,
                                      borrow_bps_annual=brw, name_cap=0.02, gross=2.0, cand_frac=0.3)
        rr = res.returns
        ann = {int(y): _slice_stats(s) for y, s in rr.groupby(rr.index.year)}
        print(f"\n-- cost {label} --")
        print(f"{'year':>6s} {'n':>4s} {'annRet':>8s} {'Sharpe':>7s}")
        for y in sorted(ann):
            flag = "  <== OOS" if y == HOLDOUT else ""
            print(f"{y:>6d} {ann[y]['n']:>4d} {ann[y]['ann_return']:>8.2%} {ann[y]['sharpe']:>7.2f}{flag}")
        ysharpe = pd.Series({y: ann[y]["sharpe"] for y in ann if y < HOLDOUT})
        s25 = ann.get(HOLDOUT, {}).get("sharpe", float("nan"))
        p = float((ysharpe < s25).mean()) if len(ysharpe) else float("nan")
        print(f"   2025 Sharpe {s25:.2f} -> {p:.0%} percentile of prior yearly Sharpes "
              f"(prior median {ysharpe.median():.2f}, range {ysharpe.min():.2f}..{ysharpe.max():.2f})")

    # ---- (3) 2025-only capacity curve vs full-sample -----------------------------------------
    print("\n===== capacity curve: 2025 (OOS) vs full sample, with market impact =====")
    print(f"{'AUM':>14s} {'tier':>7s} {'full Shrp':>10s} {'2025 Shrp':>10s} {'2025 annRet':>12s}")
    aums = sorted({v for tier in CAPITAL_TIERS.values() for v in tier})
    rows = []
    for aum in aums:
        res = turnover_aware_backtest(adj, sig, eval_dates, band, gamma=2.0, slippage_bps=5.0,
                                      borrow_bps_annual=300.0, name_cap=0.02, gross=2.0,
                                      cand_frac=0.3, aum=float(aum), adv=adv, impact_coef=0.01)
        full = _slice_stats(res.returns)
        rr = res.returns
        s25 = _slice_stats(rr[rr.index.year == HOLDOUT])
        rows.append({"aum": aum, "tier": TIER_LABEL.get(aum, "?"), "full_sharpe": full["sharpe"],
                     "oos_sharpe": s25["sharpe"], "oos_ann_return": s25["ann_return"]})
        print(f"{aum:14,.0f} {TIER_LABEL.get(aum,'?'):>7s} {full['sharpe']:>10.2f} "
              f"{s25['sharpe']:>10.2f} {s25['ann_return']:>12.2%}")
    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "crsp_dl_oos_capacity.parquet")

    print("\n[OK] out-of-sample (2025 holdout) validation. See docs/ml_zoo_study.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
