"""Rigorous inference layer for the OOS result — does the data support 'the edge decayed', and how?

An adversarial audit of crsp_gru_oos_study.py found NO bug, but flagged that 'the 2025 holdout proves the
edge failed / is overfit' OVERSTATES a single n=11 year (one-sample t=-0.71, p=0.50; not even
distinguishable from the design-period mean). The decision-relevant evidence is instead the
MULTI-YEAR trailing fade: the edge was already ~statistical-zero by 2020-2024, before any holdout.

This script computes, from the cached locked GRU signal (no GRU re-run needed):
  1. pooled rank-IC + t-stat over diagnostic windows (2010-19, 2020-24, 2021-24, last-24m, 2025),
  2. the honest significance of 2025 alone: one-sample t/p, Welch vs design, 95% CI,
     and P(a yearly mean this low | the edge were fully intact) — the '~1-in-16' sanity check.

    conda activate plutus
    python scripts/crsp_oos_inference_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from plutus.data.sources import crsp_source as crsp
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR
from plutus.research.eval.factor_eval import compute_ic
from plutus.research.backtest.metrics import month_ends

HOLDOUT = 2025


def _pooled(ic: pd.Series, lo: int, hi: int) -> dict:
    """Pooled monthly-IC stats over calendar years [lo, hi] inclusive."""
    s = ic[(ic.index.year >= lo) & (ic.index.year <= hi)]
    n = len(s)
    if n < 2:
        return {"lo": lo, "hi": hi, "n": n, "mean": float("nan"), "t": float("nan"), "p": float("nan")}
    t, p = stats.ttest_1samp(s, 0.0)
    return {"lo": lo, "hi": hi, "n": n, "mean": float(s.mean()), "std": float(s.std()),
            "t": float(t), "p": float(p)}


def main() -> int:
    adj = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_adj_close.parquet")
    cap = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_mktcap.parquet")
    signal = pd.read_parquet(BACKTESTS_DIR / "crsp_dl_smallcap_gru_signal.parquet")
    eval_dates = month_ends(adj.index)
    band = crsp.size_band_members_asof(cap)
    ic = compute_ic(signal.reindex(eval_dates), adj, eval_dates, band).ic

    print("===== trailing-window IC decay (pooled monthly rank IC, two-sided t vs 0) =====")
    print(f"{'window':>14s} {'n':>4s} {'meanIC':>8s} {'t':>7s} {'p':>7s}")
    windows = [("design 2009-24", 2009, 2024), ("strong 2010-19", 2010, 2019),
               ("2015-19", 2015, 2019), ("2020-24", 2020, 2024), ("2021-24", 2021, 2024),
               ("last-24m 24-25", 2024, 2025), ("holdout 2025", HOLDOUT, HOLDOUT)]
    for label, lo, hi in windows:
        w = _pooled(ic, lo, hi)
        print(f"{label:>14s} {w['n']:>4d} {w['mean']:>8.4f} {w['t']:>7.2f} {w['p']:>7.3f}")

    # ---- honest significance of the single holdout year --------------------------------------
    oos = ic[ic.index.year == HOLDOUT]
    prior = ic[ic.index.year < HOLDOUT]
    n = len(oos)
    mean, sd = float(oos.mean()), float(oos.std())
    se = sd / np.sqrt(n)
    t1, p1 = stats.ttest_1samp(oos, 0.0)                       # is 2025 != 0 ?
    tw, pw = stats.ttest_ind(oos, prior, equal_var=False)      # is 2025 != design period ?
    tcrit = stats.t.ppf(0.975, df=n - 1)
    ci = (mean - tcrit * se, mean + tcrit * se)

    # P(an n-month yearly mean <= observed | edge fully intact at the design-period monthly law)
    mu0, sd0 = float(prior.mean()), float(prior.std())
    z = (mean - mu0) / (sd0 / np.sqrt(n))
    p_bad = float(stats.norm.cdf(z))
    yr_means = prior.groupby(prior.index.year).mean()
    n_prior_years = len(yr_means)

    print("\n===== is the 2025 holdout (n={}) statistically informative ON ITS OWN? =====".format(n))
    print(f"2025 mean IC {mean:+.4f}  (std {sd:.4f}, se {se:.4f})")
    print(f"  vs zero        : t {t1:+.2f}  p {p1:.3f}   -> {'NOT ' if p1 > 0.05 else ''}sig. different from 0")
    print(f"  vs design mean : Welch t {tw:+.2f}  p {pw:.3f}   -> {'NOT ' if pw > 0.05 else ''}sig. different from design (+{mu0:.4f})")
    print(f"  95% CI on 2025 mean: [{ci[0]:+.4f}, {ci[1]:+.4f}]   "
          f"({'CONTAINS' if ci[0] <= mu0 <= ci[1] else 'excludes'} the design mean +{mu0:.4f})")
    print(f"  P(a year >= this bad | edge fully intact) = {p_bad:.3f}  "
          f"-> ~1 in {1/p_bad:.0f}; expected {n_prior_years * p_bad:.1f} such years over the {n_prior_years} prior years")

    neg_prior_years = int((yr_means < 0).sum())
    print(f"\n  (context: the design period already had {neg_prior_years} negative years out of "
          f"{n_prior_years}: {sorted(int(y) for y in yr_means[yr_means < 0].index)})")

    print("\n[OK] OOS inference: the multi-year fade — not the single 2025 point — is the evidence. "
          "See docs/ml_zoo_study.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
