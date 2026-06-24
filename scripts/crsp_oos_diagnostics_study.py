"""Adversarial diagnostics for the 2025 OOS failure: is it a DATA artifact or real alpha decay?

The 2025 holdout rank IC came back NEGATIVE (-0.021) while 2005-2024 was +0.019 (t=3.0). Before
calling that overfit/decay, rule out the boring explanations:
  - did the size-band universe shrink in 2025? (fewer tradable names -> noisier IC)
  - did IBES/analyst-revision COVERAGE collapse in 2025? (3 of 44 features go stale -> covariate
    shift; crsp_dl fillna(0)s missing revisions, so thin coverage silently zeros those inputs)
  - did the GRU signal's cross-sectional DISPERSION collapse in 2025? (degenerate predictions)

This reports per-year, so 2025 can be compared to the design period on each axis.

    conda activate plutus
    python scripts/crsp_oos_diagnostics_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from plutus.data.sources import crsp_source as crsp
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR
from plutus.research.backtest.metrics import month_ends


def main() -> int:
    adj = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_adj_close.parquet")
    cap = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_mktcap.parquet")
    rev1 = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_rev1.parquet")   # RAW (pre-fillna)
    signal = pd.read_parquet(BACKTESTS_DIR / "crsp_dl_smallcap_gru_signal.parquet")
    eval_dates = month_ends(adj.index)
    band = crsp.size_band_members_asof(cap)

    adj_m = adj.reindex(eval_dates)
    rev1_m = rev1.reindex(index=eval_dates, columns=adj.columns)
    sig = signal.reindex(eval_dates)

    rows = []
    for t in eval_dates:
        members = band(t)
        if not members:
            continue
        cols = [c for c in adj.columns if c in members]
        price_ok = int(adj_m.loc[t, cols].notna().sum())
        rev_cov = float(rev1_m.loc[t, cols].notna().mean()) if cols else np.nan   # frac w/ live revision
        s = sig.loc[t].reindex(cols).dropna() if t in sig.index else pd.Series(dtype=float)
        rows.append({"date": t, "year": t.year, "band_n": len(members), "price_ok": price_ok,
                     "rev_cov": rev_cov, "sig_n": len(s),
                     "sig_disp": float(s.std()) if len(s) > 1 else np.nan})
    df = pd.DataFrame(rows)

    print("Per-year diagnostics (mean over month-ends): is 2025 degraded on DATA grounds?")
    print(f"{'year':>6s} {'band_n':>7s} {'price_ok':>9s} {'rev_cov':>8s} {'sig_n':>7s} {'sig_disp':>9s}")
    g = df.groupby("year").agg(band_n=("band_n", "mean"), price_ok=("price_ok", "mean"),
                               rev_cov=("rev_cov", "mean"), sig_n=("sig_n", "mean"),
                               sig_disp=("sig_disp", "mean"))
    for yr, r in g.iterrows():
        flag = "  <== OOS" if yr == 2025 else ""
        print(f"{int(yr):>6d} {r.band_n:>7.0f} {r.price_ok:>9.0f} {r.rev_cov:>8.2%} "
              f"{r.sig_n:>7.0f} {r.sig_disp:>9.4f}{flag}")

    prior = g[g.index < 2025]
    h = g.loc[2025] if 2025 in g.index else None
    if h is not None:
        print("\n2025 vs design-period mean:")
        for col in ["band_n", "price_ok", "rev_cov", "sig_n", "sig_disp"]:
            pv = prior[col].mean()
            fmt = (lambda x: f"{x:.2%}") if col == "rev_cov" else (lambda x: f"{x:.4f}" if col == "sig_disp" else f"{x:.0f}")
            ratio = (h[col] / pv) if pv else float("nan")
            print(f"  {col:>9s}: 2025 {fmt(h[col])}  vs  prior {fmt(pv)}   ({ratio:.0%} of prior)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
