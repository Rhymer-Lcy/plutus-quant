"""Anti-skew parity gate + headline reproduction for the deployed net-payout paper strategy.

Two guarantees, on REAL CRSP data, before any forward record is trusted:
  1. PARITY -- the idempotent ledger (live.paper.replay -> live.ledger.fold_day) reproduces the
     research engine's equity curve BAR FOR BAR. The ledger's only job is to record the engine's
     decisions, never to re-decide; if these diverge, paper trading has trained/served different
     strategies (the dominant silent alpha-killer).
  2. REPRODUCTION -- the single-source deployed spec (live.strategy) reproduces the headline of
     docs/issuance_study.md (small-cap net-payout, top-50, ADV>$5M/day, 15bps, seed $1M): CAGR
     +23.7%, maxDD -32.5%, Sharpe 1.14. If the deployed signal/universe ever drift from the study,
     this fails loudly.

Backtest mode (inception=None) is used here so the curve spans the full 2005-2025 sample and can
be checked against the documented numbers. Exit code is non-zero on any failure (CI-friendly).

    conda activate plutus
    python scripts/paper_dryrun.py
"""
from __future__ import annotations

import numpy as np

from plutus.live.paper import ledger_equity, load_panels, paper_account
from plutus.live.strategy import DEPLOYED

# Documented headline (docs/issuance_study.md, ADV>$5M/day @15bps, top-50, seed $1M):
EXPECT = {"cagr": 0.237, "maxdd": -0.325, "sharpe": 1.14}
TOL = {"cagr": 0.005, "maxdd": 0.010, "sharpe": 0.03}


def main() -> int:
    adj, cap, dv = load_panels()
    d = DEPLOYED
    print(f"deployed = net-payout({d.lookback}d) top-{d.n_hold}, monthly, band[{d.exclude_top},"
          f"{d.exclude_top + d.band_size}) & ADV>${d.adv_min / 1e6:.0f}M/d, {d.slippage_bps:.0f}bps")
    print("mode: full-history BACKTEST (inception=None) -- parity gate + headline reproduction\n")

    ok = True
    for seed in (1_000_000.0, 100_000.0):
        ledger, res, report = paper_account(adj, cap, dv, seed, inception=None)
        led = ledger_equity(ledger).reindex(res.equity.index)
        gap = float(np.abs(led.values - res.equity.values).max())
        rel = gap / float(np.abs(res.equity.values).max())
        parity = rel < 1e-6
        ok = ok and parity
        print(f"  seed ${seed:>12,.0f}: parity max|ledger-engine| = {gap:.6g} "
              f"(rel {rel:.2e}) -> {'OK' if parity else 'FAIL'}")
        if parity and seed == 1_000_000.0:
            got = {"cagr": res.cagr, "maxdd": res.max_drawdown, "sharpe": report["ann_sharpe"]}
            print("\n  headline reproduction @ $1M (vs docs/issuance_study.md):")
            for k in ("cagr", "maxdd", "sharpe"):
                hit = abs(got[k] - EXPECT[k]) <= TOL[k]
                ok = ok and hit
                fmt = (lambda v: f"{v:.2f}") if k == "sharpe" else (lambda v: f"{v:+.1%}")
                print(f"    {k:8s} got {fmt(got[k]):>8}  expect {fmt(EXPECT[k]):>8}  "
                      f"-> {'OK' if hit else 'MISMATCH'}")
            print()

    print("ALL OK -- ledger == engine and the deployed spec reproduces the headline."
          if ok else "FAILED -- see mismatches above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
