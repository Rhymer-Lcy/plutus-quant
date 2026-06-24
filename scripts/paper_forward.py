"""Free-data EARLY READ on the deployed net-payout book (live.forward).

Selects the top-50 from clean CRSP at the last lake bar, prices it forward with yfinance from the
inception to today, and benchmarks vs small-cap ETFs. This is a thin, statistically-early read on
the SELECTION (net-payout is low-turnover, so a held book approximates the strategy between CRSP
refreshes) -- NOT the definitive monthly-rebalanced CRSP test (scripts/paper_live.py). Needs
network (yfinance).

    conda activate plutus
    python scripts/paper_forward.py
    python scripts/paper_forward.py --seed 1000000 --end 2026-06-24
"""
from __future__ import annotations

import argparse

from plutus.live.forward import run_forward
from plutus.live.strategy import DEPLOYED, PAPER_INCEPTION


def main() -> int:
    ap = argparse.ArgumentParser(description="Free-data early read on the deployed net-payout book.")
    ap.add_argument("--seed", type=float, default=1_000_000.0, help="seed capital (USD)")
    ap.add_argument("--inception", default=PAPER_INCEPTION, help="forward-record start (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="forward-record end (YYYY-MM-DD), default today")
    args = ap.parse_args()

    d = DEPLOYED
    print(f"deployed = net-payout({d.lookback}d) top-{d.n_hold}, band & ADV>${d.adv_min / 1e6:.0f}M/d")
    print("mode: FREE-DATA early read (CRSP selects, yfinance prices forward) -- selection-only, thin sample\n")
    try:
        r = run_forward(args.seed, inception=args.inception, end=args.end)
    except Exception:
        import traceback
        traceback.print_exc()
        print("\n[transient] run failed (most likely a yfinance/network issue). Exiting 75 so the "
              "scheduler retries; the recompute is idempotent (no state is lost).")
        return 75

    print(f"  selection @ {r['selection_asof']} (CRSP)  ->  forward {r['inception']}..{r['as_of']} "
          f"({r['n_bars']} bars, yfinance)")
    print(f"  book: {r['n_priced']}/{r['n_book']} priced "
          f"({r['n_mapped']} mapped; {r['n_corporate_actions']} corporate-action resolutions; "
          f"unresolved {r['unresolved_tickers']})")
    acts = {t: v for t, v in r["resolution"].items() if v not in ("direct", "unresolved")}
    for t, v in sorted(acts.items()):
        print(f"      {t:6s} -> {v}")
    sh = "n/a" if r["ann_sharpe"] is None else f"{r['ann_sharpe']:.2f}"
    print(f"\n  {'book':<22} totRet {r['total_return']:>+7.1%}   maxDD {r['max_drawdown']:>+7.1%}   "
          f"Sharpe {sh:>6}")
    for b, st in r["benchmarks"].items():
        bsh = "n/a" if st["ann_sharpe"] is None else f"{st['ann_sharpe']:.2f}"
        edge = r["total_return"] - st["total_return"]
        print(f"  {('benchmark ' + b):<22} totRet {st['total_return']:>+7.1%}   "
              f"{'':<14} Sharpe {bsh:>6}   (book vs {b}: {edge:+.1%})")

    print("\n  NOTE: short OOS sample -> read DIRECTION not magnitude; selection-only (no forward "
          "rebalance). Definitive test = the monthly-rebalanced CRSP ledger (paper_live.py) when "
          "fresh CRSP lands. Saved under results/paper/forward_*.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
