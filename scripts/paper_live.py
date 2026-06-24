"""Driver: replay the DEPLOYED net-payout strategy as a forward PAPER account at each capital tier
and persist the ledger + report. Re-run safe (idempotent recompute-from-seed).

Unlike the sibling hermes-quant driver (which auto-refreshes from free BaoStock), this does NOT
refresh data: CRSP is a PAID, manual pull with no free daily feed. It replays whatever is on disk.
Advance the forward record by landing a fresh CRSP pull (the build scripts), then re-running this.
Until a bar at/after the inception lands, every tier reports status="awaiting_data" -- the honest
state, not a fall-back to the in-sample backtest.

    conda activate plutus
    python scripts/paper_live.py                       # paper mode, all tiers, spec inception
    python scripts/paper_live.py --tiers 100000 1000000
    python scripts/paper_live.py --as-of 2026-03-31    # cut the lake at a date (idempotent re-run)
    python scripts/paper_live.py --backtest            # archive the full-history backtest instead
"""
from __future__ import annotations

import argparse

from plutus.live.paper import live_step, load_panels
from plutus.live.strategy import ALL_TIERS, DEPLOYED, PAPER_INCEPTION, TIER_LABEL


def main() -> int:
    ap = argparse.ArgumentParser(description="Forward paper-trade the deployed net-payout strategy.")
    ap.add_argument("--tiers", type=int, nargs="+", default=ALL_TIERS, help="capital tiers (USD)")
    ap.add_argument("--as-of", default=None, help="cut the lake at this date (YYYY-MM-DD)")
    ap.add_argument("--inception", default=PAPER_INCEPTION, help="forward-record start (YYYY-MM-DD)")
    ap.add_argument("--backtest", action="store_true",
                    help="reproduce the full-history backtest (inception=None) instead of paper mode")
    args = ap.parse_args()
    inception = None if args.backtest else args.inception

    d = DEPLOYED
    print(f"deployed = net-payout({d.lookback}d) top-{d.n_hold}, monthly, band & "
          f"ADV>${d.adv_min / 1e6:.0f}M/d, {d.slippage_bps:.0f}bps")
    print("mode: " + ("full-history BACKTEST" if args.backtest
                      else f"PAPER forward record (inception {inception})"))
    print(f"  {'tier':>13} {'label':>6} {'status':>13} {'as_of':>11} {'n_bars':>6} "
          f"{'equity':>15} {'totRet':>8} {'maxDD':>7} {'Sharpe':>7} {'vsB&H':>8} {'pos':>4}")

    panels = load_panels()          # ~0.5GB each; load once, reuse across tiers
    last = None
    for cap in sorted(args.tiers):
        r = live_step(float(cap), inception=inception, as_of=args.as_of, panels=panels)
        last = r
        sh = "n/a" if r["ann_sharpe"] is None else f"{r['ann_sharpe']:.2f}"
        vs = ("n/a" if r["status"] == "awaiting_data" or r["bh_total_return"] is None
              else f"{r['total_return'] - r['bh_total_return']:+.1%}")
        print(f"  {cap:>13,} {TIER_LABEL.get(cap, '?'):>6} {r['status']:>13} {r['as_of']:>11} "
              f"{r['n_bars']:>6} {r['equity']:>15,.0f} {r['total_return']:>+8.1%} "
              f"{r['max_drawdown']:>+7.1%} {sh:>7} {vs:>8} {r['n_positions']:>4}")

    if last and last["status"] == "awaiting_data":
        print(f"\n  {last['note']}")
    print(f"\nsaved under results/{'backtests' if args.backtest else 'paper'}/. "
          "Re-run safe (recompute-from-seed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
