"""Event-TIME PEAD: does trading the drift on the event clock (enter the day after the earnings
filing, hold N days) capture what the calendar-monthly test missed?

Builds the earnings-surprise (SUE) events from SEC filings (PIT-filtered to S&P 500 membership
at the filing), then: (1) CAAR — the cumulative abnormal-return drift by SUE quintile in event
time, to see if/when the drift accrues (front-loaded?); (2) an overlapping dollar-neutral
long-short of fresh positive vs negative surprises at several hold horizons, net of costs.
Survivorship-free CRSP returns. Compare to docs/pead_study.md (monthly, which lost).

    conda activate plutus
    python scripts/crsp_pead_event_study.py        # needs SEC_EDGAR_USER_AGENT
"""
from __future__ import annotations

import argparse

import pandas as pd

from plutus.data.sources import crsp_source as crsp
from plutus.data.sources import sec_edgar as se
from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from plutus.research.backtest.event_study import event_caar, event_time_portfolio
from plutus.research.factors import events as ev


def build_events(permno_to_ticker: dict, members_asof) -> pd.DataFrame:
    """(permno, entry_date=filing, sue) events, PIT-filtered to S&P 500 membership at the filing."""
    cikmap = se.load_ticker_cik_map()
    rows = []
    for permno, ticker in permno_to_ticker.items():
        cik = cikmap.get(str(ticker).upper())
        if cik is None:
            continue
        try:
            facts = se.company_facts(cik)
        except Exception:
            continue
        sue = ev.standardized_unexpected_earnings(
            se.discrete_quarters(se.concept_frame(facts, "NetIncomeLoss")))
        for filed, s in zip(sue["filed"], sue["sue"]):
            if permno in members_asof(filed):
                rows.append({"permno": permno, "entry_date": pd.Timestamp(filed), "sue": float(s)})
    return pd.DataFrame(rows)


def run(sue_threshold: float = 0.5, slippage_bps: float = 5.0, borrow_bps_annual: float = 50.0) -> dict:
    ensure_dirs()
    adj = pd.read_parquet(PARQUET_DIR / "crsp_adj_close.parquet")
    tmap_df = pd.read_parquet(PARQUET_DIR / "crsp_ticker_map.parquet")
    spells = pd.read_parquet(PARQUET_DIR / "crsp_members.parquet")
    permno_to_ticker = {str(int(p)): t for p, t in zip(tmap_df["permno"], tmap_df["ticker"])}
    _m = crsp.members_asof_from_spells(spells)
    members_asof = lambda d: {str(p) for p in _m(d)}
    ret = adj.pct_change(fill_method=None)

    print("building earnings-surprise events from SEC filings…")
    events = build_events(permno_to_ticker, members_asof)
    print(f"  {len(events):,} SUE events ({events['entry_date'].min().date()} -> "
          f"{events['entry_date'].max().date()})")

    # (1) CAAR — drift shape by SUE quintile, in event time
    caar = event_caar(events, ret, hold_days=60, n_groups=5)
    tmb = caar["top_minus_bottom"]
    print("\nCAAR top-minus-bottom (Q5−Q1) cumulative abnormal return, by event day:")
    for d in [5, 10, 20, 40, 60]:
        print(f"  day {d:2d}: {tmb.loc[d]:+.3%}")
    atomic_to_parquet(caar.reset_index(), BACKTESTS_DIR / "crsp_pead_caar.parquet")

    # (2) event-time long-short at several hold horizons, net of costs
    print(f"\nevent-time long-short (sue|>={sue_threshold}, slip {slippage_bps}bps + borrow "
          f"{borrow_bps_annual}bps/yr), by hold horizon:")
    print(f"{'hold':>5s} {'annRet':>8s} {'annVol':>7s} {'Sharpe':>7s} {'maxDD':>8s} "
          f"{'nLong':>6s} {'nShort':>7s}")
    rows = []
    for h in [10, 20, 40, 60]:
        r = event_time_portfolio(events, ret, hold_days=h, sue_threshold=sue_threshold,
                                 slippage_bps=slippage_bps, borrow_bps_annual=borrow_bps_annual)
        rows.append({"hold_days": h, "ann_return": r.ann_return, "ann_vol": r.ann_vol,
                     "sharpe": r.sharpe, "max_dd": r.max_drawdown,
                     "avg_long": r.avg_long, "avg_short": r.avg_short})
        print(f"{h:5d} {r.ann_return:8.2%} {r.ann_vol:7.2%} {r.sharpe:7.2f} "
              f"{r.max_drawdown:8.2%} {r.avg_long:6.1f} {r.avg_short:7.1f}")
    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "crsp_pead_event_ls.parquet")
    print("\n[OK] event-time, survivorship-free, net of costs. See docs/pead_event_study.md.")
    return {"caar": caar, "horizons": pd.DataFrame(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sue-threshold", type=float, default=0.5)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--borrow-bps-annual", type=float, default=50.0)
    args = ap.parse_args()
    run(sue_threshold=args.sue_threshold, slippage_bps=args.slippage_bps,
        borrow_bps_annual=args.borrow_bps_annual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
