"""Small/mid-cap PEAD on the event clock — does the drift get bigger (and tradeable) where the
literature says it should, away from the large-cap crowd?

Builds SUE earnings-surprise events for the mid/small cap-band names (rank 501-3000) from SEC
filings, PIT-filtered to band membership, then runs the same CAAR + overlapping long-short as
the large-cap event study (docs/pead_event_study.md). Small caps trade wider and are harder to
borrow, so costs are higher. The events build is a big one-off SEC pull (cache=False to avoid
filling the disk) and is cached to parquet; reruns of the study are fast.

    conda activate plutus
    python scripts/build_crsp_smallcap_lake.py     # once
    python scripts/crsp_smallcap_pead_study.py           # needs SEC_EDGAR_USER_AGENT; first run slow
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
from plutus.research.eval.factor_eval import compute_ic  # noqa: F401  (kept for parity/ad-hoc use)
from plutus.research.backtest.metrics import month_ends

EVENTS_PATH = PARQUET_DIR / "crsp_smallcap_pead_events.parquet"


def build_events(ticker_map_df, members_asof, eval_dates) -> pd.DataFrame:
    """SUE events for the mid/small band names, PIT-filtered to band membership at the filing.
    Big one-off SEC pull (fetch-and-discard). Saved to parquet for fast reruns."""
    permno_to_ticker = {str(int(p)): t for p, t in zip(ticker_map_df["permno"], ticker_map_df["ticker"])}
    band_union = set()
    for d in eval_dates:
        band_union |= members_asof(d)
    band = sorted(p for p in band_union if p in permno_to_ticker)
    print(f"  mid/small band union: {len(band)} PERMNOs; pulling SEC earnings (cache=False)…")
    cikmap = se.load_ticker_cik_map()
    rows, hit, miss = [], 0, 0
    for i, permno in enumerate(band):
        cik = cikmap.get(str(permno_to_ticker[permno]).upper())
        if cik is None:
            miss += 1
            continue
        try:
            facts = se.company_facts(cik, cache=False)
        except Exception:
            miss += 1
            continue
        sue = ev.standardized_unexpected_earnings(
            se.discrete_quarters(se.concept_frame(facts, "NetIncomeLoss")))
        n0 = len(rows)
        for filed, s in zip(sue["filed"], sue["sue"]):
            if permno in members_asof(filed):
                rows.append({"permno": permno, "entry_date": pd.Timestamp(filed), "sue": float(s)})
        hit += int(len(rows) > n0)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(band)} processed, {len(rows):,} events so far")
    print(f"  resolved {hit} names with events, {miss} unmapped/empty")
    return pd.DataFrame(rows)


def run(sue_threshold: float = 1.5, slippage_bps: float = 20.0, borrow_bps_annual: float = 300.0,
        entry_offset: int = 1, rebuild: bool = False) -> dict:
    ensure_dirs()
    adj = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_adj_close.parquet")
    cap = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_mktcap.parquet")
    tmap = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_ticker_map.parquet")
    dates = adj.index
    members_asof = crsp.size_band_members_asof(cap)
    eval_dates = month_ends(dates)
    ret = adj.pct_change(fill_method=None)

    if EVENTS_PATH.exists() and not rebuild:
        events = pd.read_parquet(EVENTS_PATH)
        print(f"loaded {len(events):,} cached small-cap SUE events")
    else:
        print("building small-cap SUE events from SEC…")
        events = build_events(tmap, members_asof, eval_dates)
        atomic_to_parquet(events, EVENTS_PATH)
    if events.empty:
        print("no events — aborting")
        return {}
    events["entry_date"] = pd.to_datetime(events["entry_date"])
    print(f"events span {events['entry_date'].min().date()} -> {events['entry_date'].max().date()}, "
          f"{events['permno'].nunique()} names")

    caar = event_caar(events, ret, hold_days=60, n_groups=5, entry_offset=entry_offset)
    tmb = caar["top_minus_bottom"]
    print("\nCAAR top-minus-bottom (Q5-Q1) cumulative abnormal return, by event day:")
    for d in [5, 10, 20, 40, 60]:
        print(f"  day {d:2d}: {tmb.loc[d]:+.3%}")
    atomic_to_parquet(caar.reset_index(), BACKTESTS_DIR / "crsp_smallcap_pead_caar.parquet")

    print(f"\nevent-time long-short (|sue|>={sue_threshold}, slip {slippage_bps}bps + borrow "
          f"{borrow_bps_annual}bps/yr), by hold horizon:")
    print(f"{'hold':>5s} {'annRet':>8s} {'annVol':>7s} {'Sharpe':>7s} {'maxDD':>8s} "
          f"{'nLong':>6s} {'nShort':>7s}")
    rows = []
    for h in [10, 20, 40, 60]:
        r = event_time_portfolio(events, ret, hold_days=h, sue_threshold=sue_threshold,
                                 slippage_bps=slippage_bps, borrow_bps_annual=borrow_bps_annual,
                                 entry_offset=entry_offset)
        rows.append({"hold_days": h, "ann_return": r.ann_return, "sharpe": r.sharpe,
                     "max_dd": r.max_drawdown, "avg_long": r.avg_long, "avg_short": r.avg_short})
        print(f"{h:5d} {r.ann_return:8.2%} {r.ann_vol:7.2%} {r.sharpe:7.2f} "
              f"{r.max_drawdown:8.2%} {r.avg_long:6.1f} {r.avg_short:7.1f}")
    atomic_to_parquet(pd.DataFrame(rows),
                      BACKTESTS_DIR / f"crsp_smallcap_pead_ls_{int(slippage_bps)}_{int(borrow_bps_annual)}.parquet")
    print("\n[OK] small-cap, event-time, survivorship-free, net of costs. See docs/smallcap_pead_study.md.")
    return {"caar": caar, "horizons": pd.DataFrame(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sue-threshold", type=float, default=1.5)
    ap.add_argument("--slippage-bps", type=float, default=20.0)
    ap.add_argument("--borrow-bps-annual", type=float, default=300.0)
    ap.add_argument("--entry-offset", type=int, default=1)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    run(sue_threshold=args.sue_threshold, slippage_bps=args.slippage_bps,
        borrow_bps_annual=args.borrow_bps_annual, entry_offset=args.entry_offset,
        rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
