"""Analyst-consensus PEAD on small/mid caps — does the SHARPER (IBES) surprise push the drift
over the cost line where the crude seasonal-random-walk SUE couldn't (docs/smallcap_pead_study.md)?

Pipeline: IBES actuals + estimates -> analyst-consensus surprise events (ibes_source) -> link
IBES CUSIP to CRSP PERMNO (via stocknames) -> event-clock CAAR + overlapping long-short on the
survivorship-free small/mid-cap CRSP lake. Events are cached to parquet (first build streams the
4.65GB estimates file; reruns are fast).

    conda activate plutus
    python scripts/build_crsp_smallcap_lake.py   # once (prices)
    python scripts/crsp_ibes_pead_study.py             # first run builds events (slow)
"""
from __future__ import annotations

import argparse

import pandas as pd

from plutus.data.sources import crsp_source as crsp
from plutus.data.sources import ibes_source as ibes
from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, RAW_DIR, ensure_dirs
from plutus.research.backtest.event_study import event_caar, event_time_portfolio

EVENTS_PATH = PARQUET_DIR / "crsp_ibes_pead_events.parquet"
IBES_DIR = RAW_DIR / "ibes"


def _cusip_to_permno() -> dict:
    sn = pd.read_csv(RAW_DIR / "crsp" / "stocknames.csv", dtype=str)
    sn = sn.dropna(subset=["CUSIP", "PERMNO"])
    sn = sn[sn["CUSIP"].str.len() == 8].drop_duplicates("CUSIP", keep="first")
    return {c: int(p) for c, p in zip(sn["CUSIP"], sn["PERMNO"])}


def build_events(universe_permnos: set) -> pd.DataFrame:
    """IBES analyst-consensus surprise events, linked to CRSP PERMNO, restricted to the universe."""
    cusip2permno = _cusip_to_permno()
    print("  loading IBES actuals (QTR)…")
    actuals = ibes.load_actuals(IBES_DIR / "actuals_eps_us_unadj.csv.zip", periodicity="QTR")
    actuals["permno"] = actuals["cusip"].map(cusip2permno)
    actuals = actuals.dropna(subset=["permno"])
    actuals["permno"] = actuals["permno"].astype(int).astype(str)
    actuals = actuals[actuals["permno"].isin(universe_permnos)]
    tickers = set(actuals["ticker"])
    print(f"  {len(actuals):,} linked quarterly actuals across {len(tickers)} IBES tickers; "
          f"streaming estimates…")
    estimates = ibes.stream_estimates(IBES_DIR / "detail_history_eps_us_unadj.csv.zip",
                                      tickers=tickers, start="2004-01-01")
    print(f"  {len(estimates):,} quarterly estimates; computing consensus surprise…")
    ev = ibes.build_surprise_events(actuals[["ticker", "cusip", "pends", "anndats", "actual"]],
                                    estimates)
    ev["permno"] = ev["cusip"].map(cusip2permno)
    ev = ev.dropna(subset=["permno"])
    ev["permno"] = ev["permno"].astype(int).astype(str)
    return ev[["permno", "anndats", "sue", "n_est", "actual", "consensus", "dispersion"]]


def run(sue_threshold: float = 1.0, slippage_bps: float = 5.0, borrow_bps_annual: float = 50.0,
        entry_offset: int = 1, rebuild: bool = False) -> dict:
    ensure_dirs()
    adj = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_adj_close.parquet")
    cap = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_mktcap.parquet")
    dates = adj.index
    members_asof = crsp.size_band_members_asof(cap, exclude_top=500, band_size=2500)
    ret = adj.pct_change(fill_method=None)

    if EVENTS_PATH.exists() and not rebuild:
        events = pd.read_parquet(EVENTS_PATH)
        print(f"loaded {len(events):,} cached IBES surprise events")
    else:
        print("building IBES analyst-consensus surprise events…")
        events = build_events(set(adj.columns))
        atomic_to_parquet(events, EVENTS_PATH)
    if events.empty:
        print("no events — aborting")
        return {}
    events["entry_date"] = pd.to_datetime(events["anndats"])
    # PIT-restrict to mid/small band membership at the announcement, and to names we price
    events = events[events["permno"].isin(adj.columns)]
    events = events[[p in members_asof(d) for p, d in zip(events["permno"], events["entry_date"])]]
    print(f"events in band: {len(events):,}, {events['entry_date'].min().date()} -> "
          f"{events['entry_date'].max().date()}, {events['permno'].nunique()} names "
          f"(median {events['n_est'].median():.0f} analysts/event)")

    caar = event_caar(events, ret, hold_days=60, n_groups=5, entry_offset=entry_offset)
    tmb = caar["top_minus_bottom"]
    print(f"\n(entry_offset={entry_offset} trading days after announcement — skips the "
          f"reaction-day gap)\nCAAR top-minus-bottom (Q5-Q1) by event day:")
    for d in [5, 10, 20, 40, 60]:
        print(f"  day {d:2d}: {tmb.loc[d]:+.3%}")
    atomic_to_parquet(caar.reset_index(), BACKTESTS_DIR / "crsp_ibes_pead_caar.parquet")

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
                      BACKTESTS_DIR / f"crsp_ibes_pead_ls_{int(slippage_bps)}_{int(borrow_bps_annual)}.parquet")
    print("\n[OK] analyst-consensus PEAD, small/mid-cap, survivorship-free, net of costs. "
          "See docs/ibes_pead_study.md.")
    return {"caar": caar, "horizons": pd.DataFrame(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sue-threshold", type=float, default=1.0)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--borrow-bps-annual", type=float, default=50.0)
    ap.add_argument("--entry-offset", type=int, default=1)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    run(sue_threshold=args.sue_threshold, slippage_bps=args.slippage_bps,
        borrow_bps_annual=args.borrow_bps_annual, entry_offset=args.entry_offset,
        rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
