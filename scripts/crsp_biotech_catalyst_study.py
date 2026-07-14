"""Biotech catalyst drift -- pre-registered as issue #3, implemented verbatim.

The question: when a biotech catalyst hits the tape, can someone who acts AFTER the news still
earn an abnormal return, or is the move over? Design frozen in the issue BEFORE this script
existed: pharma/biotech by SIC on the survivorship-free CRSP lake (2005-2024, price >= $5, cap
>= $100M); event = an overnight gap >= +20% (split/dividend-immune decomposition); two entries,
both strictly after the release (event-day OPEN, and event-day CLOSE), both exiting at the same
close; horizons 1/5/10/20/60 trading days; abnormal vs the same-day equal-weight biotech mean;
cost = the name's OWN CRSP half-spread on entry and exit; sub-samples = gap terciles, cap
terciles, and the two decades; inference = a clustering-robust t on MONTHLY means.

VERDICT RULE (frozen): CONFIRMED only if the CLOSE-entry mean abnormal return at 20 days is
positive NET of the round-trip half-spread with a clustering-robust t > 2. Otherwise REJECTED.
A significantly negative drift is recorded separately as "sell the news" -- and flagged as NOT
retail-harvestable (it needs shorting; small-biotech borrow is expensive or unavailable), the
same limitation already recorded for the S&P 500 ADD leg in docs/index_effect_study.md.

Implementation choices the freeze left open, fixed BEFORE any return was computed: a name needs
20 prior traded days for a gap to count as a catalyst (so a fresh listing's first noisy prints
are not events); a position in a name that delists mid-window contributes the days it actually
traded (CRSP's DlyRet carries the delisting return on the final day). CRSP quotes the CLOSING
bid/ask, so the OPEN entry is charged a closing spread it would not really get on a catalyst
day -- every open-entry NET figure below is therefore OPTIMISTIC.

    conda activate plutus
    python scripts/crsp_biotech_catalyst_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from plutus.research.backtest.gap_events import (abnormal, clustered_tstat, decompose_overnight,
                                                 event_cars, find_events, tstat)

PREFIX = "crsp_biotech"
GAP_THRESHOLD = 0.20
HORIZONS = (1, 5, 10, 20, 60)
VERDICT_HORIZON = 20
DECADES = [("2005-01-01", "2014-12-31"), ("2015-01-01", "2024-12-31")]


def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(PARQUET_DIR / f"{PREFIX}_{name}.parquet")


def _report(tag: str, cars: pd.DataFrame, col: str) -> dict:
    x = cars[col]
    ct = clustered_tstat(x, cars["date"])
    return {"tag": tag, "n": len(x), "mean": float(x.mean()), "median": float(x.median()),
            "t_event": tstat(x), "t_clustered": ct,
            "hit_rate": float((x > 0).mean())}


def _print_rows(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'sample':>22} {'N':>5} {'mean':>8} {'median':>8} {'t(event)':>9} "
          f"{'t(month)':>9} {'hit':>6}")
    for r in rows:
        print(f"  {r['tag']:>22} {r['n']:>5} {r['mean']:>+8.2%} {r['median']:>+8.2%} "
              f"{r['t_event']:>9.2f} {r['t_clustered']:>9.2f} {r['hit_rate']:>6.1%}")


def main() -> None:
    ensure_dirs()
    dlyret, open_raw, close_raw = _load("dlyret"), _load("open_raw"), _load("close_raw")
    halfspread, mktcap = _load("halfspread"), _load("mktcap")
    tmap = _load("ticker_map")
    ticker = dict(zip(tmap["permno"].astype(str), tmap["ticker"]))

    overnight, intraday = decompose_overnight(dlyret, open_raw, close_raw)
    abn_cc, abn_intra = abnormal(dlyret), abnormal(intraday)

    events = find_events(overnight, close_raw, threshold=GAP_THRESHOLD)
    cars = event_cars(events, abn_cc, abn_intra, halfspread, HORIZONS)
    cars["year"] = pd.DatetimeIndex(cars["date"]).year
    caps = [mktcap.at[d, p] if p in mktcap.columns and d in mktcap.index else np.nan
            for d, p in zip(cars["date"], cars["permno"])]
    cars["mktcap"] = caps

    span = f"{events['date'].min().date()} -> {events['date'].max().date()}"
    per_year = cars.groupby("year").size()
    print(f"events: {len(cars)} overnight gaps >= {GAP_THRESHOLD:.0%} in "
          f"{cars['permno'].nunique()} pharma/biotech names, {span}")
    print(f"  per year: min {per_year.min()}, median {int(per_year.median())}, max {per_year.max()}")
    print(f"  the gap itself (what you missed): mean {cars['gap'].mean():+.1%}, "
          f"median {cars['gap'].median():+.1%}, max {cars['gap'].max():+.1%}")
    print(f"  pre-event run-up (abnormal, t-10..t-1): mean {cars['runup'].mean():+.2%}, "
          f"t(month) {clustered_tstat(cars['runup'], cars['date']):.2f}")
    print(f"  entry half-spread on the event day: median {cars['entry_halfspread'].median():.3%}")

    # --- the headline: what is left after the news, by horizon -------------------------
    rows = []
    for h in HORIZONS:
        rows.append(_report(f"close entry, {h}d gross", cars, f"close_{h}"))
        rows.append(_report(f"close entry, {h}d NET", cars, f"close_{h}_net"))
    _print_rows(rows, "CLOSE entry (bought at the event-day close):")

    rows = []
    for h in HORIZONS:
        rows.append(_report(f"open entry, {h}d gross", cars, f"open_{h}"))
        rows.append(_report(f"open entry, {h}d NET", cars, f"open_{h}_net"))
    _print_rows(rows, "OPEN entry (bought at the event-day open; NET is optimistic -- see header):")

    # --- pre-registered sub-samples, at the verdict horizon ----------------------------
    col = f"close_{VERDICT_HORIZON}_net"
    rows = []
    cars["gap_tercile"] = pd.qcut(cars["gap"].rank(method="first"), 3, labels=["small", "mid", "large"])
    for g in ["small", "mid", "large"]:
        rows.append(_report(f"gap {g}", cars[cars["gap_tercile"] == g], col))
    cars["cap_tercile"] = pd.qcut(cars["mktcap"].rank(method="first"), 3,
                                  labels=["small", "mid", "large"])
    for g in ["small", "mid", "large"]:
        rows.append(_report(f"cap {g}", cars[cars["cap_tercile"] == g], col))
    for a, b in DECADES:
        sub = cars[(cars["date"] >= a) & (cars["date"] <= b)]
        rows.append(_report(f"{a[:4]}-{b[:4]}", sub, col))
    _print_rows(rows, f"CLOSE entry, {VERDICT_HORIZON}d NET -- pre-registered sub-samples:")

    biggest = cars.nlargest(6, "gap")
    print("\n  the largest gaps (the ones you would remember):")
    for r in biggest.itertuples():
        print(f"    {r.date.date()} {ticker.get(r.permno, r.permno):>6} gap {r.gap:>+7.1%} "
              f"-> next {VERDICT_HORIZON}d abnormal, net: "
              f"{getattr(r, f'close_{VERDICT_HORIZON}_net'):>+7.1%}")

    atomic_to_parquet(cars.drop(columns=["gap_tercile", "cap_tercile"]),
                      BACKTESTS_DIR / "biotech_catalyst_events.parquet")

    # --- the frozen verdict -------------------------------------------------------------
    v = _report("verdict", cars, col)
    confirmed = v["mean"] > 0 and v["t_clustered"] > 2.0
    sell_news = v["mean"] < 0 and v["t_clustered"] < -2.0
    print(f"\nVERDICT (issue #3 frozen rule -- close entry, {VERDICT_HORIZON}d, net, "
          f"clustering-robust t): {'CONFIRMED' if confirmed else 'REJECTED'}")
    print(f"  mean abnormal {v['mean']:+.2%}, t(month) {v['t_clustered']:.2f}, "
          f"hit rate {v['hit_rate']:.1%} over {v['n']} events")
    if sell_news:
        print("  SELL-THE-NEWS: the drift is significantly NEGATIVE -- but it is NOT "
              "retail-harvestable (it needs shorting; small-biotech borrow is expensive "
              "or unavailable).")


if __name__ == "__main__":
    main()
