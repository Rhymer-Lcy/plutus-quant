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

CORRECTIONS (see docs/biotech_catalyst_study.md for the full history):
  - 2026-07-13: two data bugs from the first run were fixed -- the SIC set had omitted CRSP's
    GROUP-level drug codes 2830/2831 (dropping Amgen for 17 years), and the price/cap floors had
    been applied as a LAKE row filter, truncating the post-event losses of any name that cratered.
  - 2026-07-14: this study was moved onto the SHARED market panels + the point-in-time SIC map,
    the same data path as the gap-lottery study (issue #5). That retired a redundant 108 MB
    biotech-only lake AND fixed a subtler bug at its source: the ">= 20 prior traded days" gate
    used to be computed on the biotech lake, whose panel only carried rows for the days a company
    was CLASSIFIED pharma, so it silently required 20 days of industry TENURE rather than 20 days
    of PRICE HISTORY. On the correct panel the sample is 1,262 events (not 1,257) and the headline
    is -4.46% (not -4.30%); the 5 recovered names had been reclassified into pharma 0-15 days
    before their gap. Industry is now point-in-time (4,698 of 13,159 names are reclassified at
    least once), so an event is tagged by what the company WAS on the day, never by what it became.

The frozen DESIGN is unchanged throughout; only the data plumbing and the corrected counts move.

Implementation choices the freeze left open, fixed BEFORE any return was computed: a name needs
20 prior traded days for a gap to count as a catalyst; a position in a name that delists
mid-window contributes the days it actually traded (CRSP's DlyRet carries the delisting return on
the final day). CRSP quotes the CLOSING bid/ask, so the OPEN entry is charged a closing spread it
would not really get on a catalyst day -- every open-entry NET figure below is OPTIMISTIC.

Note: issue #5 (docs/gap_lottery_study.md) later showed this drift is NOT catalyst-specific --
any stock that gaps up bleeds the same. This study still answers issue #3's own question (you
cannot get in after the news), and its "sell the news" reading is amended there to "post-gap
drift".

Prereq: python scripts/build_crsp_sic_map.py

    conda activate plutus
    python scripts/crsp_biotech_catalyst_study.py
"""
from __future__ import annotations

import pandas as pd
from scipy import stats

from plutus.data.sources.crsp_source import sic_membership_panel, tag_sic_asof
from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from plutus.research.backtest.gap_events import decompose_overnight, event_cars, find_events
from plutus.research.backtest.metrics import clustered_tstat, tstat

GAP_THRESHOLD = 0.20
HORIZONS = (1, 5, 10, 20, 60)
VERDICT_HORIZON = 20
PRICE_MIN, CAP_MIN = 5.0, 100e6        # the frozen tradability gate -- EVENT ELIGIBILITY only
BIOTECH_SIC = {"2830", "2831", "2833", "2834", "2835", "2836", "8731"}   # identical to issue #5
DRUG_SIC = {"2830", "2831", "2833", "2834", "2835", "2836"}   # post-hoc 283x-only split
DECADES = [("2005-01-01", "2014-12-31"), ("2015-01-01", "2024-12-31")]


def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(PARQUET_DIR / f"crsp_smallcap_{name}.parquet")


def _report(tag: str, cars: pd.DataFrame, col: str) -> dict:
    x = cars[col]
    return {"tag": tag, "n": len(x), "mean": float(x.mean()), "median": float(x.median()),
            "t_event": tstat(x), "t_clustered": clustered_tstat(x, cars["date"], freq="M"),
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
    cap = mktcap.reindex_like(close_raw)
    spells = pd.read_parquet(PARQUET_DIR / "crsp_sic_spells.parquet")
    tmap = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_ticker_map.parquet")
    ticker = dict(zip(tmap["permno"].astype(str), tmap["ticker"]))

    # Investable AND pharma/biotech ON THAT DAY. The membership panel is point-in-time, so the
    # 20-prior-day history gate in find_events now counts real PRICE history (from close_raw's
    # full series), not industry tenure -- the 2026-07-14 fix. The peer benchmark is this same
    # biotech-investable set, so a sector-wide move is not credited to the event.
    bio_member = sic_membership_panel(spells, BIOTECH_SIC, close_raw.index, close_raw.columns)
    eligible = close_raw.ge(PRICE_MIN) & cap.ge(CAP_MIN) & bio_member

    overnight, intraday = decompose_overnight(dlyret, open_raw, close_raw)
    abn_cc = dlyret.sub(dlyret.where(eligible).mean(axis=1), axis=0)
    abn_intra = intraday.sub(intraday.where(eligible).mean(axis=1), axis=0)

    events = find_events(overnight, close_raw, threshold=GAP_THRESHOLD, eligible=eligible)
    cars = event_cars(events, abn_cc, abn_intra, halfspread, HORIZONS)
    cars["year"] = pd.DatetimeIndex(cars["date"]).year
    cars["mktcap"] = [cap.at[d, p] if p in cap.columns and d in cap.index else float("nan")
                      for d, p in zip(cars["date"], cars["permno"])]
    # SIC in force on the event day (point-in-time), for the 283x post-hoc split.
    sic_on = tag_sic_asof(cars[["permno", "date"]], spells).set_index(["permno", "date"])["siccd"]
    cars["siccd"] = [sic_on.get((p, d)) for p, d in zip(cars["permno"], cars["date"])]

    span = f"{events['date'].min().date()} -> {events['date'].max().date()}"
    per_year = cars.groupby("year").size()
    inv = eligible.sum(axis=1)
    print(f"universe: pharma/biotech (point-in-time SIC); investable "
          f"(>= ${PRICE_MIN:.0f}, >= ${CAP_MIN / 1e6:.0f}M) per day: median {int(inv.median())}, "
          f"max {int(inv.max())}")
    print(f"events: {len(cars)} overnight gaps >= {GAP_THRESHOLD:.0%} in "
          f"{cars['permno'].nunique()} pharma/biotech names, {span}")
    print(f"  per year: min {per_year.min()}, median {int(per_year.median())}, max {per_year.max()}")
    print(f"  the gap itself (what you missed): mean {cars['gap'].mean():+.1%}, "
          f"median {cars['gap'].median():+.1%}, max {cars['gap'].max():+.1%}")
    print(f"  pre-event run-up (abnormal, t-10..t-1): mean {cars['runup'].mean():+.2%}, "
          f"t(month) {clustered_tstat(cars['runup'], cars['date'], freq='M'):.2f}")
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

    # POST-HOC robustness (labelled as such -- NOT part of the frozen verdict): the frozen SIC set
    # included 8731 (commercial research), which also carries devices and CDMOs. Restricting to
    # the drug manufacturers (283x) is the cleaner industry read. SIC is point-in-time here.
    is_drug = cars["siccd"].isin(DRUG_SIC)
    _print_rows([_report("drug makers (283x)", cars[is_drug], col),
                 _report("research (8731)", cars[~is_drug], col)],
                f"CLOSE entry, {VERDICT_HORIZON}d NET -- POST-HOC industry split (not the verdict):")

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

    if not sell_news:
        return
    # A significantly NEGATIVE drift is an affirmative claim, so it gets stressed harder than the
    # null did. A CAR is a SUM of simple abnormal returns, so it can print below -100% on a name
    # that collapses -- an arithmetic artifact, not realized P&L. Issue #5 further shows this drift
    # is not catalyst-specific (any gapper bleeds), so the "sell the news" label is amended there.
    x = cars[col]
    print("\n  POST-GAP DRIFT is significantly NEGATIVE (not biotech-specific -- see issue #5). "
          "Robustness of the negative-drift claim:")
    for lo, hi, tag in [(0.01, 0.99, "winsorized 1%"), (0.05, 0.95, "winsorized 5%")]:
        w = x.clip(x.quantile(lo), x.quantile(hi))
        print(f"    {tag:>16}: mean {w.mean():+.2%}, t(month) "
              f"{clustered_tstat(w, cars['date'], freq='M'):.2f}")
    keep = x > -1.0
    print(f"    {'excl CAR < -100%':>16}: mean {x[keep].mean():+.2%}, t(month) "
          f"{clustered_tstat(x[keep], cars.loc[keep, 'date'], freq='M'):.2f} "
          f"({(~keep).sum()} events dropped)")
    wins = int((x > 0).sum())
    p = stats.binomtest(wins, len(x), 0.5).pvalue
    print(f"    {'sign test':>16}: {wins}/{len(x)} positive ({wins / len(x):.1%}), "
          f"binomial p = {p:.1e}")
    per_year = cars.groupby("year")[col].mean()
    print(f"    {'per-year':>16}: {(per_year < 0).sum()}/{len(per_year)} years negative")


if __name__ == "__main__":
    main()
