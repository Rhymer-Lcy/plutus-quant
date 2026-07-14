"""Is the biotech "sell the news" drift catalyst-specific, or just the lottery effect?

Pre-registered as issue #5, implemented verbatim. It pays a debt: docs/biotech_catalyst_study.md
published an AFFIRMATIVE claim -- post-catalyst biotech names drift -4.30% over 20 days
(clustering-robust t = -3.90) -- while admitting the mechanism was unidentified. A gap MAKES a
stock lottery-like, so the drift might be the well-known MAX/lottery anomaly rather than anything
about drugs.

The trap this design avoids: matching on post-event volatility or MAX would be controlling for the
TREATMENT (the gap itself creates them) and would define the effect out of existence. The clean
control is the SAME event in a different industry:

  TREATMENT   overnight gap >= +20% in a stock that was pharma/biotech ON THAT DAY (point-in-time
              SIC -- "was ever a biotech" would add 173 phantom events to issue #3's 1,257)
  CONTROL     overnight gap >= +20% in a non-pharma/biotech stock, same universe, same gate

Both groups are measured against ONE common benchmark (the equal-weight investable universe), so
the contrast is not an artefact of two yardsticks. The primary statistic is the mean WITHIN-CELL
difference (same quarter x cap tercile x gap tercile), with a clustering-robust t on quarterly
means. VERDICT: catalyst-specific only if that difference is negative with t < -2.

No new price lake: the existing CRSP panels are reused, plus a small point-in-time SIC map.

Prereq: python scripts/build_crsp_sic_map.py

    conda activate plutus
    python scripts/crsp_gap_lottery_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from plutus.data.sources.crsp_source import sic_membership_panel, tag_sic_asof
from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from plutus.research.backtest.gap_events import (decompose_overnight, event_cars, find_events,
                                                 matched_difference)
from plutus.research.backtest.metrics import clustered_tstat

GAP = 0.20
HORIZONS = (1, 5, 10, 20, 60)
VERDICT_H = 20
PRICE_MIN, CAP_MIN = 5.0, 100e6
BIOTECH_SIC = {"2830", "2831", "2833", "2834", "2835", "2836", "8731"}   # identical to issue #3
CELLS = ("quarter", "cap_tercile", "gap_tercile")

# Consistency gate: this study must reproduce issue #3 before it is allowed to reinterpret it.
#
# Issue #3 published 1,257 biotech events. The correct number is 1,262, and the gate below expects
# that -- a 5-event (0.4%) correction this study's gate is what surfaced. Cause: issue #3 computed
# the "at least 20 prior traded days" gate on the BIOTECH lake, whose panel only carries rows for
# the days a company was CLASSIFIED pharma/biotech. That silently turned a price-history
# requirement ("do not read a fresh listing's first noisy prints as a catalyst" -- the stated
# intent) into an industry-TENURE requirement. The 5 missing events are names reclassified into
# pharma 0-15 trading days before their gap, each with 180-663 days of actual price history. The
# old sample is a strict SUBSET of the new one (nothing was wrongly included), and the headline
# number is checked below to confirm the correction does not move it.
PUBLISHED_EVENTS = 1257
CORRECTED_EVENTS = 1262
PUBLISHED_CAR20 = -0.0430
CAR20_TOL = 0.005          # the corrected sample must land within 0.5pp of the published figure


def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(PARQUET_DIR / f"crsp_smallcap_{name}.parquet")


def _report(tag: str, x: pd.Series, dates: pd.Series) -> dict:
    x = x.dropna()
    return {"tag": tag, "n": len(x), "mean": float(x.mean()), "median": float(x.median()),
            "t": clustered_tstat(x, dates.loc[x.index], freq="Q"),
            "hit": float((x > 0).mean())}


def _print(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'sample':>26} {'N':>6} {'mean':>8} {'median':>8} {'t(qtr)':>7} {'hit':>6}")
    for r in rows:
        print(f"  {r['tag']:>26} {r['n']:>6} {r['mean']:>+8.2%} {r['median']:>+8.2%} "
              f"{r['t']:>7.2f} {r['hit']:>6.1%}")


def main() -> None:
    ensure_dirs()
    dlyret, open_raw, close_raw = _load("dlyret"), _load("open_raw"), _load("close_raw")
    halfspread, mktcap = _load("halfspread"), _load("mktcap")
    cap = mktcap.reindex_like(close_raw)

    eligible = close_raw.ge(PRICE_MIN) & cap.ge(CAP_MIN)
    overnight, intraday = decompose_overnight(dlyret, open_raw, close_raw)
    events = find_events(overnight, close_raw, threshold=GAP, eligible=eligible)
    print(f"universe: {close_raw.shape[1]:,} names; investable/day median "
          f"{int(eligible.sum(axis=1).median()):,}")
    print(f"gap events >= {GAP:.0%}: {len(events):,} raw")

    # --- point-in-time industry, never "what it became" ---------------------------------
    spells = pd.read_parquet(PARQUET_DIR / "crsp_sic_spells.parquet")
    ev = tag_sic_asof(events, spells)
    dropped = len(events) - len(ev)
    ev["biotech"] = ev["siccd"].isin(BIOTECH_SIC)
    print(f"  tagged with the SIC in force that day: {len(ev):,} "
          f"({dropped} dropped for having no spell)")
    print(f"  biotech {int(ev['biotech'].sum()):,} | non-biotech {int((~ev['biotech']).sum()):,}")

    # --- CONSISTENCY GATE: reproduce issue #3 before reinterpreting it -------------------
    n_bio = int(ev["biotech"].sum())
    if n_bio != CORRECTED_EVENTS:
        raise SystemExit(f"ABORT: {n_bio} biotech events; expected {CORRECTED_EVENTS} "
                         f"(issue #3's {PUBLISHED_EVENTS} plus the 5-event tenure correction "
                         f"documented above). This study may not reinterpret a sample it cannot "
                         f"reproduce.")
    print(f"  CONSISTENCY GATE: {n_bio} biotech events = issue #3's {PUBLISHED_EVENTS} + the "
          f"5 events its industry-tenure artefact dropped")

    # --- one common benchmark for both groups -------------------------------------------
    bench = dlyret.where(eligible).mean(axis=1)
    abn = dlyret.sub(bench, axis=0)
    abn_intra = intraday.sub(intraday.where(eligible).mean(axis=1), axis=0)
    cars = event_cars(ev, abn, abn_intra, halfspread, HORIZONS)
    cars["biotech"] = ev["biotech"].to_numpy()
    cars["quarter"] = pd.PeriodIndex(cars["date"], freq="Q")
    cars["cap"] = [cap.at[d, p] if p in cap.columns else np.nan
                   for d, p in zip(cars["date"], cars["permno"])]

    # Reproduce the published number on ITS benchmark (the biotech-peer mean), as a second gate.
    # The peer group is every name CLASSIFIED pharma/biotech that day -- not the names that had an
    # event, which would net the effect against itself.
    bio_member = sic_membership_panel(spells, BIOTECH_SIC, eligible.index, eligible.columns)
    bio_elig = eligible & bio_member
    abn_peer = dlyret.sub(dlyret.where(bio_elig).mean(axis=1), axis=0)
    peer = event_cars(ev[ev["biotech"]].reset_index(drop=True), abn_peer,
                      intraday.sub(intraday.where(bio_elig).mean(axis=1), axis=0),
                      halfspread, (VERDICT_H,))
    p20 = peer[f"close_{VERDICT_H}_net"].mean()
    if abs(p20 - PUBLISHED_CAR20) > CAR20_TOL:
        raise SystemExit(f"ABORT: on issue #3's own benchmark the corrected sample gives "
                         f"{p20:+.2%}, but it published {PUBLISHED_CAR20:+.2%}. A gap that wide "
                         f"is not a 5-event rounding effect -- find it before going further.")
    print(f"  CONSISTENCY GATE: issue #3's headline reproduces on its own benchmark: "
          f"{p20:+.2%} vs published {PUBLISHED_CAR20:+.2%}")

    col = f"close_{VERDICT_H}_net"
    _print([_report(f"biotech, {h}d", cars.loc[cars['biotech'], f'close_{h}_net'], cars["date"])
            for h in HORIZONS]
           + [_report(f"non-biotech, {h}d", cars.loc[~cars['biotech'], f'close_{h}_net'],
                      cars["date"]) for h in HORIZONS],
           "LEVELS -- close entry, net, vs ONE common benchmark:")

    # --- the primary test: matched within-cell difference ---------------------------------
    cars["cap_tercile"] = pd.qcut(cars["cap"].rank(method="first"), 3, labels=False)
    cars["gap_tercile"] = pd.qcut(cars["gap"].rank(method="first"), 3, labels=False)
    md = matched_difference(cars, value=col, treated="biotech", cells=CELLS)
    t_md = clustered_tstat(md["diff"], md["date"], freq="Q")
    print(f"\nMATCHED (same quarter x cap tercile x gap tercile): {len(md)} cells holding both "
          f"sides, {int(md['n_treated'].sum())} biotech vs {int(md['n_control'].sum())} control")
    print(f"  mean within-cell difference (biotech - non-biotech): {md['diff'].mean():+.2%}")
    print(f"  clustering-robust t(qtr): {t_md:.2f}")

    # --- secondary test (pre-registered): pooled regression with quarter fixed effects -----
    reg = cars.dropna(subset=[col, "cap", "gap"]).copy()
    y = reg[col].to_numpy(dtype=float)
    X = np.column_stack([reg["biotech"].to_numpy(dtype=float),
                         np.log(reg["cap"].to_numpy(dtype=float)),
                         reg["gap"].to_numpy(dtype=float),
                         np.ones(len(reg))])
    q = reg["quarter"].astype(str).to_numpy()
    # absorb the quarter fixed effects by within-quarter demeaning, then OLS
    yd, Xd = y.copy(), X.copy()
    for qq in np.unique(q):
        m = q == qq
        yd[m] -= y[m].mean()
        Xd[m] -= X[m].mean(axis=0)
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    resid = yd - Xd @ beta
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    meat = np.zeros((Xd.shape[1], Xd.shape[1]))          # cluster-robust by quarter
    for qq in np.unique(q):
        m = q == qq
        u = Xd[m].T @ resid[m]
        meat += np.outer(u, u)
    se = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))
    print(f"\nSECONDARY -- regression of the {VERDICT_H}d net CAR, quarter fixed effects, "
          f"SE clustered by quarter (N {len(reg):,}):")
    for name, b, s in zip(["biotech dummy", "log(market cap)", "gap size"], beta[:3], se[:3]):
        print(f"  {name:>16}: {b:+.4f}  (t {b / s:+.2f})")

    atomic_to_parquet(cars.drop(columns=["quarter"]),
                      BACKTESTS_DIR / "gap_lottery_events.parquet")

    confirmed = md["diff"].mean() < 0 and t_md < -2.0
    print(f"\nVERDICT (issue #5 frozen rule -- matched biotech-minus-control, {VERDICT_H}d net, "
          f"t < -2): {'CONFIRMED: catalyst-specific' if confirmed else 'REJECTED'}")
    if not confirmed:
        print("  The biotech drift is NOT distinguishable from the drift of any other stock that "
              "gapped the same way. The issue #3 headline over-attributed it to drugs; the "
              "write-up must be amended to a general post-gap finding.")

    bio = _report("", cars.loc[cars["biotech"], col], cars["date"])
    non = _report("", cars.loc[~cars["biotech"], col], cars["date"])
    print(f"\nLEVELS at {VERDICT_H}d (recorded whatever the verdict):")
    print(f"  biotech     {bio['mean']:+.2%}  t(qtr) {bio['t']:>6.2f}  N {bio['n']:,}")
    print(f"  non-biotech {non['mean']:+.2%}  t(qtr) {non['t']:>6.2f}  N {non['n']:,}")
    if not (non["mean"] < 0 and non["t"] < -2.0):
        return
    print("  NON-BIOTECH GAPPERS ALSO BLEED, significantly -- a general 'do not chase the gap' "
          "result, broader and more useful than the biotech-only claim.")

    # That is an AFFIRMATIVE claim, so it is stressed harder than the null was. A CAR is a SUM of
    # simple abnormal returns, so a single collapse can print below -100%; the checks below show
    # the result does not lean on those tails, on one year, or on the mean at all.
    print(f"\nROBUSTNESS of the general post-gap drift (all {len(cars):,} gappers, "
          f"{VERDICT_H}d net):")
    x = cars[col].dropna()
    dts = cars.loc[x.index, "date"]
    for lo, hi, tag in [(0.01, 0.99, "winsorized 1%"), (0.05, 0.95, "winsorized 5%")]:
        w = x.clip(x.quantile(lo), x.quantile(hi))
        print(f"  {tag:>17}: mean {w.mean():+.2%}, t(qtr) {clustered_tstat(w, dts, freq='Q'):.2f}")
    keep = x > -1.0
    print(f"  {'excl CAR < -100%':>17}: mean {x[keep].mean():+.2%}, t(qtr) "
          f"{clustered_tstat(x[keep], dts[keep], freq='Q'):.2f} ({(~keep).sum()} dropped)")
    wins = int((x > 0).sum())
    print(f"  {'sign test':>17}: {wins:,}/{len(x):,} positive ({wins / len(x):.1%})")
    yr = cars.groupby(cars["date"].dt.year)[col].mean()
    print(f"  {'per-year':>17}: {(yr < 0).sum()}/{len(yr)} years negative")


if __name__ == "__main__":
    main()
