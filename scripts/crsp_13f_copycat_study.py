"""13F copycat -- pre-registered as issue #4, implemented verbatim.

The question, from a friend's thesis that investing is "taste and patience": forget what the
great investors SAY -- look at what they DO. Their US long book is public 45 days after each
quarter. By the time you can see it, is anything left? And does patience pay?

Design frozen in the issue BEFORE this script existed:
  - ARM A (primary): the hindsight-selected legends, pinned to CIK. Choosing them by present-day
    fame hands the hypothesis a God's-eye view it could never have had in 2013 -- so if copying
    even THESE earns nothing, the answer is unanswerable.
  - ARM B (control): the 20 largest filers by reported value each quarter -- observable at the
    time, no hindsight.
  - Entry at the close of the FILING DATE (never the period end); PRIMARY basket = NEW positions;
    variant = top-10 by portfolio weight; horizons 21/63/252/756 days; 5 bps per side; abnormal
    vs the cap-weighted market; pre-specified high-conviction subgroup (top-10 >= 50% of value);
    clustering-robust t on QUARTERLY means.
  - VERDICT: copying works only if ARM A's new positions earn a positive mean abnormal return at
    1 year, net, with a quarterly-clustered t > 2. PATIENCE pays only if the abnormal return
    rises monotonically across horizons AND is significantly positive at 3 years.

Prereq: build_crsp_cusip_map.py -> build_13f_lake.py -> build_13f_prices.py

    conda activate plutus
    python scripts/crsp_13f_copycat_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from plutus.research.backtest.copycat import (basket_cars, concentration, new_positions,
                                              top_weights)
from plutus.research.backtest.metrics import clustered_tstat
from plutus.research.backtest.regime import cap_weighted_index

HORIZONS = (21, 63, 252, 756)          # 1 month, 1 quarter, 1 year, 3 years
VERDICT_HORIZON = 252
PATIENCE_HORIZON = 756
COST_PER_SIDE = 5e-4
ARM_B_TOP_N = 20
CONVICTION_MIN = 0.50                  # top-10 >= 50% of reported value
PERIOD_START = "2013-06-30"            # the data set's first full quarter
FILING_END = "2024-12-31"              # the CRSP price lake ends here
MIN_FILERS_PER_PERIOD = 100            # drop delinquent stragglers reporting 1987 periods

LEGEND_CIKS = {
    "1067983": "Berkshire Hathaway", "1061768": "Baupost Group", "1336528": "Pershing Square",
    "1079114": "Greenlight Capital", "1040273": "Third Point", "1006438": "Appaloosa",
    "1656456": "Appaloosa", "921669": "Icahn", "1350694": "Bridgewater Associates",
    "1037389": "Renaissance Technologies", "1536411": "Duquesne Family Office",
    "1029160": "Soros Fund Management", "1167483": "Tiger Global Management",
    "1061165": "Lone Pine Capital", "1103804": "Viking Global Investors",
    "1135730": "Coatue Management", "1697748": "ARK Investment Management",
    "1649339": "Scion Asset Management",
}


def _report(tag: str, cars: pd.DataFrame, h: int) -> dict:
    x = cars[f"car_{h}"].dropna()
    d = cars.loc[x.index, "filing_date"]
    return {"tag": tag, "n": len(x), "mean": float(x.mean()) if len(x) else np.nan,
            "median": float(x.median()) if len(x) else np.nan,
            "t": clustered_tstat(x, d, freq="Q"),      # 13F filings cluster on the 45-day deadline
            "hit": float((x > 0).mean()) if len(x) else np.nan}


def _print(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'sample':>28} {'N':>6} {'mean':>8} {'median':>8} {'t(qtr)':>7} {'hit':>6}")
    for r in rows:
        print(f"  {r['tag']:>28} {r['n']:>6} {r['mean']:>+8.2%} {r['median']:>+8.2%} "
              f"{r['t']:>7.2f} {r['hit']:>6.1%}")


def main() -> None:
    ensure_dirs()
    filings = pd.read_parquet(PARQUET_DIR / "form13f_filings.parquet")
    holdings = (pd.read_parquet(PARQUET_DIR / "form13f_holdings.parquet")
                .drop(columns=["filing_date"], errors="ignore"))   # it lives on the filing index
    dlyret = pd.read_parquet(PARQUET_DIR / "form13f_dlyret.parquet")

    # Real quarters only: a handful of delinquent filers report periods back to 1987, which would
    # otherwise each form a "quarter" whose top-20 is whoever happened to file.
    per_n = filings.groupby("period")["cik"].nunique()
    real = set(per_n[per_n >= MIN_FILERS_PER_PERIOD].index)
    filings = filings[filings["period"].isin(real)
                      & (filings["period"] >= PERIOD_START)
                      & (filings["filing_date"] <= FILING_END)].copy()
    print(f"filings in window: {len(filings):,} from {filings['cik'].nunique():,} managers, "
          f"periods {filings['period'].min().date()} -> {filings['period'].max().date()}")

    # The market: the cap-weighted total return of the CRSP large-cap lake. Abnormal = name - market.
    adj = pd.read_parquet(PARQUET_DIR / "crsp_adj_close.parquet")
    cap = pd.read_parquet(PARQUET_DIR / "crsp_mktcap.parquet")
    mkt = cap_weighted_index(adj, cap).pct_change()
    abn = dlyret.sub(mkt.reindex(dlyret.index), axis=0)
    print(f"price panel: {abn.shape[0]:,} days x {abn.shape[1]:,} permnos")

    # --- the two arms ------------------------------------------------------------------
    legends = filings[filings["cik"].isin(LEGEND_CIKS)].copy()
    legends["manager"] = legends["cik"].map(LEGEND_CIKS)
    armb = (filings.sort_values(["period", "table_value"], ascending=[True, False])
            .groupby("period").head(ARM_B_TOP_N).copy())
    print(f"ARM A: {legends['manager'].nunique()} legends, {len(legends)} filings")
    print(f"ARM B: top {ARM_B_TOP_N}/quarter -> {armb['cik'].nunique()} CIKs, {len(armb)} filings")

    conc = concentration(holdings, top=10)
    baskets = {}
    for arm, f in [("A", legends), ("B", armb)]:
        h = holdings[holdings["accession"].isin(set(f["accession"]))]
        baskets[f"arm{arm}_new"] = new_positions(h, f)
        baskets[f"arm{arm}_top10"] = top_weights(h, f, n=10)

    cars = {k: basket_cars(v, abn, HORIZONS, COST_PER_SIDE) for k, v in baskets.items()}
    for k, v in cars.items():
        v["manager"] = v["cik"].map(LEGEND_CIKS).fillna("(arm B)")
        v["conviction"] = v["accession"].map(conc)

    # --- results ------------------------------------------------------------------------
    _print([_report(f"ARM A new positions, {h}d", cars["armA_new"], h) for h in HORIZONS],
           "PRIMARY -- copy the legends' NEW positions at the filing-date close:")
    _print([_report(f"ARM A top-10 weight, {h}d", cars["armA_top10"], h) for h in HORIZONS],
           "VARIANT -- copy the legends' biggest holdings:")
    _print([_report(f"ARM B new positions, {h}d", cars["armB_new"], h) for h in HORIZONS],
           "CONTROL (no look-ahead) -- the 20 largest filers each quarter:")

    high = cars["armA_new"][cars["armA_new"]["conviction"] >= CONVICTION_MIN]
    _print([_report(f"high conviction, {h}d", high, h) for h in HORIZONS],
           f"PRE-SPECIFIED SUBGROUP -- legends whose top-10 is >= {CONVICTION_MIN:.0%} of the book:")

    per_mgr = [_report(m, g, VERDICT_HORIZON)
               for m, g in cars["armA_new"].groupby("manager") if len(g) >= 20]
    _print(sorted(per_mgr, key=lambda r: -r["mean"]),
           f"PER MANAGER -- new positions, {VERDICT_HORIZON}d net:")

    atomic_to_parquet(cars["armA_new"].drop(columns=["seq"], errors="ignore"),
                      BACKTESTS_DIR / "copycat_13f_events.parquet")

    # --- robustness the headline needs before it can be believed -------------------------
    a = cars["armA_new"]
    col = f"car_{VERDICT_HORIZON}"
    print(f"\nROBUSTNESS on the primary ({VERDICT_HORIZON}d net):")

    # 1. Renaissance alone supplies most of the events; is the aggregate just their book?
    share = a["manager"].value_counts(normalize=True)
    print(f"  event share: {share.index[0]} {share.iloc[0]:.0%}, "
          f"{share.index[1]} {share.iloc[1]:.0%} -- the mean is mostly THEM, so:")
    per_mgr_mean = a.groupby("manager")[col].mean()
    print(f"    manager-equal-weight mean (each legend counts once): "
          f"{per_mgr_mean.mean():+.2%} over {len(per_mgr_mean)} managers")

    # 2. POST-HOC (labelled): drop the quant/macro filers whose book turns over faster than the
    #    45-day disclosure lag -- copying them was never possible in the first place.
    slow = a[~a["manager"].isin(["Renaissance Technologies", "Bridgewater Associates"])]
    _print([_report(f"ex-quant legends, {h}d", slow, h) for h in HORIZONS],
           "  POST-HOC -- excluding Renaissance and Bridgewater (turnover >> disclosure lag):")

    # 3. THE SIZE CONFOUND. The benchmark is the CAP-WEIGHTED large-cap market, but the copied
    #    names skew small, and small lost badly to mega-caps over 2013-2024. Restricting to
    #    S&P 500 members makes holding and benchmark size-comparable; if the deficit collapses
    #    here, most of the headline was a size bet, not evidence that copying destroys value.
    sp500 = set(pd.read_parquet(PARQUET_DIR / "crsp_adj_close.parquet").columns)
    big = a[a["permno"].isin(sp500)]
    _print([_report(f"S&P 500 names only, {h}d", big, h) for h in HORIZONS],
           "  SIZE CONTROL -- holdings that are S&P 500 members (size-matched to the benchmark):")

    x = a[col].dropna()
    for lo, hi, tag in [(0.01, 0.99, "winsorized 1%"), (0.05, 0.95, "winsorized 5%")]:
        w = x.clip(x.quantile(lo), x.quantile(hi))
        print(f"  {tag:>16}: mean {w.mean():+.2%}, t(qtr) "
              f"{clustered_tstat(w, a.loc[x.index, 'filing_date'], freq='Q'):.2f}")
    wins = int((x > 0).sum())
    print(f"  {'sign test':>16}: {wins:,}/{len(x):,} positive ({wins / len(x):.1%})")
    yr = a.groupby(a['filing_date'].dt.year)[col].mean()
    print(f"  {'per-year':>16}: {(yr < 0).sum()}/{len(yr)} years negative")

    # --- the two frozen verdicts ---------------------------------------------------------
    v = _report("verdict", cars["armA_new"], VERDICT_HORIZON)
    works = v["mean"] > 0 and v["t"] > 2.0
    print(f"\nVERDICT 1 -- DOES COPYING WORK? (issue #4: ARM A new positions, {VERDICT_HORIZON}d, "
          f"net, quarterly-clustered t): {'CONFIRMED' if works else 'REJECTED'}")
    print(f"  mean abnormal {v['mean']:+.2%}, t(qtr) {v['t']:.2f}, hit {v['hit']:.1%}, "
          f"N {v['n']:,}")

    seq = [_report("", cars["armA_new"], h)["mean"] for h in HORIZONS]
    p = _report("", cars["armA_new"], PATIENCE_HORIZON)
    monotone = all(b > a for a, b in zip(seq, seq[1:]))
    patient = monotone and p["mean"] > 0 and p["t"] > 2.0
    print(f"\nVERDICT 2 -- DOES PATIENCE PAY? (monotone across horizons AND significant at "
          f"{PATIENCE_HORIZON}d): {'CONFIRMED' if patient else 'REJECTED'}")
    print("  horizon means: " + ", ".join(f"{h}d {m:+.2%}" for h, m in zip(HORIZONS, seq)))
    print(f"  monotonically increasing: {monotone}; "
          f"{PATIENCE_HORIZON}d t(qtr) {p['t']:.2f} (N {p['n']:,})")


if __name__ == "__main__":
    main()
