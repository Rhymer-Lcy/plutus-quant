"""Copycat mechanics for the 13F study (issue #4): can you profit from what the greats disclose?

A 13F is public roughly 45 days after the quarter it describes. The copycat's only legal entry is
the FILING date, and the question is whether anything is left by then.

Three pieces, all pure and unit-tested:

  - `new_positions` -- names in a manager's filing that were absent from their PREVIOUS filing.
    The freshest signal, and what a copycat actually chases. Compared against the prior filing as
    the manager themselves reported it, so a name they merely re-weighted is not "new".
  - `top_weights` -- the largest holdings by portfolio weight. Weight is a RATIO, so it is immune
    to the SEC's mid-sample switch of VALUE from thousands to dollars.
  - `basket_cars` -- cumulative ABNORMAL return of each basket, entered at the close of the
    filing date (or the next trading day) and held for a fixed horizon, net of a round-trip cost.

Abnormal = the name's return minus the market's that day. A name that delists inside the horizon
realises its CRSP delisting return and the position ends there; the days it actually traded are
counted, never truncated to flatter the result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def new_positions(holdings: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    """Rows of `holdings` whose (cik, permno) was NOT in that manager's previous filing.

    `holdings` needs [accession, permno, value_usd]; `filings` needs [accession, cik, period,
    filing_date]. A manager's FIRST filing in the sample has no predecessor, so it is skipped --
    otherwise their whole opening book would count as fresh conviction. Returns the holdings rows
    plus [cik, period, filing_date, weight]."""
    h = holdings.merge(filings[["accession", "cik", "period", "filing_date"]],
                       on="accession", how="inner")
    h["weight"] = h["value_usd"] / h.groupby("accession")["value_usd"].transform("sum")

    order = (filings.sort_values(["cik", "period"])
             .assign(seq=lambda d: d.groupby("cik").cumcount())[["accession", "seq"]])
    h = h.merge(order, on="accession", how="left")

    prev = h[["cik", "permno", "seq"]].copy()
    prev["seq"] = prev["seq"] + 1                       # shift forward: held in the PRIOR filing
    prev["was_held"] = True
    out = h.merge(prev.drop_duplicates(["cik", "permno", "seq"]),
                  on=["cik", "permno", "seq"], how="left")
    fresh = out[out["was_held"].isna() & (out["seq"] > 0)]   # seq 0 has no predecessor
    return fresh.drop(columns=["was_held"]).reset_index(drop=True)


def top_weights(holdings: pd.DataFrame, filings: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The `n` largest holdings by portfolio weight in each filing (the 'what do they love' read).
    Weight is a ratio, hence immune to the VALUE units change."""
    h = holdings.merge(filings[["accession", "cik", "period", "filing_date"]],
                       on="accession", how="inner")
    h["weight"] = h["value_usd"] / h.groupby("accession")["value_usd"].transform("sum")
    return (h.sort_values(["accession", "weight"], ascending=[True, False])
            .groupby("accession").head(n).reset_index(drop=True))


def concentration(holdings: pd.DataFrame, top: int = 10) -> pd.Series:
    """accession -> share of reported value in its `top` largest positions. A ratio, so
    unit-invariant; observable at the time, so usable as an ex-ante subgroup."""
    w = holdings.sort_values(["accession", "value_usd"], ascending=[True, False])
    head = w.groupby("accession").head(top).groupby("accession")["value_usd"].sum()
    total = w.groupby("accession")["value_usd"].sum()
    return (head / total).rename("concentration")


def basket_cars(events: pd.DataFrame, abn: pd.DataFrame, horizons: tuple[int, ...],
                cost_per_side: float) -> pd.DataFrame:
    """Per-event cumulative abnormal return, entered at the close of the filing date.

    `events` needs [permno, filing_date] (extra columns are carried through). `abn` is a
    date x permno panel of ABNORMAL daily returns. Entry is the first trading day ON OR AFTER
    the filing date -- the holding is public that day, so the close is transactable. Returns run
    from the day AFTER entry (buying at a close earns nothing that day).

    A horizon is reported ONLY where the full window fits inside the panel; otherwise NaN. That
    keeps a 3-year column from silently averaging in half-length events. `n_days_h` is how many
    days the position actually survived (fewer than h means the name delisted -- those days are
    still counted, never dropped)."""
    idx = abn.index
    cols = {c: j for j, c in enumerate(abn.columns)}
    arr = abn.to_numpy(dtype=float)
    out = events.copy().reset_index(drop=True)
    pos = idx.searchsorted(pd.DatetimeIndex(out["filing_date"]), side="left")

    for h in horizons:
        cars, ndays = np.full(len(out), np.nan), np.zeros(len(out), dtype=int)
        for i, (permno, p) in enumerate(zip(out["permno"], pos)):
            j = cols.get(permno)
            if j is None or p >= len(idx) or p + h >= len(idx):
                continue                               # no entry bar, or horizon runs off the end
            seg = arr[p + 1:p + 1 + h, j]
            ok = ~np.isnan(seg)
            cars[i] = float(seg[ok].sum()) - 2.0 * cost_per_side
            ndays[i] = int(ok.sum())
        out[f"car_{h}"] = cars
        out[f"n_days_{h}"] = ndays
    return out


def quarterly_tstat(x: pd.Series, dates: pd.Series) -> float:
    """Clustering-robust t-stat: average the events inside each calendar QUARTER first, then take
    the t-stat across quarters. 13F filings pile up on the 45-day deadline, so an event-level t
    treats hundreds of same-day filings as independent draws and badly overstates significance.
    The frozen verdict rule uses THIS statistic."""
    g = pd.DataFrame({"x": np.asarray(x, dtype=float),
                      "q": pd.DatetimeIndex(dates).to_period("Q")}).dropna()
    means = g.groupby("q")["x"].mean()
    if len(means) < 2 or means.std(ddof=1) == 0:
        return float("nan")
    return float(means.mean() / (means.std(ddof=1) / np.sqrt(len(means))))
