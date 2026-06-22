"""SEC EDGAR adapter — free, official US fundamentals (no paid vendor needed).

The fundamentals backbone (the US analog of hermes-quant's Tushare). Endpoints (data.sec.gov),
verified 2026-06-21 against the SEC EDGAR API docs:

  - ticker -> CIK map :  https://www.sec.gov/files/company_tickers.json
  - company facts     :  https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json
  - company concept   :  https://data.sec.gov/api/xbrl/companyconcept/CIK{cik10}/{taxonomy}/{tag}.json
  - frames            :  https://data.sec.gov/api/xbrl/frames/{taxonomy}/{tag}/{unit}/CY{period}.json

`cik10` is the CIK zero-padded to 10 digits. The SEC requires a descriptive **User-Agent**
(name + email) on every request and rate-limits to **10 requests/second** — both enforced
here. Set SEC_EDGAR_USER_AGENT in .env.local (see config.sec_edgar_user_agent).

POINT-IN-TIME is the whole game here: a fact may only be used once it has been FILED publicly
(the `filed` date), never as of its fiscal-period `end` date — otherwise you "know" Q4
earnings on Dec 31 when they were actually filed in February (look-ahead). Restatements are
handled by letting the LATEST filing for a period win. The parsing functions
(`concept_frame` / `trailing_twelve_months` / `point_in_time_series`) are network-free and
unit-tested; the HTTP layer is a thin cached fetch on top.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from ... import config
from ...io import atomic_write_text
from ...paths import FUNDAMENTALS_DIR, RAW_DIR

_BASE_DATA = "https://data.sec.gov"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_MIN_INTERVAL = 0.11           # ~9 req/s, safely under the SEC's 10 req/s limit
_last_request_t = 0.0


def _user_agent() -> str:
    ua = config.sec_edgar_user_agent()
    if not ua:
        raise RuntimeError(
            "SEC EDGAR requires a descriptive User-Agent (name + email). "
            "Set SEC_EDGAR_USER_AGENT in the environment or .env.local."
        )
    return ua


def _get(url: str) -> requests.Response:
    """Throttled GET with the mandatory User-Agent. Raises on HTTP error."""
    global _last_request_t
    wait = _MIN_INTERVAL - (time.monotonic() - _last_request_t)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, headers={"User-Agent": _user_agent(),
                                      "Accept-Encoding": "gzip, deflate"}, timeout=30)
    _last_request_t = time.monotonic()
    resp.raise_for_status()
    return resp


def cik10(cik: int | str) -> str:
    """Zero-pad a CIK to the 10-digit form the data.sec.gov URLs require."""
    return f"{int(cik):010d}"


# --- HTTP layer (cached) ------------------------------------------------------------

def load_ticker_cik_map(refresh: bool = False) -> dict[str, int]:
    """{TICKER -> CIK}. Cached to data/raw/company_tickers.json; pass refresh=True to refetch.
    Tickers are upper-cased; on the rare ticker collision the first listing wins."""
    cache = RAW_DIR / "company_tickers.json"
    if refresh or not cache.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_text(_get(_TICKERS_URL).text, cache)
    raw = json.loads(cache.read_text(encoding="utf-8"))
    out: dict[str, int] = {}
    for row in raw.values():
        out.setdefault(str(row["ticker"]).upper(), int(row["cik_str"]))
    return out


def company_facts(cik: int | str, refresh: bool = False) -> dict:
    """Full XBRL company-facts JSON for one CIK. Cached to data/fundamentals/CIK{cik10}.json."""
    c10 = cik10(cik)
    cache = FUNDAMENTALS_DIR / f"CIK{c10}.json"
    if refresh or not cache.exists():
        FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_text(_get(f"{_BASE_DATA}/api/xbrl/companyfacts/CIK{c10}.json").text, cache)
    return json.loads(cache.read_text(encoding="utf-8"))


# --- parsing (network-free, unit-tested) --------------------------------------------

_FACT_COLS = ["start", "end", "filed", "val", "fy", "fp", "form", "frame"]


def concept_frame(facts: dict, concept: str, unit: str = "USD",
                  taxonomy: str = "us-gaap") -> pd.DataFrame:
    """Extract one XBRL concept's facts into a tidy frame, one row per reported fact.

    Columns: start (NaT for instant/balance-sheet concepts), end, filed, val, fy, fp, form,
    frame. Sorted by (filed, end). Empty frame if the concept/unit is absent. Dates are
    parsed to Timestamps; `val` to float."""
    units = (facts.get("facts", {}).get(taxonomy, {}).get(concept, {}).get("units", {}))
    rows = units.get(unit, [])
    if not rows:
        return pd.DataFrame(columns=_FACT_COLS)
    df = pd.DataFrame(rows)
    for col in _FACT_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    return df[_FACT_COLS].sort_values(["filed", "end"]).reset_index(drop=True)


def point_in_time_series(frame: pd.DataFrame, dates) -> pd.Series:
    """Daily point-in-time series over `dates`: for each date, the value from the LATEST fact
    FILED on or before it (restatements -> latest filing wins). No look-ahead — a number is
    visible only once filed.

    Correct as-is for INSTANT concepts (e.g. StockholdersEquity) and for any pre-aggregated
    frame with columns (end, filed, val); for a FLOW concept (e.g. NetIncomeLoss) pass it
    through `trailing_twelve_months` first. Within a single filing date the fact for the most
    recent period `end` is taken (the as-reported current value, not a prior-year comparative)."""
    dates = pd.DatetimeIndex(dates)
    f = frame.dropna(subset=["filed", "end", "val"]).sort_values(["filed", "end"])
    if f.empty:
        return pd.Series(index=dates, dtype=float)
    latest_per_filing = f.drop_duplicates("filed", keep="last")   # max end within each filed date
    s = latest_per_filing.set_index("filed")["val"]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    full = s.reindex(s.index.union(dates)).ffill()
    return full.reindex(dates).astype(float)


def discrete_quarters(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-fiscal-quarter discrete flow values from a flow concept, each tagged with the FILING
    date it became known. Returns columns (end, filed, val), sorted by period end.

    Keeps ~3-month facts (incl. Apple's 14-week 98d quarters) and synthesizes the missing Q4 of
    each fiscal year as (10-K annual) − (that year's Q1+Q2+Q3).

    CRITICAL: company facts repeat each period as a PRIOR-YEAR COMPARATIVE in later filings,
    each carrying that later filing's `filed` date. We dedupe every period by (start, end) and
    keep the EARLIEST filing — when the number was FIRST made public — which is both the honest
    point-in-time value and what keeps `filed` dates correct (a comparative would otherwise
    back-date a quarter)."""
    if frame.empty:
        return pd.DataFrame(columns=["end", "filed", "val"])
    f = frame.dropna(subset=["start", "end", "filed", "val", "fy"]).copy()
    if f.empty:
        return pd.DataFrame(columns=["end", "filed", "val"])
    f["dur"] = (f["end"] - f["start"]).dt.days

    q = (f[(f["dur"] >= 85) & (f["dur"] <= 100)]
         .sort_values("filed").drop_duplicates(["start", "end"], keep="first"))
    quarters = q[["fy", "end", "filed", "val"]].to_dict("records")

    annual = (f[(f["dur"] >= 340) & (f["dur"] <= 380)]
              .sort_values("filed").drop_duplicates(["start", "end"], keep="first"))
    by_year: dict = {}
    for rec in quarters:
        by_year.setdefault(rec["fy"], []).append(rec)
    for _, a in annual.iterrows():
        yr_qs = by_year.get(a["fy"], [])
        if len(yr_qs) == 3:
            q4_val = a["val"] - sum(r["val"] for r in yr_qs)
            q4_filed = max(a["filed"], *[r["filed"] for r in yr_qs])
            quarters.append({"fy": a["fy"], "end": a["end"], "filed": q4_filed, "val": q4_val})

    if not quarters:                                 # no discrete quarters (e.g. annual-only filer)
        return pd.DataFrame(columns=["end", "filed", "val"])
    qdf = pd.DataFrame(quarters).dropna(subset=["end", "val"])
    if qdf.empty:
        return pd.DataFrame(columns=["end", "filed", "val"])
    qdf = qdf.drop_duplicates("end", keep="last").sort_values("end").reset_index(drop=True)
    return qdf[["end", "filed", "val"]]


def trailing_twelve_months(frame: pd.DataFrame) -> pd.DataFrame:
    """Trailing-twelve-month values of a FLOW concept (e.g. net income), each tagged with the
    FILING date it became fully known. Returns (end, filed, val), composes with
    `point_in_time_series`. Sums each trailing 4 CONSECUTIVE discrete quarters (a ~3-month gap
    check, so a hole doesn't produce a wrong TTM); the TTM is dated by the latest component's
    filing."""
    qdf = discrete_quarters(frame)
    if qdf.empty:
        return pd.DataFrame(columns=["end", "filed", "val"])
    qdf = qdf.reset_index(drop=True)
    out = []
    for i in range(3, len(qdf)):
        win = qdf.iloc[i - 3:i + 1]
        ends = win["end"].tolist()
        gaps = [(ends[k] - ends[k - 1]).days for k in range(1, 4)]
        if all(80 <= g <= 105 for g in gaps):        # 4 consecutive quarters (~3 mo apart), no gap
            out.append({"end": ends[-1], "filed": win["filed"].max(),
                        "val": float(win["val"].sum())})
    return pd.DataFrame(out, columns=["end", "filed", "val"])


def build_fundamental_panel(facts_by_ticker: dict[str, dict], concept: str, dates,
                            kind: str = "instant", unit: str = "USD",
                            taxonomy: str = "us-gaap") -> pd.DataFrame:
    """Assemble a wide (date x ticker) point-in-time panel for one concept.

    `facts_by_ticker`: {ticker -> company_facts(cik)}. `kind`: "instant" for balance-sheet
    concepts (book equity), "flow_ttm" for income/cash-flow concepts (net income -> TTM).
    Every column is filing-date PIT (see point_in_time_series), so it is safe to feed straight
    into research.factors (after restrict_to_universe)."""
    dates = pd.DatetimeIndex(dates)
    cols = {}
    for ticker, facts in facts_by_ticker.items():
        cf = concept_frame(facts, concept, unit=unit, taxonomy=taxonomy)
        if kind == "flow_ttm":
            cf = trailing_twelve_months(cf)
        elif kind != "instant":
            raise ValueError("kind must be 'instant' or 'flow_ttm'")
        s = point_in_time_series(cf, dates)
        if s.notna().any():
            cols[ticker] = s
    return pd.DataFrame(cols, index=dates) if cols else pd.DataFrame(index=dates)
