# PEAD — the first signal with a pulse (but not yet a tradeable one)

After classic factors came up empty everywhere (docs/longshort_study.md, docs/smallcap_study.md),
the honest frontier was non-classic / event-driven signals. **Post-Earnings-Announcement Drift
(PEAD)** is the textbook candidate — it has survived better than valuation factors. Signal:
SUE (standardized unexpected earnings, seasonal-random-walk surprise), known at the SEC filing
date, held only while fresh (~one quarter). S&P 500 PIT, 2005–2024, monthly, survivorship-free.
Reproduce: `scripts/crsp_pead_study.py`.

## Result

- **Coverage:** ~234 names/month carry a fresh earnings surprise (of ~957).
- **SUE rank IC vs next-month return:** mean **+0.0080**, IC-IR 0.067, **t = 0.89**, hit 54%,
  n=176. Weak, but **positive and the strongest, most significant IC in the whole project** —
  and in the right direction, unlike the (flat/negative) valuation factors.
- **Quintile long-short, net of costs:** ann return **−1.38%**, Sharpe **−0.17**, **beta −0.07**
  (cleanly market-neutral), max DD −36%, turnover **2.29**/period.

## Read: a real but marginal anomaly, cost-dominated as implemented

PEAD has a genuine pulse — a positive, near-significant IC with ~zero market beta, exactly what
a real anomaly looks like. But the long-short still **loses after costs**: the surprise set
rotates every month (turnover 229%/side), and the per-name drift is small relative to the
round-trip cost. This matches the modern literature: **PEAD has decayed in large caps** (heavily
arbitraged since it became famous) and **its drift is front-loaded in the first ~1–20 trading
days** — a monthly rebalance captures the weak tail, not the strong head.

## Where PEAD could actually pay (next experiments)
1. **Faster reaction**: trade in the first days after the announcement (daily, not monthly).
   The drift is strongest immediately post-event — monthly rebalancing misses it. This needs an
   event-time (not calendar-month) backtest.
2. **Small/mid caps**: PEAD is documented to be stronger where less capital arbitrages it. The
   broad CRSP lake exists; the gap is fundamentals coverage (SEC ticker→CIK is patchy for small
   caps).
3. **Sharper surprise**: SUE from a seasonal random walk is crude; an analyst-estimate or
   revision-based surprise (paid data) measures it better.

## Honest standing
PEAD is the first signal worth a second look — it isn't arbitraged to zero like the valuation
factors. But in a retail, large-cap, monthly form it doesn't clear costs. The realistic edge (if
any) lives in faster execution and/or smaller caps — both of which raise data/infra demands.
The methodology again did its job: it found the faint real signal AND refused to overstate it.
