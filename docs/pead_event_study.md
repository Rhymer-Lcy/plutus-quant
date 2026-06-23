# Event-time PEAD — the drift is real, and it sits exactly on the cost boundary

> ⚠ **CORRECTED (see docs/ibes_pead_study.md).** The long-short numbers below used
> `entry_offset=0`, which captured the announcement-day overnight GAP (a look-ahead artifact),
> flattering them. The "+1.43%/Sharpe 0.07" (|SUE|≥1.5, hold-10) was the gap; de-leaked
> (`entry_offset=1`) it is **−1.1%/yr (Sharpe −0.06), negative at every horizon**. The CAAR is
> likewise gap-inflated (de-leaked Q5−Q1 ~+0.40% over 60d, not +0.68%). Conclusion only
> strengthens: post-announcement PEAD is not tradeable net of costs in large caps.

The calendar-monthly PEAD test lost because the drift is front-loaded right after the
announcement. This trades it on the EVENT clock (enter the day after the SEC filing, hold N
days, overlapping book). Earnings-surprise (SUE) events PIT-filtered to S&P 500 membership;
20,189 events 2009–2024 (SEC structured XBRL begins ~2009); survivorship-free CRSP returns.
Reproduce: `scripts/crsp_pead_event_study.py [--sue-threshold T]`.

## The drift is real and front-loaded (CAAR)

Cumulative ABNORMAL return (vs the cross-sectional mean), top-minus-bottom SUE quintile (Q5−Q1),
in event time:

| event day | 5 | 10 | 20 | 40 | 60 |
|---|---:|---:|---:|---:|---:|
| Q5−Q1 CAAR | **+0.33%** | +0.55% | +0.62% | **+0.68%** (peak) | +0.55% (reverts) |

This is textbook PEAD, confirmed on survivorship-free data: a clean drift in the direction of
the surprise, **~60% of it in the first 5 days, ~90% by day 20**, peaking ~day 40, then partly
reverting. The anomaly is unambiguously REAL — but the extreme-quintile spread is only **~0.68%**.

## …but it's not tradeable net of costs (long-short, by hold horizon)

**Broad surprises (|SUE| ≥ 0.5):** every horizon negative (Sharpe −0.09 to −0.34) — the
threshold is too loose, diluting the drift below the cost hurdle.

**Extreme surprises (|SUE| ≥ 1.5), the strongest-drift events:**

| hold (days) | ann return | Sharpe | max DD |
|---|---:|---:|---:|
| **10** | **+1.43%** | **+0.07** | −41% |
| 20 | −0.57% | −0.03 | −53% |
| 40 | −2.96% | −0.21 | −60% |
| 60 | −2.54% | −0.21 | −54% |

Only the most extreme surprises held for the shortest (most front-loaded) window is net-positive
— and at **Sharpe 0.07 with 21% vol and −41% drawdown, that is statistically indistinguishable
from break-even.** Everything else loses to costs.

## Verdict: real but at the cost boundary (which is *why* it survives)

PEAD passes the existence test (the CAAR is clean and right-signed) but fails the tradability
test for a retail, large-cap implementation: the ~0.5–0.7% drift is about the size of the
round-trip + borrow cost of harvesting it. An anomaly sitting right at the cost boundary is
exactly what an efficient-but-not-frictionless market produces — it persists *because* it can't
be arbitraged at a profit. This is a more interesting result than the valuation factors (which
had no drift at all), and a more honest one than any backtest claiming to "trade PEAD".

## Where it could cross the line (future, with more data/infra)
- **Small/mid caps**: the drift is documented to be 2–3× larger there (less arbitraged), which
  could clear the (also higher) costs. Gap: small-cap fundamentals coverage in SEC's ticker map.
- **Faster/intraday entry**: most of the day-5 drift happens in the first sessions; entering at
  the open after the release (vs next close) captures more of it.
- **Sharper surprise**: analyst-estimate or revision-based surprises (paid data) separate the
  quintiles more cleanly than a seasonal-random-walk SUE.

## Standing
The event-time harness (`research/backtest/event_study.py`: CAAR + overlapping long-short) is
the durable deliverable — it can evaluate any event signal honestly. PEAD itself is real but
break-even at retail large-cap costs; the realistic edge needs smaller caps, faster entry, or a
sharper surprise — each a data/infra step up, not a code gap.
