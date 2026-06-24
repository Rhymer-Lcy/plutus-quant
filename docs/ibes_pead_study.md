# Analyst-consensus PEAD (IBES) — and the look-ahead it exposed

The crude seasonal-random-walk SUE left small-cap PEAD just shy of the cost line. IBES gives the
real surprise — actual vs the analyst CONSENSUS just before the announcement — which should
separate the quintiles far more sharply. This is that test. It also caught a look-ahead bug that
had been quietly inflating every event-time PEAD result; the honest story is below.
Reproduce: `scripts/crsp_ibes_pead_study.py [--entry-offset N]`.

Setup: IBES US EPS unadjusted (detail + actuals) → analyst-consensus surprise events (180k
events, 6,842 mid/small names, ~7 analysts/event, 2005–2026), linked CUSIP→CRSP PERMNO,
event-clock CAAR + overlapping long-short on the survivorship-free small/mid-cap CRSP lake.

## The Sharpe-7 smoking gun

First run looked spectacular — and therefore wrong:

| (leaked) entry the day after announcement | value |
|---|---|
| CAAR Q5−Q1 **by day 5** | **+4.39%** (and flat after → a JUMP, not drift) |
| long-short hold-10, low cost | **+167%/yr, Sharpe 7.63** |

No real strategy has Sharpe 7. The tell was the CAAR shape: +4.4% by day 5 then flat — the gain
is the **announcement-day price jump**, not post-announcement drift. With close-to-close returns,
a post-close earnings release's overnight gap lands in the first day *after* the announcement
date, so "enter the next day" captures the jump — a look-ahead artifact (that move is only
available to someone positioned *before* the print, i.e. forecasting the surprise; a drift trader
can't have it). A strong surprise predicts the jump strongly, so IBES inflated the leak to absurdity.

## Fix: skip the reaction day (`entry_offset`)

Start accruing one trading day later (skip the announcement-reaction day). The truth:

| IBES small-cap, Q5−Q1 CAAR | day 5 | day 10 | day 20 | day 40 | day 60 |
|---|---:|---:|---:|---:|---:|
| leaked (offset 0) | +4.39% | +4.48% | +4.65% | +4.73% | +5.00% |
| **de-leaked (offset 1)** | **+0.26%** | +0.35% | +0.52% | +0.65% | **+0.93%** |

The real post-announcement drift is a small, *gradually accumulating* ~0.9% (the right shape) —
the +4.4% was ~all jump. Long-short, de-leaked, low cost: hold-10 −8.3% (Sharpe −0.54), …,
hold-60 +0.8% (Sharpe 0.14). Robust to offset=2 (even smaller). **Net of costs it does not pay.**

## The bug was latent in ALL earlier event-time PEAD — corrected

The same offset-0 leak inflated the earlier results too. De-leaked (offset 1), the whole PEAD
program is consistent and honest:

| study (de-leaked) | CAAR Q5−Q1 @60d | best long-short, low cost | tradeable? |
|---|---:|---|---|
| Large-cap, seasonal SUE | +0.40% | hold-10 −1.1% (Sh −0.06); all neg | no |
| Small-cap, seasonal SUE | +0.60% | hold-10 −5.2% (Sh −0.25); all neg | no |
| **Small-cap, analyst (IBES)** | **+0.93%** | hold-60 +0.8% (Sh 0.14); rest neg | no |

> This RETRACTS the earlier "first real gross edge (small-cap PEAD, Sharpe 0.41)" in
> docs/smallcap_pead_study.md and the "Sharpe 0.07" in docs/pead_event_study.md — both were the
> announcement-gap artifact. De-leaked, neither is tradeable.

## What's actually true
1. **PEAD is real** — every de-leaked CAAR is a clean, monotone, right-signed drift.
2. **IBES sharpened the SIGNAL exactly as theory predicts** — the analyst-consensus surprise
   gives the largest, cleanest drift (+0.93% vs +0.6% seasonal). The data upgrade *worked*.
3. **It still isn't tradeable** — the post-announcement drift a drift-trader can actually capture
   (~0.6–0.9% over 60d) is smaller than the turnover + borrow cost of harvesting it, even with
   the best surprise, even at idealized costs and the best horizon. The big money (the jump) is
   only available to whoever forecasts the surprise or trades the print itself — a different,
   harder game.

## The real deliverable
The platform did its job: an absurd Sharpe-7 result **exposed a latent look-ahead** that had
been mildly flattering everything, and fixing it produced the honest verdict. That truth-telling
discipline — survivorship-free + cost-aware + look-ahead-audited — is plutus's durable output;
the (null) PEAD tradability is just what it correctly found.

### Caveats
- Event-side coverage tilts to names with a current CUSIP→PERMNO link (survivor-skewed on the
  event set; prices/returns are survivorship-free).
- The drift is gross-of-impact beyond the modeled slippage; real small-cap impact would only
  make it worse.
