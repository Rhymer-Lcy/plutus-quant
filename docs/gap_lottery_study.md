# The biotech "sell the news" drift is not about drugs — every gapper bleeds

Pre-registered as [issue #5](https://github.com/Rhymer-Lcy/plutus-quant/issues/5), frozen before
any code existed. **This study pays a debt and corrects a published claim.** Reproduce:
`python scripts/build_crsp_sic_map.py && python scripts/crsp_gap_lottery_study.py`.

[biotech_catalyst_study.md](biotech_catalyst_study.md) (issue #3) published an **affirmative**
claim — after a biotech catalyst gaps a stock up, the drift is significantly negative, "sell the
news" is statistically real — while admitting in the same breath that the mechanism was
unidentified: a gap *makes* a stock lottery-like, and lottery stocks underperform in general
(the MAX anomaly). An unidentified mechanism attached to a published affirmative claim is a debt.
This settles it.

## The design trap, and the control that avoids it

The gap **itself** turns a stock into a lottery ticket: after a +40% jump the name mechanically
has a huge recent maximum daily return and a fat right tail. **Matching on post-event volatility
or MAX would be controlling for the treatment** and would define the effect out of existence. So
the control is not "a lottery-like stock" but **the same event in a different industry**:

- **TREATMENT** — overnight gap ≥ +20% in a stock that was pharma/biotech **on that day**
  (point-in-time SIC).
- **CONTROL** — overnight gap ≥ +20% in a **non**-pharma/biotech stock, same universe, same gate.

Both measured against **one common benchmark** (the equal-weight investable universe), so the
contrast is not an artefact of two yardsticks. **5,949 gap events, 2005–2024**, survivorship-free
CRSP, entry at the event-day close, 20-day hold, net of each name's own half-spread.

## Result

| 20d net abnormal | mean | median | **t(qtr)** | hit | N |
|---|---:|---:|---:|---:|---:|
| biotech gappers | −3.06% | −2.47% | −2.22 | 42.3% | 1,262 |
| **non-biotech gappers** | **−2.98%** | −1.24% | **−4.54** | 45.0% | 4,687 |
| **matched difference** (biotech − control) | **+0.28%** | | **0.01** | | 519 cells |

Matched = same calendar quarter × market-cap tercile × gap-size tercile.

Two independent methods agree. The pre-registered **regression** (quarter fixed effects, SE
clustered by quarter, N = 5,949) puts the **biotech dummy at +0.0133 (t = +1.46)** — if anything
biotech gappers do marginally *better* than comparable non-biotech gappers, and not significantly.

## VERDICT: REJECTED — the drift is not catalyst-specific

**The biotech drift is indistinguishable from the drift of any other stock that gapped the same
way.** Issue #3's headline over-attributed a general phenomenon to drugs. The write-up is amended
accordingly (see below).

## What replaces it — a bigger, sturdier finding

**Do not chase any +20% overnight gap.** Across all 5,949 gappers, the 20-day net abnormal return
is about **−3%**, and the general result is *statistically stronger* than the biotech-only claim
ever was (t = −4.5 on the control group alone, versus −2.2 for biotech):

| stress test | mean | t(qtr) |
|---|---:|---:|
| raw (all gappers) | −3.0% | — |
| winsorized 1% | −3.04% | **−5.31** |
| winsorized 5% | −2.75% | **−5.85** |
| excluding the 50 arithmetic CARs below −100% | −1.89% | −3.17 |

Sign test: only 44.4% of gappers are positive. **16 of 20 years** have a negative mean. Excluding
the 50 catastrophic events roughly halves the magnitude, so the tails carry part of it — but the
sign and the significance survive without them, so it is not a tail artefact.

The regression also recovers exactly the pattern a lottery/attention mechanism predicts:

- **log(market cap): +0.0136 (t = +4.76)** — the bigger the company, the less its gap bleeds.
- **gap size: −0.0611 (t = −3.33)** — the bigger the jump, the worse the bleed.

Small, attention-grabbing, hard-jumping stocks are the ones that hurt you. That is the MAX/lottery
story, not a drug story.

## Two consistency gates the study had to clear first

This study was not allowed to *reinterpret* issue #3 until it could *reproduce* it. Both gates are
enforced in code (`SystemExit` on failure), and both fired during development:

1. **Sample reproduction.** It rebuilds issue #3's biotech sample and gets **1,262** events, not
   the published 1,257. The 5-event (0.4%) difference is fully explained and is a **correction to
   issue #3**: that study computed its "≥ 20 prior traded days" gate on the *biotech lake*, whose
   panel only carries rows for the days a company was *classified* pharma. That silently turned a
   **price-history** requirement (its stated intent: "do not read a fresh listing's first noisy
   prints as a catalyst") into an **industry-tenure** requirement. The 5 missing events are names
   reclassified into pharma 0–15 trading days before their gap, each with 180–663 days of real
   price history. The old sample is a strict subset of the new one.
2. **Headline reproduction.** On issue #3's own biotech-peer benchmark the corrected sample gives
   **−4.46%** against the published **−4.30%** — inside the 0.5 pp tolerance, so the correction
   does not move the headline.

Point-in-time industry classification is load-bearing here and was not optional: **4,698 of the
13,159 names are reclassified at least once**. Tagging by "was this ever a biotech" would have
added 173 phantom events — gaps that happened while the company was in another industry entirely.

## What this changes

- The claim "sell the news is real in biotech" is **rescoped**: the drift is real, but it belongs
  to *gapping*, not to *drugs*.
- The practical advice gets **broader and stronger**: after any stock gaps up 20%+, the average
  buyer at that day's close loses ~3% to the market over the next month, and more if the stock is
  small or the gap was large.
- It remains **not retail-harvestable on the short side** — shorting a stock that just exploded is
  expensive or impossible to borrow, exactly when you would need to.
