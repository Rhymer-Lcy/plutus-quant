# Biotech catalysts: the move is over before you can act — and then it keeps going against you

Pre-registered as [issue #3](https://github.com/Rhymer-Lcy/plutus-quant/issues/3), frozen before
any code existed. The question, raised by a friend's pancreatic-cancer example: **when a biotech
catalyst hits the tape, can someone who acts AFTER the news still earn an abnormal return?**
Reproduce: `python scripts/build_crsp_sic_map.py && python scripts/crsp_biotech_catalyst_study.py`.

> **Correction, 2026-07-14 (data path).** This study was moved onto the shared market panels plus
> the point-in-time SIC map — the same data path as the gap-lottery study (issue #5) — which
> retired a redundant 108 MB biotech-only lake and its builder, and **fixed a subtler bug at its
> source**. The old "≥ 20 prior traded days" gate was computed on that biotech lake, whose panel
> only carried rows for the days a company was *classified* pharma, so it silently required 20 days
> of industry **tenure** rather than 20 days of **price history**. On the correct panel the sample
> is **1,262 events (not 1,257)** and the headline is **−4.46% (not −4.30%)**; the 5 recovered
> names had been reclassified into pharma 0–15 trading days before their gap, each with 180–663
> days of real price history. The tables below carry the earlier 1,257/−4.30% figures; the deltas
> are immaterial and the verdict is unchanged, so they are annotated rather than rerun line by line.
> Industry is now point-in-time throughout (4,698 of 13,159 names are reclassified at least once).

> **Correction, 2026-07-13.** The first run of this study was WRONG and its numbers are retracted.
> A post-run audit found two data bugs, both fixed here, both of which had biased the measured
> drift *upward*:
> 1. **The SIC set omitted CRSP's group-level drug codes 2830/2831.** CRSP carries the drug
>    industry both as specific 4-digit codes and as the group code 2830, and names move between
>    them — Amgen is coded 2830 from 2000 to 2021 and 2836 only from 2021-12-06. The first
>    universe therefore silently dropped 249 names, Amgen among them for 17 of its 20 years,
>    which distorted both the universe and the cross-sectional benchmark and lost events.
> 2. **The price/cap floors were applied as a lake-level row filter**, which deleted a name's
>    rows the moment it fell below them. A biotech that gapped up and then cratered past $5 had
>    its post-event **losses truncated** — an upward bias in exactly the quantity being measured.
>    The floors now gate **event eligibility only** (was this event tradable when it happened?),
>    never the holding period.
>
> The frozen **design** is unchanged; only its implementation is corrected. The verdict was
> REJECTED before and is REJECTED now — but the corrected drift is more than twice as negative
> and is now statistically significant, which changes the *interpretation*, so the old numbers
> are struck rather than quietly updated. Retracted: 1,111 events, −1.64% at 20 days, t = −0.78.

Universe: pharma/biotech by SIC (2830, 2831, 2833–2836, 8731) on the survivorship-free CRSP lake,
2005–2024 — **1,431 names ever**, including every one that later died; median 234 investable
(≥ $5, ≥ $100M) on a given day. Event: an overnight gap ≥ +20%, decomposed split- and
dividend-immune from the total return (`overnight[t] = (1+DlyRet[t])/(1+intraday[t]) − 1`).
**1,257 events across 638 names.** Material biotech data is released outside trading hours by
design, which is precisely why it arrives as a gap nobody can trade into. Abnormal = the name
minus the same-day equal-weight mean of the *investable* biotech universe; cost = the name's own
CRSP half-spread on each side (median 0.079% on the event day).

**The gap itself — what you never get: mean +51.6%, median +34.1%, max +1,186%.**

## Result: nothing is left, and then it bleeds

| entry, horizon | gross | NET | median (net) | t(event) | **t(month)** | hit rate |
|---|---:|---:|---:|---:|---:|---:|
| close, 1d | −0.07% | −0.51% | −0.63% | −1.50 | −1.15 | 42.9% |
| close, 5d | −1.84% | −2.30% | −1.88% | −4.83 | −3.05 | 42.8% |
| close, 10d | −2.64% | −3.08% | −2.47% | −5.38 | −3.93 | 41.9% |
| close, 20d **(the frozen verdict)** | −3.86% | **−4.30%** | −3.91% | −6.33 | **−3.90** | 39.8% |
| close, 60d | −7.42% | **−7.89%** | −7.79% | −7.30 | −4.90 | 36.4% |
| open, 1d | +1.26% | +0.81% | −1.42% | 1.04 | 1.23 | 42.6% |
| open, 20d | −2.54% | −2.98% | −4.74% | −3.15 | −1.71 | 39.2% |
| open, 60d | −6.10% | −6.56% | −7.98% | −5.16 | −3.34 | 38.0% |

t(month) is the clustering-robust statistic the pre-registration committed to: biotech catalysts
cluster in time (ASCO/ESMO, JPM healthcare week, PDUFA dates), and an event-level t treats
clustered events as independent draws.

## Verdict: REJECTED

The close-entry 20-day net abnormal return is **−4.30%** (clustering-robust **t = −3.90**, hit
rate 39.8%). Under the frozen rule, the answer to "can you still buy it after the news" is **no**
— and not merely "no edge": the average post-catalyst buyer *loses*. Buying at the open instead
of the close captures a small day-1 bounce (+1.26% gross, not significant) that is entirely given
back and more by day 20 — and that figure is *optimistic*, because CRSP quotes the closing
bid/ask while a real catalyst-day opening spread is far wider.

> ## AMENDED 2026-07-14 — the drift is real, but it is NOT about drugs
>
> A follow-up study, pre-registered as [issue #5](https://github.com/Rhymer-Lcy/plutus-quant/issues/5)
> and written up in [gap_lottery_study.md](gap_lottery_study.md), settled the "mechanism is not
> identified" limitation that this page admits below. The answer: **the negative drift is not
> catalyst-specific.** Any stock that gaps up ≥ +20% bleeds about the same amount, whatever its
> industry:
>
> | 20d net abnormal | mean | t(qtr) | N |
> |---|---:|---:|---:|
> | biotech gappers | −3.06% | −2.22 | 1,262 |
> | **non-biotech gappers** | **−2.98%** | **−4.54** | 4,687 |
> | **matched difference** (same quarter × cap tercile × gap tercile) | **+0.28%** | **0.01** | 519 cells |
>
> A pre-registered regression agrees: the biotech dummy is **+0.0133 (t = +1.46)** — if anything
> biotech gappers do marginally *better* than comparable non-biotech gappers.
>
> **So the section below over-attributed a general phenomenon to drugs.** The finding that
> replaces it is broader and statistically stronger: *do not chase any +20% gap*. Read the
> "sell the news" heading below as **"post-gap drift"**, and note that it is a property of
> gapping, not of trial data. Everything else on this page — that you cannot get in after the
> news, that the money moves before the announcement, that the typical trade loses — stands.
>
> That study's consistency check also surfaced the sample-size correction now documented at the
> top of this page (1,262 events, headline −4.46%). The deltas are immaterial and do not touch
> any conclusion here.

## Post-gap drift is real — and still not a trade for you

Unlike the retracted first run, the negative drift now clears the pre-registered bar. Because this
is an **affirmative** claim, it is stressed harder than the null was — and it survives everything
(though see the amendment above: the effect is not specific to biotech):

| stress test | mean | t(month) |
|---|---:|---:|
| raw | −4.30% | −3.90 |
| winsorized 1% | −4.32% | −4.19 |
| winsorized 5% | −4.38% | −5.10 |
| excluding the 4 arithmetic CARs below −100% | −3.94% | −3.73 |

Plus: the sign test rejects a coin flip outright (500/1,257 positive, binomial **p = 4.3e−13**),
and **18 of 20 years** have a negative mean. It is **not a bid-ask bounce** — a bounce is a
one-day artifact, and this drift grows monotonically from −0.51% (1d) to −7.89% (60d). Median and
mean now agree (−3.91% vs −4.30%), so it is broad-based, not a few disasters.

**Three honest limits on that finding:**

- **Not retail-harvestable.** Harvesting it requires shorting, and small-biotech borrow is
  expensive or unavailable — most of all in the days right after a huge gap up, which is exactly
  when you would need it. Same limitation already recorded for the S&P 500 ADD leg in
  [index_effect_study.md](index_effect_study.md).
- ~~**The mechanism is not identified.** Post-gap names are lottery-like, and lottery stocks
  underperform in general (the MAX anomaly). This may be that well-known effect wearing a biotech
  costume rather than a catalyst-specific "sell the news." This study does not separate them, and
  does not claim to.~~ **RESOLVED by [gap_lottery_study.md](gap_lottery_study.md): it IS the
  general effect. Non-biotech gappers bleed the same −3%, and the matched biotech-minus-control
  difference is +0.28% (t = 0.01). The lottery/attention reading wins; the drug-specific reading
  is rejected.**
- **A CAR is a sum of simple abnormal returns**, so an individual event can print below −100% (4
  of them do). That is an arithmetic artifact, not a realized P&L; the robustness table above
  shows the result does not depend on those tails.

## What is actually going on

- **The money moves BEFORE the announcement.** Pre-event abnormal run-up over t−10..t−1 is
  **+9.84%, clustering-robust t = 3.73** — consistent with anticipation or leakage. **It is not a
  signal you can trade**: it is measured conditional on the gap having occurred, so it is visible
  only in hindsight. You cannot know ex ante which names will gap. The finding says where the
  money is, not how to get it.
- **It has gotten worse.** 2015–2024: −5.02% (t = −3.70) versus 2005–2014: −2.77% (t = −1.86).
- **Small and mid caps bleed; large caps do not.** Cap terciles: small −6.83% (t = −2.94), mid
  −5.43% (t = −3.30), large −0.66% (t = −0.52). The damage lives exactly where a retail buyer
  chasing a biotech headline would be shopping.
- Post-hoc industry split (**not** part of the frozen verdict): drug makers (283x) −3.88%
  (t = −3.62), commercial research (8731) −6.27% (t = −2.90). Both negative.
- The events are real catalysts: the largest gaps are VTGN 2023 (+1,186% → −57.2% over the next
  20 days), VNDA's 2009 FDA approval (+822% → +41.7%), TPST 2023 (+788% → −127.1%), Tobira's
  2016 Allergan takeover (+596% → +8.6%).

## The live out-of-sample case

Revolution Medicines / daraxonrasib (2026) sits **outside** this data window and matches the
historical pattern exactly: the stock gapped **+41%** on the topline press release, and on the day
of the ASCO plenary standing ovation it moved +3.9% and gave back −7.6% the next day. The
breakthrough was real; the tradeable part was over before anyone outside could act. One case
proves nothing — that is why the 1,257-event sample above exists.
