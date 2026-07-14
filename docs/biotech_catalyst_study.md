# Biotech catalysts: the move is over before you can act — and the pre-registration says so

Pre-registered as [issue #3](https://github.com/Rhymer-Lcy/plutus-quant/issues/3), frozen
before any code existed. The question, raised by a friend's pancreatic-cancer example: **when a
biotech catalyst hits the tape, can someone who acts AFTER the news still earn an abnormal
return?** Reproduce: `python scripts/build_crsp_biotech_lake.py && python scripts/crsp_biotech_catalyst_study.py`.

Universe: pharma/biotech by SIC (2833–2836, 8731) on the survivorship-free CRSP lake, 2005–2024,
common stock, price ≥ $5, cap ≥ $100M — 1,100 names including every one that later died. Event:
an overnight gap ≥ +20%, decomposed split- and dividend-immune from the total return
(`overnight[t] = (1+DlyRet[t])/(1+intraday[t]) − 1`). **1,111 events across 571 names.** Material
biotech data is released outside trading hours by design, which is precisely why it arrives as a
gap nobody can trade into. Abnormal = the name minus the same-day equal-weight biotech mean; cost
= the name's OWN CRSP half-spread on each side (median 0.078% on the event day).

**The gap itself — what you never get: mean +50.4%, median +33.8%, max +822%.**

## Result: nothing is left after the news

| entry, horizon | gross | NET | median (net) | t(event) | **t(month)** | hit rate |
|---|---:|---:|---:|---:|---:|---:|
| close, 1d | +0.63% | +0.15% | −0.54% | 0.43 | −0.34 | 42.8% |
| close, 5d | −0.64% | −1.08% | −1.23% | −2.34 | −1.54 | 44.9% |
| close, 20d **(the frozen verdict)** | −1.24% | **−1.64%** | −2.62% | −2.57 | **−0.78** | 42.2% |
| close, 60d | −2.33% | −2.73% | −4.97% | −2.77 | −1.50 | 39.3% |
| open, 1d | +1.85% | +1.38% | −1.35% | 1.70 | 1.46 | 42.9% |
| open, 20d | −0.01% | −0.42% | −3.99% | −0.43 | 0.70 | 41.1% |

t(month) is the clustering-robust statistic the pre-registration committed to: biotech catalysts
cluster in time (ASCO/ESMO, JPM healthcare week, PDUFA dates), and an event-level t treats
clustered events as independent draws.

## Verdict: REJECTED

The close-entry 20-day net abnormal return is **−1.64%** with a clustering-robust **t = −0.78** —
not positive, not significant. Under the frozen rule, the answer to "can you still buy it after
the news" is **no**. Buying at the open instead of the close captures a small day-1 bounce
(+1.85% gross) that neither persists nor clears significance — and that figure is *optimistic*,
because CRSP quotes the closing bid/ask while a real catalyst-day opening spread is far wider.

**"Sell the news" is NOT established either, and the distinction matters.** The event-level
t of −2.57 looks significant, and quoting it would be the easy story. The pre-registered
statistic is the clustering-robust one (t = −0.78), and it does not clear the bar. The point
estimate is negative and the median is clearly negative, but this study does not claim a
significant negative drift. Even if it did, harvesting it needs shorting, and small-biotech borrow
is expensive or unavailable — the same limitation already recorded for the S&P 500 ADD leg in
[index_effect_study.md](index_effect_study.md).

## What is actually going on

- **The typical trade loses; the mean is propped up by a handful of monsters.** Median 20-day net
  is −2.62% (close entry) and −3.99% (open entry); the hit rate is 42%. The top 20 events — **1.8%
  of the sample** — contribute +1.32 pp to the mean; excluding them, the mean is **−3.02%**. The
  dispersion is brutal (10th percentile −25.8%, 90th +21.3%). This is a lottery, and it is the
  exact machine that manufactures survivor stories: roughly one event in fifty is a monster
  everyone remembers, while the other forty-nine quietly bleed.
- **The money moves BEFORE the announcement.** The one statistically robust effect in the whole
  study is the pre-event run-up: **+4.98% abnormal over t−10..t−1, clustering-robust t = 3.47** —
  consistent with anticipation or leakage. **This is not a signal you can trade**: it is measured
  conditional on the gap having occurred, so it is visible only in hindsight. You cannot know
  ex ante which names will gap. The finding says where the money is, not how to get it.
- **It has gotten worse, not better.** 2015–2024 net −2.17% (t = −1.99) versus 2005–2014 −0.33%
  (t = +0.66) — the post-catalyst window is more thoroughly picked over now than it was.
- The events are real: the largest gaps are VNDA's 2009 FDA approval (+822%), TPST 2023 (+788%),
  Tobira's 2016 Allergan takeover (+596%), Seres' 2020 phase-3 success (+334%).

## The live out-of-sample case

Revolution Medicines / daraxonrasib (2026) sits **outside** this data window and matches the
historical pattern exactly: the stock gapped **+41%** on the topline press release, and on the day
of the ASCO plenary standing ovation it moved +3.9% and gave back −7.6% the next day. The
breakthrough was real; the tradeable part was over before anyone outside could act. One case
proves nothing — that is why the 1,111-event sample above exists.
