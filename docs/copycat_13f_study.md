# Copying the great investors' 13F filings: nothing is left, and patience does not rescue it

Pre-registered as [issue #4](https://github.com/Rhymer-Lcy/plutus-quant/issues/4), frozen before
any code existed. It began with a friend's thesis — *"股票主要靠 taste 和 patience"*, and the
follow-up, *can we extract "taste" from what the great investors say?*

**"Taste" as stated is not testable, and mining it out of interviews would be pseudo-science.**
You only hear from the ones who won; the managers with identical instincts who blew up do not give
interviews, and an interview is a story told *after* the outcome is known. Anything extracted from
that corpus is the shape of a survivor's narration — self-consistent, compelling, and with zero
ex-ante validity.

So this tests the falsifiable version: **forget what they say, look at what they do.** Institutional
managers must disclose their US long book on Form 13F, roughly 45 days after quarter end (measured
here: median 41 days). By the time you can see it, is anything left? And does patience pay?
Reproduce: `python scripts/build_crsp_cusip_map.py && python scripts/build_13f_lake.py && python scripts/build_13f_prices.py && python scripts/crsp_13f_copycat_study.py`.

## The design choice that decides whether the study is honest

Picking managers by who is famous *today* is survivorship selection — famous **because** they won.
Rather than pretend otherwise, the bias was placed **deliberately on the side of the hypothesis**:
ARM A is a hindsight-selected list of 17 legends, pinned to CIK, chosen with full knowledge of who
turned out great. That hands the hypothesis a **God's-eye view it could never have had in 2013**.
If copying *even these* earns nothing, the answer is unanswerable — because in real time you could
not have known whom to copy.

Data: SEC Form 13F structured data sets (2013Q2–2024Q3 periods, filings through 2024-12), 245,166
original 13F-HR filings from 10,943 managers; holdings joined to survivorship-free CRSP via a
point-in-time CUSIP → PERMNO map. Entry is **the close of the FILING DATE**, never the period end.
Abnormal = the name's return minus the cap-weighted market. 5 bps per side. Inference uses a
clustering-robust t on **quarterly** means (13F filings pile up on the 45-day deadline).

## Result: the primary basket (copy the legends' NEW positions)

| horizon | mean | median | **t(qtr)** | hit rate | N |
|---|---:|---:|---:|---:|---:|
| 21d | −0.37% | −0.36% | −0.47 | 48.2% | 35,102 |
| 63d | −1.11% | −0.69% | −1.08 | 48.2% | 34,406 |
| **252d (the frozen verdict)** | **−3.88%** | −1.24% | **−2.08** | 48.3% | 32,129 |
| 756d | −8.32% | −1.14% | −2.71 | 48.9% | 25,890 |

## VERDICT 1 — does copying work? **REJECTED**

No positive abnormal return, at any horizon. **But the headline number is confounded, and saying
"copying loses you 3.9% a year" would be the dishonest version of this result.**

## The size confound — the most important number in this study

The benchmark is the **cap-weighted** market, while the copied names skew **small**, and 2013–2024
was the decade mega-caps crushed small caps. Restricting to holdings that are S&P 500 members makes
holding and benchmark size-comparable:

| 252d net | mean | median | t(qtr) | hit |
|---|---:|---:|---:|---:|
| all holdings | −3.88% | −1.24% | −2.08 | 48.3% |
| **S&P 500 members only** | **+1.09%** | +1.42% | **0.59** | 52.7% |
| S&P 500 members, 756d | +2.17% | +2.76% | 1.26 | 53.4% |

**The deficit vanishes.** Most of the −3.88% was a size bet, not evidence that copying destroys
value. The honest conclusion is the plainer one: **copying the greats gives you no reliable edge —
the abnormal return is statistically indistinguishable from zero once you compare like with like.**
Forty-five days later, there is nothing left to harvest.

## VERDICT 2 — does patience pay? **REJECTED**

Frozen rule: the abnormal return must rise monotonically across horizons AND be significantly
positive at 3 years. Raw horizon means go the other way (−0.37% → −1.11% → −3.88% → −8.32%), so the
rule fails. **But that downward path is the same size artifact**: within the size-matched S&P 500
subset, longer holds are mildly *better* (+1.09% at 1 year → +2.17% at 3 years, hit rate 52.7% →
53.4%) — and still not significant. Honest reading: **patience neither reliably helps nor reliably
hurts a copied book.** It does not rescue a signal that was never there.

## Per manager (new positions, 252d net)

| manager | mean | t(qtr) | N |    | manager | mean | t(qtr) | N |
|---|---:|---:|---:|---|---|---:|---:|---:|
| Scion (Burry) | **+6.36%** | 1.37 | 156 |  | Soros | −4.40% | −2.75 | 2,537 |
| Pershing Square | +2.87% | 0.00 | 28 |  | Renaissance | −4.70% | −1.46 | 18,967 |
| Appaloosa | +1.47% | −0.71 | 282 |  | Baupost | −4.78% | −0.97 | 187 |
| Duquesne | +1.25% | 0.20 | 720 |  | Third Point | −6.60% | −1.80 | 485 |
| Coatue | +0.03% | −1.79 | 1,253 |  | Tiger Global | −11.83% | −0.85 | 397 |
| Lone Pine | −0.76% | 0.24 | 314 |  | Greenlight | −13.31% | −2.75 | 293 |
| Bridgewater | −0.97% | −1.46 | 5,090 |  | **ARK (Wood)** | **−13.84%** | −1.12 | 443 |
| Viking Global | −2.15% | −0.47 | 841 |  | Icahn | −18.12% | −2.77 | 47 |
| **Berkshire (Buffett)** | **−2.71%** | 0.07 | 89 |  |  |  |  |  |

- **Burry's +6.36% is not evidence of anything.** With 17 managers tested, a best-of-17 t of 1.37 is
  exactly what chance produces. No multiple-comparison adjustment survives it.
- **Buffett's new positions did not beat the market either** (−2.71%, t = 0.07). He opens few (89 in
  eleven years) — and by the time you see them, they are priced.
- **ARK is the archetype the friend's thesis is really about** — technological conviction — and
  copying her new positions lost 13.8% a year against the market. (Part of that is the same size and
  growth-style confound; it is not a clean skill verdict, and is not claimed as one.)

## Robustness

- Renaissance alone supplies **59%** of ARM A's events and Bridgewater another 16%, so the aggregate
  is mostly *them*. Weighting each legend equally instead: **−4.25%** — same conclusion.
- Dropping the quant/macro filers whose book turns over far faster than the disclosure lag
  (post-hoc, labelled): −3.80% at 252d (t = −2.06). Unchanged.
- Winsorized 1% / 5%: −4.28% (t = −2.48) / −4.15% (t = −3.14). Sign test 48.3% positive. 8 of 12
  years negative. The raw negative number is stable — it is just *confounded*, per the size control.

## Limitations, stated in advance and kept

- A 13F shows only **long US equity**: no shorts, no bonds, no foreign listings. A manager's
  disclosed book is **not their actual book**, so a "new position" may be one leg of a hedge. This
  measures what a copycat can see — which is the question — not the manager's skill.
- 69% of share positions and 71% of reported dollars map to priceable CRSP common stock; the rest
  are ETFs, foreign issues and junk CUSIPs.
- **ARM B (the no-look-ahead control — the 20 largest filers each quarter) did not do its job.** The
  largest 13F filers are index giants, whose "new positions" are index inclusions, not stock picks.
  Its −11.97% at 252d is the size artifact in its purest form. Reported because it was
  pre-registered; it should not be read as a skill test.

## What this says about the original claim

**Taste:** cannot be extracted from what the greats say — and what they *do*, once you can see it,
carries no harvestable edge. **Patience:** does not rescue a signal that was never there. The one
thing the data does support is narrower and less romantic: *if you must copy, copy the big, liquid,
high-conviction names and hold them* — and even then you are buying the market, not an edge.
