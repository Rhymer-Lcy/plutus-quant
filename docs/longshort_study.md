# Market-neutral long-short — the honest verdict on classic factors

The long-only studies showed factors can't beat the index (a long book is ~90% market beta).
The real test of a factor is its **dollar-neutral spread**: long the top quintile, short the
bottom quintile, beta stripped, after turnover + borrow costs. If there's alpha, it shows up
here as a positive net Sharpe with beta ≈ 0. Survivorship-free CRSP, PIT S&P 500, 2005–2024
(ML combiner from 2012), monthly, 5 bps/side slippage + 50 bps/yr borrow. Reproduce:
`scripts/crsp_longshort_study.py`.

## Quintile long-short, net of costs

| signal | ann return | ann vol | **Sharpe** | max DD | market beta | turnover |
|---|---:|---:|---:|---:|---:|---:|
| earnings_yield | −1.88% | 11.3% | −0.17 | −44.6% | 0.18 | 0.46 |
| book_yield | −2.53% | 13.4% | −0.19 | −57.3% | 0.26 | 0.28 |
| roe (quality) | **+1.07%** | 10.0% | **+0.11** | −27.6% | −0.18 | 0.28 |
| reversal_1m | −2.09% | 14.2% | −0.15 | −53.8% | 0.35 | 3.13 |
| momentum_12_1 | −4.53% | 19.5% | −0.23 | −75.2% | −0.51 | 0.95 |
| low_vol | −3.46% | 21.4% | −0.16 | −78.9% | −0.94 | 0.28 |
| candidate (val+rev) | −3.06% | 13.7% | −0.22 | −57.1% | 0.33 | 2.18 |
| ML combiner | −4.19% | 7.6% | −0.55 | −42.9% | 0.13 | 2.19 |

## Verdict: no tradeable edge in classic factors on large-cap US

- **Every factor's net long-short Sharpe is ≈0 or negative.** Only quality (roe) is barely
  positive (+1.1%/yr, Sharpe 0.11) — economically negligible and not robust.
- **The ML combiner is the worst (Sharpe −0.55)** — combining edgeless factors just overfits.
- Betas aren't perfectly zero (low_vol −0.94, momentum −0.51): those factors carry structural
  market exposure by construction, so they aren't even cleanly neutral — and still no alpha.
- reversal's turnover is 313%/side/month → costs alone would bury any raw signal.

This is the correct, well-documented modern result: **classic academic factors on large-cap US
equities have been largely arbitraged away (especially post-~2003), and after realistic costs +
borrow a retail trader cannot extract them.** Long-only they're just beta; long-short they're
flat-to-negative.

## What this whole arc established
Across [first_study](first_study.md) → [survivorship_study](survivorship_study.md) →
[multifactor_study](multifactor_study.md) → this: a rigorous, survivorship-free, out-of-sample,
cost-aware, benchmarked, long-only AND market-neutral evaluation. The honest output is **"no
edge here"** — which is the *valuable* result. The +372% we started with was a survivorship +
look-ahead mirage; under rigor it vanishes. Most retail "edges" are exactly such artifacts; this
pipeline is built to refuse them.

## Where edge could still plausibly live (honest research frontier)
- **Smaller / less-liquid universe** (mid/micro caps): arbitrage capital can't fish there, so
  factor premia may survive — but borrow is hard/expensive and capacity is tiny. The CRSP lake
  can be widened beyond the S&P 500 to test this directly (the next clean experiment).
- **Non-classic signals**: alternative data, shorter horizons, cross-sectional interactions —
  validated with this same survivorship-free, cost-aware harness.
- **Sober meta-conclusion**: beating efficient large-cap US markets is genuinely hard. The
  durable contribution of plutus is the *methodology that tells the truth*, reusable for any
  future signal — not this (null) result on stale factors.
