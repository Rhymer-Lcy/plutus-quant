# Small-cap PEAD — the first real gross edge, gated by cost

> ⚠ **RETRACTED / CORRECTED (see docs/ibes_pead_study.md).** The headline result below
> (+10.3%/yr, Sharpe 0.41 at low cost) was a **look-ahead artifact**: with close-to-close
> returns, entering the day after the announcement captured the post-close overnight GAP (the
> announcement jump), not tradeable drift. Fixed with `entry_offset=1` (skip the reaction day).
> De-leaked, the same strategy is **−5.2%/yr (Sharpe −0.25) at hold-10, negative at every
> horizon** — NOT a tradeable edge. The CAAR below is similarly gap-inflated (de-leaked Q5−Q1
> drift is ~+0.6% over 60 days, not +1.3%). Read this doc only with that correction in mind.

The large-cap event-time PEAD sat on the cost boundary (docs/pead_event_study.md). The
literature says the drift is larger in small caps. This tests it: SUE earnings-surprise events
for the mid/small cap-band (rank 501–3000), event-clock CAAR + overlapping long-short, on
survivorship-free CRSP. 74,194 events, 2,567 names, 2010–2024 (SEC XBRL era). Reproduce:
`scripts/crsp_smallcap_pead.py [--slippage-bps --borrow-bps-annual]`.

## The drift is ~2× the large-cap drift (CAAR)

Top-minus-bottom (Q5−Q1) cumulative abnormal return, in event time:

| event day | 5 | 10 | 20 | 40 | 60 |
|---|---:|---:|---:|---:|---:|
| small-cap Q5−Q1 | **+1.09%** | +1.29% | +1.03% | +1.20% | **+1.35%** |
| (large-cap, for ref) | +0.33% | +0.55% | +0.62% | +0.68% | +0.55% |

Confirmed: the PEAD drift is **roughly twice as large in mid/small caps**, and front-loaded
(~+1.1% by day 5). The hypothesis ("premia survive where capital can't fish") is right — gross.

## At low cost it WORKS; at realistic small-cap cost it's destroyed

Event-time long-short, |SUE| ≥ 1.5, by hold horizon:

| hold (days) | low cost (5 bps + 50 bps/yr) | realistic small-cap (20 bps + 300 bps/yr) |
|---|---|---|
| **10** | **+10.27%, Sharpe 0.41**, DD −43% | −9.54%, Sharpe −0.38, DD −90% |
| 20 | +5.48%, Sharpe 0.28 | −7.83%, Sharpe −0.40 |
| 40 | +4.05%, Sharpe 0.33 | −4.64%, Sharpe −0.38 |
| 60 | +1.90%, Sharpe 0.18 | −3.30%, Sharpe −0.32 |

**This is the first real gross edge in the whole project** — small-cap PEAD held ~10 days
returns ~+10%/yr at Sharpe 0.41 at idealized costs. But the same strategy at realistic small-cap
frictions (wide spreads + ~3%/yr borrow on harder-to-borrow names) flips to −9.5%: the bigger
drift is more than offset by the bigger cost of accessing it.

## The conclusion flips: not "no signal" — a signal gated by execution cost

Unlike the (genuinely empty) valuation factors, small-cap PEAD **has a real, economically-
meaningful gross edge**. Whether it is *net* tradeable is decided entirely by execution cost and
borrow, which sit right around the edge:
- The honest answer lives **between** the two runs; real costs for a rank-501–3000 book are
  ~10–25 bps/side slippage and 50–1000+ bps/yr borrow depending on the name and the date.
- High turnover is the enemy: hold-10 has the biggest gross edge but the most turnover; longer
  holds cut turnover but annualize less drift. There is a cost-dependent optimal hold.
- So small-cap PEAD is a **borderline-tradeable execution problem**, not a dead signal — the
  first result that points somewhere real.

## How to push it over the line (the actionable path)
1. **Sharper surprise (IBES analyst consensus)** — separating the surprise quintiles better
   gives more drift PER NAME, so fewer/larger trades clear the cost hurdle. The seasonal-random-
   walk SUE here is crude; an analyst-consensus surprise (via IBES, obtainable through WRDS) is
   the highest-value upgrade and plugs straight into this same event harness.
2. **Cost/borrow-aware selection** — trade only easy-to-borrow names; size by liquidity; optimize
   the hold for net (not gross) drift.
3. **Faster entry (next open vs next close)** — CRSP has `DlyOpen`; entering at the open captures
   more of the front-loaded day-1 jump.

## Caveats
- **Event-side survivorship**: only 2,567 of 8,395 band names resolved to a current SEC CIK
  (delisted/no-XBRL small-caps drop out), so the EVENT set tilts to survivors and the gross edge
  is likely somewhat optimistic. Prices/returns are survivorship-free; the surprise events are not.
- Concentrated long-short → large drawdowns (−33% to −90%); needs risk sizing.
- SEC XBRL begins ~2009, so the test is 2010–2024 (post-GFC).
