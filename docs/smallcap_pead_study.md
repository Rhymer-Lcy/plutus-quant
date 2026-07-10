# Small-cap PEAD — the first real gross edge, gated by cost

> **RETRACTED / CORRECTED (see docs/ibes_pead_study.md).** The headline result below
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
`scripts/crsp_smallcap_pead_study.py [--slippage-bps --borrow-bps-annual]`.

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
   *(Executed — see [ibes_pead_study.md](ibes_pead_study.md): the drift sharpened but stayed
   untradeable.)*
2. **Cost/borrow-aware selection** — trade only easy-to-borrow names; size by liquidity; optimize
   the hold for net (not gross) drift. *(The cost half executed below.)*
3. **Faster entry (next open vs next close)** — CRSP has `DlyOpen`; entering at the open captures
   more of the front-loaded day-1 jump. *(Executed below.)*

## Paths #2 and #3, executed: open entry + per-name half-spread costs — still REJECTED

`scripts/crsp_smallcap_pead_open_study.py` (de-leaked throughout; window pinned to 2010-2024;
same |SUE| >= 1.5 and holds as above; open-price coverage 99.9% of tradable events). Entry moves
to the next day's OPEN — leak-free, since the filing is public before the open and entry-day
accrual is the same-day open-to-close ratio — and turnover is charged at each name's own quoted
half-spread (median 7.7 bps) instead of a flat 20 bps per side.

| long-short variant | hold 10 | hold 20 | hold 40 | hold 60 |
|---|---:|---:|---:|---:|
| close entry, flat 20 bps, borrow 300 (baseline) | −22.2% | −16.3% | −11.5% | −5.6% |
| OPEN entry, flat 20 bps, borrow 300 | −19.9% | −14.5% | −9.6% | −5.7% |
| OPEN entry, per-name half-spread, borrow 300 | −5.8% | −4.9% | −3.6% | −2.7% |
| OPEN entry, per-name half-spread, borrow 50 | −4.3% | −3.3% | −1.9% | −0.9% |
| **long leg only (retail form), half-spread** | **+1.7%** | **+7.0%** | **+8.0%** | **+9.7%** |

Three findings, two of which correct this document's own cost narrative:

1. **Faster entry is real but small**: ~+2 pp/yr at the short holds — not the missing edge.
2. **The flat 20 bps "realistic" cost was over-punitive by ~14 pp/yr at hold-10.** The names this
   strategy actually trades (SEC filers with analyst attention) quote far tighter than a
   small-cap-wide average; per-name honest costs shrink the loss dramatically. The earlier
   "destroyed at realistic cost" magnitude overstated the destruction — but does not reverse it.
3. **The boundary still is not crossed, and the failure now has a precise shape.** The long-short
   stays negative at every hold even at low borrow, because the SHORT leg subtracts alpha plus
   borrow — and the long leg alone, though positive (+9.7%/yr, Sharpe 0.53 at hold-60), sits
   BELOW the band's buy-and-hold (+10.5%/yr, Sharpe 0.61): it is beta minus costs, not
   harvestable drift. A retail account, which cannot short small caps anyway, does better
   indexing the band than trading its PEAD events.

Per the pre-registration, no threshold or hold tuning follows. The family is closed: every named
path has now been executed and measured, and small-cap PEAD remains a real anomaly that is not
retail-harvestable in any tested form.

## Caveats
- **Event-side survivorship**: only 2,567 of 8,395 band names resolved to a current SEC CIK
  (delisted/no-XBRL small-caps drop out), so the EVENT set tilts to survivors and the gross edge
  is likely somewhat optimistic. Prices/returns are survivorship-free; the surprise events are not.
- Concentrated long-short → large drawdowns (−33% to −90%); needs risk sizing.
- SEC XBRL begins ~2009, so the test is 2010–2024 (post-GFC).
