# Do factor premia survive in mid/small caps? — no, under rigor

The large-cap arc concluded classic factors have no tradeable edge (docs/longshort_study.md).
The standard hope is that premia survive where arbitrage capital can't fish — **mid/small caps**.
This tests it directly: a broad survivorship-free CRSP universe (11,219 common stocks, major
exchange, price ≥ $5, cap ≥ $100M, 2005–2024), restricted to the cap-rank **band 501–3000**
(mid/small, ~2,500 names), market-neutral quintile long-short on price factors, two cost levels.
Reproduce: `scripts/build_crsp_smallcap_lake.py` → `scripts/crsp_smallcap_longshort_study.py`.

## Quintile long-short on the mid/small band

**Low cost (5 bps/side slip + 50 bps/yr borrow — same as the large-cap test, "does the premium exist?")**

| factor | ann return | Sharpe | max DD | beta | turnover |
|---|---:|---:|---:|---:|---:|
| reversal_1m | +1.83% | 0.16 | −26.1% | 0.28 | **3.13** |
| momentum_12_1 | −2.13% | −0.14 | −62.2% | −0.26 | 1.08 |
| low_vol | −3.41% | −0.20 | −69.6% | −0.68 | 0.39 |
| mom+lowvol | −2.13% | −0.13 | −65.8% | −0.55 | 0.91 |

**Realistic small-cap (15 bps/side slip + 300 bps/yr borrow — small caps trade wide & are hard to borrow)**

| factor | ann return | Sharpe | max DD |
|---|---:|---:|---:|
| reversal_1m | −4.29% | −0.37 | −59.0% |
| momentum_12_1 | −5.77% | −0.38 | −74.3% |
| low_vol | −6.24% | −0.37 | −78.9% |
| mom+lowvol | −5.58% | −0.34 | −78.6% |

## Verdict: the small-cap hypothesis is rejected under rigor

- At low cost, the only positive is reversal (+1.8%/yr, Sharpe 0.16) — but it churns **313% per
  side per month**. It is a pure **transaction-cost illusion**: at realistic small-cap costs it
  flips to **−4.3%/yr (Sharpe −0.37)**.
- Momentum and low-vol are **negative even gross** in this band/period (the 2009 momentum crash
  + their structural negative beta), and worse net.
- At realistic costs **every factor is solidly negative (Sharpe −0.34 to −0.38)**.
- Practical reality check: shorting mid/small caps is genuinely hard/expensive (locate + high
  borrow), so even the gross spreads aren't really harvestable long-short — the realizable
  version is a long-only tilt, which is just beta plus these (absent) premia.

So classic price factors have **no tradeable edge in US equities — large OR small cap — after
realistic costs.** The "go smaller" escape hatch closes too.

## Where this leaves the research
Across the whole arc (large-cap long-only → survivorship → multifactor/benchmark → long-short →
mid/small-cap), the rigorous, survivorship-free, cost-aware, OOS verdict is consistent: **the
stale academic factors are arbitraged/cost-dominated everywhere we can actually trade.** The
contribution is the methodology that establishes this honestly. The honest remaining frontier is
NOT classic factors anywhere — it is **non-classic / alternative signals** (alt-data, shorter
horizons, event-driven, cross-asset), each to be run through this same harness, which will keep
refusing the survivorship/overfitting/turnover mirages that make most retail "edges" evaporate.

### Caveats
- Momentum's negative gross result is influenced by the 2009 crash and the specific band/period;
  crash-filtered momentum variants differ academically, but the net-of-cost reality holds.
- Betas aren't 0 (low_vol −0.68): these price factors carry structural market exposure, so they
  aren't cleanly neutral — yet still no positive alpha.
- Fundamentals (value/quality) weren't tested in small caps (SEC ticker-join coverage is poor
  there); the price-factor result is the clean, decisive test.
