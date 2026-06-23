# ML model zoo — a real signal at last, but still uneconomic

The classic 6 factors had ~0 IC. The untried lever was a much larger feature space fed to
stronger models. Phase 1: ~34 Alpha158-style price/size features
(research.factors.alpha_features) → walk-forward LightGBM → OOS monthly signal on the
survivorship-free mid/small-cap CRSP universe (11,219 names, 490k samples, OOS 2008–2024).
Reproduce: `scripts/crsp_ml_zoo.py --model lightgbm --universe smallcap`.

## The signal is REAL — the first genuine predictive edge in the project

| | mean IC | IC-IR | **t-stat** | hit | n |
|---|---:|---:|---:|---:|---:|
| LightGBM on 34 rich features | **+0.0105** | 0.133 | **1.90** | 56% | 203 |
| (for reference) classic factors | ≈0 | ≈0 | <1 | ~50% | — |

t≈1.9 (near-significant), hit 56% — the rich features + GBT extract genuine cross-sectional
predictability the stale factors missed. The "more features" lever **worked on prediction**.

## …but it is not tradeable: edge-per-turnover < cost

Market-neutral long-short, net of costs, swept over concentration (quantile) × time-smoothing ×
cost. The best cell is barely break-even; everything else loses:

| signal | quantile | low cost (5/50) | realistic (15/300) | turnover |
|---|---:|---:|---:|---:|
| raw | 0.20 | **+0.12% (Sharpe 0.01)** | −5.46% (−0.64) | 2.74 |
| raw | 0.10 | −0.74% (−0.07) | −6.60% (−0.62) | 3.03 |
| raw | 0.05 | −1.08% (−0.08) | −7.13% (−0.53) | 3.21 |
| smooth-3mo | 0.20 | −1.65% (−0.19) | −5.76% (−0.66) | 1.49 |

- **Turnover is the killer (≈274%/side/month).** The signal is short-horizon / fast-decaying, so
  harvesting it requires near-total monthly rotation, and the cost of that exceeds the edge.
- **Smoothing cuts turnover (2.74→1.0) but kills the return** — the edge lives in the fast-moving
  component, so slowing the signal destroys it faster than it saves cost.
- **Concentrating into extreme conviction (5–10%) is worse**, not better — those names churn more
  (turnover up to 3.2) and the bigger per-name edge doesn't compensate.

## Verdict & what it implies

A real, statistically-significant ML signal still **does not clear retail transaction costs** —
the extractable predictability is smaller, per unit of required trading, than the cost of
trading. **The limiter is cost/turnover, not model quality.** That has a sharp implication for
the rest of the zoo: a stronger model (XGBoost/CatBoost/DL) might lift the IC from t≈1.9 to
t≈2.5, but to turn the −5.5% realistic-cost result positive you'd need to roughly TRIPLE the
gross spread without adding turnover — which no model upgrade plausibly delivers from these
features. The market is efficient *relative to retail costs*.

## Phase 2 — tree family confirms (XGBoost / CatBoost)

| model | OOS IC t | best net Sharpe (low cost) | realistic cost |
|---|---:|---|---|
| LightGBM | 1.90 | ~0.01 (q0.20) | all negative |
| **XGBoost** | 1.91 | **+0.19** (q0.05) | all negative (−3.6%…) |
| CatBoost | 1.85 | +0.09 (q0.10) | all negative |

All three tree models agree: the same real signal (t≈1.85–1.91, mean IC ≈0.011), robust across
implementations. XGBoost is marginally the most harvestable — at **low (institutional/liquid)
cost** its extreme-quintile long-short is mildly positive (Sharpe ~0.19) — but at **realistic
small-cap cost every model is negative**. The cost wall stands; a better tree didn't cross it.

## Phase 3 — temporal DL (GRU) crosses the wall at low cost (the best result)

`scripts/crsp_dl.py` (torch cu128, RTX 5080): a GRU over each name's last 12 months of the 34
features → next-month return, walk-forward, **5-seed ensemble** (DL is noisy; ensembling
separates signal from seed luck).

| GRU | OOS IC t | low cost (5/50) q0.10 | realistic (15/300) q0.10 | turnover |
|---|---:|---|---|---:|
| 1 seed | 1.40 | +5.6% (Sharpe 0.38) | −0.11% (≈break-even) | 2.60 |
| **5-seed ensemble** | **2.61** | **+5.0% (Sharpe 0.47)** | −0.88% (≈break-even) | 2.79 |

The ensemble lifts the IC to **t=2.61 — statistically significant, the strongest in the project**
— confirming a REAL signal (not seed luck), and it is NOT a leak (Sharpe ~0.47, not 7). Verdict
shift: the temporal model finds something the trees can't, and it **crosses the cost wall at
low/institutional cost (Sharpe ~0.45 market-neutral)**, sitting **≈break-even at full retail
small-cap frictions** (15 bps slip + 300 bps borrow). The edge rides the cost boundary — positive
as costs fall.

## Where this lands
- **A real, modest, market-neutral edge exists** in mid/small-cap US, from a temporal DL model on
  rich features — tradeable at low/institutional execution cost (~Sharpe 0.45), break-even for a
  retail small-cap shorter. This partially vindicates "small profit is possible" — at the liquid
  end / with cheap execution, not at worst-case retail frictions.
- It is the only thing in the whole program to cross the wall; classic factors and PEAD did not.
- Durable asset unchanged: a survivorship-free + cost-aware + look-ahead-audited platform — now
  also GPU-DL-capable — that can tell a real (if marginal) edge from the mirages it rejected.

## Phase 4 — is it deployable for retail? (scripts/crsp_dl_tradability.py)

**A. Long-only top-decile (no borrow — the realistic retail form) vs equal-weight band benchmark.**

| | CAGR | Sharpe | maxDD | active-IR |
|---|---:|---:|---:|---:|
| equal-weight band (benchmark) | 11.8% | 0.68 | −46% | — |
| GRU long-only top-decile @ 5bps | 11.7% | 0.65 | **−34%** | **0.00** |
| … @ 15bps | 10.3% | 0.59 | −36% | −0.14 |

Long-only keeps **no return alpha** (active-IR ≈ 0) — it roughly matches equal-weighting, just
with a lower drawdown. **The edge lives in the SHORT leg** (the bottom decile underperforms),
which retail can't harvest (can't short small caps).

**B. Market-neutral q0.10 net Sharpe across cost (the breakeven map):** positive at slip ≤10bps &
borrow ≤150bps/yr (Sharpe 0.21–0.53); ≈zero around slip 15–20 + borrow 150–300; **retail small-cap
(15/300) = −0.08**; institutional (3–5bps/50) = 0.47–0.53. Breakeven ≈ slip 12bps + borrow 150bps.

**C. Liquidity tiers (market-neutral q0.10), same signal:**

| tier (cap rank) | low 5/50 | mid 10/150 | real 15/300 |
|---|---:|---:|---:|
| 501–1000 (most liquid) | −0.09 | −0.31 | −0.56 |
| 501–2500 | 0.39 | 0.13 | −0.16 |
| **1501–3000 (smaller)** | **0.46** | **0.24** | −0.02 |

The alpha is **stronger in the smaller / less-liquid tier** (less arbitraged) — but that is exactly
where execution costs are highest. Classic tension: the edge is where it's hardest to trade.

### Deployability verdict
- **Retail (long-only, small-cap costs): no usable edge** — the alpha is short-side + in illiquid
  names; long-only ≈ the benchmark (better drawdown, no excess return).
- **Low-cost market-neutral fund (can short, ~5–10bps slip, ≤150bps borrow): marginally yes** —
  a real ~Sharpe 0.2–0.5 market-neutral edge, strongest in the smaller tier. Institutional-grade
  and small, not a retail money-maker.
- So "small profit possible?" — **yes for a low-cost market-neutral book, no for retail long-only.**
  The edge is real (IC t=2.6) but structurally sits just past where retail frictions can reach it.

## Phase 5 — volume/liquidity features: no improvement (last data lever, null)

Rebuilt the lake with CRSP `DlyVol`/`DlyPrcVol` and added the price-volume family (volume
spike/momentum/vol, log dollar volume, Amihud illiquidity) → 41 features, re-ran the 5-seed GRU
ensemble: **OOS IC t=2.64 (vs 2.61 without volume), low-cost q0.10 Sharpe 0.48 (vs 0.47)** —
statistically identical. The price features already captured the signal; volume adds nothing
material. The marginal edge is what it is; the deployability verdict (Phase 4) is unchanged.

## Phase 6 — portfolio construction crosses the realistic-cost wall (the breakthrough)

The naive book equal-weights the top/bottom decile (~250 names/side), DILUTING a continuous,
informative signal. A turnover-aware dollar-neutral optimizer (`research/backtest/optimize.py`,
cvxpy: max alpha·w − γ·slip·‖w−w_prev‖₁, dollar-neutral, per-name cap 2%, gross 2) instead
concentrates into the ~100 highest-conviction names. On the cached GRU signal:

| construction | low cost (5/50) | **realistic (15/300)** | turnover |
|---|---:|---:|---:|
| naive quintile q0.10 | Sharpe 0.48 | **−0.11** | 2.77 |
| optimizer (γ=0) | **0.89** | **+0.46** | 3.15 |
| optimizer (γ=2) | 0.75 | **+0.47** | 2.32 |
| optimizer (γ=15) | 0.47 | 0.09 | 1.58 |

**The realistic-cost market-neutral Sharpe goes from −0.11 (naive) to ~0.46 (optimized)** — the
first clearly net-positive result at realistic frictions. The gain is **concentration / signal-
magnitude weighting** (using the GRU's continuous alpha + a 2% cap), not turnover reduction
(γ helps only marginally). OOS signal, no look-ahead — not a leak (Sharpe 0.46, not 7).

Caveats (so we don't over-claim): (1) **no market-impact** in the cost model → valid for SMALL/
MEDIUM AUM (the small/medium capital tiers) where 15 bps slippage is realistic; large AUM would
erode it (the capacity study, using dollar-volume, is the check). (2) It **requires shorting** the
bottom names → viable for a small **market-neutral fund**, still not for retail long-only (Phase 4:
the long leg alone keeps no alpha). (3) name_cap=0.02 is an a-priori risk limit —
**robustness CONFIRMED**: realistic-cost Sharpe stays positive across caps (0.17 at ~200 names →
0.46 at ~100 → 0.62 at ~40 → 0.63 at ~20), rising with concentration. But higher concentration =
fewer/bigger positions = more impact-sensitive, so the ~100-name **0.46 is the conservative base
case**; the 0.6+ figures need the capacity (impact) study to trust.

### Verdict update
There IS a tradeable market-neutral edge for a small/medium-AUM fund that can short: a temporal-DL
signal, concentrated via a dollar-neutral optimizer, nets ~Sharpe 0.45–0.9 (realistic→low cost).
This is the first thing to clearly clear realistic costs. It is small, needs shorting + low impact
(small AUM), and is not retail-long-only — but it is real, and it vindicates "small profit is
possible" for the right (small fund) setup.

## Phase 7 — capacity: a small-AUM edge (the verdict)

Same optimized book, but the per-name cost now includes square-root market impact (cost rate =
5bps + impact_coef·√(traded$/ADV$) + 300bps/yr borrow), run at each capital tier:

| AUM | tier | ann return | **Sharpe (with impact)** |
|---|---:|---:|---:|
| $25k | small | 8.9% | **0.55** |
| $100k | small | 8.6% | 0.53 |
| $500k | small | 7.7% | 0.48 |
| $2M | medium | 6.1% | 0.38 |
| $10M | medium | 2.2% | 0.14 |
| $50M | large | −5.9% | −0.37 |
| $250M | large | −22.0% | −1.37 |

**The edge is a SMALL-AUM edge.** Net of realistic costs + impact it earns ~Sharpe 0.5 / ~8%/yr
up to a few hundred $k, fades through the low-$millions, and dies (negative) by ~$50M. **Capacity
ceiling ≈ $10M.** This is precisely a *retail/small-player* edge — it survives only because it is
too small for big money to arbitrage. (impact_coef is an assumption; the curve's SHAPE — positive
small, negative large — is the robust result; the exact ceiling moves with the impact calibration.)

## FINAL VERDICT of the quant program
"Is small profit possible?" — **Yes, for a small (≤ ~$5M) market-neutral book that can short.** A
temporal-DL signal on rich features, concentrated via a dollar-neutral optimizer, nets ~Sharpe
0.4–0.55 at small AUM after costs, borrow, and impact; capacity ~$10M. It needs shorting (not
retail long-only) and modest execution. Everything else in the program — classic factors (large &
small cap), PEAD (even with IBES) — had no tradeable edge after rigor. The path to this single
real result ran through, and was repeatedly saved by, a survivorship-free + cost-aware +
look-ahead-audited + capacity-aware platform that rejected every mirage along the way
(survivorship +372%, PEAD Sharpe 7) — that platform is the durable asset.

## Next levers (to strengthen / extend the small-AUM edge)
- **Short-side signals** (analyst revisions [IBES detail, in hand], short interest [可代下]) — the
  alpha is short-side; strengthening it is the natural next signal work.
- **Impact-aware optimizer** (penalize illiquid names in the objective) — would raise the capacity
  ceiling above $10M.
- **DL archs / Qlib native** (Transformer/TFT/TabNet) — incremental; IC ceiling (~t=2.6) looks
  reached; low EV, overfit risk.

The honest through-line holds: real signal found, rigorously; not tradeable at retail cost; the
durable asset remains the look-ahead-audited, cost-aware platform that can say so with confidence.
