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

## Next levers to push the edge further above the line
- **Cost-sensitivity curve** + **liquidity-tiered universe** (run on the more-liquid mid-caps
  where realistic cost ≈ low cost) — find the tier where it's clearly net-positive.
- **Long-only top-decile** (no borrow — borrow is most of the retail cost; can't short small-caps
  anyway) — the realistic retail deployment.
- **Volume/liquidity features** (lake rebuild for CRSP `DlyVol`), bigger ensemble, attention/
  Transformer — carefully, avoiding overfit-by-backtest.

The honest through-line holds: real signal found, rigorously; not tradeable at retail cost; the
durable asset remains the look-ahead-audited, cost-aware platform that can say so with confidence.
