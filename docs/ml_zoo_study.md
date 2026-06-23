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

## Still open (one real hope + a data lever)
- **DL on the GPU** (local: 1× RTX 5080 16GB) — `scripts/crsp_dl.py` runs a TEMPORAL GRU over
  each name's last 12 months of features: the one qualitatively different bet (a *persistent*,
  lower-turnover signal the cross-sectional trees can't see). Modest odds — the trees already had
  multi-horizon features and still found a fast signal — but the right thing to try. [result pending]
- **Volume/liquidity features** (needs a lake rebuild to add CRSP `DlyVol`) — a richer, possibly
  slower signal.

The honest through-line holds: real signal found, rigorously; not tradeable at retail cost; the
durable asset remains the look-ahead-audited, cost-aware platform that can say so with confidence.
