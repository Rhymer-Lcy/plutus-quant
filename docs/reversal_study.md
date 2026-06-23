# Short-term reversal: a real liquidity-provision premium, not a retail edge

The factor arc (`longshort_study.md`, `smallcap_study.md`) tested *monthly* reversal and dismissed it
as a high-turnover illusion. This closes out the canonical **short-term reversal** properly — the
1–2 week horizon, weekly rebalance, where the effect is largest — and separates the two reasons it
is not tradeable, head on.

Dollar-neutral quintile long-short (`research/backtest/long_short.py`), survivorship-free CRSP, both
the large-cap S&P 500 lake and the mid/small-cap cap-rank band. No-look-ahead: signal read at t,
return realized t→t+1. Reproduce: `python scripts/crsp_reversal_study.py`.

## Two illusions, tested directly

1. **Bid-ask bounce.** A name that closed near the *bid* looks like a loser and "reverts" when it
   next closes near the *ask* — microstructure, not tradeable profit, because the engine reads the
   signal and enters at the same closing print. Control: a **skip-1-day** variant forms the signal
   through close[t−1] (via `.shift(1)`) while still entering at close[t], so the last signal price
   and the entry price are different prints. The drop from no-skip to skip-1d is the bounce
   contribution (Jegadeesh 1990 / Lehmann 1990 / Nagel 2012).
2. **The turnover cost wall.** Weekly reversal turns the book over almost completely each week
   (two-sided turnover ≈ 3.1), so even light per-trade slippage compounds into a large annual drag.

## Result — gross is real, net is dead

Weekly, quintile (20/20), beta-reported. Realistic cost: large-cap 5bps/side + 50bps borrow;
mid/small 15bps/side + 300bps borrow.

| universe | signal | gross Sharpe | gross ann | **net Sharpe** | net ann | turnover | beta |
|---|---|--:|--:|--:|--:|--:|--:|
| large | rev_1w no-skip | 0.59 | +11.0% | **0.10** | +1.9% | 3.09 | 0.41 |
| large | rev_1w skip-1d | 0.49 | +8.9% | **−0.00** | −0.1% | 3.08 | 0.34 |
| small | rev_1w no-skip | 1.01 | +15.3% | **−0.79** | −12.0% | 3.07 | 0.33 |
| small | rev_1w skip-1d | 0.73 | +11.0% | **−1.02** | −15.2% | 3.06 | 0.29 |

Pooled inference on the weekly long-short return series (vs 0):

| universe | series | n | ann | Sharpe | t | p |
|---|---|--:|--:|--:|--:|--:|
| large | gross no-skip | 1042 | +11.0% | 0.65 | +2.90 | 0.004 |
| large | gross skip-1d | 1042 | +8.8% | 0.56 | +2.50 | 0.013 |
| large | **net** | 1042 | +1.9% | 0.19 | +0.85 | 0.397 |
| small | gross no-skip | 1094 | +15.3% | 1.01 | +4.63 | 0.000 |
| small | gross skip-1d | 1094 | +10.9% | 0.77 | +3.53 | 0.000 |
| small | **net** | 1094 | −11.9% | −0.76 | −3.50 | 0.000 |
| small | net 2005-2019 | 780 | −13.3% | −1.09 | −4.21 | 0.000 |
| small | net holdout 2025 | 53 | −21.9% | −2.23 | −2.25 | 0.029 |

Three things follow:

- **The gross premium is real and large**, strongest in small caps (gross Sharpe 1.01, t=4.6). It is
  not a survivorship or look-ahead artifact (clean engine, PIT universe).
- **It is only partly the bid-ask bounce.** Skipping one day removes ~17% of the large-cap and ~28%
  of the small-cap gross Sharpe, but the rest survives (skip-1d still t=2.5 / 3.5) — there is genuine
  short-horizon mean reversion underneath. This is the textbook **liquidity-provision premium**: the
  compensation for stepping in front of short-term order-flow imbalances.
- **Net of realistic cost it is not tradeable.** The weekly turnover (≈3.1) at retail spreads
  erases the large-cap edge to a statistical zero (net Sharpe 0.10, t=0.85) and turns the small-cap
  edge significantly *negative* (−0.76, t=−3.50) — in the full sample, in the 2005-2019 sub-period,
  and in the 2025 holdout. Large-cap rev_1w reaches a positive net Sharpe (~0.40, +7.5%) *only* under
  an aggressive 2bps/side, zero-borrow assumption — which is not a credible fill for the names
  reversal actually trades: the week's biggest losers and winners carry the widest spreads and the
  most impact, exactly where 2bps is unrealistic. At a realistic 5bps/side + borrow it is a
  statistical zero (large-cap) or significantly negative (small-cap).

## Why it persists, and the verdict

Gross reversal is biggest exactly in the years it cannot be captured — crisis/high-volatility years
(2008, 2020, 2022), when spreads and impact are widest. That is the signature of a
liquidity-provision premium (Nagel 2012): the return is **the price of immediacy**, and it accrues to
the market maker who *posts* liquidity and earns the spread, not to a retail trader who *crosses* the
spread to chase it. At retail you are on the wrong side of the bounce and you pay the turnover wall.

Short-term reversal joins the list of US-equity families with no retail-tradeable edge after rigor
(classic factors large+small, PEAD incl. IBES, ML/GRU cross-sectional, pairs stat-arb). It is real —
and it is owned by whoever provides the liquidity, which is not us.
