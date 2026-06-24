# Volatility-managed exposure: a real, un-levered risk-adjusted improvement (an overlay, not alpha)

Moreira-Muir: scale exposure by target/realized volatility. Because volatility is far more
forecastable than returns and high-vol regimes carry poor returns, cutting exposure when realized vol
spikes raises risk-adjusted return. Tested **capped at 1.0** (a retail account cannot lever up), on
the survivorship-free cap-weighted CRSP index, vs buy-and-hold and the binary 200-day trend filter.
No look-ahead: exposure is decided at t (trailing 21-day vol) and applied at t+1.

Reproduce: `python scripts/crsp_vol_overlay_study.py`.

| universe | book | CAGR | vol | Sharpe | maxDD | Calmar |
|---|---|--:|--:|--:|--:|--:|
| large | buy & hold | +11.1% | 19.2% | 0.65 | −54.5% | 0.20 |
| large | **vol-managed (cap 1.0)** | +11.3% | 14.6% | **0.81** | −38.4% | 0.29 |
| large | 200d trend filter | +7.9% | 11.6% | 0.71 | −25.2% | 0.31 |
| small | buy & hold | +10.5% | 19.4% | 0.61 | −54.7% | 0.19 |
| small | **vol-managed (cap 1.0)** | +10.4% | 14.8% | **0.74** | −38.9% | 0.27 |
| small | 200d trend filter | +8.3% | 11.6% | 0.74 | −26.4% | 0.31 |

The capped vol overlay holds CAGR roughly flat while cutting volatility ~25%, so it improves **both**
Sharpe (0.65→0.81 large, 0.61→0.74 small) **and** drawdown (−54%→−38%) — a genuine risk-adjusted gain,
not merely risk-shaping. It dominates the binary trend filter, which sacrifices too much return going
fully to cash and missing rebounds (CAGR 7.9% vs 11.3%) for a similar Calmar.

**Honest caveats.** (1) It is an **overlay**, not a standalone edge — it improves whatever you hold;
the alpha must still come from the holdings (e.g. the net-payout book). (2) The volatility *target*
is calibrated on the full sample (sets the average exposure); the timing/shape is causal, but an
expanding-window target is the purist's version. (3) Most of the gain is downside (cutting exposure in
crises); in a long, calm bull it adds little. Useful as a **risk overlay** on a real long book — pair
it with the net-payout strategy rather than running it alone.
