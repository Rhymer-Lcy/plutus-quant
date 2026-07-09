# First study — PIT S&P 500 factor read (2018–2025)

**Purpose:** exercise the whole stack end-to-end on real data and get an honest first read. This
is a **capability readout, NOT a tradeable result** — read the caveats before drawing any
conclusion. Reproduce with `scripts/build_price_lake.py` then `scripts/yfinance_factor_study.py`.

## Setup

- Universe: point-in-time S&P 500 (`members_asof`), monthly rebalance, long-only.
- Window: 2018-01-02 → 2025-12-30, 96 monthly eval dates.
- Prices: yfinance adjusted close (returns) + unadjusted close × SEC shares (market cap).
- Fundamentals: SEC EDGAR, filing-date PIT — TTM net income (407/445 names), book equity
  (424/445), shares (415/445).
- Frictions: `USEquityCosts` defaults ($0 commission, SEC §31 + FINRA TAF on sells, 5 bps slip).

## Single-factor rank IC

| factor | mean IC | IC-IR | t-stat | hit | n |
|---|---:|---:|---:|---:|---:|
| earnings_yield | 0.0120 | 0.071 | 0.70 | 0.55 | 95 |
| book_yield | 0.0055 | 0.030 | 0.29 | 0.51 | 95 |
| reversal_1m | 0.0057 | 0.034 | 0.32 | 0.51 | 94 |
| momentum_12_1 | −0.0004 | −0.002 | −0.02 | 0.54 | 83 |
| low_vol | −0.0195 | −0.070 | −0.64 | 0.48 | 83 |

**Read:** every factor is weak and statistically insignificant (|t| < 1). Earnings yield is the
least-bad, but on this universe/window there is **no tradeable single-factor edge to claim**.
That is the honest, expected result for plain factors on a large-cap, survivorship-pruned,
single-regime sample.

## Candidate value+reversal backtest (top-20, monthly, PIT, frictions)

- total return: **+372.6%** · CAGR **21.45%** · max drawdown **−44.95%** · ~19.8/20 names held
  · $10,380 costs over 95 rebalances.

**Do not misread this.** With factor IC ≈ 0, the +372% is almost entirely **market beta +
concentration**, not alpha: 2018–2025 was a strong bull market and a top-20 slice of the S&P
500 is concentrated long equity. The −45% drawdown is large for that reason.

## Caveats (why this is not a result)

1. **Survivorship bias.** Free data is missing ~16% of PIT members (84/529 delisted/acquired
   names yfinance drops), and the 445 it has are disproportionately survivors. This inflates
   both IC and returns. *Top open data problem* — see [data_sources.md](data_sources.md).
2. **Large-cap only.** S&P 500; many factor premia live in smaller caps not sampled here.
3. **Single window / no out-of-sample.** One 2018–2025 regime, no walk-forward holdout. No
   alpha claim is admissible without OOS survival (the `research/model` walk-forward harness
   exists for exactly this — not yet run at scale).
4. **Market-cap approximation.** unadjusted price × most-recent-filed shares (shares between
   filings are stale); TTM uses first-reported values (no restatement look-back).

## What this DID establish

The full pipeline runs end-to-end on real data and is point-in-time honest: PIT universe,
filing-date fundamentals, friction-faithful backtest, all reproducible from cached lakes.
Running it at scale also shook out two real bugs (SEC prior-year comparatives corrupting TTM;
empty-quarter filers), both now regression-tested.

## Next to make it real

1. Source delisted price series (kill survivorship) — the gating data problem.
2. Broaden the universe beyond large caps.
3. Walk-forward out-of-sample (the ML combiner) before any alpha claim.
