# Forward paper-trading the deployed net-payout strategy (the out-of-sample gate)

`docs/issuance_study.md` found the one retail-operable edge plutus has: net-payout / buyback in
liquid mid/small-cap US stocks (long-only top-50, ADV > $5M/day), Sharpe ~1.14 full / ~1.40 since
2021 net of 15bps, vs a cap-weighted buy-and-hold bar of 0.61. That result is **in-sample**
(2005-2025) and its magnitude is above the published issuance literature, so it has not earned real
capital. This module is the gate it must clear first: a forward, out-of-sample paper record, the
same discipline the sibling hermes-quant A-share book is held to.

## Architecture — the research engine IS the strategy brain (no train/serve skew)

Train/serve skew (the served strategy quietly differing from the researched one) is the dominant
silent alpha-killer. It is eliminated here by construction:

- **One signal definition.** `plutus.research.factors.library.net_payout` is the single source of
  truth. Both `scripts/crsp_issuance_study.py` (research) and `live.strategy.deployed_signal`
  (serving) call it. `tests/test_paper.py::test_deployed_signal_is_the_net_payout_factor` locks
  this; the dry-run proves the deployed spec reproduces the study headline to the digit.
- **One universe definition.** `live.strategy.deployed_members` is the cap-rank band [500, 3000)
  intersected with the ADV > $5M/day liquidity screen — the exact universe of the study's headline.
- **One decision engine.** `live.paper.replay` runs the research backtest
  (`signal_portfolio_backtest(..., collect_trades=True)`) and folds its per-fill trade log, day by
  day, into an idempotent `live.ledger.LedgerState` valued by the SAME `valuation_panel`. The
  ledger only *records* the engine's decisions; it never re-decides.

The anti-skew **parity gate** (`scripts/paper_dryrun.py`) asserts the ledger equity equals the
engine equity bar-for-bar, and that the deployed spec reproduces the documented headline:

    seed $1,000,000: parity max|ledger-engine| = 5.96e-08 (rel 6.5e-16) -> OK
    headline @ $1M (vs docs/issuance_study.md): CAGR +23.7% / maxDD -32.5% / Sharpe 1.14 -> OK

## The deployed spec (frozen as of this commit; `live.strategy.DEPLOYED`)

| field | value | rationale (docs/issuance_study.md) |
|---|---|---|
| signal | net-payout, 1y (252d) | the post-publication-robust, low-turnover, survivorship-free anomaly |
| book | long-only top-50, equal weight | the realizable retail form; top-30/50/100 all Sharpe 1.33-1.52 |
| universe | cap-rank band [500, 3000) ∩ ADV > $5M/day | the liquidity-screened (tradeable) headline universe |
| rebalance | monthly, no turnover buffer | low-turnover signal does not need a band |
| cost | $0 commission + 15bps slippage | the validated headline (50bps stress still clears B&H at 1.07) |

## Inception and the out-of-sample discipline

`PAPER_INCEPTION = 2026-01-02`. The study used the full 2005-2025 sample (the "2025 holdout" was
reported, so 2025 is **in-sample**); the first genuinely out-of-sample bar is the first trading day
of 2026. The seed is invested at the first available close ≥ inception into the then-current top-50,
and `total_return` / `max_drawdown` are measured from there. **The spec above is frozen as of this
commit and must not be refit on post-2025 data** — refitting on the holdout would destroy the gate.

### Data constraint (the key difference from hermes-quant)

hermes auto-refreshes from free BaoStock. plutus runs on **CRSP — a paid, manual pull with no free
daily feed**, so this module does NOT auto-refresh; it replays whatever is on disk. The lake
currently ends **2025-12-31**, before the inception, so every tier honestly reports
`status="awaiting_data"` (seeded, no positions) rather than silently falling back to the in-sample
backtest:

    tier      label  status         as_of       n_bars  equity     totRet  ...
    25,000    small  awaiting_data  2025-12-31   0       25,000     +0.0%
    ...
    seeded; lake ends 2025-12-31 < inception 2026-01-02 -- land a fresh CRSP pull
    covering post-inception bars, then re-run to extend the record.

To advance the forward record: land a fresh CRSP pull that includes 2026 bars (the build scripts),
then re-run `scripts/paper_live.py`. The ledger is idempotent (recompute-from-seed), so re-running
any date reproduces it.

## Running it

    conda activate plutus
    python scripts/paper_dryrun.py                     # parity gate + headline reproduction (CI)
    python scripts/paper_live.py                       # paper mode, all tiers, persists to results/paper/
    python scripts/paper_live.py --tiers 100000 1000000
    python scripts/paper_live.py --as-of 2026-03-31    # cut the lake at a date (idempotent re-run)
    python scripts/paper_live.py --backtest            # archive the full-history backtest to results/backtests/

Each run persists, per tier, `paper_curve_<seed>.parquet` (equity), `paper_trades_<seed>.parquet`
(full fill log), and `paper_report_<seed>.json` (the snapshot below) under `results/paper/`.

## Reading the forward record

The report benchmarks the strategy against the cap-weighted buy-and-hold over the **same forward
window** (`bh_total_return`, `bh_ann_sharpe`) — the out-of-sample version of the study's comparison.
`ann_sharpe` is `null` until ≥ 21 forward bars accrue (a Sharpe on a handful of days is noise).
Judge the strategy by whether it clears the B&H bar OOS, not by its absolute return.

A machinery smoke test on the in-sample 2025 window (NOT the forward record) confirms the active
path reproduces the studied edge on real data: +45.4% vs B&H +18.3%, Sharpe 1.81 vs 0.99 — i.e. the
live code yields the same strong-recent-years result the study found.

## Free-data early read — start now, before the next CRSP pull (`live.forward`)

Because the CRSP path is `awaiting_data`, `live.forward` gives an immediate, lower-rigor forward
read from FREE data, without waiting for a paid pull. The split keeps it honest: the book is
**selected from clean CRSP** at the last lake bar (survivorship-clean decision, validated signal +
universe), then **priced forward with yfinance** adjusted closes and benchmarked against small-cap
ETFs (VB, which tracks the CRSP US Small Cap index; IWM, Russell 2000). It tests the *selection*
(do the picked names beat the index OOS), not the monthly rebalance — net-payout is low-turnover, so
a held book approximates the strategy between CRSP refreshes.

Coverage is good: of the 50-name 2025-12-31 book, **47 (94%)** have full free 2026 data; 3 are
unresolved tickers (changes/acquisitions) and excluded. Run: `python scripts/paper_forward.py`.

**First read (2026-01-02 .. 2026-06-23, 118 bars, seed $1M) — a caution flag, not a verdict:**

| book | total return | maxDD | Sharpe |
|---|--:|--:|--:|
| net-payout top-50 (held) | +4.6% | −16.5% | 0.55 |
| benchmark VB (CRSP small-cap) | +13.2% | | 1.60 |
| benchmark IWM (Russell 2000) | +19.2% | | 1.93 |

The book **lagged** the small-cap index by 9-15 points over the first ~6 OOS months — the opposite
of the in-sample story (where it beat the B&H bar). Read this with heavy caveats: (1) the sample is
thin (~6 months — the Sharpe is noise-dominated, read direction not magnitude); (2) it is
selection-only (no forward rebalance); (3) early 2026 was a sharp small-cap beta rally that a value-
tilted net-payout book would be expected to trail; (4) 3/50 names excluded, bias direction unknown.
It is **not** the definitive test, but it is the first real out-of-sample evidence, and it does not
corroborate the in-sample edge. The monthly-rebalanced CRSP ledger (`paper_live.py`), run when a
fresh CRSP pull lands, remains the decisive gate.

## Honest caveats (carried from docs/issuance_study.md, plus the feed constraint)

- **Magnitude is above the literature** (~0.5-0.8 published issuance Sharpe). The forward record is
  expected to come in *below* the in-sample 1.1-1.4; treat the in-sample level as an optimistic
  upper bound and watch whether it merely clears B&H, not whether it matches 1.4.
- **One searched signal.** The IC t=4.4 and cross-signal controls make this far more than a lucky
  fit, but the forward test is precisely what converts "survived the in-sample controls" into
  evidence. Do not size real capital until the forward record holds up.
- **Long-only ≈ beta + a tilt**, carrying full small-cap market beta; pair with the capped
  volatility overlay (docs/vol_overlay_study.md) if the drawdown is to be managed.
- **No live feed.** Because CRSP is a manual pull, the forward record advances only as fast as new
  pulls land — the `lake_lag_days` field flags how stale the lake is on each run.
