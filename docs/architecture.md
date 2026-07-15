# Architecture

## Why a separate repo from hermes-quant

hermes-quant is an A-share system. Its defining value — the friction gate (T+1 holding lock,
±10%/±20% daily price limits, stamp tax, ¥5 minimum commission, 100-share lots) — does not
transfer to US equities, where none of those rules exist. The data vendors (BaoStock/Tushare/
AKShare), the universe (CSI 300/500), and the execution stack (vnpy/miniQMT) are equally
China-specific. Bolting a second market onto hermes would also destabilize a system that is
close to taking live capital. So plutus is a sibling repo that **reuses the market-agnostic
research core** and **rebuilds the market-specific layer**.

### Reused from hermes (copy-and-diverge, market-agnostic)

- `io.py`, `paths.py`, `config.py` — atomic writes, path single-source-of-truth, secrets.
- `research/eval/` — single-factor rank IC, quantile returns, probability calibration.
- `research/model/walk_forward.py` — walk-forward LightGBM cross-sectional combiner.
- `research/factors/library.py` — cross-sectional processing (winsorize/z-score/blend) and
  the survivorship discipline (`restrict_to_universe`); price-based factors are identical,
  fundamental factors take SEC-derived panels instead of BaoStock ratios.
- `research/backtest/portfolio.py`, `sizing.py` — the rebalance engine and inverse-vol
  weighting. The engine keeps hermes's no-look-ahead (signal at t, execute t+1), delisting
  force-liquidation, and top-N hysteresis band.
- `live/ledger.py` — the idempotent, immutable-rebuild paper-trading ledger.

Why copy-and-diverge rather than a shared `quant-core` package: for a solo developer, a
shared package adds version-coupling and release-coordination overhead before the US friction
and data layers have stabilized. Extract a shared core later, if duplication actually hurts.

### Rebuilt for US (market-specific)

- `research/backtest/frictions.py` — US cost model (see [MARKET_FACTS.md](MARKET_FACTS.md)).
- `data/sources/` — yfinance (price), Stooq (cross-check), SEC EDGAR (fundamentals), CRSP
  (licensed survivorship-free extract), IBES (licensed estimates). No Alpaca adapter exists here.
- the universe / point-in-time membership (S&P 500, from fja05680/sp500) — the hard, survivorship-
  sensitive part (see [data_sources.md](data_sources.md)). Nasdaq-100 is not implemented.
- `live/forward.py` — the EOD forward record's data path, built on yfinance (an earlier separate
  feed module was retired as unused). `execution/` — deferred Alpaca live gateway, an unused
  stub; Alpaca supplies no data anywhere in the pipeline today.

## No framework fork

The engine is hand-rolled and depends on no trading framework. `zipline-reloaded` is intended
only as an *independent* friction cross-check (the RQAlpha analog) — installed editable under
`external/`, unmodified. Forking a framework's internals is deferred until a concrete need
arises; it is not required to build or run plutus.

## Staged pipeline

Backtest (offline, friction gate) → realtime paper trading (simulated EOD ledger) → live
(small real capital, gateway swapped). The SAME strategy object and the SAME rebalance
primitives drive every stage, so the served signal cannot drift from the researched one —
train/serve skew is the dominant silent alpha-killer.
