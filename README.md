# plutus-quant

US-equity quantitative research, backtesting, and paper-trading system. Codename **Plutus**.

Sibling of the A-share system [hermes-quant](https://github.com/Rhymer-Lcy/hermes-quant):
same staged-pipeline philosophy and reusable research core, a US-specific data/friction/
execution layer. Built as a separate repo, not a hermes subpackage, because the friction
model, data adapters, universe, and execution are all market-specific.

> Status: **scaffold**. The market-agnostic research core (factor eval, walk-forward ML
> combiner, cross-sectional backtest engine, position sizing, idempotent paper ledger) is
> ported and unit-tested. The US data layer (yfinance price backbone, **SEC EDGAR
> fundamentals**, **point-in-time S&P 500 membership**) and the US friction model are in place
> and tested, and a **first end-to-end PIT S&P 500 factor study** runs on real data
> ([docs/first_study.md](docs/first_study.md)) — a capability readout with a known ~16%
> survivorship gap, NOT a tradeable result. Next: source delisted price series (the gating data
> problem), broaden the universe, and run walk-forward OOS. No alpha has been researched yet —
> the strategy in `live/strategy.py` is a *starting prior to test*, not validated.

## Architecture

Offline research and online execution are separated deliberately. A strategy advances a
stage only when the prior stage holds up:

```
            ┌─────────────────────────────┐         ┌──────────────────────────┐
            │  RESEARCH  (offline)         │ signals │  EXECUTION  (online)     │
            │  local PC (Windows)          │ ──────▶ │  local PC (Windows)      │
            │                              │ (files) │                          │
            │  factors · ML combiner       │         │  EOD paper ledger        │
            │  friction-faithful backtest  │         │  → (later) Alpaca live   │
            └─────────────────────────────┘         └──────────────────────────┘
                         │
                         ▼
            (optional) zipline-reloaded friction cross-check  ($0 commission · SEC §31 fee · FINRA TAF · slippage)
```

1. **Backtest** on historical data, offline, through a US-faithful friction model
   (`research/backtest/frictions.py`). An optional independent cross-check via
   `zipline-reloaded` plays the role RQAlpha plays for hermes.
2. **Realtime paper trading** (simulated): a lightweight idempotent end-of-day ledger
   (`live/ledger.py`) that replays the SAME research engine forward, so there is no
   train/serve skew. A monthly-rebalance strategy needs only an EOD feed.
3. **Live** (small real capital): deferred. Same strategy object, Alpaca gateway swapped in.

**US vs A-share — what changes, what doesn't.** The research core (eval/model/factors/
backtest mechanics/sizing/IO) is market-agnostic and shared with hermes by copy-and-diverge.
The market-specific layer is rebuilt: data vendors, the friction model, the universe, and
execution. Crucially, **US frictions are much lighter** — $0 commission, no stamp tax, no
100-share lot, no daily price limit, no T+1 holding lock — so the dominant risk shifts from
*frictions* (the A-share crown jewel) to **data quality: survivorship bias and a
point-in-time universe** (see [docs/data_sources.md](docs/data_sources.md)).

The exact US market rules and fee rates baked into the friction model are tracked, with
primary-source citations and as-of dates, in [docs/MARKET_FACTS.md](docs/MARKET_FACTS.md).

## Environment

A dedicated conda env **`plutus`** (Python 3.12) — separate from hermes's env, because the
dependency stack differs (yfinance/SEC-EDGAR/Alpaca, not BaoStock/Tushare).

```
conda create -n plutus python=3.12 -y
conda activate plutus
pip install -e ".[dev]"
pytest                                  # the research-core unit suite
python scripts/probes/smoke_yfinance.py # verify the free data link
```

## Data sources

Free-tier first (the US analog of hermes starting on anonymous BaoStock):

| Source | Auth | Role |
|---|---|---|
| **yfinance** (Yahoo) | none | free daily adjusted OHLCV backbone — start here ✓ |
| **SEC EDGAR** | User-Agent header (free) | official fundamentals for value/quality factors ✓ |
| **fja05680/sp500** | none | point-in-time S&P 500 membership (survivorship-free universe) ✓ |
| **Stooq** | none | daily CSV cross-check / partial delisted coverage — adapter built, but the endpoint is currently behind a JS bot-check (best-effort) ⚠ |
| **Alpaca** | free key | paper-trading account + EOD/data API (planned) |
| **Tiingo** | free key | cleaner EOD source, rate-limited (optional) |

⚠️ **Survivorship bias is the main free-data weakness.** Yahoo drops delisted tickers, and a
free point-in-time S&P 500 / Nasdaq-100 membership history is hard to assemble cleanly. See
[docs/data_sources.md](docs/data_sources.md) for the approach and its limits.

## Layout

```
src/plutus/        the engine — importable package (src-layout); no trading-framework dependency
  config.py        secret/token loading (env → .env.local)
  paths.py, io.py  on-disk locations; atomic file writes
  data/            sources/ (yfinance prices, SEC EDGAR fundamentals; Stooq planned);
                   universe.py (point-in-time S&P 500 membership) → adjusted parquet lake
  research/
    backtest/      friction-faithful backtest: frictions (US), portfolio, sizing
    factors/       factor library (value, reversal, momentum, low-vol, quality, size)
    eval/, model/  single-factor IC + calibration; walk-forward LightGBM combiner
  live/            EOD paper trading: candidate strategy spec, data feed, idempotent ledger
  execution/       Alpaca live-gateway adapters — deferred stub, unused
scripts/           research experiments (*_study.py) and operational drivers; probes/
tests/             pytest suite: engine invariants, no-look-ahead, friction model
data/              local data lake — INPUTS (gitignored)
results/           generated OUTPUTS: signals, backtests, figures, paper ledgers (gitignored)
external/          upstream checkouts (zipline-reloaded), pip install -e — gitignored, unmodified
docs/              architecture, market facts (cited), and curated research findings (tracked)
notebooks/         research scratch
```
