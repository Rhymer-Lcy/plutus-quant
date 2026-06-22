# plutus-quant

US-equity quantitative research, backtesting, and paper-trading system. Codename **Plutus**.

Sibling of the A-share system [hermes-quant](https://github.com/Rhymer-Lcy/hermes-quant):
same staged-pipeline philosophy and reusable research core, a US-specific data/friction/
execution layer. Built as a separate repo, not a hermes subpackage, because the friction
model, data adapters, universe, and execution are all market-specific.

> Status: **scaffold**. The market-agnostic research core (factor eval, walk-forward ML
> combiner, cross-sectional backtest engine, position sizing, idempotent paper ledger) is
> ported and unit-tested. The US data layer (yfinance, **SEC EDGAR fundamentals**, **PIT S&P
> 500 membership**) and the US friction model are in place and tested. Backtests now run on a
> **survivorship-free CRSP lake** (total-return, delisting-aware prices + PIT membership by
> PERMNO; bring-your-own licensed extract). Measured impact: survivorship bias had inflated the
> candidate strategy's CAGR from a true ~8% to a fake ~21% and hid 14 pts of drawdown — and the
> full-history run reveals an −88% 2008 near-ruin the biased data couldn't show
> ([docs/survivorship_study.md](docs/survivorship_study.md)). A multi-factor walk-forward ML
> combiner + a market-regime risk overlay were then tested
> ([docs/multifactor_study.md](docs/multifactor_study.md)): the regime filter is crash insurance
> (−88%→−43% in 2008) but whipsaws in bulls; the ML combiner modestly beats the fixed candidate
> — **but a passive S&P 500 buy-and-hold beats every long-only variant on CAGR, Sharpe and
> Calmar.** Taken further to a **market-neutral long-short** test
> ([docs/longshort_study.md](docs/longshort_study.md)), every classic factor's net-of-cost
> spread is ≈0 or negative (only quality barely positive) and the ML combiner overfits to
> negative — **no tradeable edge in classic factors on large-cap US after rigor** (the known
> "factors are arbitraged" result). The "go smaller" escape hatch was then tested too
> ([docs/smallcap_study.md](docs/smallcap_study.md)) on a broad 11,219-stock survivorship-free
> universe (mid/small band): at realistic small-cap costs every price factor is negative
> (Sharpe −0.34 to −0.38) — the only gross-positive one (reversal) is a 313%/mo turnover
> illusion. So classic factors have no tradeable edge in US equities, **large OR small cap**.
> The durable output is a methodology that tells the truth; the honest frontier is non-classic /
> alternative signals. First such test — **PEAD** (post-earnings-announcement drift,
> [docs/pead_study.md](docs/pead_study.md)) — is the first signal with a *pulse*: a positive,
> near-significant SUE rank IC (t≈0.9) with ~zero market beta, unlike the flat/negative
> valuation factors. Re-run on the **event clock** ([docs/pead_event_study.md](docs/pead_event_study.md))
> the drift is confirmed real and front-loaded (CAAR: a clean +0.68% extreme-quintile drift
> peaking ~day 40, ~90% of it by day 20) — but net of costs it sits exactly on the cost boundary
> (only extreme surprises held ~10 days is marginally positive, Sharpe ~0.07; longer/looser
> loses). An anomaly at the cost boundary is *why* it persists. Realistic edge would need smaller
> caps / faster entry / a sharper surprise (data+infra steps). No validated alpha —
> `live/strategy.py` is a prior, not a recommendation.

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
| **fja05680/sp500** | none | point-in-time S&P 500 membership (free universe) ✓ |
| **CRSP** (via WRDS) | bring-your-own licensed extract | survivorship-free total-return prices + PIT membership — the backtest backbone ✓ |
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
