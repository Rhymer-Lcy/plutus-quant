# plutus-quant

US-equity quantitative research, backtesting, and paper-trading system. Codename **Plutus**.

Sibling of the A-share system [hermes-quant](https://github.com/Rhymer-Lcy/hermes-quant): the same
staged-pipeline philosophy and reusable research core, with a US-specific data / friction /
execution layer. Kept as a separate repository (not a hermes subpackage) because the friction
model, data adapters, universe, and execution are all market-specific.

## Status

A rigorous, survivorship-free research platform with **one strategy on a forward out-of-sample
watch and no live capital deployed**. The market-agnostic research core (factor evaluation,
walk-forward ML combiner, cross-sectional backtest engine, position sizing, idempotent paper
ledger) and the US data / friction layer are implemented and unit-tested (110 tests).

Headline of the research program: after every classic family was tested under survivorship-free,
cost-aware, look-ahead-audited rigor, **only one edge survived as a deployable core strategy** — a
net-payout / buyback tilt in liquid mid/small-cap stocks — and it is now being forward paper-traded
out-of-sample, **not yet validated**. One further edge is retail-operable but too small and episodic
to be a core: the S&P 500 index delete-reversal, kept as a *satellite*.

### Research log (honest findings)

The negative results are kept on the record, not buried. The major families are below; the remaining
studies are written up in [docs/](docs/).

- **Survivorship bias, quantified** ([docs/survivorship_study.md](docs/survivorship_study.md)) — on a
  survivorship-free CRSP lake (total-return, delisting-aware prices + point-in-time membership), the
  candidate strategy's CAGR falls from a biased ~21% to a true ~8%, ~14 points of drawdown reappear,
  and the full history exposes a −88% 2008 near-ruin the biased data hid.
- **Multi-factor + regime overlay** ([docs/multifactor_study.md](docs/multifactor_study.md)) — the
  regime filter is crash insurance (−88% to −43% in 2008) but whipsaws in bull markets; a passive
  S&P 500 buy-and-hold beats every long-only variant on CAGR, Sharpe, and Calmar.
- **Market-neutral long-short** ([docs/longshort_study.md](docs/longshort_study.md)) — every classic
  factor's net-of-cost spread is approximately zero or negative: no tradeable edge in classic factors
  on large-cap US (the known "factors are arbitraged" result).
- **Small-cap** ([docs/smallcap_study.md](docs/smallcap_study.md)) — on a broad 11,219-stock
  survivorship-free universe, every price factor is negative at realistic small-cap costs; the only
  gross-positive one (reversal) is a turnover illusion.
- **PEAD, including IBES analyst surprise** ([docs/ibes_pead_study.md](docs/ibes_pead_study.md)) — the
  drift is real (clean, monotone CAAR; IBES gives the largest +0.93% / 60d) but not tradeable net of
  costs. An early Sharpe-7.6 result was caught and **retracted** as a look-ahead artifact
  (announcement-gap capture); the fix (skip the reaction day) de-leaked all earlier event-time PEAD
  numbers.
- **ML / temporal-DL cross-sectional zoo** ([docs/ml_zoo_study.md](docs/ml_zoo_study.md)) — a GRU
  small-cap market-neutral signal looked tradeable in-sample (~Sharpe 0.6–0.67), but its information
  coefficient, real in 2010–2019 (t=3.4), **decayed to statistical zero by 2020–2024** (t=0.57) before
  the 2025 holdout printed the weakest reading on record. Verdict: on-watch, **do not deploy**. The
  most detailed negative result in the program.
- **S&P 500 index reconstitution** ([docs/index_effect_study.md](docs/index_effect_study.md)) — the
  ADD run-up is dead post-effective, but the **DELETE-reversal is real**: dropped names earn ~+3.4% to
  +5.1% abnormal return net of a ~0.30% round-trip over 20–60 days. Retail-operable, but only ~11
  deletions a year in distressed, volatile names: a genuine **satellite**, not a core strategy.
- **Net-payout / buyback** ([docs/issuance_study.md](docs/issuance_study.md),
  [docs/paper_trading.md](docs/paper_trading.md)) — the one deployable core. A long-only top-50 book in the
  liquid mid/small-cap band clears the buy-and-hold bar in-sample (Sharpe ~1.14, 2005–2025),
  signal-specific and cost-robust. It is frozen as the deployed spec (`live/strategy.py`) and
  paper-traded forward from 2026-01-02; the first ~6-month out-of-sample read **lags** the small-cap
  index, so it is on watch, not validated.

- **Pre-registered outside claims** ([docs/topgainer_study.md](docs/topgainer_study.md),
  [docs/sp1_study.md](docs/sp1_study.md), [docs/biotech_catalyst_study.md](docs/biotech_catalyst_study.md),
  [docs/copycat_13f_study.md](docs/copycat_13f_study.md))
  — four claims from a friend, each frozen in a public issue (#1–#4) before any code existed.
  Daily top-gainer rotation is rejected at every reading (gross −29%, net −99.5% over 2005–2024 vs
  +724% for the benchmark). "Always hold the largest market cap" beats on raw return (+1210%) and
  DCA but fails risk-adjusted (Sharpe 0.60 vs 0.65), with all of its outperformance confined to the
  2015–24 mega-cap regime — rejected under the frozen rule. **Biotech catalysts**: across 1,257
  overnight gaps ≥ +20% (2005–2024, survivorship-free), buying after the news earns **−4.30%**
  abnormal over 20 days net (clustering-robust t = −3.90, hit rate 40%) and −7.9% over 60 days —
  the move is over before you can act, and then it bleeds. "Sell the news" is statistically real
  but not retail-harvestable (it needs shorting hard-to-borrow small biotech). The money moves
  *before* the announcement (+9.8% run-up), which is visible only in hindsight. An earlier run of
  this study was wrong and is retracted in the write-up. **Copying the greats' 13F filings**:
  17 hindsight-selected legends (a God's-eye view the hypothesis could never have had), entered at
  the filing-date close — no edge at any horizon, and once size-matched to the benchmark the
  abnormal return is +1.1%/yr with t = 0.6, indistinguishable from zero. Patience does not rescue
  it; Buffett's own new positions did not beat the market either.

Also written up in [docs/](docs/): the first point-in-time factor read; event-time PEAD, small-cap
PEAD, short-term reversal and overnight returns — each a real gross effect that is not
retail-tradeable net of cost; pairs trading, whose mean-reversion alpha is no longer there at all;
and volatility-managed exposure, a genuine risk-adjusted improvement that is an overlay, not alpha.

The durable asset is the methodology — survivorship-free, cost-aware, and look-ahead-audited — that
reports the truth regardless of the result.

## Architecture

Offline research and online execution are separated deliberately. A strategy advances a stage only
when the prior stage holds up:

```
            +-----------------------------+         +--------------------------+
            |  RESEARCH  (offline)        | signals |  EXECUTION  (online)     |
            |  local PC (Windows)         | ------> |  local PC (Windows)      |
            |                             | (files) |                          |
            |  factors / ML combiner      |         |  EOD paper ledger        |
            |  friction-faithful backtest |         |  -> (later) Alpaca live  |
            +-----------------------------+         +--------------------------+
                         |
                         v
            (optional) zipline-reloaded friction cross-check
            ($0 commission / SEC Section 31 fee / FINRA TAF / slippage)
```

1. **Backtest** on historical data, offline, through a US-faithful friction model
   (`research/backtest/frictions.py`). An optional independent cross-check via `zipline-reloaded`
   plays the role RQAlpha plays for hermes.
2. **Paper trading** (simulated): a lightweight idempotent end-of-day ledger (`live/ledger.py`) that
   replays the SAME research engine forward, so there is no train/serve skew. A monthly-rebalance
   strategy needs only an EOD feed.
3. **Live** (small real capital): deferred. Same strategy object, with an Alpaca gateway swapped in.

**US vs A-share — what changes, what does not.** The research core (eval / model / factors /
backtest mechanics / sizing / IO) is market-agnostic and shared with hermes by copy-and-diverge. The
market-specific layer is rebuilt: data vendors, the friction model, the universe, and execution.
Crucially, **US frictions are much lighter** — $0 commission, no stamp tax, no 100-share lot, no
daily price limit, no T+1 holding lock — so the dominant risk shifts from *frictions* (the A-share
crown jewel) to **data quality: survivorship bias and a point-in-time universe** (see
[docs/data_sources.md](docs/data_sources.md)).

The exact US market rules and fee rates baked into the friction model are tracked, with
primary-source citations and as-of dates, in [docs/MARKET_FACTS.md](docs/MARKET_FACTS.md).

## Environment

A dedicated conda environment **`plutus`** (Python 3.12), separate from the hermes environment
because the dependency stack differs (yfinance / SEC EDGAR / Alpaca, not BaoStock / Tushare).

```
conda create -n plutus python=3.12 -y
conda activate plutus
pip install -e ".[dev]"                  # research core + the pytest suite
cp .env.template .env.local              # then fill in any keys you use (all optional to start)
pytest                                   # the research-core unit suite
python scripts/probes/smoke_yfinance.py  # verify the free data link
```

Secrets are read from the real environment first, then from `.env.local` (gitignored; never
committed). yfinance and Stooq need no key, so research runs with nothing configured. Optional
extras: `pip install -e ".[broker]"` (Alpaca) and `pip install -e ".[notebooks]"` (JupyterLab).

## Data sources

Free-tier first (the US analog of hermes starting on anonymous BaoStock):

| Source | Auth | Role |
|---|---|---|
| yfinance (Yahoo) | none | free daily adjusted OHLCV backbone — the default |
| SEC EDGAR | free User-Agent header | official fundamentals for value / quality factors |
| fja05680/sp500 | none | point-in-time S&P 500 membership (free universe) |
| CRSP (via WRDS) | bring-your-own licensed extract | survivorship-free total-return prices + PIT membership; **not included** |
| Stooq | none | daily CSV cross-check; endpoint currently behind a JS bot-check (best-effort) |
| Alpaca | free key | paper-trading account + market-data API (registered, parked) |

**Survivorship bias is the main free-data weakness:** Yahoo drops delisted tickers, and a free
point-in-time membership history is hard to assemble cleanly. The survivorship-free backtests use a
licensed CRSP extract that is **not part of this repository** (see Disclaimer); see
[docs/data_sources.md](docs/data_sources.md) for the approach and its limits.

## Layout

```
src/plutus/        the engine — importable package (src-layout); no trading-framework dependency
  config.py        secret / token loading (environment -> .env.local)
  paths.py, io.py  on-disk locations; atomic file writes
  data/            sources/ (yfinance prices, SEC EDGAR fundamentals, CRSP, IBES, Stooq);
                   universe.py (point-in-time S&P 500 membership) -> adjusted parquet lake
  research/
    backtest/      friction-faithful backtest: frictions (US), portfolio, long-short, pairs,
                   optimize, regime, sizing, shared metrics
    factors/       factor library (value, reversal, momentum, low-vol, quality, net-payout)
    eval/, model/  single-factor IC + calibration; walk-forward LightGBM combiner
  live/            EOD paper trading: deployed net-payout spec (strategy.py), idempotent ledger,
                   CRSP replay (paper.py), free-data forward read (forward.py), data feed
  execution/       Alpaca live-gateway adapters — deferred stub, unused
scripts/           research studies (*_study.py), data-lake builders, and paper-trading drivers; probes/
tests/             pytest suite: engine invariants, no-look-ahead, friction model, ledger parity
data/              local data lake — INPUTS (gitignored)
results/           generated OUTPUTS: signals, backtests, figures, paper ledgers (gitignored)
external/          upstream checkouts (zipline-reloaded), pip install -e — gitignored, unmodified
docs/              architecture, market facts (cited), and curated research findings (tracked)
notebooks/         research scratch (gitignored)
```

## Disclaimer

This is a personal research project. **Nothing here is investment advice.** The strategies are
research hypotheses under test, not recommendations, and no live capital is deployed. The CRSP and
IBES data used for the backtests is licensed for personal research use only; it is **not included in
this repository and must not be redistributed** — supply your own licensed extract. The code is
provided as-is, without warranty of any kind.
