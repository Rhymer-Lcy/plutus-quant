# US market facts baked into plutus

The numeric rules and fee rates the friction model and engine depend on, with primary-source
citations and **as-of dates**. Verified 2026-06-21 via the cited sources. These rates are
re-adjusted periodically (the SEC fee annually, the FINRA TAF on its own schedule), so
**re-check before trusting net P&L** and update `research/backtest/frictions.py` to match.

> Provenance: a background research workflow was attempted but its subagents' web access was
> permission-blocked; these facts were then verified directly in the main session. If a fact
> below lacks a date or source, treat it as LOW confidence and re-verify.

## 1. Settlement & trading rules

- **Settlement cycle: T+1**, effective **2024-05-28** (was T+2). Securities settle one
  business day after the trade. Source: [SEC press release 2024-62](https://www.sec.gov/newsroom/press-releases/2024-62),
  [Investor.gov T+1 bulletin](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins/new-t1-settlement-cycle-what-investors-need-know-investor-bulletin).
- **No A-share-style same-day-sell lock.** US retail can buy and sell the same security the
  same day; "T+1" here is *settlement*, not a holding restriction. Same-day round trips are
  legal.
- **Pattern Day Trader (PDT) $25,000 rule: ELIMINATED, effective 2026-06-04.** FINRA amended
  Rule 4210 to remove the $25,000 minimum and the "pattern day trader" label, replacing them
  with an intraday risk-based margin standard. (Before this, a margin account making ≥4 day
  trades in 5 business days needed ≥$25k.) **Irrelevant to plutus's daily/monthly cadence
  either way** — a monthly-rebalance book does no day trading. Source:
  [FINRA Regulatory Notice 26-10](https://www.finra.org/rules-guidance/notices/26-10),
  [Federal Register filing 2026-00519](https://www.federalregister.gov/documents/2026/01/14/2026-00519/self-regulatory-organizations-financial-industry-regulatory-authority-inc-notice-of-filing-of-a).
- **Cash accounts:** sale proceeds are available to re-trade after settlement (T+1); trading
  unsettled funds can trigger good-faith / free-riding violations. Not binding on a monthly
  book. (General rule; re-verify specifics with your broker.)

## 2. Frictions & fees  (constants in `frictions.py`)

- **Commission: $0** on US stocks at most retail brokers (Alpaca, Schwab, Fidelity,
  Robinhood, IBKR Lite). IBKR Pro is per-share (~$0.0035/sh, $0.35 min, 1% of trade value
  cap) — model via `commission_per_share` + `min_commission` if used.
- **SEC Section 31 fee (SELL side only): $20.60 per $1,000,000 of proceeds = `0.0000206`**,
  effective **2026-04-04**. NOTE: for FY2026 charge dates from Oct 2025 through **Apr 3, 2026
  the rate was $0.00/million** (prior-year over-collection), then $20.60/M from Apr 4. The
  SEC resets this each fiscal year and has set it to $0 before. Source:
  [SEC FY2026 Fee Rate Advisory](https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2),
  [Federal Register order 2026-04233](https://www.federalregister.gov/documents/2026/03/04/2026-04233/order-making-fiscal-year-2026-annual-adjustments-to-transaction-fee-rates).
- **FINRA Trading Activity Fee (TAF), covered equity (SELL side only): `0.000195`/share, cap
  `$9.79`/trade**, effective **2026-01-01** (raised from $0.000166/share, $8.30 cap, which
  applied through 2025). Source:
  [FINRA Trading Activity Fee](https://www.finra.org/rules-guidance/guidance/trading-activity-fee).
- **No US stamp duty / transfer tax** on stock trades (contrast: A-share 0.05% stamp tax on
  sells; HK stamp duty). The SEC §31 fee + FINRA TAF above are the only regulatory charges.
- **No 100-share lot.** Trading is 1-share granular, and many brokers (Alpaca, Fidelity,
  Schwab, Robinhood) support **fractional shares** — so the A-share small-account feasibility
  floor (100-share lots + minimum commission) does not exist. `frictions.lot_size = 1`.
- **Short selling** requires a locate/borrow; hard-to-borrow names carry a borrow fee. Not
  modeled (the candidate strategy is long-only).

## 3. Free data sources (daily)

Use as the free backbone (verify current free-tier limits at each vendor before relying):

- **yfinance** (Yahoo): free, no key; daily adjusted OHLCV (use `auto_adjust=True`).
  Unofficial/scraped — can break/rate-limit; **drops delisted tickers** (survivorship risk).
- **Stooq**: free CSV daily bars, no key; useful independent cross-check, has some delisted.
- **Tiingo**: free tier with API key; clean EOD history; rate/symbol limited.
- **Alpaca**: free paper-trading account + market-data API (free tier is IEX feed; full SIP
  is paid). The natural US paper-trading endpoint.
- **Alpha Vantage**: free key, heavily rate-limited (tight daily request cap).

## 4. Fundamentals & point-in-time universe

- **SEC EDGAR** (data.sec.gov) — free, official fundamentals. Endpoints (verified; wired up in
  `data/sources/sec_edgar.py`), CIK zero-padded to 10 digits:
  - ticker→CIK: `https://www.sec.gov/files/company_tickers.json`
  - company facts: `https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json`
  - company concept: `https://data.sec.gov/api/xbrl/companyconcept/CIK{cik10}/{taxonomy}/{tag}.json`
  - frames: `https://data.sec.gov/api/xbrl/frames/{taxonomy}/{tag}/{unit}/CY{period}.json`
  - **Mandatory descriptive User-Agent** (name + email), not an API key; **rate limit 10
    req/s**. Source: [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).
    Income/cash-flow concepts are FLOWS → aggregate to TTM (the adapter synthesizes the
    missing Q4 = 10-K annual − Q1−Q2−Q3); balance-sheet concepts are INSTANTs. Everything is
    aligned to the `filed` date (PIT), never the fiscal-period `end`.
- **Point-in-time index membership (survivorship-free):** the hard, free-data-weak part. Wired
  up in `data/universe.py` from **fja05680/sp500**, "S&P 500 Historical Components & Changes"
  (membership since 1996 as (date, comma-sep tickers) rows). Source:
  [github.com/fja05680/sp500](https://github.com/fja05680/sp500). Gold standard (CRSP/Norgate)
  is paid. **Survivorship bias is the dominant correctness risk in the free US stack** — this
  gives PIT *membership*, but delisted *price* series still need sourcing (yfinance drops most
  delisted tickers), so flag any early backtest that lacks both. See [data_sources.md](data_sources.md).

## 5. Backtest framework cross-check

- **zipline-reloaded** `3.1.1` (released 2025-07-19) provides **Python 3.12 wheels incl.
  Windows (win_amd64)**, models US commissions/slippage and a US trading calendar — suitable
  as the independent friction cross-check (the RQAlpha analog), installed **unmodified** via
  pip. Source: [zipline-reloaded on PyPI](https://pypi.org/project/zipline-reloaded/),
  [GitHub releases](https://github.com/stefan-jansen/zipline-reloaded/releases). The plutus
  engine is hand-rolled; no framework fork is required.
