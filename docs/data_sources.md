# Data sources & the survivorship problem

plutus starts on **free** data, the way hermes started on anonymous BaoStock. The exact
free-tier capabilities, rate limits, and auth requirements (verified against vendor docs) are
summarized in [MARKET_FACTS.md](MARKET_FACTS.md §3). This doc is the *design* rationale.

## The backbone

- **Price (daily, adjusted):** `yfinance` (Yahoo) — free, anonymous, the place to start.
  Use `auto_adjust=True` so OHLC are split- and dividend-adjusted (the right series for a
  total-return backtest). `Stooq` is a free, independent second source for cross-checking a
  questionable bar and for some delisted names Yahoo lacks.
- **Fundamentals:** `SEC EDGAR` company-facts API — free and official (no paid vendor needed
  for US financials). Map ticker → CIK, pull the XBRL facts, and **align every datum to its
  FILING date, not the fiscal-period-end date** — otherwise look-ahead leaks (you would
  "know" Q4 earnings on Dec 31 when they were filed in February).
- **Paper trading:** `Alpaca` — free paper account + commission-free API; the natural US
  analog of hermes's vnpy paper account.

## The survivorship problem (the main free-data weakness)

A backtest universe built from *currently* listed tickers is survivorship-contaminated: names
that delisted (bankruptcy, takeover, index removal) are silently absent, so the backtest only
ever "buys" companies that survived to today. This **inflates returns** and is the single
biggest correctness risk in a free US data stack.

Two distinct needs:

1. **Delisted price series.** Yahoo drops most delisted tickers. Stooq has some; the clean
   solution (Norgate / CRSP) is paid. The backtest engine already handles a name whose price
   series ends — it force-liquidates at the last real bar (`portfolio.valuation_panel`) — so
   the gap is *data availability*, not engine logic.
2. **Point-in-time index membership.** "What was in the S&P 500 on 2018-03-31?" Free options:
   the current constituents from Wikipedia plus its change history, and community-maintained
   reconstructions on GitHub. These are approximate; the gold standard (CRSP/Norgate) is paid.

**Approach:** build a point-in-time `members_asof(date) -> set[ticker]` from the best free
reconstruction, feed it everywhere (`restrict_to_universe`, the backtest `members_asof`), and
treat survivorship-free results as the ones that count. Document the membership source's known
gaps. Until delisted price coverage is solved, be explicit that early backtests carry a
survivorship-bias caveat rather than pretending it away. See [MARKET_FACTS.md](MARKET_FACTS.md §4)
for the concrete free sources and their limits.
