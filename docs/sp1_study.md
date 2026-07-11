# sp1 (always hold the largest market cap): rejected on the frozen rule, instructive in how

The companion pre-registered claim
([#2](https://github.com/Rhymer-Lcy/plutus-quant/issues/2)): only ever hold the company
with the largest market capitalization, keep adding to it. Frozen before code: PIT S&P 500
members on survivorship-free CRSP (2005-2024), seat check at each month-end close,
rotation at the next trading day's close at 5 bps per side, a DCA variant (equal monthly
contributions into the then-#1 vs the same stream into the benchmark), benchmark = the
cap-weighted total-return index, and a verdict rule requiring net total return AND net
Sharpe above the benchmark AND the DCA variant ending ahead. Reproduce:
`python scripts/crsp_sp1_study.py`.

## Result

| run | net total | CAGR | Sharpe | maxDD | 2005-14 | 2015-24 |
|---|---:|---:|---:|---:|---:|---:|
| sp1 | **+1210.2%** | +13.7% | 0.60 | −47.0% | +92.5% | **+587.2%** |
| benchmark | +723.8% | +11.1% | **0.65** | −54.5% | **+128.2%** | +261.1% |

DCA (238 monthly units): sp1 ends at 1,712 vs 1,050 for the same stream into the
benchmark (**+63%**). Seat history: a GE/XOM tug-of-war through 2006, XOM's long reign to
2011, an XOM/AAPL contest to 2013, then the AAPL/MSFT (once AMZN) era -- NVDA never held
a month-end #1 seat in this window.

## Verdict: REJECTED (frozen rule), with the honest reading spelled out

- Two of the three legs pass -- raw return (+1210% vs +724%) and DCA (+63%) -- but the
  Sharpe leg fails (0.60 vs 0.65), so the pre-registered verdict is REJECTED. A
  single-name book that carries −47% drawdowns and beats the index by LESS than its
  extra risk is a concentration bet that happened to pay, not a better rule.
- The sub-periods are the regime split the pre-registration predicted: the first decade
  the top dog LAGGED (+92.5% vs +128.2% -- the documented "too big to succeed" drag);
  ALL of the outperformance is the 2015-24 AAPL/MSFT mega-cap run. The claim as stated
  is not a law; it is a bet that the mega-cap regime continues.
- What survives is the modest form: over THIS window, buying the biggest name did not
  cost a raw-return premium (the historical drag did not repeat), so as a
  minimum-effort retail heuristic it vastly outperformed the top-gainer rule tested in
  [topgainer_study.md](topgainer_study.md). That is a statement about the window, not
  a validated edge.
