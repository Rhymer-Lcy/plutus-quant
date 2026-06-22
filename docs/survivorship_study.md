# Survivorship bias, quantified — yfinance vs CRSP

The point of moving to CRSP was to kill survivorship bias. This is what it bought, measured.
Same candidate strategy throughout: value (E/P) + light 1-month-reversal blend (5:1), top-20,
monthly, long-only, US frictions. Reproduce: `scripts/factor_study.py` (yfinance) and
`scripts/build_crsp_lake.py` → `scripts/crsp_study.py [--start --end]` (CRSP).

## The candidate backtest under three lenses

| data / window | universe | total ret | **CAGR** | **max DD** |
|---|---|---:|---:|---:|
| **yfinance**, 2018–2025 (survivorship-BIASED) | 445 survivors | +372.6% | **21.45%** | −44.95% |
| **CRSP**, 2018–2024 (survivorship-FREE, ~same window) | 732 PIT members incl. delisted | +72.2% | **8.08%** | −58.62% |
| **CRSP**, 2005–2024 (survivorship-FREE, full history) | 957 PIT members incl. delisted | +139.7% | **4.47%** | **−88.05%** |

### Finding 1 — survivorship inflated CAGR ~2.6× and hid 14 pts of drawdown
Holding the window fixed (2018–2024) and changing ONLY the universe (445 surviving tickers →
732 point-in-time members that include names which were later delisted/acquired):
- CAGR collapses **21.45% → 8.08%** — the biased figure was ~2.6× the truth.
- Max drawdown WORSENS **−45% → −59%** — the bias didn't just inflate returns, it concealed risk.

(Minor caveat: the yfinance run ends 2025 vs CRSP 2024, a ~1-year window difference; the gap is
far too large to be explained by that — it is survivorship.)

### Finding 2 — full history reveals a near-ruin the short window hid
Over 2005–2024 the candidate strategy returns CAGR **4.47%** (below a passive S&P 500 ~10%)
with an **−88% max drawdown, troughing 2008-11-20**. Mechanism: a value + reversal tilt buys
the cheapest, most beaten-down names *into* the 2008 crisis — i.e. financials that then went to
**zero** (Lehman, WaMu, …). CRSP includes those bankruptcies; yfinance drops them, so the
biased dataset literally cannot show the wipeout. Equity: $126k peak (2007-06) → **$15k**
(2008-11) → $240k (2024).

### Finding 3 — no tradeable single-factor edge (clean data)
CRSP 2005–2024 rank IC, all weak/insignificant (|t| < 1):

| factor | mean IC | t-stat | n |
|---|---:|---:|---:|
| earnings_yield | −0.0028 | −0.21 | 172 |
| book_yield | −0.0093 | −0.73 | 187 |
| reversal_1m | +0.0086 | 0.81 | 238 |
| momentum_12_1 | +0.0028 | 0.21 | 227 |
| low_vol | +0.0108 | 0.66 | 227 |

## Lessons → what changes next
1. **Survivorship-free is non-negotiable.** The biased pipeline produced a plausible, totally
   false 21% CAGR. All future research uses the CRSP lake.
2. **The naive candidate is non-viable** (−88% DD, sub-market CAGR). It exists only to expose
   the data effect. Real work needs: a risk/drawdown overlay (regime filter, vol targeting),
   multi-factor combination (the walk-forward ML harness), and out-of-sample validation.
3. **Plain factors are weak on large caps** — expect to need better construction (sector/beta
   neutralization, quality screens to avoid value traps) and a broader universe.

## Honest scope / caveats
- Prices, returns, and the universe are fully survivorship-free (CRSP `DlyRet` is
  delisting-aware; PIT membership by PERMNO). This is the dominant fix.
- The SEC-fundamentals JOIN is still survivor-skewed: delisted PERMNOs don't resolve in SEC's
  current ticker map, so value-factor coverage is 559–595 / 957 names and tilts to survivors.
  Price factors and the backtest returns are clean.
- Data is a personal CRSP/WRDS extract — **personal research only, not redistributable**.
