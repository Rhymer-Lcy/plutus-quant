# Multi-factor + risk overlay — and the humbling benchmark

Two fixes for the non-viable candidate (docs/survivorship_study.md): a **risk overlay** (market
trend filter → exposure) and a **multi-factor ML combiner** (walk-forward LightGBM over the
6-factor battery). All survivorship-free CRSP, PIT S&P 500, top-20 monthly, US frictions.
A passive **buy-and-hold of the cap-weighted market** is included as the benchmark that matters.
Reproduce: `scripts/crsp_multifactor_study.py [--eval-start YYYY-MM-DD]`.

## Full history, 2005–2024 (incl. the GFC) — risk overlay as crash insurance

| strategy | CAGR | max DD | Calmar | Sharpe |
|---|---:|---:|---:|---:|
| candidate (val+rev) | 4.47% | −88.05% | 0.05 | 0.30 |
| candidate + regime | 5.11% | **−42.61%** | 0.12 | 0.35 |
| ML multi-factor¹ | 5.69% | −48.58% | 0.12 | 0.40 |
| ML + regime¹ | 3.97% | −40.05% | 0.10 | 0.36 |
| **S&P 500 proxy (buy & hold)** | **11.12%** | −54.46% | **0.20** | **0.65** |

The trend filter does its job as **crash insurance**: it turns the candidate's −88% 2008
drawdown into −43% (and lifts return). ¹But the ML rows only trade from 2012-08 (walk-forward
warm-up), so they are not comparable to the others here — that is what the fair window below is for.

## Fair common window, 2012-08 → 2024 (all strategies trade) — does anything beat the index?

| strategy | CAGR | max DD | Calmar | Sharpe |
|---|---:|---:|---:|---:|
| candidate (val+rev) | 8.69% | −58.76% | 0.15 | 0.45 |
| candidate + regime | 5.14% | −37.86% | 0.14 | 0.36 |
| ML multi-factor | 9.32% | −48.58% | 0.19 | 0.51 |
| ML + regime | 6.47% | −40.05% | 0.16 | 0.45 |
| **S&P 500 proxy (buy & hold)** | **15.19%** | **−33.89%** | **0.45** | **0.93** |

## Findings (honest)

1. **The benchmark wins on everything.** Passive buy-and-hold beats every active variant on
   CAGR, Sharpe, AND Calmar — and in 2012–2024 it even has the **smallest drawdown** (−34% vs
   the strategies' −38% to −59%). Long-only large-cap factor strategies **do not beat the index
   here**, before even asking about robustness.
2. **The ML combiner modestly beats the fixed candidate** (fair window: Sharpe 0.51 vs 0.45,
   Calmar 0.19 vs 0.15, CAGR 9.3% vs 8.7%). The walk-forward harness adds value — just not
   enough to clear the index.
3. **The regime filter is crash insurance, not alpha.** It halves the GFC drawdown (−88%→−43%)
   but in the 2012–2024 bull it whipsaws — cutting CAGR (8.7%→5.1%) while barely changing
   Calmar. A binary trend filter pays its way only when there is a crash to avoid.

This is the academically-expected result (the equity premium dominates; large-cap factor alpha
is weak/arbitraged; long-only can't isolate factor spreads; 20-name concentration adds risk,
not return). The contribution here is the RIGOR that surfaces it — survivorship-free, OOS, and
benchmarked — instead of the survivorship-biased fantasy (+372%) we started with.

## Where edge might actually live (next directions)

- **Long/short, market-neutral**: harvest the factor *spread* (top minus bottom) and strip the
  market beta, instead of long-only (which is ~90% just equity beta). This is the standard way
  factors are actually traded.
- **Broader / less-efficient universe**: small/mid caps, where factor premia are stronger and
  large institutions can't fish — the survivorship-free CRSP universe can be widened beyond the
  S&P 500.
- **Better signals**: the stale academic factors are arbitraged; alternative data / shorter-
  horizon / interaction features, validated OOS.
- **Reframe the goal**: if beating the index is the (hard) bar, an honest alternative is
  *risk-managed index-plus* (benchmark return with smaller drawdowns) — the regime overlay is a
  first step there.

## Caveats
- ML OOS starts 2012-08 (24-month min train + factor warm-up); a longer ML history needs an
  earlier data start or a shorter warm-up.
- The SEC-fundamentals join is survivor-skewed (value-factor coverage tilts to survivors);
  prices/returns/universe and the benchmark are fully survivorship-free.
- The benchmark is a cap-weighted proxy built from the same universe, not the official SPX TR
  index — close, but not identical.
