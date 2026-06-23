# Statistical-arbitrage pairs trading on survivorship-free US large-cap CRSP

After the monthly cross-sectional alpha program failed out-of-sample (see `ml_zoo_study.md` Phase 9),
this tests a **structurally different** family the user picked from the menu: **daily time-series
mean-reversion between co-moving pairs** — not a cross-sectional return forecast. Pairs trading is
**capacity-limited**, which is the one regime where a retail-size book has an *advantage* (big money
can't fit, so the spread is less competed away). Two canonical selection methods, same harness.

## Method

Universe: the survivorship-free large-cap CRSP lake (`crsp_adj_close`, 957 names, 2005-01-03 →
2024-12-31) — liquid and shortable, which pairs requires. Engine: `research/backtest/pairs.py`.

Walk-forward, **non-overlapping** windows: formation 252d → trading 126d, step 126d (≈38 windows).

- **Distance method (Gatev-Goetzmann-Rouwenhorst 2006):** in the formation window, normalize each
  name to a total-return index (the lake is DlyRet-adjusted, so price ratios *are* cumulative TR);
  pick the top-K pairs by smallest SSD of normalized prices; open when the spread exceeds
  `entry_z ×` its *formation* std, close on reversion through zero.
- **Cointegration method (Engle-Granger):** OLS hedge ratio `Pᵢ = α + β·Pⱼ + ε`, keep pairs whose
  residual passes ADF stationarity (`< −2.86`, ≈5%), trade the residual **z-score** (formation
  mean/std), `|z| > entry_z`. The practitioners' default and a genuinely different selection.

**No look-ahead by construction:** pair selection, the normalization base / hedge ratio, and the
trigger scale (std or ADF) use **only formation data**; each day's position is decided at close *d*
and realized *d → d+1*; survivorship is handled by the lake (a delisted leg carries its DlyRet
delisting return). Execution is dollar-neutral 50/50 with per-leg slippage on open/close + short
borrow (tests in `tests/test_pairs.py`, incl. a formation-only-scale leak guard).

## Result — both methods are dead in liquid US large-cap, net of cost

| method | low 2/30 | **realistic 5/50** | high 10/100 | in-mkt |
|---|---:|---:|---:|---:|
| distance | 0.15 | **−0.07** | −0.46 | 68% |
| cointegration | −0.06 | **−0.20** | −0.45 | 90% |

(Sharpe; formation 252 / trading 126 / top_k 20 / entry_z 2.0.) The distance method is negative at
realistic cost and **robustly negative across the whole grid** (top_k ∈ {10,20,50} × entry_z ∈
{1.5,2.0,2.5}: every one of the 9 cells is −0.07 … −0.18). Cointegration is *worse* (it stays in the
market ~90% of the time, bleeding borrow). Neither half of the sample is positive:

| period | distance mean yearly Sharpe | cointegration |
|---|---:|---:|
| 2005-2014 | −0.11 | −0.18 |
| 2015-2024 | −0.11 | −0.25 |

**The only thing pairs trading does is harvest crisis reversals.** Both methods make money precisely
in the violent-dislocation-then-snap-back years — 2009 (+1.54 / +1.72) and 2020 (+1.07 / +1.22), plus
2012 and 2022 — and lose in trending years. That is the known signature of a long-mean-reversion /
short-momentum book: it is effectively short volatility, paid off in reversals and bled dry by
drifts. Over the full cycle, net of realistic cost, it does not pay.

This matches the literature: the GGR distance profits were strong 1962-2002 and **decayed to
insignificance after ~2002** as the trade was crowded — exactly what we see from 2005 on.

## Caveats (and why they don't rescue it)

- **Large-cap only.** Pairs needs shortable, low-borrow names; small-caps have more dislocation but
  borrow is expensive/hard, so a small-cap pairs book is unlikely to fare better net of cost.
- **Dollar-neutral 50/50 execution** (cointegration uses the β-residual only for the *signal*).
  β-hedging would trim residual market exposure and a little vol, but cannot turn a −0.20 Sharpe
  positive — the spreads simply don't revert enough to clear cost.
- The result is a clean walk-forward already (non-overlapping windows), so it needs no separate OOS
  year — every window is traded on formation data that precedes it.

## Conclusion

Classic equity statistical arbitrage (distance **and** cointegration) is arbitraged out of liquid US
large-cap — the same verdict the cross-sectional factors and PEAD reached. It is genuinely
capacity-limited (the retail zone), but capacity was never the binding constraint here; **robust
mean-reversion alpha net of cost simply isn't there anymore.** One more retail-accessible US-equity
family, rigorously ruled out.
