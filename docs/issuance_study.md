# Net share issuance / net-payout: the first genuinely retail-tradeable edge found in plutus

After every classic family was ruled out (factors large+small, PEAD, ML/GRU, pairs, short-term
reversal, overnight), a 5-lens residual scan ranked **net share issuance / buyback** the #1 untested
candidate: the most post-publication-robust anomaly (McLean-Pontiff), **low turnover** (so it escapes
the cost wall that killed everything fast), and uniquely **buildable from CRSP data alone** — hence
fully survivorship-free, unlike any SEC-fundamental signal whose ticker-join drops delisted names.
This tested it, and unlike everything before it, **it survived every adversarial control.**

Reproduce: `python scripts/crsp_issuance_study.py`.

## Signal (split- and dividend-immune, survivorship-free, no look-ahead)

    issuance(H) = log(mktcap_t / mktcap_{t-H}) − log(adj_close_t / adj_close_{t-H})
    factor (higher = attractive) = −issuance

Market cap is split-invariant; `adj_close` is the total return. Their difference is the growth in
market cap not explained by return = net equity raised. Price-measurement error at each date enters
*both* terms (mktcap ∝ price, adj_close ∝ price) and cancels, so the signal reduces to clean
split-adjusted share change. Because `adj_close` is the **total** return, this is the **net-payout**
form (net issuance minus buybacks **and** dividends) — a documented predictor (Boudoukh et al. net
payout; Daniel-Titman composite issuance). H = 252 (1y) and 1260 (5y). Higher factor = net
distributor / buyback firm = attractive.

## Result — a real signal everywhere, a tradeable EDGE in liquid small/mid-caps

Rank IC is positive and significant in both universes (small-cap 1y IC +0.025, t=4.4; large-cap 5y
+0.019, t=2.5) — net-payout genuinely predicts returns. The dollar-neutral long-short spread is
modest (Sharpe 0.2–0.35) and has faded recently in small-cap (2025 holdout −6.6%); the **realizable
retail form is the long-only tilt**, and that is where the edge is large and persistent.

**Small-cap net-payout, long-only top-50, monthly, net of cost, survivorship-free:**

| universe / cost | CAGR | maxDD | Sharpe | Sharpe 2021+ |
|---|--:|--:|--:|--:|
| full band @15bps | +29.8% | −24.1% | 1.44 | 1.62 |
| ADV>$1M/day @15bps | +26.3% | −22.4% | 1.32 | 1.51 |
| **ADV>$5M/day @15bps** | **+23.7%** | **−32.5%** | **1.14** | **1.40** |
| ADV>$5M/day @50bps | +21.9% | −33.9% | 1.07 | 1.33 |
| ADV>$20M/day @15bps | +18.9% | −46.1% | 0.87 | 1.11 |

vs the cap-weighted small-cap buy-and-hold bar: **Sharpe 0.61**. It survives a liquidity screen
(tradeable names only), a 50bps cost stress, and concentration (top-30/50/100 all Sharpe 1.33–1.52),
and it is **stronger in recent years, not faded** (2021+ Sharpe 1.40 at the $5M-ADV screen).

### It is the SIGNAL, not the universe (same $5M-ADV liquid subset, top-50 EW, 25bps)

| book | Sharpe | Sharpe 2021+ |
|---|--:|--:|
| equal-weight liquid B&H | 0.56 | 0.51 |
| no-signal top-50 (arbitrary) | 0.49 | 0.54 |
| smallest-50 (pure size) | 0.77 | 0.78 |
| momentum top-50 | 0.45 | 0.65 |
| reversal top-50 | 0.49 | 0.46 |
| low-vol top-50 | 0.54 | 0.49 |
| **net-payout top-50** | **1.12** | **1.38** |

Every other price signal on the identical liquid universe lands at 0.45–0.77; net-payout alone
reaches 1.12 (1.38 recent). The edge is specific to the signal, not to small-cap concentration, the
equal-weight tilt, or the universe. It is also crisis-protective: in 2008 the book returned +1.5%
while the band fell −37%.

## Honest discounts (this is strong, so discount it hard)

- **Magnitude is above the literature.** Published issuance is ~0.5–0.8 Sharpe long-short; a 1.1–1.4
  long-only Sharpe is flattered by small-cap concentration and a 2005–2025 sample that contained two
  large small-cap booms (2009, 2021). The *direction and significance* are trustworthy; treat the
  *level* as an optimistic upper bound.
- **Drawdown protection is partly an illiquidity artifact.** The headline −24% maxDD widens to −32%
  ($5M ADV) and −46% ($20M ADV) — the most-illiquid names damp drawdowns via stale marks. The honest
  tradeable drawdown is ~−32%, still well inside the index's −55%.
- **Long-only ≈ beta + a tilt.** Most of the academic issuance alpha lives in the short (heavy-
  issuer) leg, which retail cannot borrow; here the long leg alone is enough, but the book carries
  full small-cap market beta.
- **One searched signal.** The IC t=4.4 and the cross-signal control make this far more than a lucky
  fit, but it is still in-sample. It must earn a **forward paper-trading record** before real money —
  the same gate hermes-quant's A-share book is held to.

## Verdict

This is the **first retail-operable, signal-specific, liquidity- and cost-robust, recently-strong,
survivorship-free edge** the plutus program has found. Net-payout / buyback in liquid small/mid-cap
US stocks (long-only top-50, ≥$5M ADV) clears the buy-and-hold bar by a wide, signal-specific margin
(Sharpe ~1.1 full / ~1.4 recent vs 0.61) net of realistic cost. It is capacity-limited (a concentrated
small-cap book) — which is precisely the zone where a small retail account has an advantage. With the
magnitude discounted and a forward test still owed, it is the one candidate that genuinely deserves to
be carried forward rather than filed under "ruled out."
