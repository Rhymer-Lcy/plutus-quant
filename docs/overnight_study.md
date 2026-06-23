# The overnight-return anomaly: real, the closest near-miss, still not a standalone retail edge

For US equities almost the entire realized return accrues **overnight** (close → next open); the
intraday session (open → close) is roughly flat (Cooper-Cliff-Gulen; Lou-Polk-Skouras, "A tug of
war", 2019). That looks like a free lunch: be long only overnight. This tests whether a retail
trader can actually capture it, on the survivorship-free large-cap S&P 500 CRSP lake, with the real
bid-ask spread measured from CRSP closing quotes.

Build the open/quote panels once (`scripts/build_crsp_open_lake.py` streams DlyOpen/DlyClose/DlyRet/
DlyBid/DlyAsk for the large-cap union), then `python scripts/crsp_overnight_study.py`.

## Decomposition (split- and dividend-immune by construction)

```
intraday[t]  = close[t]/open[t] - 1                       (same-day; no overnight event contaminates it)
overnight[t] = (1 + DlyRet[t]) / (1 + intraday[t]) - 1     (close[t-1]->open[t] TOTAL return)
```
so `(1+overnight)*(1+intraday) - 1 == DlyRet` identically — verified residual ~3e-16. No look-ahead:
each overnight return is realized at the next open from a position taken at the prior close;
membership is point-in-time.

## Result — the premium is overnight; harvesting it is the problem

Equal-weight PIT universe, daily, 957 names, 2005-2024:

| leg | mean/day | annualized | t vs 0 |
|---|--:|--:|--:|
| overnight | 3.48 bp | **+8.4%** | 3.3 |
| intraday | 1.39 bp | +2.1% | 0.9 |
| total (buy & hold) | 4.93 bp | +10.7% | 2.6 |

~78% of the total return accrues overnight — the anomaly is real and significant. Now harvest it by
buying each close and selling each next open (one round trip every day), charging the **actual
measured** CRSP closing spread, crossed twice per round trip:

| | bp/day |
|---|--:|
| gross overnight | +3.48 |
| round-trip spread (measured) | −4.81 |
| **net** | **−1.33** (Sharpe −0.28, −4.0%/yr) |

The spread you must cross to harvest it slightly exceeds the return. But this verdict is genuinely
cost-sensitive, and that is where the honesty lives.

## How close is it? (the three checks that decide it)

**(4) Cost sensitivity.** The closing quoted spread is the *continuous-session* NBBO; the MOC/MOO
auctions you would actually use typically clear *inside* it, so ×1.0 is an upper bound.

| spread charged | net bp/day | Sharpe | ann |
|---|--:|--:|--:|
| ×1.00 (full quoted) | −1.33 | −0.28 | −4.0% |
| ×0.50 | +1.07 | +0.23 | +2.0% |
| ×0.25 | +2.27 | +0.48 | +5.1% |

At half the quoted spread — a plausible auction cost — the long-only timing book is net-positive.

**(5) The tradeable subset.** The equal-weight loss is dominated by wide-spread names. Restricting to
the genuinely tradeable book (below-median spread that day): gross 3.15 bp vs spread 1.80 bp →
**net +1.36 bp/day (+2.8%/yr, Sharpe 0.31) even at the full quoted spread.**

So overnight is the closest near-miss in the whole project: on the liquid names, or at a realistic
auction cost, the long-only timing strategy is net-positive. Two things still sink it as an edge:

- **It is dominated by costless buy-and-hold.** Even net-positive (+2.8%), it is far below simply
  holding the index (+10.7%, zero trading): you give up the positive intraday return and pay a daily
  spread to capture a subset of the same premium. There is no reason to run it as a timing overlay.
- **(6) The proper cross-sectional form is buried by its own turnover.** The real anomaly is
  cross-sectional overnight momentum (long high / short low trailing-overnight names). Gross it is
  strong and beta-neutral — **+10.6%/yr, Sharpe 1.36, t=6.1** — and would be additive to buy-and-hold.
  But an overnight-only book must round-trip *every name every day*, and that daily spread wall turns
  +10.6% gross into **−15.2% net (Sharpe −2.13)**. Holding longer to cut turnover just converts it
  into ordinary cross-sectional momentum, already shown to have no net edge.

## Verdict

The overnight anomaly is **real, large, and the closest thing to a tradeable retail signal found in
US equities** — on liquid names or at a realistic auction cost the long-only timing form clears zero.
But it is **not a standalone edge**: the timing form is dominated by costless buy-and-hold, and the
cross-sectional form (which *would* add to buy-and-hold) is destroyed by the daily round-trip it
inherently requires. Like PEAD, it survives precisely because it sits *on* the cost boundary.

The one honest, usable residual is not a strategy but an **execution refinement**: the overnight/
intraday asymmetry says *when* to execute trades you are doing anyway — for a position you must buy,
the close tends to be a better entry than the open; for one you must sell, the open. That trims
implementation shortfall on an existing book; it does not, on its own, pay. Consistent with the rest
of the program: a real anomaly, owned by whoever already provides the liquidity and is trading at the
auctions regardless — not by a retail trader crossing the spread to chase it.
