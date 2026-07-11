# Daily top-gainer rotation: pre-registered, and rejected at every reading

A friend's live heuristic -- "just find the day's biggest gainer and buy it" -- and the
repo's first study whose design was frozen in a public issue
([#1](https://github.com/Rhymer-Lcy/plutus-quant/issues/1)) before any code existed. The
issue fixed the universe (PIT S&P 500 members, survivorship-free CRSP with delisting
returns, 2005-2024), the execution (rank at the day-t close; the PRIMARY buys the top-1
name at the t+1 close and is replaced at the t+2 close -- the ranking does not exist
before the t close prints), the variants (top-10 equal weight; t+1 OPEN entry), the costs
(5 bps per side, gross also reported), the benchmark (cap-weighted total-return index of
the same universe), and the verdict rule. Reproduce: `python scripts/crsp_topgainer_study.py`.

## Result (net of 5 bps per side unless marked)

| run | net total | CAGR | Sharpe | maxDD | 2005-14 | 2015-24 |
|---|---:|---:|---:|---:|---:|---:|
| top-1 (primary) | **−99.5%** | −23.3% | −0.02 | −99.7% | −96.6% | −85.8% |
| top-1 gross (no costs) | −29.4% | −1.7% | 0.35 | −98.7% | −59.3% | +69.4% |
| top-1, t+1 OPEN entry | −100.0% | −40.8% | −0.42 | −100.0% | −100.0% | −93.4% |
| top-10 equal weight | −84.7% | −9.0% | −0.12 | −91.0% | −49.7% | −69.7% |
| benchmark | +723.8% | +11.1% | 0.65 | −54.5% | +128.2% | +261.1% |

## Verdict: REJECTED (frozen rule -- and by every variant, gross included)

- The claim loses even before costs: gross −29% over twenty years while the market made
  +724%. Daily winners revert; the strategy is a machine for buying the top tick of
  short-term attention. Costs then compound the wreck (~100% one-way turnover per day).
- The "open the app in the morning and buy" reading is the worst of all (−40.8%/yr):
  buying the t+1 open pays the overnight gap after a winner day and then eats the
  intraday fade -- the same overnight/intraday asymmetry documented in
  [overnight_study.md](overnight_study.md), from the buying side.
- Diversifying to top-10 dilutes but does not change the sign. Nothing here is rescuable
  by execution detail: the underlying gross effect points the other way.
- The survivor-memory mechanism, made concrete: the most-picked names over the window are
  AMD (69 picks), FSLR, NFLX, NVDA, MU (43) -- the anecdotes one remembers are exactly
  the volatile names this rule keeps touching, and the book still ends near zero. A
  remembered winning pick is not evidence the RULE wins.
