# S&P 500 index-reconstitution: the delete-reversal is a real, small, capacity-limited retail edge

Buildable now from the PIT membership spells (`crsp_members.parquet`) — no new data. When index funds
are forced to sell a deleted name, the overshoot can reverse; going **long the dropped name** needs no
shorting, so it is retail-shaped. Event-time cumulative ABNORMAL return (name minus the cross-
sectional mean), survivorship-free large-cap CRSP, entered the day after the effective date.

Reproduce: `python scripts/crsp_index_effect_study.py`.

| leg | n | d5 | d10 | d20 | d40 | d60 |
|---|--:|--:|--:|--:|--:|--:|
| ADD (run-up) | 473 | −1.3% | −1.6% | −1.9% | −2.4% | −3.0% |
| **DELETE (reversal)** | 223 | −0.3% | **+3.0%** | **+3.7%** | **+5.8%** | **+5.4%** |

- **ADD is dead post-effective** — additions *underperform* after the effective date (the run-up
  front-runs the ~5-day-earlier announcement, which the spells don't carry; by the effective date it
  is over and reverses). Not tradeable from this data.
- **DELETE-reversal is real**: deleted names earn a +3.7% (20d) to +5.4% (60d) abnormal bounce, ~+3.4%
  to +5.1% net of a ~0.30% round-trip. Retail-operable (buy the dropped name, hold ~1–3 months).

**Caveat — capacity.** ~223 deletions over 2005-2024 ≈ ~11/year; held 20–60 days you hold only a
handful at a time. The dropped names are distressed/volatile, so a real book's Sharpe is moderate
despite the clean average. This is a genuine but **small, episodic, capacity-limited** edge — which
is exactly the niche where a small retail account, not a fund, can operate. A worthwhile *satellite*,
not a core strategy.
