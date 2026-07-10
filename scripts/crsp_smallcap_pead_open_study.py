"""Small-cap PEAD with OPEN-price entry and per-name half-spread costs -- the two upgrades the
close-entry study named but never ran.

docs/smallcap_pead_study.md's verdict was "a real gross edge, gated by execution cost": the
de-leaked close-entry book loses -5.2%/yr at idealized costs and worse at realistic ones. Its own
"how to push it over the line" section names the upgrades tested here, on data already on disk:

  - OPEN entry (path #3): the drift is front-loaded, and a close-entry book waits a full session
    after the filing is public. Entering at the next day's OPEN is leak-free (the announcement
    predates the open; the overnight gap is excluded because entry-day accrual is open-to-close,
    from the same-day raw ratio -- split/dividend-immune) and captures the day-1 intraday drift.
  - Per-name half-spread costs (path #2): a flat 20 bps charges a liquid mid-cap and an illiquid
    micro-cap the same toll. The quote-derived half-spread panel prices each name's actual toll,
    day by day (median 7.7 bps; the illiquid tail is far wider).

PRE-REGISTERED: |SUE| >= 1.5 and holds {10, 20, 40, 60} exactly as the close-entry study; entry at
the next trading day's OPEN (entry_offset=0 with the intraday panel -- the same calendar day the
de-leaked close-entry book enters, one session earlier); costs = per-name half-spread with 300
bps/yr borrow (realistic) and 50 bps/yr (optimistic bound); plus the LONG-LEG-ONLY readout (a
retail account cannot short small caps). Window pinned to the documented 2010-2024. No threshold
or hold tuning follows a weak read.

    python scripts/crsp_smallcap_pead_open_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from plutus.research.backtest.event_study import event_time_portfolio

EVENTS_PATH = PARQUET_DIR / "crsp_smallcap_pead_events.parquet"
LAKE_END = "2024-12-31"          # the documented window; see crsp_smallcap_longshort_study.py
SUE_THRESHOLD = 1.5
HOLDS = [10, 20, 40, 60]
BORROW_REALISTIC, BORROW_LOW = 300.0, 50.0
FLAT_SLIP_BPS = 20.0             # the close-entry study's realistic flat slippage, for reference


def main() -> None:
    ensure_dirs()
    adj = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_adj_close.parquet").loc[:LAKE_END]
    open_raw = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_open_raw.parquet").loc[:LAKE_END]
    close_raw = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_close_raw.parquet").loc[:LAKE_END]
    halfspread = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_halfspread.parquet").loc[:LAKE_END]
    events = pd.read_parquet(EVENTS_PATH)
    ret = adj.pct_change(fill_method=None)
    intraday = (close_raw / open_raw - 1.0).reindex(index=ret.index, columns=ret.columns)
    halfspread = halfspread.reindex(index=ret.index, columns=ret.columns)

    tradable = events[events["sue"].abs() >= SUE_THRESHOLD]
    ev_named = tradable[tradable["permno"].isin(ret.columns)]
    entry_pos = ret.index.searchsorted(pd.DatetimeIndex(ev_named["entry_date"]), side="right")
    in_range = entry_pos < len(ret.index)
    covered = [not np.isnan(intraday.iloc[p][c]) if c in intraday.columns else False
               for p, c in zip(entry_pos[in_range], ev_named["permno"].to_numpy()[in_range])]
    print(f"{len(events):,} cached SUE events; {len(tradable):,} at |SUE|>={SUE_THRESHOLD}")
    print(f"open-price coverage on entry days: {np.mean(covered):.1%} of tradable events "
          f"(the rest fall back to close-to-close entry-day accrual)")
    print(f"median half-spread across the panel: {halfspread.stack().median():.4%}\n")

    rows = []
    print(f"  {'variant':>44} {'hold':>5} {'annRet':>8} {'Sharpe':>7} {'maxDD':>8} {'nL':>6} {'nS':>6}")

    def line(tag, **kw):
        for h in HOLDS:
            r = event_time_portfolio(events, ret, hold_days=h, sue_threshold=SUE_THRESHOLD, **kw)
            print(f"  {tag:>44} {h:>5} {r.ann_return:>+8.2%} {r.sharpe:>7.2f} "
                  f"{r.max_drawdown:>8.1%} {r.avg_long:>6.1f} {r.avg_short:>6.1f}")
            rows.append({"variant": tag, "hold": h, "ann_return": r.ann_return,
                         "sharpe": r.sharpe, "max_dd": r.max_drawdown})
        print()

    # A. close-entry de-leaked baselines (reference; reproduce the documented reads)
    line("close entry, flat 20bps, borrow 300 (baseline)",
         slippage_bps=FLAT_SLIP_BPS, borrow_bps_annual=BORROW_REALISTIC, entry_offset=1)
    # B. open entry, same flat costs (isolates the TIMING upgrade)
    line("OPEN entry, flat 20bps, borrow 300",
         slippage_bps=FLAT_SLIP_BPS, borrow_bps_annual=BORROW_REALISTIC, entry_offset=0,
         intraday_entry=intraday)
    # C. open entry + per-name half-spread (the full upgrade), realistic and low borrow
    line("OPEN entry, per-name halfspread, borrow 300",
         slippage_bps=FLAT_SLIP_BPS, borrow_bps_annual=BORROW_REALISTIC, entry_offset=0,
         intraday_entry=intraday, halfspread_panel=halfspread)
    line("OPEN entry, per-name halfspread, borrow 50",
         slippage_bps=FLAT_SLIP_BPS, borrow_bps_annual=BORROW_LOW, entry_offset=0,
         intraday_entry=intraday, halfspread_panel=halfspread)

    # D. the retail form: LONG LEG ONLY (no shorting), open entry, per-name half-spread
    long_events = events[events["sue"] >= SUE_THRESHOLD]
    print("  LONG-LEG ONLY (retail form; absolute return, no borrow):")
    for h in HOLDS:
        r = event_time_portfolio(long_events, ret, hold_days=h, sue_threshold=SUE_THRESHOLD,
                                 slippage_bps=FLAT_SLIP_BPS, borrow_bps_annual=0.0,
                                 entry_offset=0, intraday_entry=intraday,
                                 halfspread_panel=halfspread)
        print(f"  {'OPEN entry, halfspread, long leg only':>44} {h:>5} {r.ann_return:>+8.2%} "
              f"{r.sharpe:>7.2f} {r.max_drawdown:>8.1%} {r.avg_long:>6.1f} {r.avg_short:>6.1f}")
        rows.append({"variant": "long leg only, open entry, halfspread", "hold": h,
                     "ann_return": r.ann_return, "sharpe": r.sharpe, "max_dd": r.max_drawdown})

    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "crsp_smallcap_pead_open.parquet")
    print(f"\nwrote crsp_smallcap_pead_open.parquet to {BACKTESTS_DIR}")
    print("Reading: B-minus-A isolates what faster entry buys; C tests whether honest per-name "
          "costs beat the flat assumption or worsen it (illiquid tails trade WIDER than 20 bps). "
          "The upgrade is adopt-worthy only if the realistic long-short (C, borrow 300) turns "
          "positive AND the long leg alone (D) clears zero -- a retail account cannot short small "
          "caps. Per the pre-registration, a weak read closes the family; no tuning follows.")


if __name__ == "__main__":
    main()
