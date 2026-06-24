"""S&P 500 index-reconstitution effect (#3) -- buildable NOW from the PIT membership spells.

When a stock is ADDED to the S&P 500, index funds must buy it (historic run-up); when DELETED,
they must sell (a forced-selling overshoot that may REVERSE). The delete-reversal is the
retail-shaped leg: you go LONG the dropped name (no shorting needed). This tests both legs as the
cumulative ABNORMAL return (name return minus the cross-sectional mean) in event time, on the
survivorship-free large-cap CRSP lake, and nets a realistic round-trip cost off the tradeable leg.

Data: crsp_members.parquet (PERMNO spells: start = addition, end = deletion). No look-ahead: enter
the trading day AFTER the event. CAVEAT: the spells carry only the EFFECTIVE date, not the ~5-day-
earlier ANNOUNCEMENT date, so the add run-up (which front-runs the announcement) is understated here
-- but the post-effective delete-reversal, the part retail could trade, is measured cleanly.

    conda activate plutus
    python scripts/crsp_index_effect_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs

HOLD = 60
RT_COST = 0.0030   # ~2 x 15bps round trip for one buy + one sell of the event leg


def _caar(ret: pd.DataFrame, events: list[tuple[str, pd.Timestamp]], hold: int) -> tuple[np.ndarray, int]:
    """Mean cumulative ABNORMAL return (name minus same-day cross-sectional mean) over event
    days 1..hold, entered the trading day strictly after each event date."""
    idx = ret.index
    abn = ret.sub(ret.mean(axis=1), axis=0)
    sums = np.zeros(hold)
    counts = np.zeros(hold)
    for permno, d in events:
        if permno not in abn.columns:
            continue
        epos = idx.searchsorted(pd.Timestamp(d), side="right")
        seg = abn[permno].to_numpy()[epos:epos + hold]
        if len(seg) == 0:
            continue
        car = np.nancumsum(np.where(np.isnan(seg), 0.0, seg))
        m = min(len(car), hold)
        sums[:m] += car[:m]
        counts[:m] += 1
    caar = np.where(counts > 0, sums / np.where(counts == 0, 1, counts), np.nan)
    return caar, int(counts[0]) if counts[0] else 0


def main() -> int:
    ensure_dirs()
    adj = pd.read_parquet(PARQUET_DIR / "crsp_adj_close.parquet")
    spells = pd.read_parquet(PARQUET_DIR / "crsp_members.parquet")
    ret = adj.pct_change(fill_method=None)
    dmin, dmax = adj.index.min(), adj.index.max()

    def has_data_after(permno, d):
        c = str(int(permno))
        return c in adj.columns and adj[c].loc[pd.Timestamp(d):].notna().sum() > HOLD

    adds, dels = [], []
    for _, r in spells.iterrows():
        p, s, e = r["permno"], pd.Timestamp(r["start"]), pd.Timestamp(r["end"])
        if dmin < s < dmax and has_data_after(p, s):
            adds.append((str(int(p)), s))
        if dmin < e < dmax - pd.Timedelta(days=120) and has_data_after(p, e):
            dels.append((str(int(p)), e))   # left the index but kept trading (a real deletion, not a delisting end-of-data)

    print("=" * 84)
    print("S&P 500 INDEX-RECONSTITUTION EFFECT -- survivorship-free CRSP large-cap, event-time CAAR")
    print(f"  {len(adds)} additions, {len(dels)} deletions (effective dates; announcement ~5d earlier not in spells)")
    print("=" * 84)
    rows = []
    for label, ev in [("ADD (run-up)", adds), ("DELETE (reversal)", dels)]:
        caar, n = _caar(ret, ev, HOLD)
        print(f"\n  {label}: n={n}  cumulative abnormal return (gross):")
        print("    " + "  ".join(f"d{d}:{caar[d-1]:+.2%}" for d in (1, 5, 10, 20, 40, 60) if d <= HOLD))
        for d in (5, 10, 20, 40, 60):
            rows.append({"leg": label, "day": d, "caar_gross": float(caar[d-1]),
                         "caar_net_rt": float(caar[d-1] - RT_COST), "n": n})
    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "crsp_index_effect.parquet")
    print(f"\n  net of a ~{RT_COST:.2%} round-trip cost, the tradeable DELETE-reversal leg is "
          f"'gross CAAR minus {RT_COST:.2%}'. A few-dozen-events/yr, capacity-limited trade.")
    print("Reading: positive DELETE CAAR that exceeds the round-trip cost = a real (if small, "
          "capacity-limited) long-the-dropped-name edge. See docs/index_effect_study.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
