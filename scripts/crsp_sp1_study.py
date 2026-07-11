"""sp1 -- always hold the largest market cap: a friend's heuristic, pre-registered as issue #2.

The claim: only ever buy the company with the largest market capitalization, keep adding
to it. Design frozen in the issue BEFORE this script existed, implemented verbatim:
survivorship-free CRSP, PIT S&P 500 members, 2005-2024; seat check at each month-end
close (largest market cap); on a seat change, rotate at the NEXT trading day's close;
PRIMARY = fully-invested single-name book at 5 bps per side; DCA variant = equal monthly
contributions at each month's first close into the then-held name vs the same stream into
the benchmark; benchmark = the cap-weighted total-return index of the same universe;
sub-periods 2005-2014 / 2015-2024. Verdict rule (frozen): CONFIRMED only if the primary
beats the benchmark on BOTH net total return AND net Sharpe, AND the DCA variant ends
above its same-stream benchmark.

    conda activate plutus
    python scripts/crsp_sp1_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from plutus.data.sources import crsp_source as crsp
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from plutus.research.backtest.metrics import month_ends
from plutus.research.backtest.regime import cap_weighted_index

COST_PER_SIDE = 5e-4
SUBPERIODS = [("2005-01-03", "2014-12-31"), ("2015-01-01", "2024-12-31")]


def perf(nav: pd.Series) -> dict:
    ret = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return {
        "total": nav.iloc[-1] / nav.iloc[0] - 1.0,
        "cagr": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1.0,
        "sharpe": float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else float("nan"),
        "maxdd": float((nav / nav.cummax() - 1.0).min()),
    }


def row(tag: str, nav: pd.Series) -> None:
    p = perf(nav)
    subs = "  ".join(f"{nav.loc[a:b].iloc[-1] / nav.loc[a:b].iloc[0] - 1.0:>+9.1%}"
                     for a, b in SUBPERIODS)
    print(f"  {tag:>10} {p['total']:>+10.1%} {p['cagr']:>+7.1%} {p['sharpe']:>7.2f} "
          f"{p['maxdd']:>7.1%}  {subs}")


def main() -> None:
    ensure_dirs()
    adj = pd.read_parquet(PARQUET_DIR / "crsp_adj_close.parquet")
    dlyret = pd.read_parquet(PARQUET_DIR / "crsp_dlyret.parquet")
    cap = pd.read_parquet(PARQUET_DIR / "crsp_mktcap.parquet")
    spells = pd.read_parquet(PARQUET_DIR / "crsp_members.parquet")
    tmap = pd.read_parquet(PARQUET_DIR / "crsp_ticker_map.parquet")
    ticker = dict(zip(tmap["permno"].astype(str), tmap["ticker"]))
    members_asof = crsp.members_asof_from_spells(spells)
    dates = dlyret.index

    # The seat at each month-end close: the member with the largest market cap.
    seats = {}
    for t in month_ends(dates):
        m = [str(p) for p in members_asof(t)]
        c = cap.loc[t, [p for p in m if p in cap.columns]].dropna()
        if len(c):
            seats[t] = c.idxmax()
    seats = pd.Series(seats)

    # held.loc[d] = the name earning day d's return. A seat change at month-end t trades
    # at the next day's close e (cost charged on e); the new name earns from e+1 on.
    held = pd.Series(np.nan, index=dates, dtype=object)
    cost = pd.Series(0.0, index=dates)
    cur = None
    changes = []
    for t, s in seats.items():
        if s == cur:
            continue
        pos = dates.get_loc(t)
        if pos + 2 >= len(dates):
            break
        cost.loc[dates[pos + 1]] += 2.0 * COST_PER_SIDE if cur is not None else COST_PER_SIDE
        held.loc[dates[pos + 2]] = s
        changes.append((t, s))
        cur = s
    held = held.ffill()

    ret = pd.Series([dlyret.loc[d, h] if isinstance(h, str) else np.nan
                     for d, h in held.items()], index=dates).fillna(0.0)
    net = (1.0 + ret) * (1.0 - cost) - 1.0
    nav = (1.0 + net).cumprod()
    bench = cap_weighted_index(adj, cap)
    bench = bench / bench.iloc[0]

    print("seat history (month-end winner, rotation at the next close):")
    for t, s in changes:
        print(f"   {t.date()}  {ticker.get(s, s)}")

    print(f"\n  {'run':>10} {'net total':>10} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7}  "
          f"{'2005-14':>9}  {'2015-24':>9}")
    row("sp1", nav)
    row("benchmark", bench)
    nav.to_csv(BACKTESTS_DIR / "sp1_primary.csv")
    bench.to_csv(BACKTESTS_DIR / "sp1_benchmark.csv")

    # DCA: 1 unit at each month's first close into the then-held name (buy leg costed),
    # the same stream into the frictionless benchmark index.
    first_days = pd.Series(dates, index=dates).groupby(dates.to_period("M")).min()
    contrib = set(first_days[first_days >= held.first_valid_index()])
    v = vb = paid = 0.0
    bret = bench.pct_change().fillna(0.0)
    for d in dates:
        v *= 1.0 + net.loc[d]
        vb *= 1.0 + bret.loc[d]
        if d in contrib:
            v += 1.0 - COST_PER_SIDE
            vb += 1.0
            paid += 1.0
    print(f"\n  DCA ({int(paid)} monthly units): sp1 {v:,.1f} vs benchmark {vb:,.1f} "
          f"({v / vb - 1.0:+.1%})")

    s, b = perf(nav), perf(bench)
    confirmed = s["total"] > b["total"] and s["sharpe"] > b["sharpe"] and v > vb
    print(f"\nVERDICT (issue #2 frozen rule): {'CONFIRMED' if confirmed else 'REJECTED'} -- "
          f"net {s['total']:+.1%} vs {b['total']:+.1%}, Sharpe {s['sharpe']:.2f} vs "
          f"{b['sharpe']:.2f}, DCA {v / vb - 1.0:+.1%}")


if __name__ == "__main__":
    main()
