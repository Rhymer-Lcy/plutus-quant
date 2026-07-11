"""Daily top-gainer rotation -- a friend's live heuristic, pre-registered as issue #1.

The claim: "just find the day's biggest gainer and buy it." The design was frozen in the
issue BEFORE this script existed and is implemented here verbatim: PIT S&P 500 members,
survivorship-free CRSP dlyret (delisting returns included); signal = rank day-t total
returns; PRIMARY = hold the top-1 name from the t+1 close to the t+2 close (daily
rotation -- the ranking does not exist before the t close prints); variants = top-10
equal weight, and t+1 OPEN entry (raw opens adjusted by the close's adjustment factor --
a dividend-timing approximation, disclosed); 5 bps per side, gross also reported;
benchmark = the cap-weighted total-return index of the same universe; sub-periods
2005-2014 / 2015-2024. Verdict rule (frozen): CONFIRMED only if the primary beats the
benchmark on BOTH net total return AND net Sharpe over the full window.

Selection ties break by panel column order (first PERMNO wins) -- deterministic. A held
name whose return is missing (post-delisting gap) earns 0 (cash) until the next rotation.

    conda activate plutus
    python scripts/crsp_topgainer_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from plutus.data.sources import crsp_source as crsp
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from plutus.research.backtest.regime import cap_weighted_index

COST_PER_SIDE = 5e-4
N_TOP = 10                          # the diversified variant
SUBPERIODS = [("2005-01-03", "2014-12-31"), ("2015-01-01", "2024-12-31")]


def rotation_returns(picks: pd.Series, dlyret: pd.DataFrame, cost_per_side: float) -> pd.Series:
    """Net daily returns of a book that holds picks[t] (a tuple of permnos, equal weight)
    from the t+1 close to the t+2 close. Day-d return accrues to the names picked at d-2;
    the rotation trade happens at the d-1 close and its cost is charged there."""
    held = picks.shift(2)
    ret = pd.Series(0.0, index=picks.index)
    cost = pd.Series(0.0, index=picks.index)
    prev: tuple = ()
    for d in picks.index:
        names = held.loc[d] if isinstance(held.loc[d], tuple) else ()
        if names:
            r = dlyret.loc[d, list(names)]
            ret.loc[d] = float(np.nanmean(r.to_numpy(dtype=float))) if r.notna().any() else 0.0
        if names != prev:
            if prev:
                k = len(set(names) ^ set(prev)) / 2          # names swapped out for new ones
                cost.loc[d] = cost_per_side * 2.0 * k / max(len(names), 1)   # sell + buy legs
            elif names:
                cost.loc[d] = cost_per_side                  # the initial buy: one side, full book
        prev = names
    # the trade at the d-1 close is charged on d-1; shift the cost series back one day
    cost = cost.shift(-1).fillna(0.0)
    return (1.0 + ret) * (1.0 - cost) - 1.0


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
    print(f"  {tag:>16} {p['total']:>+10.1%} {p['cagr']:>+7.1%} {p['sharpe']:>7.2f} "
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

    # Ranking day by day among that day's members with a printed return.
    top1, topn = {}, {}
    for d in dlyret.index:
        m = [str(p) for p in members_asof(d)]
        r = dlyret.loc[d, [c for c in m if c in dlyret.columns]].dropna()
        if r.empty:
            continue
        order = r.sort_values(ascending=False)
        top1[d] = (order.index[0],)
        topn[d] = tuple(order.index[:N_TOP])
    top1 = pd.Series(top1).reindex(dlyret.index)
    topn = pd.Series(topn).reindex(dlyret.index)

    # The open-entry variant prices the same picks open-to-open: raw opens adjusted by the
    # close's adjustment factor (adj_close / raw_close), a disclosed approximation.
    raw_close = pd.read_parquet(PARQUET_DIR / "crsp_close_raw.parquet")
    raw_open = pd.read_parquet(PARQUET_DIR / "crsp_open_raw.parquet")
    adj_open = raw_open * (adj / raw_close)
    open_ret = adj_open.pct_change(fill_method=None)

    bench = cap_weighted_index(adj, cap)
    runs = {
        "top1 (primary)": (1.0 + rotation_returns(top1, dlyret, COST_PER_SIDE)).cumprod(),
        "top1 gross": (1.0 + rotation_returns(top1, dlyret, 0.0)).cumprod(),
        "top1 open-entry": (1.0 + rotation_returns(top1, open_ret, COST_PER_SIDE)).cumprod(),
        f"top{N_TOP} EW": (1.0 + rotation_returns(topn, dlyret, COST_PER_SIDE)).cumprod(),
        "benchmark": bench / bench.iloc[0],
    }
    print(f"\n  {'run':>16} {'net total':>10} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7}  "
          f"{'2005-14':>9}  {'2015-24':>9}")
    for tag, nav in runs.items():
        slug = tag.replace(" ", "_").replace("(", "").replace(")", "")
        nav.to_csv(BACKTESTS_DIR / f"topgainer_{slug}.csv")
        row(tag, nav)

    picked = pd.Series([n for t in top1.dropna() for n in t]).value_counts().head(8)
    print("\n  most-picked names:",
          ", ".join(f"{ticker.get(p, p)} x{c}" for p, c in picked.items()))

    s, b = perf(runs["top1 (primary)"]), perf(runs["benchmark"])
    confirmed = s["total"] > b["total"] and s["sharpe"] > b["sharpe"]
    print(f"\nVERDICT (issue #1 frozen rule): {'CONFIRMED' if confirmed else 'REJECTED'} -- "
          f"net {s['total']:+.1%} vs benchmark {b['total']:+.1%}, "
          f"Sharpe {s['sharpe']:.2f} vs {b['sharpe']:.2f}")


if __name__ == "__main__":
    main()
