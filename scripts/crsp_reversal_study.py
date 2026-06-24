"""Short-term reversal — the last untested cross-sectional US-equity family, closed out honestly.

The factor arc (docs/longshort_study.md, smallcap_study.md) tested MONTHLY reversal and found it a
high-turnover illusion. This isolates the canonical SHORT-TERM reversal (1-2 week horizon, weekly
rebalance), which is where the effect is documented to be largest — and where two well-known
illusions live, both tested here head-on:

  1. BID-ASK BOUNCE. A name that closed near the BID looks like a "loser" and "reverts" when it next
     closes near the ASK — pure microstructure, not tradeable alpha, because the engine reads the
     signal AND enters at the same closing print. Control: a SKIP-1-DAY variant forms the signal
     through close[t-1] (one trading day before the rebalance) while still entering at close[t], so
     the last signal price and the entry price are different prints. If reversal collapses under the
     skip, its gross profit was the bounce (Jegadeesh 1990 / Lehmann 1990 / Nagel 2012).
  2. THE TURNOVER COST WALL. Weekly reversal turns the book over almost completely each week, so
     even light per-trade slippage compounds into a huge annual drag. Reported as gross vs net.

Dollar-neutral quintile long-short (research.backtest.long_short), survivorship-free CRSP, both the
large-cap S&P 500 lake and the mid/small-cap band. No-look-ahead: signal read at t, return realized
t->t+1; the skip variant additionally severs the signal/entry price overlap. Per-year and a 2025
holdout (small-cap lake only; the large-cap lake ends 2024) show whether anything is a recent
phenomenon or a permanent cost-walled illusion.

    conda activate plutus
    python scripts/crsp_reversal_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from plutus.data.sources import crsp_source as crsp
from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from plutus.research.backtest.long_short import quantile_long_short
from plutus.research.backtest.regime import cap_weighted_index
from plutus.research.factors import library as fl

PPY = 52  # weekly


def _week_ends(dates: pd.DatetimeIndex) -> list:
    """Last trading day of each ISO week — the weekly rebalance dates."""
    s = pd.Series(dates, index=dates)
    return s.groupby(dates.to_period("W")).max().tolist()


def _load(universe: str):
    """Return (adj_close, mktcap, members_asof, slippage_bps, borrow_bps). Realistic frictions
    differ by liquidity: liquid large-caps ~5bps/side + 50bps borrow; mid/small ~15bps + 300bps."""
    if universe == "large":
        adj = pd.read_parquet(PARQUET_DIR / "crsp_adj_close.parquet")
        cap = pd.read_parquet(PARQUET_DIR / "crsp_mktcap.parquet")
        spells = pd.read_parquet(PARQUET_DIR / "crsp_members.parquet")
        _m = crsp.members_asof_from_spells(spells)
        members = lambda d: {str(p) for p in _m(d)}          # noqa: E731 (CRSP panels keyed by str PERMNO)
        return adj, cap, members, 5.0, 50.0
    adj = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_adj_close.parquet")
    cap = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_mktcap.parquet")
    members = crsp.size_band_members_asof(cap)
    return adj, cap, members, 15.0, 300.0


def _ann(returns: pd.Series) -> float:
    """Geometric annualized return of a weekly return series."""
    if len(returns) < 1:
        return float("nan")
    return float((1.0 + returns).prod() ** (PPY / len(returns)) - 1.0)


def _sharpe(returns: pd.Series) -> float:
    sd = returns.std()
    return float(returns.mean() / sd * np.sqrt(PPY)) if sd and sd > 0 else float("nan")


def _ls(adj, sig, eval_dates, members, market, slip, borrow):
    return quantile_long_short(adj, sig, eval_dates, members, quantile=0.2,
                               slippage_bps=slip, borrow_bps_annual=borrow, market_index=market)


def run(universe: str) -> list[dict]:
    adj, cap, members, slip, borrow = _load(universe)
    eval_dates = _week_ends(adj.index)
    market = cap_weighted_index(adj, cap)

    print("=" * 96)
    print(f"SHORT-TERM REVERSAL -- {universe}-cap, survivorship-free CRSP, weekly dollar-neutral "
          f"quintile long-short")
    print(f"  {adj.shape[1]} names, {adj.index.min().date()} -> {adj.index.max().date()}, "
          f"{len(eval_dates)} weekly rebalances; realistic cost {slip:.0f}bps/side + {borrow:.0f}bps borrow")
    print("=" * 96)

    signals = {
        "rev_1w  no-skip": fl.reversal(adj, 5),
        "rev_1w  skip-1d": fl.reversal(adj, 5).shift(1),
        "rev_2w  no-skip": fl.reversal(adj, 10),
        "rev_2w  skip-1d": fl.reversal(adj, 10).shift(1),
    }

    # [1] gross vs net, no-skip vs skip-1 — the two diagnostic axes side by side.
    print("\n[1] gross (zero-cost) vs net (realistic); no-skip vs skip-1-day (bid-ask-bounce control)")
    print(f"  {'signal':<18} {'grSharpe':>9} {'grAnn':>8} | {'netSharpe':>10} {'netAnn':>8} "
          f"{'turn':>6} {'beta':>6} {'n':>5}")
    rows = []
    for name, sig in signals.items():
        g = _ls(adj, sig, eval_dates, members, market, 0.0, 0.0)
        nt = _ls(adj, sig, eval_dates, members, market, slip, borrow)
        print(f"  {name:<18} {g.sharpe:>9.2f} {g.ann_return:>8.1%} | {nt.sharpe:>10.2f} "
              f"{nt.ann_return:>8.1%} {nt.avg_turnover:>6.2f} {nt.market_beta:>6.2f} {nt.n_periods:>5}")
        rows.append({"universe": universe, "signal": name, "gross_sharpe": g.sharpe,
                     "gross_ann": g.ann_return, "net_sharpe": nt.sharpe, "net_ann": nt.ann_return,
                     "turnover": nt.avg_turnover, "beta": nt.market_beta, "n": nt.n_periods})

    # [2] per-year + holdout decomposition of the headline rev_1w (gross & net) — is any of it a
    #     recent phenomenon, or a permanent cost-walled illusion?
    head = fl.reversal(adj, 5)
    g = _ls(adj, head, eval_dates, members, market, 0.0, 0.0)
    nt = _ls(adj, head, eval_dates, members, market, slip, borrow)
    sk = _ls(adj, fl.reversal(adj, 5).shift(1), eval_dates, members, market, 0.0, 0.0)
    print("\n[2] rev_1w per-year annualized return (grNoSkip = gross no-skip; grSkip = gross skip-1d;"
          " net = realistic):")
    print(f"  {'year':>6} {'grNoSkip':>9} {'grSkip':>8} {'net':>8}")
    for yr in sorted(set(g.returns.index.year)):
        gy = _ann(g.returns[g.returns.index.year == yr])
        sy = _ann(sk.returns[sk.returns.index.year == yr])
        ny = _ann(nt.returns[nt.returns.index.year == yr])
        print(f"  {yr:>6} {gy:>9.1%} {sy:>8.1%} {ny:>8.1%}")

    # pooled inference on the WEEKLY net return series (is the net edge != 0?) + skip collapse
    def _pool(r, lo, hi):
        s = r[(r.index.year >= lo) & (r.index.year <= hi)]
        if len(s) < 2:
            return None
        t, p = stats.ttest_1samp(s, 0.0)
        return {"n": len(s), "ann": _ann(s), "sharpe": _sharpe(s), "t": float(t), "p": float(p)}

    yr_max = int(g.returns.index.year.max())
    print("\n  pooled inference (weekly returns vs 0):")
    print(f"  {'window/series':<22} {'n':>5} {'ann':>8} {'Sharpe':>7} {'t':>6} {'p':>6}")
    for label, series, lo, hi in [
        ("gross no-skip FULL", g.returns, 2005, yr_max),
        ("gross skip-1d FULL", sk.returns, 2005, yr_max),
        ("NET FULL", nt.returns, 2005, yr_max),
        ("NET 2005-2019", nt.returns, 2005, 2019),
        (f"NET holdout {yr_max}", nt.returns, yr_max, yr_max),
    ]:
        w = _pool(series, lo, hi)
        if w:
            print(f"  {label:<22} {w['n']:>5} {w['ann']:>8.1%} {w['sharpe']:>7.2f} "
                  f"{w['t']:>6.2f} {w['p']:>6.3f}")
    return rows


def main() -> int:
    ensure_dirs()
    rows = []
    for universe in ["large", "small"]:
        rows += run(universe)
        print()
    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "crsp_reversal_summary.parquet")
    print("Reading: short-term reversal's gross profit is the BID-ASK BOUNCE (collapses under "
          "skip-1d) and/or is buried by the weekly TURNOVER cost wall (net Sharpe). If both legs "
          "kill it across years and the 2025 holdout, the family is closed. See docs/reversal_study.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
