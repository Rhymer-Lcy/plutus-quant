"""The overnight-return anomaly on survivorship-free large-cap CRSP -- real, and untradeable at retail.

Well documented (Cooper-Cliff-Gulen; Lou-Polk-Skouras "A tug of war", 2019): for US equities almost
the ENTIRE realized return accrues OVERNIGHT (close -> next open), while the intraday session
(open -> close) is flat or negative. That looks like a free lunch: be long only overnight. This tests
whether a retail trader can actually capture it, on the survivorship-free large-cap CRSP lake.

DECOMPOSITION (split- and dividend-immune by construction):
  intraday[t]  = close[t] / open[t] - 1            (same-day open->close; no overnight event can
                                                    contaminate it, so splits/dividends are immune)
  overnight[t] = (1 + DlyRet[t]) / (1 + intraday[t]) - 1   (close[t-1] -> open[t] TOTAL return;
                                                    DlyRet carries the dividends/splits exactly)
so (1+overnight)*(1+intraday) - 1 == DlyRet identically (verified as a sanity residual).

THE TRADEABILITY TEST. Capturing the overnight return means buying at each close and selling at the
next open -- one round trip EVERY day. You buy near the ASK and sell near the BID, so you pay the
full quoted spread per day. We charge the ACTUAL measured CRSP closing half-spread (crsp_halfspread),
not an assumed bps, twice per round trip, and compare the gross overnight return to that spread head
on. If the daily overnight return is smaller than the daily spread you must cross to harvest it, the
anomaly is a liquidity-provision premium that accrues to market makers, not to anyone crossing the
spread -- the same verdict short-term reversal reached (docs/reversal_study.md).

Large-cap is the BEST case (tightest spreads); if it fails here it fails worse in small caps, whose
spreads are far wider. No-look-ahead: each day's overnight return is realized at the next open from a
position taken at the prior close; membership is point-in-time.

    conda activate plutus
    python scripts/build_crsp_open_lake.py     # once (streams open/close/quote)
    python scripts/crsp_overnight_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from plutus.data.sources import crsp_source as crsp
from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from plutus.research.factors import library as fl

PPY = 252  # daily


def _ann(r: pd.Series) -> float:
    return float((1.0 + r).prod() ** (PPY / len(r)) - 1.0) if len(r) else float("nan")


def _sharpe(r: pd.Series) -> float:
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(PPY)) if sd and sd > 0 else float("nan")


def main() -> int:
    ensure_dirs()
    open_raw = pd.read_parquet(PARQUET_DIR / "crsp_open_raw.parquet")
    close_raw = pd.read_parquet(PARQUET_DIR / "crsp_close_raw.parquet")
    ret = pd.read_parquet(PARQUET_DIR / "crsp_dlyret.parquet")
    half = pd.read_parquet(PARQUET_DIR / "crsp_halfspread.parquet")
    spells = pd.read_parquet(PARQUET_DIR / "crsp_members.parquet")
    _m = crsp.members_asof_from_spells(spells)
    members = lambda d: {str(p) for p in _m(d)}              # noqa: E731

    # align all panels on the open/close grid
    idx = open_raw.index.intersection(close_raw.index).intersection(ret.index)
    cols = open_raw.columns.intersection(close_raw.columns).intersection(ret.columns)
    open_raw, close_raw, ret = open_raw.loc[idx, cols], close_raw.loc[idx, cols], ret.loc[idx, cols]
    half = half.reindex(index=idx, columns=cols)

    intraday = (close_raw / open_raw - 1.0).where((open_raw > 0) & (close_raw > 0))
    overnight = (1.0 + ret) / (1.0 + intraday) - 1.0

    # sanity: the decomposition must reproduce DlyRet exactly
    resid = ((1.0 + overnight) * (1.0 + intraday) - 1.0 - ret).abs().stack()
    print("=" * 92)
    print("OVERNIGHT-RETURN ANOMALY -- large-cap S&P 500, survivorship-free CRSP")
    print(f"  {len(cols)} names, {idx.min().date()} -> {idx.max().date()}, {len(idx)} days")
    print(f"  decomposition identity residual: max {resid.max():.2e}, mean {resid.mean():.2e} "
          f"(should be ~0)")
    print(f"  median relative half-spread: {half.stack().median():.4%}  "
          f"(round-trip cost to harvest = ~2x this per day)")
    print("=" * 92)

    # restrict to PIT members, equal-weight across members each day
    on_pit = fl.restrict_to_universe(overnight, members)
    in_pit = fl.restrict_to_universe(intraday, members)
    half_pit = fl.restrict_to_universe(half, members)

    on_d = on_pit.mean(axis=1).dropna()                      # daily EW overnight return
    in_d = in_pit.reindex(on_d.index).mean(axis=1)           # daily EW intraday return
    tot_d = (1.0 + on_d) * (1.0 + in_d) - 1.0                # daily EW total (close-to-close)

    # [1] the anomaly: where does the return live?
    print("\n[1] where the return lives (equal-weight PIT universe, daily):")
    print(f"  {'leg':<14} {'mean/day':>10} {'ann':>9} {'t vs 0':>8}")
    for label, s in [("overnight", on_d), ("intraday", in_d), ("total (B&H)", tot_d)]:
        t, _ = stats.ttest_1samp(s, 0.0)
        print(f"  {label:<14} {s.mean()*1e4:>8.2f}bp {_ann(s):>9.1%} {t:>8.1f}")

    # [2] the spread wall: gross overnight vs the spread you must cross to capture it
    rt_cost = (half_pit + half_pit.shift(1)).reindex(on_d.index)     # buy@close + sell@next-open
    rt_cost_d = rt_cost.where(on_pit.reindex(on_d.index).notna()).mean(axis=1)
    net_d = (on_d - rt_cost_d).dropna()
    on_aligned = on_d.reindex(net_d.index)
    print("\n[2] tradeability -- harvest overnight by buying each close, selling each open:")
    print(f"  gross overnight : {on_aligned.mean()*1e4:>7.2f} bp/day  (Sharpe {_sharpe(on_aligned):>5.2f}, "
          f"ann {_ann(on_aligned):>6.1%})")
    print(f"  round-trip spread: {rt_cost_d.reindex(net_d.index).mean()*1e4:>7.2f} bp/day  "
          f"(the cost to cross, measured from CRSP bid/ask)")
    print(f"  NET overnight    : {net_d.mean()*1e4:>7.2f} bp/day  (Sharpe {_sharpe(net_d):>5.2f}, "
          f"ann {_ann(net_d):>6.1%})")

    # [3] per-year gross vs net (is it ever positive net of the spread?)
    print("\n[3] overnight strategy per-year (annualized):")
    print(f"  {'year':>6} {'gross':>9} {'net':>9} {'spread bp/d':>12}")
    rows = []
    for yr in sorted(set(net_d.index.year)):
        gy = _ann(on_aligned[on_aligned.index.year == yr])
        ny = _ann(net_d[net_d.index.year == yr])
        sp = rt_cost_d.reindex(net_d.index)[net_d.index.year == yr].mean() * 1e4
        print(f"  {yr:>6} {gy:>9.1%} {ny:>9.1%} {sp:>11.2f}bp")
        rows.append({"year": yr, "gross_ann": gy, "net_ann": ny, "spread_bp": sp})
    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "crsp_overnight_peryear.parquet")

    # [4] cost sensitivity: the verdict turns on the true MOC/MOO auction cost. The closing quoted
    #     spread is the CONTINUOUS-session NBBO; auctions usually clear INSIDE it, so x1.0 is an
    #     upper bound. Show where net flips positive.
    print("\n[4] cost sensitivity (full-sample EW; auctions clear inside the quoted spread, so x1.0 "
          "is conservative):")
    print(f"  {'spread mult':>12} {'net bp/day':>11} {'Sharpe':>7} {'ann':>8}")
    for mult in [1.0, 0.5, 0.25, 0.0]:
        net = (on_d - mult * rt_cost_d).dropna()
        print(f"  {('x%.2f' % mult):>12} {net.mean()*1e4:>10.2f} {_sharpe(net):>7.2f} {_ann(net):>8.1%}")

    # [5] liquid subset: restrict to names with a below-median half-spread that day (the genuinely
    #     tradeable book). The EW-universe loss is dominated by wide-spread names.
    med = half_pit.median(axis=1)
    liq = half_pit.le(med, axis=0)
    on_liq = on_pit.where(liq)
    half_liq = half_pit.where(liq)
    on_liq_d = on_liq.mean(axis=1).dropna()
    rt_liq = (half_liq + half_liq.shift(1)).reindex(on_liq_d.index)
    rt_liq_d = rt_liq.where(on_liq.reindex(on_liq_d.index).notna()).mean(axis=1)
    net_liq = (on_liq_d - rt_liq_d).dropna()
    print(f"\n[5] liquid subset (below-median spread): gross {on_liq_d.reindex(net_liq.index).mean()*1e4:.2f}bp "
          f"vs spread {rt_liq_d.reindex(net_liq.index).mean()*1e4:.2f}bp -> NET {net_liq.mean()*1e4:+.2f}bp/day "
          f"(Sharpe {_sharpe(net_liq):+.2f}, ann {_ann(net_liq):+.1%}) at FULL quoted spread")

    # [6] cross-sectional overnight long-short (the actual anomaly form): long high / short low
    #     trailing-20d overnight return, realize the NEXT overnight, dollar-neutral, daily. No
    #     look-ahead: signal uses overnight through t (known by close t); realize overnight[t+1].
    sig = on_pit.rolling(20).mean()
    on_next = on_pit.shift(-1)
    g_ls, n_ls, dts = [], [], []
    for t in on_d.index:
        s = sig.loc[t].dropna()
        if t not in on_next.index:
            continue
        rn = on_next.loc[t]
        s = s[rn.reindex(s.index).notna()]
        if len(s) < 50:
            continue
        k = max(int(len(s) * 0.2), 1)
        hi_, lo_ = s.nlargest(k).index, s.nsmallest(k).index
        g = float(rn[hi_].mean() - rn[lo_].mean())
        c = 2.0 * (float(half_pit.loc[t, hi_].mean()) + float(half_pit.loc[t, lo_].mean()))  # round-trip both legs daily
        g_ls.append(g); n_ls.append(g - c); dts.append(t)
    g_ls, n_ls = pd.Series(g_ls, index=dts), pd.Series(n_ls, index=dts)
    tg, _ = stats.ttest_1samp(g_ls, 0.0)
    print(f"\n[6] cross-sectional overnight long-short (20d signal, dollar-neutral, daily):")
    print(f"  gross {g_ls.mean()*1e4:+.2f}bp/day (Sharpe {_sharpe(g_ls):+.2f}, ann {_ann(g_ls):+.1%}, "
          f"t={tg:+.1f})  ->  NET {n_ls.mean()*1e4:+.2f}bp/day (Sharpe {_sharpe(n_ls):+.2f}, ann {_ann(n_ls):+.1%})")
    print("  (overnight-only requires a round trip of the WHOLE book every day -- the daily spread "
          "wall buries a strong gross anomaly.)")

    print("\nReading: if overnight carries the premium (gross) but the round-trip spread you must "
          "cross every day EXCEEDS the overnight return (net <= 0), the anomaly is a liquidity-\n"
          "provision premium for market makers, not a retail-harvestable edge. See docs/overnight_study.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
