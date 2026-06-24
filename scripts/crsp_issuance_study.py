"""Net share issuance / net-payout factor -- the highest-conviction untested candidate.

The residual-scan (5-lens workflow) ranked net share issuance #1: the most post-publication-robust
anomaly (McLean-Pontiff), LOW turnover (so it escapes the cost wall that killed reversal / overnight
/ pairs / the ML zoo), and buildable NOW from CRSP data only -- so it is fully survivorship-free,
unlike any SEC-fundamental signal (whose ticker-join drops delisted names).

SIGNAL (split- and dividend-immune, no re-stream, no shares needed):
    issuance(H) = log(mktcap_t / mktcap_{t-H}) - log(adj_t / adj_{t-H})
mktcap is split-invariant (price*shares); adj_close is the total return. Their difference is the
growth in market cap NOT explained by total return = net equity raised. Because adj_close is TOTAL
return, this is precisely the NET-PAYOUT form (net issuance minus buybacks AND dividends) -- a
documented return predictor (Boudoukh et al. net payout yield; Daniel-Titman composite issuance).
The factor (higher = attractive) = -issuance: net distributors / buyback names are attractive,
net issuers unattractive.

H = 252 (1y, ~the simple net-issuance horizon) and 1260 (5y, ~Daniel-Titman composite).
Universes: large-cap S&P (2005-2024) and the mid/small-cap band (2005-2025, has a 2025 holdout).

Three reads per (universe, horizon), all net of realistic cost, survivorship-free, no look-ahead:
  - rank IC (does it predict next-month returns?),
  - dollar-neutral quintile long-short (the clean factor alpha; beta-reported),
  - LONG-ONLY top-50 vs the cap-weighted buy-and-hold index (the realizable RETAIL form -- the
    decisive question, judged against the B&H bar, with the short-leg discount made explicit).

    conda activate plutus
    python scripts/crsp_issuance_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from plutus.data.sources import crsp_source as crsp
from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from plutus.research.backtest.long_short import quantile_long_short
from plutus.research.backtest.portfolio import signal_portfolio_backtest
from plutus.research.backtest.regime import cap_weighted_index
from plutus.research.eval.factor_eval import compute_ic
from plutus.research.backtest.frictions import USEquityCosts
from plutus.research.factors import library as fl

N_HOLD = 50
CAPITAL = 1_000_000.0


def _month_ends(dates: pd.DatetimeIndex) -> list:
    s = pd.Series(dates, index=dates)
    return s.groupby(dates.to_period("M")).max().tolist()


def _issuance(mktcap: pd.DataFrame, adj: pd.DataFrame, H: int) -> pd.DataFrame:
    """-(log market-cap growth - log total return) over H trading days. Higher = net payout."""
    lc = np.log(mktcap.where(mktcap > 0))
    la = np.log(adj.where(adj > 0))
    iss = (lc - lc.shift(H)) - (la - la.shift(H))
    return -iss


def _stats(equity: pd.Series, ppy: int = 252) -> dict:
    eq = equity.dropna()
    r = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1)
    dd = float((eq / eq.cummax() - 1).min())
    sh = float(r.mean() / r.std() * np.sqrt(ppy)) if r.std() > 0 else float("nan")
    return {"cagr": cagr, "maxdd": dd, "sharpe": sh, "calmar": cagr / abs(dd) if dd < 0 else float("nan")}


def _load(universe: str):
    if universe == "large":
        adj = pd.read_parquet(PARQUET_DIR / "crsp_adj_close.parquet")
        cap = pd.read_parquet(PARQUET_DIR / "crsp_mktcap.parquet")
        spells = pd.read_parquet(PARQUET_DIR / "crsp_members.parquet")
        _m = crsp.members_asof_from_spells(spells)
        members = lambda d: {str(p) for p in _m(d)}          # noqa: E731
        return adj, cap, members, 5.0, 50.0
    adj = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_adj_close.parquet")
    cap = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_mktcap.parquet")
    members = crsp.size_band_members_asof(cap, exclude_top=500, band_size=2500)
    return adj, cap, members, 15.0, 300.0


def run(universe: str) -> list[dict]:
    adj, cap, members, slip, borrow = _load(universe)
    cap = cap.reindex(index=adj.index, columns=adj.columns)
    eval_dates = _month_ends(adj.index)
    market = cap_weighted_index(adj, cap)
    bh = _stats(market.reindex(adj.index))

    print("=" * 96)
    print(f"NET ISSUANCE / NET-PAYOUT -- {universe}-cap, survivorship-free CRSP, monthly, net of cost")
    print(f"  {adj.shape[1]} names, {adj.index.min().date()} -> {adj.index.max().date()}; "
          f"cost {slip:.0f}bps/side + {borrow:.0f}bps borrow")
    print(f"  buy&hold cap-weighted index bar: CAGR {bh['cagr']:+.1%}  Sharpe {bh['sharpe']:.2f}  "
          f"maxDD {bh['maxdd']:.1%}  Calmar {bh['calmar']:.2f}")
    print("=" * 96)
    print(f"  {'horizon':>8} {'IC':>8} {'IC t':>6} | {'LS Sharpe':>9} {'LS ann':>7} {'beta':>5} {'turn':>5} |"
          f" {'LO CAGR':>8} {'LO Shrp':>7} {'LO Calmar':>9} {'LO-vs-BH':>8}")
    rows = []
    ls_keep = {}
    for H in (252, 1260):
        fac = _issuance(cap, adj, H)
        ic = compute_ic(fac, adj, eval_dates, members)
        ls = quantile_long_short(adj, fac, eval_dates, members, quantile=0.2,
                                 slippage_bps=slip, borrow_bps_annual=borrow, market_index=market)
        lo = signal_portfolio_backtest(adj, fac, CAPITAL, n_hold=N_HOLD,
                                       costs=USEquityCosts(slippage_bps=slip), members_asof=members)
        los = _stats(lo.equity)
        beat = los["sharpe"] - bh["sharpe"]
        print(f"  {('%dd' % H):>8} {ic.mean_ic:>+8.4f} {ic.t_stat:>+6.2f} | {ls.sharpe:>+9.2f} "
              f"{ls.ann_return:>+7.1%} {ls.market_beta:>+5.2f} {ls.avg_turnover:>5.2f} | "
              f"{los['cagr']:>+8.1%} {los['sharpe']:>+7.2f} {los['calmar']:>+9.2f} {beat:>+8.2f}")
        rows.append({"universe": universe, "horizon": H, "ic": ic.mean_ic, "ic_t": ic.t_stat,
                     "ls_sharpe": ls.sharpe, "ls_ann": ls.ann_return, "ls_beta": ls.market_beta,
                     "lo_cagr": los["cagr"], "lo_sharpe": los["sharpe"], "lo_calmar": los["calmar"],
                     "bh_sharpe": bh["sharpe"], "lo_minus_bh_sharpe": beat})
        ls_keep[H] = ls.returns

    # per-year + holdout on the 1y LS (the simple net-issuance horizon)
    r = ls_keep[252].dropna()
    print(f"\n  1y-horizon LS net return by year (is any of it recent / does it survive a holdout?):")
    yrs = sorted(set(r.index.year))
    print("   " + "  ".join(f"{y}:{((1+r[r.index.year==y]).prod()-1):+.0%}" for y in yrs))
    ymax = max(yrs)
    hold = r[r.index.year == ymax]
    if len(hold) >= 2:
        t, p = stats.ttest_1samp(hold, 0.0)
        print(f"   holdout {ymax}: ann {((1+hold).prod()-1):+.1%}  t={t:+.2f} p={p:.3f}")
    return rows


def deep_dive_liquid() -> None:
    """The headline finding: small-cap net-payout LONG-ONLY top-50 survives a LIQUIDITY screen
    and is SIGNAL-SPECIFIC (beats momentum/reversal/low-vol/size/arbitrary on the same liquid
    universe). This is the rigorous gate that separates a real edge from a small-cap-microstructure
    artifact -- ADV screen (you can only trade liquid names), cost stress, and same-universe controls."""
    adj = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_adj_close.parquet")
    cap = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_mktcap.parquet").reindex(index=adj.index, columns=adj.columns)
    dv = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_dollarvol.parquet").reindex(index=adj.index, columns=adj.columns)
    adv = dv.rolling(60, min_periods=20).mean()
    band = crsp.size_band_members_asof(cap, exclude_top=500, band_size=2500)
    netpay = _issuance(cap, adj, 252)

    def liq_members(thr):
        def f(d):
            s = adv.loc[:pd.Timestamp(d)]
            if s.empty:
                return set()
            row = s.iloc[-1]
            return band(d) & {str(c) for c in row.index[row > thr]}
        return f

    def s21(eq):
        return _stats(eq[eq.index.year >= 2021])["sharpe"]

    print("=" * 96)
    print("HEADLINE -- small-cap NET-PAYOUT long-only top-50: liquidity screen + cost stress")
    print("=" * 96)
    print(f"  {'universe / cost':<26} {'avgN':>5} {'CAGR':>8} {'maxDD':>7} {'Sharpe':>7} {'Shrp21+':>8}")
    for label, thr in [("full band", 0.0), ("ADV>$1M/d", 1e6), ("ADV>$5M/d", 5e6), ("ADV>$20M/d", 2e7)]:
        mem = liq_members(thr) if thr > 0 else band
        for slip in (15.0, 50.0):
            r = signal_portfolio_backtest(adj, netpay, CAPITAL, n_hold=50,
                                          costs=USEquityCosts(slippage_bps=slip), members_asof=mem)
            st = _stats(r.equity)
            print(f"  {label + (' @%dbps' % slip):<26} {r.avg_names_held:>5.0f} {st['cagr']:>+8.1%} "
                  f"{st['maxdd']:>7.1%} {st['sharpe']:>7.2f} {s21(r.equity):>8.2f}")

    print("\n  signal-specificity on the SAME $5M-ADV liquid subset (top-50 EW, 25bps) -- is it the "
          "SIGNAL or just the universe?")
    mem = liq_members(5e6)
    const = pd.DataFrame(1.0, index=adj.index, columns=adj.columns)
    controls = {"EW-liquid B&H (all)": (const, 3000), "no-signal top-50": (const, 50),
                "smallest-50 (size)": (-cap, 50), "momentum top-50": (fl.momentum(adj, 252, 21), 50),
                "reversal top-50": (fl.reversal(adj, 21), 50), "low-vol top-50": (fl.low_vol(adj, 120), 50),
                "NET-PAYOUT top-50": (netpay, 50)}
    print(f"  {'book':<24} {'CAGR':>8} {'maxDD':>7} {'Sharpe':>7} {'Shrp21+':>8}")
    for label, (sig, n) in controls.items():
        r = signal_portfolio_backtest(adj, sig, CAPITAL, n_hold=n,
                                      costs=USEquityCosts(slippage_bps=25.0), members_asof=mem)
        st = _stats(r.equity)
        print(f"  {label:<24} {st['cagr']:>+8.1%} {st['maxdd']:>7.1%} {st['sharpe']:>7.2f} {s21(r.equity):>8.2f}")
    print()


def main() -> int:
    ensure_dirs()
    rows = []
    for u in ("large", "small"):
        rows += run(u)
        print()
    deep_dive_liquid()
    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "crsp_issuance_summary.parquet")
    print("Reading: judge LO against the B&H bar (LO-vs-BH = LO Sharpe minus index Sharpe; >0 means the "
          "realizable long-only tilt beats indexing). LS Sharpe with beta~0 = the clean factor alpha. "
          "Apply the short-leg discount: if the edge is only in the LS (short leg) but LO<=BH, it is "
          "not retail-harvestable. See docs/issuance_study.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
