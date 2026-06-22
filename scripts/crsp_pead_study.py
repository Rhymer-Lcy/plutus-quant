"""Post-Earnings-Announcement Drift (PEAD) on survivorship-free CRSP — a NON-classic, event-
driven signal, the honest frontier after classic factors came up empty.

For each name, compute SUE (standardized unexpected earnings) at each SEC filing, hold it as a
point-in-time signal only while fresh (~one quarter), then test: (1) does SUE predict the next
month's return (rank IC)? (2) does a market-neutral long-short on SUE pay after costs? S&P 500
PIT universe, 2005-2024, monthly, survivorship-free CRSP returns.

    conda activate plutus
    python scripts/crsp_pead_study.py        # needs SEC_EDGAR_USER_AGENT
"""
from __future__ import annotations

import argparse

import pandas as pd

from plutus.data.sources import crsp_source as crsp
from plutus.data.sources import sec_edgar as se
from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from plutus.research.backtest.long_short import quantile_long_short
from plutus.research.backtest.regime import cap_weighted_index
from plutus.research.eval.factor_eval import compute_ic
from plutus.research.factors import events

from crsp_study import _month_ends


def build_sue_panel(permno_to_ticker: dict, dates, freshness_days: int) -> pd.DataFrame:
    """SUE PEAD signal panel (date x PERMNO) from SEC net-income filings."""
    cikmap = se.load_ticker_cik_map()
    sue_by_permno: dict[str, pd.DataFrame] = {}
    for permno, ticker in permno_to_ticker.items():
        cik = cikmap.get(str(ticker).upper())
        if cik is None:
            continue
        try:
            facts = se.company_facts(cik)
        except Exception:
            continue
        sue = events.standardized_unexpected_earnings(
            se.discrete_quarters(se.concept_frame(facts, "NetIncomeLoss")))
        if not sue.empty:
            sue_by_permno[permno] = sue
    return events.pit_event_signal(sue_by_permno, dates, freshness_days=freshness_days)


def run(quantile: float = 0.2, slippage_bps: float = 5.0, borrow_bps_annual: float = 50.0,
        freshness_days: int = 63) -> dict:
    ensure_dirs()
    adj = pd.read_parquet(PARQUET_DIR / "crsp_adj_close.parquet")
    cap = pd.read_parquet(PARQUET_DIR / "crsp_mktcap.parquet")
    tmap_df = pd.read_parquet(PARQUET_DIR / "crsp_ticker_map.parquet")
    spells = pd.read_parquet(PARQUET_DIR / "crsp_members.parquet")
    dates = adj.index
    permno_to_ticker = {str(int(p)): t for p, t in zip(tmap_df["permno"], tmap_df["ticker"])}
    _m = crsp.members_asof_from_spells(spells)
    members_asof = lambda d: {str(p) for p in _m(d)}

    print(f"building SUE PEAD signal (freshness {freshness_days}d) from SEC filings…")
    sue = build_sue_panel(permno_to_ticker, dates, freshness_days)
    sue = sue.reindex(index=dates, columns=adj.columns)
    eval_dates = _month_ends(dates)
    cover = sue.reindex(eval_dates).notna().sum(axis=1)
    print(f"  signal coverage: {int(cover.mean())} names/month with a fresh surprise "
          f"(of ~{adj.shape[1]} PERMNOs)")

    market = cap_weighted_index(adj, cap)
    ic = compute_ic(sue, adj, eval_dates, members_asof)
    print(f"\nSUE rank IC vs next-month return:")
    print(f"  mean IC {ic.mean_ic:.4f}  IC-IR {ic.ic_ir:.3f}  t {ic.t_stat:.2f}  "
          f"hit {ic.hit_rate:.2f}  n {ic.n_periods}")

    r = quantile_long_short(adj, sue, eval_dates, members_asof, quantile=quantile,
                            slippage_bps=slippage_bps, borrow_bps_annual=borrow_bps_annual,
                            market_index=market)
    print(f"\nPEAD quintile long-short (monthly, slip {slippage_bps}bps + borrow "
          f"{borrow_bps_annual}bps/yr, survivorship-free):")
    print(f"  ann return {r.ann_return:.2%}  vol {r.ann_vol:.2%}  Sharpe {r.sharpe:.2f}  "
          f"maxDD {r.max_drawdown:.2%}  beta {r.market_beta:.2f}  turn {r.avg_turnover:.2f}  "
          f"n {r.n_periods}")
    atomic_to_parquet(pd.DataFrame([{"signal": "PEAD_SUE", "mean_ic": ic.mean_ic,
                                     "t_stat": ic.t_stat, "ann_return": r.ann_return,
                                     "sharpe": r.sharpe, "max_dd": r.max_drawdown,
                                     "beta": r.market_beta, "turnover": r.avg_turnover}]),
                      BACKTESTS_DIR / "crsp_pead_summary.parquet")
    print("\n[OK] survivorship-free, market-neutral, net of costs. See docs/pead_study.md.")
    return {"ic": ic, "longshort": r}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quantile", type=float, default=0.2)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--borrow-bps-annual", type=float, default=50.0)
    ap.add_argument("--freshness-days", type=int, default=63)
    args = ap.parse_args()
    run(quantile=args.quantile, slippage_bps=args.slippage_bps,
        borrow_bps_annual=args.borrow_bps_annual, freshness_days=args.freshness_days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
