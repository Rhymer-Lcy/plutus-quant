"""Market-neutral long-short factor study, on survivorship-free CRSP data.

The long-only studies showed factors can't beat the index (a long book is ~90% market beta).
This strips the beta: for each signal, go long the top quintile and short the bottom quintile,
dollar-neutral, monthly, after turnover + borrow costs (research.backtest.long_short). The
question it answers honestly: does any factor (or the ML combiner) have a tradeable
market-neutral SPREAD — net Sharpe and return with ~0 market beta?

    conda activate plutus
    python scripts/build_crsp_lake.py            # once
    python scripts/crsp_longshort_study.py       # needs SEC_EDGAR_USER_AGENT
"""
from __future__ import annotations

import argparse

import pandas as pd

from plutus.data.sources import crsp_source as crsp
from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from plutus.research.backtest.long_short import quantile_long_short
from plutus.research.backtest.regime import cap_weighted_index
from plutus.research.factors import library as fl
from plutus.research.model.walk_forward import build_dataset, walk_forward_predict

from build_fundamentals import build_panels
from crsp_study import _month_ends, _ticker_panel_to_permno


def run(quantile: float = 0.2, slippage_bps: float = 5.0, borrow_bps_annual: float = 50.0) -> dict:
    ensure_dirs()
    adj = pd.read_parquet(PARQUET_DIR / "crsp_adj_close.parquet")
    cap = pd.read_parquet(PARQUET_DIR / "crsp_mktcap.parquet")
    tmap_df = pd.read_parquet(PARQUET_DIR / "crsp_ticker_map.parquet")
    spells = pd.read_parquet(PARQUET_DIR / "crsp_members.parquet")
    dates = adj.index
    permno_to_ticker = {str(int(p)): t for p, t in zip(tmap_df["permno"], tmap_df["ticker"])}
    _m = crsp.members_asof_from_spells(spells)
    members_asof = lambda d: {str(p) for p in _m(d)}

    funds = build_panels(sorted(set(permno_to_ticker.values())), dates)
    ni = _ticker_panel_to_permno(funds["net_income_ttm"], permno_to_ticker).reindex(index=dates, columns=adj.columns)
    book = _ticker_panel_to_permno(funds["book_equity"], permno_to_ticker).reindex(index=dates, columns=adj.columns)
    cap = cap.reindex(index=dates, columns=adj.columns)

    factors = {
        "earnings_yield": fl.earnings_yield(ni, cap),
        "book_yield": fl.book_yield(book, cap),
        "roe": fl.roe(ni, book),
        "reversal_1m": fl.reversal(adj, 21),
        "momentum_12_1": fl.momentum(adj, 252, 21),
        "low_vol": fl.low_vol(adj, 252),
    }
    eval_dates = _month_ends(dates)
    market = cap_weighted_index(adj, cap)

    candidate = fl.blend([fl.restrict_to_universe(factors["earnings_yield"], members_asof),
                          fl.restrict_to_universe(factors["reversal_1m"], members_asof)], [5.0, 1.0])
    print(f"walk-forward ML combiner over {len(eval_dates)} months…")
    data, cols = build_dataset(factors, adj, eval_dates, members_asof)
    ml = walk_forward_predict(data, cols, min_train=24, window=36)

    signals = {**factors, "candidate(val+rev)": candidate, "ML combiner": ml}
    print(f"\nquintile long-short ({int(quantile*100)}/{int(quantile*100)}), monthly, "
          f"slip {slippage_bps}bps/side + borrow {borrow_bps_annual}bps/yr, survivorship-free CRSP\n")
    print(f"{'signal':18s} {'annRet':>8s} {'annVol':>7s} {'Sharpe':>7s} {'maxDD':>8s} "
          f"{'beta':>6s} {'turn':>6s} {'n':>4s}")
    rows = []
    for name, sig in signals.items():
        r = quantile_long_short(adj, sig, eval_dates, members_asof, quantile=quantile,
                                slippage_bps=slippage_bps, borrow_bps_annual=borrow_bps_annual,
                                market_index=market)
        rows.append({"signal": name, "ann_return": r.ann_return, "ann_vol": r.ann_vol,
                     "sharpe": r.sharpe, "max_dd": r.max_drawdown, "beta": r.market_beta,
                     "turnover": r.avg_turnover, "n": r.n_periods})
        print(f"{name:18s} {r.ann_return:8.2%} {r.ann_vol:7.2%} {r.sharpe:7.2f} "
              f"{r.max_drawdown:8.2%} {r.market_beta:6.2f} {r.avg_turnover:6.2f} {r.n_periods:4d}")
    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "crsp_longshort_summary.parquet")
    print("\n[OK] survivorship-free, market-neutral, net of costs. Sharpe with beta~0 = real "
          "factor edge (if any). See docs/longshort_study.md.")
    return {"summary": pd.DataFrame(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quantile", type=float, default=0.2)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--borrow-bps-annual", type=float, default=50.0)
    args = ap.parse_args()
    run(quantile=args.quantile, slippage_bps=args.slippage_bps,
        borrow_bps_annual=args.borrow_bps_annual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
