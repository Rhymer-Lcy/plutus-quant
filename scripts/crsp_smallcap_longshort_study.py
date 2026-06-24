"""Do factor premia survive in mid/small caps? Market-neutral long-short on the broad CRSP
universe (build_crsp_smallcap_lake), restricted to a cap-rank BAND (mid/small).

Tests PRICE-BASED factors (reversal, momentum, low-vol, and a momentum+low-vol blend) — no
fundamentals needed, and momentum is the factor most documented to be stronger in small caps.
Survivorship-free; costs are tunable because small caps trade wider and are harder to borrow.

CAVEAT: shorting small caps is genuinely hard/expensive (locate + high borrow); the long-short
SPREAD measures whether the premium EXISTS, but realizable tradability is limited — hence the
realistic high-borrow run. Compare to the large-cap null result (docs/longshort_study.md).

    conda activate plutus
    python scripts/build_crsp_smallcap_lake.py        # once
    python scripts/crsp_smallcap_longshort_study.py --slippage-bps 15 --borrow-bps-annual 300
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

from plutus.research.backtest.metrics import month_ends


def run(quantile: float = 0.2, slippage_bps: float = 15.0, borrow_bps_annual: float = 300.0,
        exclude_top: int = 500, band_size: int = 2500) -> dict:
    ensure_dirs()
    adj = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_adj_close.parquet")
    cap = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_mktcap.parquet")
    dates = adj.index
    members_asof = crsp.size_band_members_asof(cap, exclude_top=exclude_top, band_size=band_size)
    market = cap_weighted_index(adj, cap)
    eval_dates = month_ends(dates)

    mom = fl.momentum(adj, 252, 21)
    lvol = fl.low_vol(adj, 252)
    factors = {
        "reversal_1m": fl.reversal(adj, 21),
        "momentum_12_1": mom,
        "low_vol": lvol,
        "mom+lowvol": fl.blend([fl.restrict_to_universe(mom, members_asof),
                                fl.restrict_to_universe(lvol, members_asof)], [1.0, 1.0]),
    }

    print(f"mid/small universe: cap-rank {exclude_top+1}-{exclude_top+band_size}, "
          f"{adj.shape[1]} names total, {dates.min().date()} -> {dates.max().date()}")
    print(f"quintile long-short, monthly, slip {slippage_bps}bps/side + borrow "
          f"{borrow_bps_annual}bps/yr, survivorship-free\n")
    print(f"{'factor':14s} {'annRet':>8s} {'annVol':>7s} {'Sharpe':>7s} {'maxDD':>8s} "
          f"{'beta':>6s} {'turn':>6s} {'n':>4s}")
    rows = []
    for name, sig in factors.items():
        r = quantile_long_short(adj, sig, eval_dates, members_asof, quantile=quantile,
                                slippage_bps=slippage_bps, borrow_bps_annual=borrow_bps_annual,
                                market_index=market)
        rows.append({"factor": name, "ann_return": r.ann_return, "ann_vol": r.ann_vol,
                     "sharpe": r.sharpe, "max_dd": r.max_drawdown, "beta": r.market_beta,
                     "turnover": r.avg_turnover, "n": r.n_periods})
        print(f"{name:14s} {r.ann_return:8.2%} {r.ann_vol:7.2%} {r.sharpe:7.2f} "
              f"{r.max_drawdown:8.2%} {r.market_beta:6.2f} {r.avg_turnover:6.2f} {r.n_periods:4d}")
    tag = f"{int(slippage_bps)}_{int(borrow_bps_annual)}"
    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / f"crsp_smallcap_ls_{tag}.parquet")
    print("\n[OK] mid/small-cap, survivorship-free, market-neutral, net of costs.")
    return {"summary": pd.DataFrame(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quantile", type=float, default=0.2)
    ap.add_argument("--slippage-bps", type=float, default=15.0)
    ap.add_argument("--borrow-bps-annual", type=float, default=300.0)
    ap.add_argument("--exclude-top", type=int, default=500)
    ap.add_argument("--band-size", type=int, default=2500)
    args = ap.parse_args()
    run(quantile=args.quantile, slippage_bps=args.slippage_bps,
        borrow_bps_annual=args.borrow_bps_annual, exclude_top=args.exclude_top,
        band_size=args.band_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
