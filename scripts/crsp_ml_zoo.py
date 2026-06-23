"""ML model zoo on the rich feature set — does a larger feature space + stronger models find a
small cost-clearing monthly edge the 6 classic factors couldn't?

Phase 1: rich Alpha158-style features (research.factors.alpha_features) -> walk-forward model
-> OOS monthly signal -> rigorous evaluation (rank IC + market-neutral quintile long-short, net
of costs, low AND realistic). All survivorship-free CRSP; monthly cross-section. Models added
incrementally (lightgbm now; xgboost/catboost/torch next). Compare OOS IC to the (~0) 6-factor
baseline — if the rich features lift it, escalate to DL on the GPU.

    conda activate plutus
    python scripts/build_crsp_smallcap_lake.py    # once (universe + prices)
    python scripts/crsp_ml_zoo.py --model lightgbm --universe smallcap
"""
from __future__ import annotations

import argparse

import pandas as pd

from plutus.data.sources import crsp_source as crsp
from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from plutus.research.backtest.long_short import quantile_long_short
from plutus.research.eval.factor_eval import compute_ic
from plutus.research.factors import alpha_features as af
from plutus.research.model.walk_forward import DEFAULT_PARAMS, build_dataset

from crsp_study import _month_ends


def _make_model(model: str):
    """Model factory (sklearn-like regressors). Tree models handle NaN/scale natively."""
    if model == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**DEFAULT_PARAMS)
    if model == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbosity=0)
    if model == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=400, depth=5, learning_rate=0.03, l2_leaf_reg=3.0,
                                 verbose=0, thread_count=-1, allow_writing_files=False)
    raise ValueError(f"unknown model {model}")


def _walk_forward(data: pd.DataFrame, cols: list, model: str, min_train: int = 24,
                  window: int = 36) -> pd.DataFrame:
    """Generic rolling walk-forward -> OOS signal panel (date x ticker), for any sklearn-like
    model. Mirrors research.model.walk_forward but model-agnostic."""
    udates = sorted(data["date"].unique())
    preds = {}
    for idx, t in enumerate(udates):
        if idx < min_train:
            continue
        tr = data[data["date"].isin(udates[max(0, idx - window):idx])]
        te = data[data["date"] == t]
        if len(tr) < 200 or te.empty:
            continue
        m = _make_model(model)
        m.fit(tr[cols], tr["fwd_ret"])
        preds[t] = pd.Series(m.predict(te[cols]), index=te["ticker"].to_numpy())
    signal = pd.DataFrame(preds).T
    signal.index = pd.to_datetime(signal.index)
    return signal.sort_index()


def _load(universe: str):
    if universe == "smallcap":
        adj = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_adj_close.parquet")
        cap = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_mktcap.parquet")
        members_asof = crsp.size_band_members_asof(cap, exclude_top=500, band_size=2500)
    else:  # large-cap S&P 500
        adj = pd.read_parquet(PARQUET_DIR / "crsp_adj_close.parquet")
        cap = pd.read_parquet(PARQUET_DIR / "crsp_mktcap.parquet")
        spells = pd.read_parquet(PARQUET_DIR / "crsp_members.parquet")
        _m = crsp.members_asof_from_spells(spells)
        members_asof = lambda d: {str(p) for p in _m(d)}
    return adj, cap, members_asof


def run(model: str = "lightgbm", universe: str = "smallcap", rebuild: bool = False) -> dict:
    ensure_dirs()
    adj, cap, members_asof = _load(universe)
    dates = adj.index
    eval_dates = _month_ends(dates)
    print(f"{universe}: {adj.shape[1]} names, {len(eval_dates)} monthly eval dates")

    sig_path = BACKTESTS_DIR / f"crsp_mlzoo_{universe}_{model}_signal.parquet"
    if sig_path.exists() and not rebuild:
        signal = pd.read_parquet(sig_path)
        print(f"loaded cached OOS signal: {signal.shape[0]} months x {signal.shape[1]} names")
    else:
        feats = af.build_features(adj, cap)
        print(f"features: {len(feats)} ({', '.join(list(feats)[:8])}, …)")
        data, cols = build_dataset(feats, adj, eval_dates, members_asof)
        print(f"dataset: {len(data):,} samples x {len(cols)} features; walk-forward {model}…")
        signal = _walk_forward(data, cols, model, min_train=24, window=36)
        signal.to_parquet(sig_path)              # cache BEFORE eval (don't lose the slow fit)
        print(f"OOS signal: {signal.shape[0]} months x {signal.shape[1]} names "
              f"(from {signal.index.min().date()})")

    signal = signal.reindex(eval_dates)          # align to the full grid (early months -> NaN, skipped)
    ic = compute_ic(signal, adj, eval_dates, members_asof)
    print(f"\nOOS signal rank IC: mean {ic.mean_ic:.4f}  IC-IR {ic.ic_ir:.3f}  "
          f"t {ic.t_stat:.2f}  hit {ic.hit_rate:.2f}  n {ic.n_periods}")

    # the limiter is cost-per-turnover, not the signal — sweep CONCENTRATION (extreme quantiles
    # have bigger per-name edge) x time-smoothing x cost, to find any net-positive cell.
    variants = {"raw": signal, "smooth3": signal.rolling(3, min_periods=1).mean()}
    print(f"\nlong-short (monthly, market-neutral), net of costs:")
    print(f"{'signal':8s} {'q':>5s} {'costs':>16s} {'annRet':>8s} {'Sharpe':>7s} {'maxDD':>8s} {'turn':>6s}")
    rows = []
    for vname, sig in variants.items():
        for q in (0.05, 0.10, 0.20):
            for label, slp, brw in [("low 5/50", 5.0, 50.0), ("realistic 15/300", 15.0, 300.0)]:
                r = quantile_long_short(adj, sig, eval_dates, members_asof, quantile=q,
                                        slippage_bps=slp, borrow_bps_annual=brw, market_index=None)
                rows.append({"signal": vname, "q": q, "costs": label, "ann_return": r.ann_return,
                             "sharpe": r.sharpe, "max_dd": r.max_drawdown, "turnover": r.avg_turnover})
                print(f"{vname:8s} {q:5.2f} {label:>16s} {r.ann_return:8.2%} {r.sharpe:7.2f} "
                      f"{r.max_drawdown:8.2%} {r.avg_turnover:6.2f}")
    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / f"crsp_mlzoo_{universe}_{model}.parquet")
    print(f"\n[OK] {model} on the rich feature set, {universe}, survivorship-free, net of costs. "
          "See docs/ml_zoo_study.md.")
    return {"ic": ic, "rows": pd.DataFrame(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--universe", default="smallcap", choices=["smallcap", "largecap"])
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    run(model=args.model, universe=args.universe, rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
