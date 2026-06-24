"""Volatility-managed exposure overlay (#6, Moreira-Muir) -- risk-shaping, not new alpha.

Scale next-period market exposure by target/realized volatility, CAPPED at 1.0 (a retail account
cannot lever up), vs plain buy-and-hold and vs the binary 200-day trend filter. The honest question:
for an un-levered retail book, does vol-management improve risk-adjusted return (Sharpe) or only the
drawdown/tail (Calmar)? On the survivorship-free cap-weighted CRSP index.

    conda activate plutus
    python scripts/crsp_vol_overlay_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from plutus.research.backtest.regime import cap_weighted_index


def _stats(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    years = max(len(r) / 252.0, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / years) - 1)
    vol = float(r.std() * np.sqrt(252))
    sh = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")
    dd = float((eq / eq.cummax() - 1).min())
    return {"cagr": cagr, "vol": vol, "sharpe": sh, "maxdd": dd, "calmar": cagr / abs(dd) if dd < 0 else float("nan")}


def run(universe: str, adj_f: str, cap_f: str) -> list[dict]:
    adj = pd.read_parquet(PARQUET_DIR / adj_f)
    cap = pd.read_parquet(PARQUET_DIR / cap_f).reindex(index=adj.index, columns=adj.columns)
    idx = cap_weighted_index(adj, cap)
    r = idx.pct_change(fill_method=None)

    # vol-managed: exposure = min(target / realized_vol, 1), applied to NEXT day (no look-ahead)
    rv = r.rolling(21).std()
    target = float(r.std())                                  # full-sample daily vol -> mean exposure ~1
    expo = (target / rv).clip(upper=1.0).shift(1)            # decided at t, applied t+1
    vm = (expo * r)

    # binary 200d trend filter (the already-built regime overlay), also applied next day
    ma = idx.rolling(200).mean()
    trend = (idx >= ma).astype(float).shift(1)
    tr = (trend * r)

    print(f"\n{universe}: cap-weighted index, {adj.index.min().date()}..{adj.index.max().date()}")
    print(f"  {'book':<22} {'CAGR':>8} {'vol':>7} {'Sharpe':>7} {'maxDD':>8} {'Calmar':>7}")
    rows = []
    for label, series in [("buy & hold", r), ("vol-managed (cap 1.0)", vm), ("200d trend filter", tr)]:
        s = _stats(series)
        print(f"  {label:<22} {s['cagr']:>+8.1%} {s['vol']:>7.1%} {s['sharpe']:>7.2f} "
              f"{s['maxdd']:>8.1%} {s['calmar']:>7.2f}")
        rows.append({"universe": universe, "book": label, **s})
    return rows


def main() -> int:
    ensure_dirs()
    rows = []
    rows += run("large-cap", "crsp_adj_close.parquet", "crsp_mktcap.parquet")
    rows += run("small-cap", "crsp_smallcap_adj_close.parquet", "crsp_smallcap_mktcap.parquet")
    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / "crsp_vol_overlay.parquet")
    print("\nReading: an un-levered (cap 1.0) vol overlay can only lower vol/drawdown -- judge it on "
          "Calmar/maxDD, not Sharpe. If Sharpe ~ B&H but Calmar/maxDD improve, it is risk-shaping, "
          "not new alpha. See docs/overnight_study.md / the residual scan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
