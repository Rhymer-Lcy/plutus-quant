"""Analyst earnings-estimate REVISION signal (#4) — the alpha is on the short side, so a signal
that flags downgrades (and upgrades) is the natural strengthener.

Builds the monthly FY1 consensus EPS per name from IBES (ibes_source.monthly_consensus), forms
percentage revisions (1m & 3m, fiscal-year-rollover masked) + analyst dispersion, links IBES
CUSIP -> CRSP PERMNO, and tests STANDALONE on the survivorship-free mid/small-cap band: rank IC +
market-neutral long-short, net of costs. Caches the revision panels for later GRU integration.

    conda activate plutus
    python scripts/crsp_revision_study.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from plutus.data.sources import crsp_source as crsp
from plutus.data.sources import ibes_source as ibes
from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, RAW_DIR, ensure_dirs
from plutus.research.backtest.long_short import quantile_long_short
from plutus.research.eval.factor_eval import compute_ic
from crsp_study import _month_ends

IBES_DIR = RAW_DIR / "ibes"


def _cusip_to_permno():
    sn = pd.read_csv(RAW_DIR / "crsp" / "stocknames.csv", dtype=str).dropna(subset=["CUSIP", "PERMNO"])
    sn = sn[sn["CUSIP"].str.len() == 8].drop_duplicates("CUSIP", keep="first")
    return {c: int(p) for c, p in zip(sn["CUSIP"], sn["PERMNO"])}


def _rev_to_permno(panel: pd.DataFrame, ticker_to_permno: dict, cols) -> pd.DataFrame:
    ren = {t: ticker_to_permno[t] for t in panel.columns if t in ticker_to_permno}
    p = panel.rename(columns=ren)
    p = p.loc[:, [c for c in p.columns if isinstance(c, str)]]   # keep permno-str cols
    return p.loc[:, ~p.columns.duplicated()].reindex(columns=cols)


def run() -> dict:
    ensure_dirs()
    adj = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_adj_close.parquet")
    cap = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_mktcap.parquet")
    eval_dates = _month_ends(adj.index)
    band = crsp.size_band_members_asof(cap, exclude_top=500, band_size=2500)

    cusip2permno = _cusip_to_permno()
    actuals = ibes.load_actuals(IBES_DIR / "actuals_eps_us_unadj.csv.zip", periodicity="QTR")
    actuals["permno"] = actuals["cusip"].map(cusip2permno)
    actuals = actuals.dropna(subset=["permno"])
    actuals["permno"] = actuals["permno"].astype(int).astype(str)
    band_union = set().union(*[band(d) for d in eval_dates])
    bact = actuals[actuals["permno"].isin(band_union)]
    ticker_to_permno = dict(zip(bact["ticker"], bact["permno"]))
    tickers = set(bact["ticker"])
    print(f"linked {len(tickers)} IBES tickers in the mid/small band; streaming FY1 estimates…")

    fy1 = ibes.stream_estimates(IBES_DIR / "detail_history_eps_us_unadj.csv.zip",
                                tickers=tickers, fpi={"1"}, start="2004-01-01")
    print(f"{len(fy1):,} FY1 estimates; building monthly consensus…")
    cons, disp, fpe = ibes.monthly_consensus(fy1, eval_dates, freshness_days=180)
    cons = _rev_to_permno(cons, ticker_to_permno, adj.columns)
    disp = _rev_to_permno(disp, ticker_to_permno, adj.columns)
    fpe = _rev_to_permno(fpe, ticker_to_permno, adj.columns)
    print(f"consensus coverage: {int(cons.notna().any().sum())} names")

    def revision(k):
        r = (cons - cons.shift(k)) / cons.shift(k).abs()
        rolled = fpe.ne(fpe.shift(k))           # fiscal-year rollover over the window -> mask
        return r.where(~rolled)
    rev1, rev3 = revision(1), revision(3)
    dispersion = -(disp / cons.abs())           # higher dispersion = more uncertain -> negative
    # cache all three panels (for GRU feature integration)
    for nm, pnl in [("rev1", rev1), ("rev3", rev3), ("disp", dispersion)]:
        atomic_to_parquet(pnl, PARQUET_DIR / f"crsp_smallcap_{nm}.parquet")

    print(f"\nstandalone rank IC vs next-month return (mid/small band):")
    sigs = {"rev_1m": rev1, "rev_3m": rev3, "dispersion": dispersion}
    for name, s in sigs.items():
        ic = compute_ic(s, adj, eval_dates, band)
        print(f"  {name:12s} mean IC {ic.mean_ic:+.4f}  t {ic.t_stat:+.2f}  hit {ic.hit_rate:.2f}  n {ic.n_periods}")

    print(f"\nrev_3m market-neutral quintile long-short, net of costs:")
    for label, slp, brw in [("low 5/50", 5.0, 50.0), ("realistic 15/300", 15.0, 300.0)]:
        r = quantile_long_short(adj, rev3, eval_dates, band, quantile=0.2,
                                slippage_bps=slp, borrow_bps_annual=brw)
        print(f"  {label:>16s} annRet {r.ann_return:+.2%} Sharpe {r.sharpe:+.2f} turn {r.avg_turnover:.2f}")
    print("\n[OK] analyst-revision signal, survivorship-free, net of costs. See docs/ml_zoo_study.md.")
    return {"rev3": rev3}


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
