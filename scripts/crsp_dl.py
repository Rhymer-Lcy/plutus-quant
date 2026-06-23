"""Temporal deep learning (GRU) on the rich features — the one qualitatively different bet.

The tree models found a real but FAST (high-turnover) signal that costs eat. A sequence model
that sees each name's last K months of features might instead find a more PERSISTENT (slower,
lower-turnover) signal — the only way the cost wall plausibly gets crossed. Walk-forward, GPU,
evaluated with the same survivorship-free + cost-aware harness as the tree zoo.

    conda activate plutus
    python scripts/crsp_dl.py --universe smallcap --seq 12
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from plutus.data.sources import crsp_source as crsp
from plutus.io import atomic_to_parquet
from plutus.paths import BACKTESTS_DIR, PARQUET_DIR, ensure_dirs
from plutus.research.backtest.long_short import quantile_long_short
from plutus.research.eval.factor_eval import compute_ic
from plutus.research.factors import alpha_features as af
from plutus.research.factors import library as fl

from crsp_study import _month_ends


class GRUNet(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):                       # x: [B, K, F]
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _build_sequences(adj, cap, members_asof, eval_dates, seq_len, volume=None, dollar_vol=None,
                     extra=None):
    """Return X [n, seq_len, F], y [n] (xs-demeaned fwd ret), di [n] (date idx), tk [n] (ticker),
    and feature names. Standardized features over the PIT universe; sequences are seq_len
    consecutive monthly feature vectors with no gaps; target is the next-month return.
    `extra`: optional {name -> monthly date×ticker panel} of additional features (e.g. analyst
    revisions) merged into the price/volume set."""
    raw = af.build_features(adj, cap, volume=volume, dollar_vol=dollar_vol)
    if extra:
        raw.update(extra)
    cols = list(raw)
    aligned = {k: fl.restrict_to_universe(v.reindex(index=eval_dates, columns=adj.columns), members_asof)
               for k, v in raw.items()}
    std = {k: fl.standardize(v) for k, v in aligned.items()}
    F = np.stack([std[k].to_numpy(dtype=np.float32) for k in cols], axis=-1)   # [T, N, F]
    close = adj.reindex(eval_dates)
    fwd = (close.shift(-1) / close - 1.0)                                       # next-month return
    fwd = fwd.sub(fwd.mean(axis=1), axis=0).to_numpy(dtype=np.float32)          # xs-demeaned
    tickers = list(adj.columns)
    T, N = len(eval_dates), len(tickers)

    X, y, di, tk = [], [], [], []
    for j in range(N):
        for i in range(seq_len - 1, T - 1):           # need seq history; predict i -> i+1
            seq = F[i - seq_len + 1:i + 1, j, :]
            target = fwd[i, j]
            if np.isnan(seq).any() or np.isnan(target):
                continue
            X.append(seq); y.append(target); di.append(i); tk.append(tickers[j])
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32),
            np.asarray(di), np.asarray(tk), cols, eval_dates)


def _train(model, Xtr, ytr, device, epochs=15, batch=2048, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.MSELoss()
    Xt = torch.from_numpy(Xtr).to(device)
    yt = torch.from_numpy(ytr).to(device)
    n = len(Xt)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()


def run(universe: str = "smallcap", seq_len: int = 12, min_train: int = 48, refresh: int = 12,
        hidden: int = 64, epochs: int = 15, ensemble: int = 1) -> dict:
    ensure_dirs()
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vol = dvol = None
    extra: dict = {}
    if universe == "smallcap":
        adj = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_adj_close.parquet")
        cap = pd.read_parquet(PARQUET_DIR / "crsp_smallcap_mktcap.parquet")
        members_asof = crsp.size_band_members_asof(cap, exclude_top=500, band_size=2500)
        vp = PARQUET_DIR / "crsp_smallcap_volume.parquet"
        dp = PARQUET_DIR / "crsp_smallcap_dollarvol.parquet"
        if vp.exists() and dp.exists():               # volume/liquidity features (if lake has them)
            vol = pd.read_parquet(vp).reindex(columns=adj.columns)
            dvol = pd.read_parquet(dp).reindex(columns=adj.columns)
        for nm in ("rev1", "rev3", "disp"):           # analyst-revision features (if cached)
            rp = PARQUET_DIR / f"crsp_smallcap_{nm}.parquet"
            if rp.exists():
                extra[nm] = pd.read_parquet(rp).reindex(columns=adj.columns).fillna(0.0)
    else:
        adj = pd.read_parquet(PARQUET_DIR / "crsp_adj_close.parquet")
        cap = pd.read_parquet(PARQUET_DIR / "crsp_mktcap.parquet")
        spells = pd.read_parquet(PARQUET_DIR / "crsp_members.parquet")
        _m = crsp.members_asof_from_spells(spells)
        members_asof = lambda d: {str(p) for p in _m(d)}
    eval_dates = _month_ends(adj.index)
    print(f"{universe}: {adj.shape[1]} names, {len(eval_dates)} months, device={device}, "
          f"volume={'yes' if vol is not None else 'no'}, extra={sorted(extra)}")

    X, y, di, tk, cols, edates = _build_sequences(adj, cap, members_asof, eval_dates, seq_len,
                                                  volume=vol, dollar_vol=dvol, extra=extra or None)
    print(f"sequences: {len(X):,} (seq_len {seq_len}, {len(cols)} features); GRU walk-forward…")

    preds: dict = {}
    models: list = []
    for t in range(min_train, len(edates) - 1):
        if (t - min_train) % refresh == 0:            # periodic retrain on all history < t
            tr = di < t
            if tr.sum() < 5000:
                continue
            models = []                                # ENSEMBLE: K seeds, averaged (DL is noisy)
            for s in range(ensemble):
                torch.manual_seed(s)
                m = GRUNet(len(cols), hidden=hidden).to(device)
                _train(m, X[tr], y[tr], device, epochs=epochs)
                m.eval()
                models.append(m)
        if not models:
            continue
        te = di == t
        if not te.any():
            continue
        Xte = torch.from_numpy(X[te]).to(device)
        with torch.no_grad():
            p = np.mean([mdl(Xte).cpu().numpy() for mdl in models], axis=0)
        preds[edates[t]] = pd.Series(p, index=tk[te])
    signal = pd.DataFrame(preds).T.sort_index()
    signal.index = pd.to_datetime(signal.index)
    sig_path = BACKTESTS_DIR / f"crsp_dl_{universe}_gru_signal.parquet"
    signal.to_parquet(sig_path)
    print(f"OOS signal: {signal.shape[0]} months x {signal.shape[1]} names "
          f"(from {signal.index.min().date()})")

    signal = signal.reindex(eval_dates)
    ic = compute_ic(signal, adj, eval_dates, members_asof)
    print(f"\nGRU OOS rank IC: mean {ic.mean_ic:.4f}  IC-IR {ic.ic_ir:.3f}  t {ic.t_stat:.2f}  "
          f"hit {ic.hit_rate:.2f}  n {ic.n_periods}")
    print(f"\nlong-short (market-neutral), net of costs:")
    print(f"{'signal':8s} {'q':>5s} {'costs':>16s} {'annRet':>8s} {'Sharpe':>7s} {'turn':>6s}")
    variants = {"raw": signal, "smooth3": signal.rolling(3, min_periods=1).mean()}
    rows = []
    for vname, sig in variants.items():
        for q in (0.1, 0.2):
            for label, slp, brw in [("low 5/50", 5.0, 50.0), ("realistic 15/300", 15.0, 300.0)]:
                r = quantile_long_short(adj, sig, eval_dates, members_asof, quantile=q,
                                        slippage_bps=slp, borrow_bps_annual=brw)
                rows.append({"signal": vname, "q": q, "costs": label, "ann_return": r.ann_return,
                             "sharpe": r.sharpe, "turnover": r.avg_turnover})
                print(f"{vname:8s} {q:5.2f} {label:>16s} {r.ann_return:8.2%} {r.sharpe:7.2f} "
                      f"{r.avg_turnover:6.2f}")
    atomic_to_parquet(pd.DataFrame(rows), BACKTESTS_DIR / f"crsp_dl_{universe}_gru.parquet")
    print("\n[OK] temporal GRU, survivorship-free, net of costs. See docs/ml_zoo_study.md.")
    return {"ic": ic, "rows": pd.DataFrame(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", default="smallcap", choices=["smallcap", "largecap"])
    ap.add_argument("--seq", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--ensemble", type=int, default=1)
    args = ap.parse_args()
    run(universe=args.universe, seq_len=args.seq, hidden=args.hidden, epochs=args.epochs,
        ensemble=args.ensemble)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
