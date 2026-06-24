"""EOD paper trading: run the DEPLOYED net-payout strategy forward on daily closes, recording
every fill in an idempotent ledger (live.ledger).

Architecture (monthly EOD): the research backtest engine IS the strategy brain. `replay()` runs
`signal_portfolio_backtest(..., collect_trades=True)` and folds its per-fill trade log, day by day,
into a LedgerState valued with the SAME `valuation_panel`. So paper P&L is reconstructed from the
seed by the exact research code -- no re-implementation, hence no train/serve drift (the dominant
silent alpha-killer). The ledger is seeded at PAPER_INCEPTION (invested fully that day) and tracked
forward, so `total_return` / `max_drawdown` are the FORWARD (out-of-sample) paper record -- NOT the
2005-> backtest (which `paper_account(inception=None)` reproduces, archived separately).

DATA CONSTRAINT (the key difference from the sibling hermes-quant, which auto-refreshes from free
BaoStock): plutus runs on CRSP, a PAID, manual pull -- there is no free daily feed. So this module
does NOT auto-refresh; it replays from whatever is on disk. The lake is advanced by landing a fresh
CRSP pull (the build scripts), after which re-running paper_live extends the ledger. Until a bar at
or after PAPER_INCEPTION exists, the account is SEEDED AND AWAITING DATA (status="awaiting_data") --
the honest state, never a silent fall-back to the in-sample backtest. See docs/paper_trading.md.
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd

from ..io import atomic_to_parquet, atomic_write_text
from ..paths import BACKTESTS_DIR, PARQUET_DIR, PAPER_DIR, ensure_dirs
from ..research.backtest.portfolio import signal_portfolio_backtest, valuation_panel
from ..research.backtest.regime import cap_weighted_index
from .ledger import LedgerState, fold_day
from .strategy import (DEPLOYED, PAPER_INCEPTION, DeployedStrategy, deployed_costs,
                       deployed_members, deployed_signal)

ADJ_FILE = "crsp_smallcap_adj_close.parquet"
CAP_FILE = "crsp_smallcap_mktcap.parquet"
DV_FILE = "crsp_smallcap_dollarvol.parquet"


def replay(price: pd.DataFrame, signal: pd.DataFrame, seed_cash: float, *,
           n_hold: int = 50, costs=None, members_asof=None, weight_asof=None,
           rebalance_band: int = 0, initial_rebalance: bool = False) -> tuple[LedgerState, object]:
    """Reconstruct the strategy's P&L as an idempotent ledger. Returns (ledger, result):
    `result` is the underlying PortfolioResult (for parity checks / stats); `ledger` is built by
    folding `result.trades` day by day from `seed_cash`, valued with `valuation_panel`.

    `initial_rebalance` invests the seed on the FIRST bar (paper inception); default off keeps the
    research backtest and the parity tests on the natural month-end schedule. The two equity series
    MUST agree (live.paper's only job is to record the engine's decisions, not re-decide) --
    scripts/paper_dryrun.py asserts this as the anti-skew gate."""
    result = signal_portfolio_backtest(
        price, signal, seed_cash, n_hold=n_hold, costs=costs, members_asof=members_asof,
        weight_asof=weight_asof, rebalance_band=rebalance_band, collect_trades=True,
        initial_rebalance=initial_rebalance,
    )
    valuation, _, _ = valuation_panel(price)

    fills_by_day: dict[pd.Timestamp, list[dict]] = {}
    for t in result.trades:
        fills_by_day.setdefault(t["date"], []).append(t)

    state = LedgerState(seed_cash=seed_cash)
    for d in valuation.index:
        marks = valuation.loc[d].to_dict()
        state = fold_day(state, d.strftime("%Y-%m-%d"), fills_by_day.get(d, []), marks)
    return state, result


def ledger_equity(state: LedgerState) -> pd.Series:
    """The ledger's equity curve as a date-indexed Series (for plotting / comparison)."""
    idx = pd.to_datetime([d for d, _ in state.equity_curve])
    return pd.Series([v for _, v in state.equity_curve], index=idx, name="equity")


def _ann_sharpe(equity: pd.Series, min_bars: int = 21, ppy: int = 252):
    """Annualized Sharpe of an equity curve, or None until enough bars accrue (a Sharpe on a
    handful of forward days is noise, not signal)."""
    eq = equity.dropna()
    if len(eq) < min_bars:
        return None
    r = eq.pct_change().dropna()
    if len(r) < 2 or r.std() == 0:
        return None
    return float(r.mean() / r.std() * (ppy ** 0.5))


def paper_account(adj: pd.DataFrame, cap: pd.DataFrame, dollar_volume: pd.DataFrame,
                  seed_cash: float, *, spec: DeployedStrategy = DEPLOYED, costs=None,
                  inception: str | None = PAPER_INCEPTION, as_of: str | None = None
                  ) -> tuple[LedgerState | None, object | None, dict]:
    """Pure core (no disk I/O): given the CRSP panels, return (ledger, result, report).

    `inception` seeds the FORWARD paper record at the first bar >= that date (invested fully there);
    `inception=None` reproduces the full-history backtest curve. `as_of` cuts the data (the lake's
    last bar) before anything is computed, so a holiday/stale re-run idempotently reproduces the
    prior bar. In AWAITING-DATA mode (inception set, but no bar at/after it on disk) ledger/result
    are None and the report carries status="awaiting_data" -- never a silent backtest fall-back.
    The report's bh_* fields benchmark the strategy against the cap-weighted buy-and-hold over the
    SAME forward window (the out-of-sample version of the docs/issuance_study.md comparison)."""
    adj = adj.sort_index()
    if as_of is not None:
        adj = adj.loc[adj.index <= pd.Timestamp(as_of)]
    if adj.empty:
        raise ValueError("no price data on/before as_of")
    lake_last = adj.index[-1]
    run_dt = date.today()
    incept = pd.Timestamp(inception) if inception is not None else None

    # AWAITING-DATA short-circuit FIRST (needs only adj.index): skip the heavy signal/ADV build
    # when the lake has no bar at/after inception yet -- and never fall back to the in-sample backtest.
    if incept is not None and not (adj.index >= incept).any():
        report = {
            "status": "awaiting_data",
            "as_of": lake_last.strftime("%Y-%m-%d"),
            "run_date": run_dt.strftime("%Y-%m-%d"),
            "lake_lag_days": (run_dt - lake_last.date()).days,
            "inception": incept.strftime("%Y-%m-%d"),
            "seed_cash": float(seed_cash),
            "n_bars": 0,
            "equity": float(seed_cash),
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "ann_sharpe": None,
            "bh_total_return": None,
            "bh_ann_sharpe": None,
            "n_positions": 0,
            "avg_names_held": 0.0,
            "positions": {},
            "today_trades": [],
            "n_trades_total": 0,
            "note": (f"seeded; lake ends {lake_last.date()} < inception {incept.date()} -- land a "
                     "fresh CRSP pull covering post-inception bars, then re-run to extend the record."),
        }
        return None, None, report

    # ACTIVE path: build the signal/universe/benchmark (reindex applies the as_of row-cut to cap/dv).
    cap = cap.reindex(index=adj.index, columns=adj.columns)
    dollar_volume = dollar_volume.reindex(index=adj.index, columns=adj.columns)
    costs = costs or deployed_costs(spec)
    signal = deployed_signal(cap, adj, spec)                # full-history signal (lookbacks satisfied)
    members = deployed_members(cap, dollar_volume, spec)
    market = cap_weighted_index(adj, cap)                   # benchmark on the (as_of-cut) full history
    initial = False
    if incept is not None:
        adj = adj.loc[adj.index >= incept]
        signal = signal.loc[signal.index >= incept]
        initial = True
    ledger, res = replay(adj, signal, seed_cash, n_hold=spec.n_hold, costs=costs,
                         members_asof=members, weight_asof=spec.weight_asof,
                         rebalance_band=spec.rebalance_band, initial_rebalance=initial)

    fwd_mkt = market.loc[market.index >= incept] if incept is not None else market
    bh_tot = float(fwd_mkt.iloc[-1] / fwd_mkt.iloc[0] - 1.0) if len(fwd_mkt) >= 2 else None
    today = adj.index[-1]
    today_fills = [{**t, "date": t["date"].strftime("%Y-%m-%d")} for t in res.trades
                   if t["date"] == today]
    report = {
        "status": "active" if incept is not None else "backtest",
        "as_of": today.strftime("%Y-%m-%d"),               # last data bar (the strategy's clock)
        "run_date": run_dt.strftime("%Y-%m-%d"),            # wall-clock date this was computed
        "lake_lag_days": (run_dt - today.date()).days,      # run_date - as_of (calendar days, freshness)
        "inception": adj.index[0].strftime("%Y-%m-%d") if incept is not None else None,
        "seed_cash": float(seed_cash),
        "n_bars": int(len(res.equity)),
        "equity": float(res.equity.iloc[-1]),
        "total_return": float(res.total_return),
        "max_drawdown": float(res.max_drawdown),
        "ann_sharpe": _ann_sharpe(res.equity),              # None until >= 21 forward bars
        "bh_total_return": bh_tot,
        "bh_ann_sharpe": _ann_sharpe(fwd_mkt),
        "n_positions": int(sum(1 for s in ledger.positions.values() if s > 0)),
        "avg_names_held": float(res.avg_names_held),
        "positions": dict(sorted(ledger.positions.items())),
        "today_trades": today_fills,
        "n_trades_total": len(res.trades),
    }
    return ledger, res, report


def load_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load (adj_close, market_cap, dollar_volume) for the deployed small/mid-cap universe. The
    panels are ~0.5GB each, so a multi-tier driver should load ONCE and pass `panels=` to live_step."""
    return (pd.read_parquet(PARQUET_DIR / ADJ_FILE),
            pd.read_parquet(PARQUET_DIR / CAP_FILE),
            pd.read_parquet(PARQUET_DIR / DV_FILE))


def live_step(seed_cash: float, *, spec: DeployedStrategy = DEPLOYED,
              inception: str | None = PAPER_INCEPTION, as_of: str | None = None,
              persist: bool = True, out_dir=None, panels=None) -> dict:
    """Run `paper_account` on the CRSP small/mid-cap lake and (idempotently) persist the equity
    curve, full trade log, and a JSON report. Paper mode (inception set) writes `paper_*` under
    PAPER_DIR; backtest mode (inception=None) writes `backtest_*` under BACKTESTS_DIR. Pass
    `panels=(adj, cap, dv)` to reuse already-loaded panels (a multi-tier driver should). Re-run
    safe (recompute-from-seed); returns today's report."""
    adj, cap, dv = panels if panels is not None else load_panels()
    ledger, res, report = paper_account(adj, cap, dv, seed_cash, spec=spec,
                                        inception=inception, as_of=as_of)
    if persist:
        ensure_dirs()
        out = out_dir or (BACKTESTS_DIR if inception is None else PAPER_DIR)
        out.mkdir(parents=True, exist_ok=True)
        prefix = "backtest_" if inception is None else "paper_"
        tag = f"{int(seed_cash)}"
        atomic_write_text(json.dumps(report, ensure_ascii=False, indent=2),
                          out / f"{prefix}report_{tag}.json")
        if ledger is not None:
            atomic_to_parquet(ledger_equity(ledger).to_frame(), out / f"{prefix}curve_{tag}.parquet")
            atomic_to_parquet(pd.DataFrame(res.trades), out / f"{prefix}trades_{tag}.parquet")
    return report
