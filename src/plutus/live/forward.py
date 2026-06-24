"""Free-data EARLY READ on the deployed net-payout book's out-of-sample selection.

The CRSP lake ends 2025-12-31, so the CRSP-driven paper ledger (live.paper) is `awaiting_data`.
This module gives an immediate forward read with FREE data, without waiting for the next paid CRSP
pull. The split keeps it honest:

  - the book is SELECTED from clean CRSP at inception (the decision is survivorship-clean and uses
    the validated net-payout signal + liquidity-screened universe), then
  - it is PRICED FORWARD with yfinance adjusted closes and benchmarked against a small-cap ETF.

So this tests the SELECTION -- do the picked names beat the small-cap index out-of-sample -- NOT
the monthly rebalance. Net-payout is low-turnover, so a held book is a fair approximation between
CRSP refreshes; the definitive test remains the CRSP monthly-rebalanced live.paper ledger, run when
a fresh CRSP pull lands.

Deliberate scope limits (why this is an "early read", not the verdict):
  - SHORT sample (months), so the Sharpe is statistically thin -- read direction, not magnitude.
  - The net-payout signal is NOT recomputed forward: that needs point-in-time SHARE COUNTS, which
    free sources do not provide reliably. Hence selection-only, no forward rebalance.
  - yfinance coverage of mid/small-caps is good for the liquid ADV-screened book (~94% out of the
    box) but imperfect; unresolved tickers (changes / acquisitions) are reported and excluded.
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd

from ..io import atomic_to_parquet, atomic_write_text
from ..paths import PAPER_DIR, PARQUET_DIR, ensure_dirs
from .paper import _ann_sharpe, load_panels
from .strategy import (DEPLOYED, PAPER_INCEPTION, DeployedStrategy, deployed_members,
                       deployed_signal)

TICKER_MAP_FILE = "crsp_smallcap_ticker_map.parquet"
BENCHMARKS = ["VB", "IWM"]   # VB tracks the CRSP US Small Cap index (apt); IWM = Russell 2000 (context)

# Curated 2026 corporate-action resolutions for book names whose 2025-12-31 CRSP ticker no longer
# resolves on yfinance. Each was verified against primary sources (see docs/paper_trading.md); a
# frozen equal-DOLLAR holding's forward return depends only on its forward PRICE PATH, so a
# share-exchange ratio cancels and is noted only for documentation.
#   ("ticker", SUCCESSOR): the holding CONTINUES -> price it forward via SUCCESSOR's adjusted close
#                          (Yahoo migrates the full total-return history to the surviving symbol).
#   ("cash",):             a CASH acquisition near the deal price -> the holding becomes cash,
#                          carried flat for the rest of the window (a < ~1% approximation: the
#                          stock was already pinned at the deal price before close).
CORPORATE_ACTIONS = {
    # CommScope sold its CCS unit to Amphenol ($10.5B, closed 2026-01-12), renamed Vistance Networks
    # (COMM -> VISN, 2026-01-14, 1:1) and paid a special dividend >= $10/sh -- VISN's adjusted close
    # carries that total return, so dropping COMM understated the book.
    "COMM": ("ticker", "VISN"),
    # First Foundation merged into FirstSun (all-stock, 0.16083 FSUN/FFWM, effective 2026-04-01); a
    # fixed-ratio all-stock holding's return tracks the acquirer FSUN.
    "FFWM": ("ticker", "FSUN"),
    # Denny's taken private by a TriArtisan-led consortium for $6.25 cash/share (closed 2026-01-16).
    "DENN": ("cash",),
}


def inception_book(adj: pd.DataFrame, cap: pd.DataFrame, dollar_volume: pd.DataFrame,
                   spec: DeployedStrategy = DEPLOYED, asof=None) -> list[str]:
    """The deployed top-N book (list of str PERMNOs) selected from clean CRSP at `asof` (default:
    the last CRSP bar) -- the same net-payout signal ranked within the liquidity-screened universe
    that live.paper would hold. The selection is survivorship-clean; only the forward pricing uses
    free data."""
    cap = cap.reindex(index=adj.index, columns=adj.columns)
    dollar_volume = dollar_volume.reindex(index=adj.index, columns=adj.columns)
    asof = pd.Timestamp(asof) if asof is not None else adj.index[-1]
    signal = deployed_signal(cap, adj, spec)
    members = deployed_members(cap, dollar_volume, spec)
    row = signal.loc[asof].dropna()
    row = row[row.index.isin(members(asof))]
    return [str(p) for p in row.sort_values(ascending=False).head(spec.n_hold).index]


def ticker_map() -> dict[str, str]:
    """PERMNO (str) -> ticker, from the prebuilt crsp_smallcap_ticker_map lake."""
    tm = pd.read_parquet(PARQUET_DIR / TICKER_MAP_FILE)
    return {str(int(p)): str(t) for p, t in zip(tm["permno"], tm["ticker"]) if pd.notna(t)}


def frozen_book_forward(forward_prices: pd.DataFrame, seed_cash: float,
                        slippage_bps: float = 15.0) -> tuple[pd.Series, list[str]]:
    """Equity curve of an equal-dollar, buy-and-HOLD basket of `forward_prices` (date x ticker,
    adjusted closes), seeded at the first bar. Shares are fixed at inception (fractional allowed),
    bought at the entry close x (1 + slippage), then marked at the adjusted close each day; the
    one-time entry slippage shows up as equity[0] < seed. Returns (equity, names_held).

    Columns with no valid entry price are dropped (a name that could not have been bought at
    inception); interior gaps are forward-filled."""
    fp = forward_prices.sort_index()
    entry = fp.iloc[0]
    valid = [c for c in fp.columns if pd.notna(entry.get(c)) and float(entry.get(c)) > 0]
    if not valid:
        raise ValueError("no names with a valid inception price")
    fp = fp[valid].ffill()
    slip = slippage_bps * 1e-4
    alloc = seed_cash / len(valid)
    shares = alloc / (fp.iloc[0] * (1.0 + slip))           # fixed share count, entry incl. slippage
    invested = float((shares * fp.iloc[0] * (1.0 + slip)).sum())
    cash = seed_cash - invested                            # ~0 (fully invested), residual rounding
    equity = (fp * shares).sum(axis=1) + cash
    equity.name = "equity"
    return equity, valid


def assemble_forward_prices(book_tickers: list[str], prices: pd.DataFrame,
                            succ_prices: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Combine the directly-priced names (`prices`) with corporate-action resolutions: a continuing
    holding uses its successor's price path (from `succ_prices`, e.g. VISN/FSUN); a cash acquisition
    becomes a flat series (carried at the deal value). Returns (full_panel, resolution_log), where
    the log maps each book ticker to how it was priced: 'direct', 'successor:<SYM>', 'cash', or
    'unresolved'. Pure (no I/O) so it is unit-tested offline."""
    idx = prices.index
    out: dict[str, pd.Series] = {}
    log: dict[str, str] = {}
    for t in book_tickers:
        if t in prices.columns:
            out[t] = prices[t]
            log[t] = "direct"
            continue
        act = CORPORATE_ACTIONS.get(t)
        if act and act[0] == "ticker" and act[1] in succ_prices.columns:
            out[t] = succ_prices[act[1]].reindex(idx).ffill().bfill()
            log[t] = f"successor:{act[1]}"
        elif act and act[0] == "cash":
            out[t] = pd.Series(1.0, index=idx)         # held at the deal value -> cash -> flat
            log[t] = "cash"
        else:
            log[t] = "unresolved"
    return pd.DataFrame(out), log


def run_forward(seed_cash: float = 1_000_000.0, *, spec: DeployedStrategy = DEPLOYED,
                inception: str = PAPER_INCEPTION, end: str | None = None,
                persist: bool = True, panels=None, prices: pd.DataFrame | None = None,
                succ_prices: pd.DataFrame | None = None,
                bench_prices: pd.DataFrame | None = None) -> dict:
    """Select the book from CRSP, price it forward with free data from `inception` to `end`
    (default: today) -- resolving 2026 corporate actions (ticker changes / mergers) so no held name
    is silently dropped -- and benchmark vs small-cap ETFs. Pass `prices`/`succ_prices`/`bench_prices`
    to inject panels (tests / offline); otherwise they are pulled from yfinance. Persists a report +
    curve under PAPER_DIR. Returns the report."""
    adj, cap, dv = panels if panels is not None else load_panels()
    select_asof = adj.index[-1]
    permnos = inception_book(adj, cap, dv, spec, select_asof)
    tmap = ticker_map()
    pairs = [(p, tmap[p]) for p in permnos if p in tmap]
    tickers = [t for _, t in pairs]
    end = end or date.today().strftime("%Y-%m-%d")

    if prices is None or succ_prices is None or bench_prices is None:
        from ..data.sources import yfinance_source as yfs
        if prices is None:
            prices = yfs.adjusted_close_panel(tickers, inception, end)
        if succ_prices is None:
            missing = [t for t in tickers if t not in prices.columns]
            succ_syms = sorted({CORPORATE_ACTIONS[t][1] for t in missing
                                if CORPORATE_ACTIONS.get(t, (None,))[0] == "ticker"})
            succ_prices = (yfs.adjusted_close_panel(succ_syms, inception, end)
                           if succ_syms else pd.DataFrame(index=prices.index))
        if bench_prices is None:
            bench_prices = yfs.adjusted_close_panel(BENCHMARKS, inception, end)

    full, resolution = assemble_forward_prices(tickers, prices, succ_prices)
    equity, valid = frozen_book_forward(full, seed_cash, slippage_bps=spec.slippage_bps)
    eq = equity.dropna()
    benches = {}
    for b in BENCHMARKS:
        if b in bench_prices.columns:
            s = bench_prices[b].reindex(eq.index).ffill().dropna()
            if len(s) >= 2:
                benches[b] = {"total_return": float(s.iloc[-1] / s.iloc[0] - 1.0),
                              "ann_sharpe": _ann_sharpe(s)}

    report = {
        "mode": "free_data_forward_early_read",
        "selection_source": "CRSP (survivorship-clean)",
        "selection_asof": pd.Timestamp(select_asof).strftime("%Y-%m-%d"),
        "pricing_source": "yfinance (adjusted close)",
        "inception": eq.index[0].strftime("%Y-%m-%d"),
        "as_of": eq.index[-1].strftime("%Y-%m-%d"),
        "run_date": date.today().strftime("%Y-%m-%d"),
        "seed_cash": float(seed_cash),
        "n_book": len(permnos),
        "n_mapped": len(tickers),
        "n_priced": len(valid),
        "resolution": resolution,
        "n_corporate_actions": sum(1 for v in resolution.values() if v != "direct" and v != "unresolved"),
        "unresolved_tickers": sorted(t for t, v in resolution.items() if v == "unresolved"),
        "n_bars": int(len(eq)),
        "equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / seed_cash - 1.0),
        "max_drawdown": float((eq / eq.cummax() - 1.0).min()),
        "ann_sharpe": _ann_sharpe(eq),
        "benchmarks": benches,
        "names": valid,
    }
    if persist:
        ensure_dirs()
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        atomic_to_parquet(eq.to_frame(), PAPER_DIR / f"forward_curve_{int(seed_cash)}.parquet")
        atomic_write_text(json.dumps(report, ensure_ascii=False, indent=2),
                          PAPER_DIR / f"forward_report_{int(seed_cash)}.json")
    return report
