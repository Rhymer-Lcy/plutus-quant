"""End-to-end smoke test of the research pipeline on a few real tickers.

Exercises the full stack on live free data — yfinance prices (adjusted + unadjusted) + SEC
EDGAR fundamentals (TTM net income, book equity, shares) -> factors -> rank IC -> frictioned
backtest. Proves the plumbing works; it is NOT a research result (6 names is not a universe).

Requires SEC_EDGAR_USER_AGENT (name + email) in the env / .env.local.

    conda activate plutus
    python scripts/probes/smoke_pipeline.py
"""
from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))   # make scripts/ importable

from crsp_value_reversal_study import run_study   # noqa: E402

TICKERS = ["AAPL", "MSFT", "NVDA", "XOM", "JPM", "PG"]


def main() -> int:
    out = run_study(TICKERS, "2020-01-01", "2023-12-31", capital=100_000.0, n_hold=3)
    res = out["backtest"]
    ok = (
        math.isfinite(res.total_return)
        and res.avg_names_held > 0
        and any(r.n_periods > 0 for r in out["ic"].values())
    )
    print("\nSMOKE:", "OK — full pipeline ran end-to-end" if ok else "FAILED — see metrics above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
