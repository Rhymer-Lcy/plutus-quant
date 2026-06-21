"""Single source of truth for on-disk locations. Import these — never hardcode dirs.

Centralizing paths keeps scripts portable and makes the data layout self-documenting.
(Convention carried over from the sibling hermes-quant project.)
"""
from __future__ import annotations

from pathlib import Path

# src/plutus/paths.py -> package dir, then repo root (src-layout: two levels up).
PACKAGE_DIR = Path(__file__).resolve().parent          # src/plutus
REPO_ROOT = PACKAGE_DIR.parent.parent                  # repo root

# --- local data lake (gitignored) ---
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # vendor-native dumps (immutable, as-fetched)
PARQUET_DIR = DATA_DIR / "parquet"    # cleaned + adjusted columnar lake (the working set)
FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"  # SEC EDGAR pulls (company facts, by CIK)
CACHE_DIR = DATA_DIR / "cache"        # fitted models / computed factors (fingerprinted)

# --- code & artefacts ---
EXTERNAL_DIR = REPO_ROOT / "external"   # editable-installed cross-check frameworks (zipline, ...)
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
DOCS_DIR = REPO_ROOT / "docs"

# --- generated OUTPUTS (gitignored; derived, regenerable from data + code) ---
RESULTS_DIR = REPO_ROOT / "results"
SIGNALS_DIR = RESULTS_DIR / "signals"      # prediction / signal panels (parquet)
BACKTESTS_DIR = RESULTS_DIR / "backtests"  # equity curves, trade logs, metrics
FIGURES_DIR = RESULTS_DIR / "figures"      # plots
MODELS_DIR = RESULTS_DIR / "models"        # trained model dumps
PAPER_DIR = RESULTS_DIR / "paper"          # paper-trading ledgers, daily reports, curves

_RUNTIME_DIRS = [DATA_DIR, RAW_DIR, PARQUET_DIR, FUNDAMENTALS_DIR, CACHE_DIR, NOTEBOOKS_DIR,
                 RESULTS_DIR, SIGNALS_DIR, BACKTESTS_DIR, FIGURES_DIR, MODELS_DIR, PAPER_DIR]


def ensure_dirs() -> None:
    """Create the local data/runtime directories if they don't exist."""
    for d in _RUNTIME_DIRS:
        d.mkdir(parents=True, exist_ok=True)
