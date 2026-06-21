"""The candidate strategy spec -- ONE definition, shared by research demos AND paper trading,
so the served signal can never drift from the researched one (train/serve skew is the
dominant silent alpha-killer).

>>> STATUS: this is a STARTING POINT, NOT validated alpha. Unlike the sibling hermes-quant
>>> (whose value+reversal blend passed an A-share friction gate), plutus has run NO research
>>> yet. The factor choice below (value + a light reversal tilt) is a sensible prior to TEST
>>> with factor_eval + signal_portfolio_backtest, not a result to deploy. Do not size real
>>> capital on it until the backtest and paper record hold up.

Equal weight, top-10, monthly rebalance -- deliberately the simplest baseline to beat.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..research.factors import library as fl


@dataclass(frozen=True)
class CandidateStrategy:
    n_hold: int = 10
    value_weight: float = 5.0          # value : reversal = 5 : 1 (a prior, to be tuned)
    reversal_weight: float = 1.0
    reversal_lookback: int = 21        # ~1 month of trading days
    rebalance_band: int = 0            # turnover buffer OFF until shown to help
    weight_asof = None                 # equal weight (inverse-vol must earn its place)


CANDIDATE = CandidateStrategy()

# Paper-trading capital tiers (USD), grouped small / medium / large. NOTE: unlike A-shares,
# US has NO 100-share lot and (at many brokers) supports fractional shares, so there is no
# hard small-account feasibility floor -- a 10-name book is feasible even at a few thousand
# dollars. Tiers here are a SCALE reference (does behaviour change with size?), and assume
# negligible market impact (flat slippage, no size-dependent impact term -- NOT modeled).
CAPITAL_TIERS: dict[str, list[int]] = {
    "small": [5_000, 10_000, 25_000],     # (the old $25k pattern-day-trader minimum was
                                          # ELIMINATED 2026-06-04; irrelevant to a monthly
                                          # book anyway -- kept only as a round reference point)
    "medium": [100_000, 500_000],
    "large": [1_000_000, 5_000_000],
}
ALL_TIERS: list[int] = [v for tier in CAPITAL_TIERS.values() for v in tier]
TIER_LABEL: dict[int, str] = {v: label for label, tier in CAPITAL_TIERS.items() for v in tier}


def candidate_signal(close: pd.DataFrame, net_income_ttm: pd.DataFrame,
                     market_cap: pd.DataFrame, members_asof,
                     spec: CandidateStrategy = CANDIDATE) -> pd.DataFrame:
    """The candidate score panel: value (E/P) + a light short-term-reversal tilt, each
    restricted to the PIT members BEFORE the cross-sectional blend (else the survivorship
    union leaks into the z-scores). Higher score = more attractive.

    `net_income_ttm` / `market_cap` are point-in-time panels (filing-date aligned) built from
    plutus.data.sources.sec_edgar joined to price * shares outstanding."""
    ep = fl.restrict_to_universe(fl.earnings_yield(net_income_ttm, market_cap), members_asof)
    rev = fl.restrict_to_universe(fl.reversal(close, spec.reversal_lookback), members_asof)
    return fl.blend([ep, rev], [spec.value_weight, spec.reversal_weight])
