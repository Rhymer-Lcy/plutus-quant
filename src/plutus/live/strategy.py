"""The DEPLOYED strategy spec -- ONE definition, shared by the research study AND paper trading,
so the served signal/universe can never drift from the researched one (train/serve skew is the
dominant silent alpha-killer).

Deployed = NET-PAYOUT (net issuance / buyback), long-only top-50, monthly rebalance, equal
weight, on the liquid mid/small-cap band (cap-rank [500, 3000) intersected with an ADV > $5M/day
liquidity screen). Established in docs/issuance_study.md as the first retail-operable,
signal-specific, liquidity- and cost-robust, recently-strong, survivorship-free edge plutus has
found: long-only top-50 at the $5M-ADV screen earns Sharpe ~1.14 (1.40 since 2021) net of 15bps,
vs the cap-weighted small-cap buy-and-hold bar of 0.61 -- and beats momentum / reversal / low-vol
/ size / arbitrary controls on the IDENTICAL liquid universe (so it is the signal, not the
universe). It is the only candidate that earned a forward paper test rather than "ruled out".

>>> HONEST STATUS: validated IN-SAMPLE (2005-2025), with the magnitude discounted (above the
>>> ~0.5-0.8 published issuance Sharpe; flattered by small-cap concentration and two small-cap
>>> booms). The forward paper record this module produces is the OUT-OF-SAMPLE gate; the spec
>>> below is FROZEN as of this commit and must NOT be refit on post-2025 data.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data.sources import crsp_source as crsp
from ..research.backtest.frictions import USEquityCosts
from ..research.factors import library as fl


@dataclass(frozen=True)
class DeployedStrategy:
    n_hold: int = 50                   # long-only top-50 (concentration-robust: top-30/50/100 all 1.33-1.52)
    lookback: int = 252                # 1y net-payout horizon (the simple net-issuance window)
    exclude_top: int = 500             # size band: drop the largest 500 (mega/large caps)
    band_size: int = 2500              # keep the next 2500 (the mid/small-cap band)
    adv_min: float = 5_000_000.0       # liquidity screen: trade only names with ADV > $5M/day
    adv_window: int = 60               # ADV = trailing 60-trading-day mean dollar volume
    adv_min_periods: int = 20          # ... requiring >= 20 observations
    slippage_bps: float = 15.0         # the validated headline cost (50bps stress still passes at 1.07)
    rebalance_band: int = 0            # turnover buffer OFF (low-turnover signal does not need it)
    weight_asof = None                 # equal weight


DEPLOYED = DeployedStrategy()

# Paper-trading inception: the date the FORWARD (out-of-sample) paper record starts. The research
# study used the FULL 2005-2025 sample (the "2025 holdout" was reported, so 2025 is in-sample), so
# the first genuinely out-of-sample bar is the first trading day of 2026. The seed is invested at
# the first available close >= this date into the then-current top-N, and total_return /
# max_drawdown are measured from there -- NOT the 2005-> backtest (that curve is archived
# separately via live_step(inception=None)). The CRSP lake currently ends 2025-12-31, so until a
# fresh pull lands a 2026 bar the paper account is SEEDED AND AWAITING DATA (live.paper reports
# status="awaiting_data"); it cannot auto-refresh because CRSP is a paid, manual pull. See
# docs/paper_trading.md.
PAPER_INCEPTION = "2026-01-02"

# Capital tiers (USD). The US constraint is the OPPOSITE of A-shares: with fractional shares and
# no 100-share lot there is NO small-account feasibility floor (a 50-name book is fine at $25k),
# so the tiers instead bracket the CAPACITY knee -- the AUM at which MARKET IMPACT in mid/small
# caps (position size vs a name's average daily dollar volume, ADV) starts to erode the edge.
# Impact is NOT modeled by the flat-slippage engine; read the large tiers as "does it scale".
# CAVEAT: the repo's only capacity curve (scripts/crsp_gru_capacity_study.py) was computed on the
# RETIRED GRU market-neutral book, not on net-payout. It indicates roughly where a mid/small-cap
# book's impact knee sits; it is not a net-payout measurement, and no net-payout one has been run.
#   small  [$25k, $100k, $500k]  -- retail: impact negligible even in small caps (the edge's
#                                   home turf is fully accessible here).
#   medium [$2M, $10M]           -- serious individual / small fund: impact modest, working regime.
#   large  [$50M, $250M]         -- fund scale: moving the mid/small basket costs real impact ->
#                                   the capacity ceiling where the (small) edge degrades.
CAPITAL_TIERS: dict[str, list[int]] = {
    "small": [25_000, 100_000, 500_000],
    "medium": [2_000_000, 10_000_000],
    "large": [50_000_000, 250_000_000],
}
ALL_TIERS: list[int] = [v for tier in CAPITAL_TIERS.values() for v in tier]
TIER_LABEL: dict[int, str] = {v: label for label, tier in CAPITAL_TIERS.items() for v in tier}


def deployed_signal(market_cap: pd.DataFrame, adj_close: pd.DataFrame,
                    spec: DeployedStrategy = DEPLOYED) -> pd.DataFrame:
    """The deployed score panel = the net-payout factor (higher = net distributor / buyback =
    attractive). Computed over the FULL price history so the `lookback`-day window is satisfied;
    the point-in-time universe is applied separately (deployed_members) inside the engine, and
    the raw factor is a ranking score (no cross-sectional standardize/blend), so no
    restrict_to_universe is needed before it. Single source of truth: same fl.net_payout the
    research study uses."""
    return fl.net_payout(market_cap, adj_close, spec.lookback)


def deployed_members(market_cap: pd.DataFrame, dollar_volume: pd.DataFrame,
                     spec: DeployedStrategy = DEPLOYED):
    """The deployed point-in-time universe = the cap-rank band [exclude_top, exclude_top+band_size)
    INTERSECTED with the ADV liquidity screen (only names whose trailing-`adv_window` mean dollar
    volume, as of the signal date, exceeds `adv_min`). Returns members_asof(date)->set[ticker].
    This IS part of the strategy (the headline result is the liquidity-screened book), so it lives
    in the deployed spec, mirroring scripts/crsp_issuance_study.py's deep_dive_liquid exactly."""
    adv = dollar_volume.rolling(spec.adv_window, min_periods=spec.adv_min_periods).mean()
    band = crsp.size_band_members_asof(market_cap, exclude_top=spec.exclude_top,
                                       band_size=spec.band_size)

    def members_asof(date) -> set:
        s = adv.loc[:pd.Timestamp(date)]
        if s.empty:
            return set()
        row = s.iloc[-1]
        return band(date) & {str(c) for c in row.index[row > spec.adv_min]}

    return members_asof


def deployed_costs(spec: DeployedStrategy = DEPLOYED) -> USEquityCosts:
    """The deployed friction model: a $0-commission retail broker with the validated headline
    slippage. (The 50bps stress in docs/issuance_study.md still clears the B&H bar, so this is
    not knife-edge.)"""
    return USEquityCosts(slippage_bps=spec.slippage_bps)
