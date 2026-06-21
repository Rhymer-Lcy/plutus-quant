"""US-equity transaction cost model.

US retail frictions are MUCH lighter than A-share (no stamp tax, no 100-share lot, no daily
price limit, no T+1 holding lock). What remains, for a daily-rebalance strategy:

  - commission:  $0 at most retail brokers (Alpaca, Schwab, Fidelity, Robinhood, IBKR Lite).
                 IBKR Pro is per-share (~$0.0035/sh, $0.35 min, 1% cap) -- model via
                 commission_per_share + min_commission if you use it.
  - SEC Section 31 fee:  SELL side only, charged per dollar of proceeds. Set by the SEC and
                 RE-ADJUSTED periodically (sometimes to $0 when prior collections overshoot),
                 so this rate MUST be checked against the current SEC fee-rate advisory.
  - FINRA TAF (Trading Activity Fee):  SELL side only, per share, with a per-trade cap.
  - slippage:    modeled as bps on the execution price, both sides.
  - lot size:    1 share (US has no 100-share lot; fractional shares exist at many brokers,
                 in which case lot is effectively continuous -- left at 1 here, the
                 broadly-supported whole-share case).

>>> The exact numeric rates below are tracked in docs/MARKET_FACTS.md with primary-source
>>> citations and as-of dates. Re-verify before trusting net P&L; these are the heart of the
>>> friction gate.

The fee methods take BOTH share count and dollar turnover, because US fees are mixed:
per-share (TAF, IBKR commission) and per-dollar (SEC fee, $0-broker commission is just 0).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class USEquityCosts:
    # commission: defaults model a $0-commission retail broker.
    commission_per_share: float = 0.0     # e.g. IBKR Pro ~0.0035
    commission_per_trade: float = 0.0     # flat per-ticket, if any
    min_commission: float = 0.0           # e.g. IBKR Pro 0.35

    # regulatory fees (SELL side only). VALUES + CITATIONS + as-of dates in docs/MARKET_FACTS.md.
    sec_fee_rate: float = 0.0000206       # SEC Section 31: $20.60 per $1M, eff. 2026-04-04
                                          #   (was $0.00/M Oct 2025–Apr 3 2026; rate is reset
                                          #   ~annually, sometimes to $0 — re-check each FY)
    taf_per_share: float = 0.000195       # FINRA TAF, per covered-equity share, eff. 2026-01-01
    taf_cap: float = 9.79                 # FINRA TAF per-trade cap, eff. 2026-01-01

    slippage_bps: float = 5.0             # slippage, one side, basis points
    lot_size: int = 1                     # 1-share granularity; no 100-share lot

    def _commission(self, shares: float, turnover: float) -> float:
        if shares <= 0 or turnover <= 0:
            return 0.0
        c = self.commission_per_trade + shares * self.commission_per_share
        c = max(c, self.min_commission)
        return min(c, turnover)           # IBKR-style: commission never exceeds trade value

    def buy_fees(self, shares: float, turnover: float) -> float:
        """Cash fees on a BUY (slippage handled via exec price). SEC fee and TAF are
        sell-side only, so a buy pays commission only."""
        return self._commission(shares, turnover)

    def sell_fees(self, shares: float, turnover: float) -> float:
        """Cash fees on a SELL: commission + SEC Section 31 fee + FINRA TAF (capped)."""
        sec = turnover * self.sec_fee_rate
        taf = min(shares * self.taf_per_share, self.taf_cap)
        return self._commission(shares, turnover) + sec + taf

    @property
    def slip(self) -> float:
        return self.slippage_bps * 1e-4


# All-zero costs — used to compute the "gross" (frictionless) curve for comparison.
ZERO_COSTS = USEquityCosts(
    commission_per_share=0.0, commission_per_trade=0.0, min_commission=0.0,
    sec_fee_rate=0.0, taf_per_share=0.0, taf_cap=0.0, slippage_bps=0.0,
)
