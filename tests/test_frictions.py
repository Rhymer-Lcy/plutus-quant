"""US cost model: $0-commission default, sell-side SEC fee + FINRA TAF (capped), zero-cost
sanity, IBKR-Pro-style per-share commission floor."""
import math

from plutus.research.backtest.frictions import ZERO_COSTS, USEquityCosts


def test_default_broker_is_zero_commission():
    c = USEquityCosts()
    # a buy at a $0-commission broker pays nothing (SEC fee + TAF are sell-side only)
    assert c.buy_fees(shares=100, turnover=10_000.0) == 0.0


def test_sell_charges_sec_fee_and_taf_only_over_buy():
    c = USEquityCosts()
    shares, turnover = 100, 10_000.0
    sec = turnover * c.sec_fee_rate
    taf = min(shares * c.taf_per_share, c.taf_cap)
    # sell minus buy: commission cancels (both 0), leaving exactly SEC fee + TAF
    assert math.isclose(c.sell_fees(shares, turnover) - c.buy_fees(shares, turnover), sec + taf)


def test_taf_is_capped_per_trade():
    c = USEquityCosts()
    huge = int(c.taf_cap / c.taf_per_share) + 10_000   # well past the cap
    sell = c.sell_fees(shares=huge, turnover=1_000_000.0)
    sec = 1_000_000.0 * c.sec_fee_rate
    assert math.isclose(sell - sec, c.taf_cap)          # TAF component pinned at the cap


def test_ibkr_pro_commission_hits_minimum_floor():
    c = USEquityCosts(commission_per_share=0.0035, min_commission=0.35,
                      sec_fee_rate=0.0, taf_per_share=0.0, taf_cap=0.0)
    # 10 shares * 0.0035 = 0.035 -> floored to the 0.35 minimum
    assert math.isclose(c.buy_fees(shares=10, turnover=2_000.0), 0.35)
    # 1000 shares * 0.0035 = 3.50 > 0.35, so no floor
    assert math.isclose(c.buy_fees(shares=1000, turnover=200_000.0), 3.50)


def test_commission_never_exceeds_trade_value():
    c = USEquityCosts(commission_per_share=0.0, min_commission=5.0,
                      sec_fee_rate=0.0, taf_per_share=0.0, taf_cap=0.0)
    assert c.buy_fees(shares=1, turnover=2.0) == 2.0    # 5.0 min capped at the $2 trade value


def test_zero_costs_are_zero():
    assert ZERO_COSTS.buy_fees(1000, 1e6) == 0.0
    assert ZERO_COSTS.sell_fees(1000, 1e6) == 0.0
    assert ZERO_COSTS.slip == 0.0


def test_zero_turnover_no_fee():
    c = USEquityCosts()
    assert c.buy_fees(0, 0.0) == 0.0
    assert c.sell_fees(0, 0.0) == 0.0


def test_slip_bps_conversion():
    assert math.isclose(USEquityCosts(slippage_bps=5.0).slip, 5e-4)


def test_default_lot_is_one_share():
    assert USEquityCosts().lot_size == 1     # US: no 100-share lot
