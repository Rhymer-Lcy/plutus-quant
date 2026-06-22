"""Dollar-neutral quantile long-short: it captures the factor spread, is market-neutral
(beta ~ 0), and turnover costs reduce the return."""
import numpy as np
import pandas as pd

from plutus.research.backtest.long_short import quantile_long_short

NAMES = [f"N{i}" for i in range(10)]
EVAL = list(pd.bdate_range("2015-01-30", periods=25, freq="BME"))   # 24 monthly periods
MKT = [0.03, -0.02, 0.01, -0.04, 0.05, -0.01, 0.02, -0.03] * 3      # varying market shocks


def _panels(flip: bool = False):
    """Returns = market shock (common) + a signal-driven effect, so a long-short spread cancels
    the market. `flip` alternates the ranking each period (forces full turnover)."""
    price = pd.DataFrame(index=EVAL, columns=NAMES, dtype=float)
    signal = pd.DataFrame(index=EVAL, columns=NAMES, dtype=float)
    price.iloc[0] = 100.0
    for p in range(len(EVAL) - 1):
        rank = {n: (i if (not flip or p % 2 == 0) else 9 - i) for i, n in enumerate(NAMES)}
        signal.iloc[p] = pd.Series(rank)
        for n in NAMES:
            ret = MKT[p] + 0.01 * (rank[n] - 4.5)      # top-ranked names go up, market cancels in LS
            price.loc[EVAL[p + 1], n] = price.loc[EVAL[p], n] * (1 + ret)
    market = pd.Series(100.0 * np.cumprod([1.0] + MKT[:len(EVAL) - 1]), index=EVAL)
    return price, signal, market


def test_captures_spread_and_is_market_neutral():
    price, signal, market = _panels()
    res = quantile_long_short(price, signal, EVAL, quantile=0.2, slippage_bps=0.0,
                              borrow_bps_annual=0.0, market_index=market)
    assert res.ann_return > 0.10            # a real long-short spread (~8%/mo gross before annualizing)
    assert res.sharpe > 1.0                 # steady spread -> high Sharpe
    assert abs(res.market_beta) < 0.10      # dollar-neutral -> market beta ~ 0
    assert res.avg_turnover < 0.5           # constant ranking -> little turnover after setup


def test_costs_reduce_return():
    price, signal, market = _panels(flip=True)         # flipping ranking -> full turnover each period
    free = quantile_long_short(price, signal, EVAL, slippage_bps=0.0, borrow_bps_annual=0.0)
    costly = quantile_long_short(price, signal, EVAL, slippage_bps=200.0, borrow_bps_annual=0.0)
    assert costly.ann_return < free.ann_return          # turnover cost drags net return
    assert costly.avg_turnover > 1.5                     # ranking flips -> heavy two-sided turnover


def test_borrow_fee_drags_return():
    price, signal, market = _panels()
    no_borrow = quantile_long_short(price, signal, EVAL, slippage_bps=0.0, borrow_bps_annual=0.0)
    borrow = quantile_long_short(price, signal, EVAL, slippage_bps=0.0, borrow_bps_annual=500.0)
    assert borrow.ann_return < no_borrow.ann_return
