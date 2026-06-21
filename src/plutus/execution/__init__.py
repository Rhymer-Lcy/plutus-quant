"""Live broker execution (Alpaca) — DEFERRED stub, unused.

The staged pipeline is: backtest (offline, friction gate) -> realtime paper trading
(simulated, end-of-day ledger) -> live (small real capital, gateway swapped). Live execution
is the LAST stage and is intentionally not implemented until a paper-trading record holds up.

When built, this package will wrap Alpaca's trading API (alpaca-py, the `broker` optional
dependency) behind the SAME order interface the paper ledger uses, so the strategy object is
unchanged and only the gateway swaps. Keeping decisions in the shared research engine (see
live.strategy / research.backtest.portfolio) is what prevents train/serve skew.
"""
