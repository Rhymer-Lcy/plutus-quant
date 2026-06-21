"""Online execution: end-of-day paper trading (idempotent ledger), the deployed strategy
spec, and broker/data feeds. The SAME research engine drives decisions here, so paper
trading cannot drift from backtest (train/serve skew is the dominant silent alpha-killer)."""
