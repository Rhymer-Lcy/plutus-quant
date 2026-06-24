"""Paper-trading: the anti-skew parity guarantee (the ledger reproduces the research engine's
equity exactly), the inception seeding flag, the deployed spec/universe locks, and the honest
awaiting-data vs active paths. The plain fold_day mechanics are covered in test_ledger.py."""
import numpy as np
import pandas as pd

from plutus.live.forward import frozen_book_forward
from plutus.live.paper import ledger_equity, paper_account, replay
from plutus.live.strategy import (DEPLOYED, DeployedStrategy, deployed_members,
                                  deployed_signal)
from plutus.research.factors import library as fl


def test_replay_reproduces_engine_equity_exactly():
    # 3 names over ~3 months, monotone prices, fixed ranking a>b>c so top-2 is stable. The ledger
    # equity (replay -> fold_day) must match the engine's PortfolioResult.equity bar for bar.
    dates = pd.bdate_range("2020-01-01", "2020-03-31")
    n = len(dates)
    price = pd.DataFrame({"a": np.linspace(10.0, 13.0, n), "b": np.linspace(20.0, 24.0, n),
                          "c": np.linspace(30.0, 33.0, n)}, index=dates)
    signal = pd.DataFrame({"a": 3.0, "b": 2.0, "c": 1.0}, index=dates)

    ledger, res = replay(price, signal, 1_000_000.0, n_hold=2)
    led = ledger_equity(ledger).reindex(res.equity.index)
    assert float(np.abs(led.values - res.equity.values).max()) < 1e-6
    assert len(ledger.folded_days) == n          # every trading day folded
    assert res.trades                            # at least one rebalance happened


def test_replay_idempotent_rebuild():
    dates = pd.bdate_range("2020-01-01", "2020-02-28")
    price = pd.DataFrame({"a": np.linspace(10, 12, len(dates)),
                          "b": np.linspace(20, 19, len(dates))}, index=dates)
    signal = pd.DataFrame({"a": 2.0, "b": 1.0}, index=dates)
    e1 = ledger_equity(replay(price, signal, 100_000.0, n_hold=1)[0])
    e2 = ledger_equity(replay(price, signal, 100_000.0, n_hold=1)[0])
    assert e1.equals(e2)


def test_initial_rebalance_invests_on_the_first_bar():
    # With initial_rebalance the seed is invested on day 0 (paper inception); the default (research)
    # schedule stays idle until the first month-end. Same panel, the two must differ on the first bar.
    dates = pd.bdate_range("2020-01-01", "2020-02-28")
    price = pd.DataFrame({"a": np.linspace(10, 12, len(dates)),
                          "b": np.linspace(20, 22, len(dates))}, index=dates)
    signal = pd.DataFrame({"a": 2.0, "b": 1.0}, index=dates)
    _, res_default = replay(price, signal, 1_000_000.0, n_hold=1)
    _, res_incept = replay(price, signal, 1_000_000.0, n_hold=1, initial_rebalance=True)
    assert not [t for t in res_default.trades if t["date"] == dates[0]]   # idle on day 0
    assert [t for t in res_incept.trades if t["date"] == dates[0]]        # invested on day 0


def test_deployed_signal_is_the_net_payout_factor():
    # Anti-skew lock: the single-source deployed_signal must equal fl.net_payout at the spec
    # lookback. If someone edits the spec, this forces the docs/research to move with it.
    dates = pd.bdate_range("2020-01-01", periods=300)
    n = len(dates)
    cap = pd.DataFrame({"a": np.linspace(100, 200, n), "b": np.linspace(50, 40, n)}, index=dates)
    adj = pd.DataFrame({"a": np.linspace(10, 12, n), "b": np.linspace(20, 25, n)}, index=dates)
    pd.testing.assert_frame_equal(deployed_signal(cap, adj), fl.net_payout(cap, adj, DEPLOYED.lookback))
    assert (DEPLOYED.n_hold, DEPLOYED.rebalance_band, DEPLOYED.weight_asof) == (50, 0, None)


def test_deployed_members_is_band_intersect_adv():
    # Universe lock: members = cap-rank band INTERSECTED with the ADV liquidity screen.
    dates = pd.bdate_range("2025-01-01", periods=4)
    cap = pd.DataFrame({"A": 100.0, "B": 80.0, "C": 60.0, "D": 40.0}, index=dates)   # A largest
    dv = pd.DataFrame({"A": 20e6, "B": 10e6, "C": 3e6, "D": 2e6}, index=dates)       # C, D below $5M
    spec = DeployedStrategy(exclude_top=1, band_size=2, adv_min=5e6, adv_window=2, adv_min_periods=1)
    members = deployed_members(cap, dv, spec)
    # band drops the largest (A), keeps the next 2 = {B, C}; ADV>$5M = {A, B}; intersection = {B}.
    assert members(dates[-1]) == {"B"}


def test_paper_account_awaiting_data_when_inception_is_in_the_future():
    # The lake ends before inception (CRSP cannot auto-refresh): seeded, no positions, and NEVER a
    # silent fall-back to the in-sample backtest. Data ends ~2025-02, well before inception.
    dates = pd.bdate_range("2024-01-01", periods=300)
    n = len(dates)
    cap = pd.DataFrame({"a": np.linspace(100, 200, n), "b": np.linspace(80, 90, n)}, index=dates)
    adj = pd.DataFrame({"a": np.linspace(10, 12, n), "b": np.linspace(20, 22, n)}, index=dates)
    dv = pd.DataFrame({"a": 1e7, "b": 1e7}, index=dates)
    ledger, res, report = paper_account(adj, cap, dv, 100_000.0, inception="2026-01-02")
    assert ledger is None and res is None
    assert report["status"] == "awaiting_data"
    assert report["equity"] == 100_000.0 and report["n_positions"] == 0 and report["positions"] == {}
    assert report["n_bars"] == 0


def test_paper_account_active_seeds_at_inception_and_benchmarks():
    # Active path: the seed is invested at inception, the forward record accrues, and the report
    # carries a same-window buy-and-hold benchmark.
    dates = pd.bdate_range("2024-01-01", periods=120)
    n = len(dates)
    # constant caps (descending: a largest) -> a is excluded by the band; b,c,d,e form the universe.
    cap = pd.DataFrame({k: v for k, v in zip("abcde", (500.0, 400.0, 300.0, 200.0, 100.0))},
                       index=dates)
    rates = {"a": 0.001, "b": 0.004, "c": 0.003, "d": 0.002, "e": 0.0005}
    adj = pd.DataFrame({k: 10.0 * (1 + r) ** np.arange(n) for k, r in rates.items()}, index=dates)
    dv = pd.DataFrame({k: 1e6 for k in "abcde"}, index=dates)
    spec = DeployedStrategy(lookback=60, n_hold=2, exclude_top=1, band_size=3,
                            adv_min=0.0, adv_window=5, adv_min_periods=1)
    incept = dates[90].strftime("%Y-%m-%d")
    ledger, res, report = paper_account(adj, cap, dv, 1_000_000.0, spec=spec, inception=incept)
    assert report["status"] == "active"
    assert report["inception"] == dates[90].strftime("%Y-%m-%d")
    assert report["n_bars"] == n - 90            # forward window only (seeded at inception)
    assert report["n_positions"] >= 1 and report["n_trades_total"] > 0
    assert report["bh_total_return"] is not None  # >= 2 forward bars -> benchmark defined
    # ledger equity must still equal the engine equity on the forward window (parity holds here too).
    led = ledger_equity(ledger).reindex(res.equity.index)
    assert float(np.abs(led.values - res.equity.values).max()) < 1e-6


def test_frozen_book_forward_buy_and_hold():
    # Free-data early read core: equal-dollar buy-and-hold. With 0 slippage equity[0] == seed; a
    # name that doubles while the other is flat lifts the 50/50 book to 1.5x. A NaN-entry column
    # is dropped (could not be bought at inception).
    dates = pd.bdate_range("2026-01-02", periods=4)
    fp = pd.DataFrame({"X": [100.0, 120.0, 200.0, 200.0],
                       "Y": [50.0, 50.0, 50.0, 50.0],
                       "Z": [float("nan"), 10.0, 10.0, 10.0]}, index=dates)
    eq, valid = frozen_book_forward(fp, 1_000_000.0, slippage_bps=0.0)
    assert valid == ["X", "Y"]                       # Z dropped (no inception price)
    assert abs(eq.iloc[0] - 1_000_000.0) < 1e-6      # no slippage -> seeded at par
    assert abs(eq.iloc[-1] - 1_500_000.0) < 1e-6     # X doubled (+500k), Y flat

    # slippage drags equity[0] below the seed (one-time entry cost).
    eq2, _ = frozen_book_forward(fp, 1_000_000.0, slippage_bps=50.0)
    assert eq2.iloc[0] < 1_000_000.0
