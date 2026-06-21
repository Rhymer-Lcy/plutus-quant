"""Idempotent ledger: folding fills moves cash/positions correctly, and replaying the same
day sequence from the seed reproduces identical state (the immutable-rebuild contract)."""
import copy
import math

from plutus.live.ledger import LedgerState, fold_day


def test_fold_day_applies_fills_and_marks():
    s = LedgerState(seed_cash=100_000.0)
    fills = [{"ticker": "AAPL", "shares": 100, "price": 150.0, "fee": 1.0}]
    s1 = fold_day(s, "2025-01-02", fills, marks={"AAPL": 155.0})
    assert s1.positions == {"AAPL": 100}
    assert math.isclose(s1.cash, 100_000.0 - 100 * 150.0 - 1.0)
    # equity = cash + 100 * 155 mark
    assert math.isclose(s1.equity_curve[-1][1], s1.cash + 100 * 155.0)


def test_sell_credits_cash_and_closes_position():
    s = LedgerState(seed_cash=100_000.0)
    s = fold_day(s, "2025-01-02", [{"ticker": "MSFT", "shares": 10, "price": 400.0, "fee": 0.5}],
                 marks={"MSFT": 400.0})
    s = fold_day(s, "2025-01-03", [{"ticker": "MSFT", "shares": -10, "price": 410.0, "fee": 0.5}],
                 marks={})
    assert "MSFT" not in s.positions                 # flat after the sell
    assert math.isclose(s.cash, 100_000.0 - 10 * 400.0 - 0.5 + 10 * 410.0 - 0.5)


def test_replay_from_seed_is_idempotent():
    seed = LedgerState(seed_cash=50_000.0)
    days = [
        ("2025-01-02", [{"ticker": "AAA", "shares": 50, "price": 20.0, "fee": 0.2}], {"AAA": 21.0}),
        ("2025-01-03", [{"ticker": "BBB", "shares": 30, "price": 10.0, "fee": 0.1}], {"AAA": 22.0, "BBB": 10.0}),
    ]
    a = copy.deepcopy(seed)
    for d, f, m in days:
        a = fold_day(a, d, f, m)
    b = copy.deepcopy(seed)
    for d, f, m in days:
        b = fold_day(b, d, f, m)
    assert a.positions == b.positions
    assert math.isclose(a.cash, b.cash)
    assert a.equity_curve == b.equity_curve         # full replay reproduces the curve
