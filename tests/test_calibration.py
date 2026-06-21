"""Probability calibration: proper scoring rules and reliability/ECE."""
import math

import numpy as np

from plutus.research.eval.calibration import brier, log_loss, reliability


def test_perfect_confident_predictions_score_zero():
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([0, 1])
    assert math.isclose(log_loss(p, y), 0.0, abs_tol=1e-9)
    assert math.isclose(brier(p, y), 0.0, abs_tol=1e-12)


def test_brier_of_coin_flip():
    p = np.array([[0.5, 0.5], [0.5, 0.5]])
    y = np.array([0, 1])
    # each row contributes (0.5^2 + 0.5^2) = 0.5
    assert math.isclose(brier(p, y), 0.5)


def test_reliability_returns_ece_and_bins():
    rng = np.random.default_rng(0)
    p_up = rng.uniform(0, 1, size=2000)
    p = np.column_stack([1 - p_up, p_up])
    y = (rng.uniform(0, 1, size=2000) < p_up).astype(int)   # well-calibrated by construction
    ece, rows = reliability(p, y, n_bins=10)
    assert 0.0 <= ece < 0.1            # calibrated -> small ECE
    assert len(rows) > 0
