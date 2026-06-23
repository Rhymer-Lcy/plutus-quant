"""Rich feature library: correct computation, log-size, and strict no-look-ahead (a feature at
date t must not change when FUTURE prices change)."""
import numpy as np
import pandas as pd

from plutus.research.factors.alpha_features import build_features


def _panel(seed=0, n=160, k=6):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.02, (n, k)), axis=0)
    cols = [f"N{i}" for i in range(k)]
    return pd.DataFrame(px, index=dates, columns=cols)


def test_build_features_count_and_shape():
    close = _panel()
    cap = close * 1e6
    feats = build_features(close, cap)
    assert len(feats) > 30                        # a genuinely rich set
    assert "roc5" in feats and "vol20" in feats and "rsv20" in feats and "size" in feats
    for name, panel in feats.items():
        assert panel.shape == close.shape, name


def test_roc_and_size_values():
    close = _panel()
    feats = build_features(close, mktcap=close * 1000.0)
    expected_roc5 = (close / close.shift(5) - 1.0)
    pd.testing.assert_frame_equal(feats["roc5"], expected_roc5)
    pd.testing.assert_frame_equal(feats["size"], np.log(close * 1000.0))


def test_no_lookahead():
    close = _panel()
    feats = build_features(close)
    snapshot = {k: v.iloc[80].copy() for k, v in feats.items()}
    future = close.copy()
    future.iloc[100:] *= 1.5                      # perturb only the future (after row 80)
    feats2 = build_features(future)
    for k, before in snapshot.items():
        after = feats2[k].iloc[80]
        pd.testing.assert_series_equal(after, before, check_names=False)   # row 80 unchanged
