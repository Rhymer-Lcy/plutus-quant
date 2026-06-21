"""Walk-forward ML combiner: dataset construction is point-in-time correct (target is the
cross-sectionally demeaned forward return; only members are kept) and prediction is
strictly out-of-sample (no signal before `min_train`)."""
import numpy as np
import pandas as pd

from plutus.research.model.walk_forward import build_dataset, walk_forward_predict


def _panels(n_dates=12, n_names=80, seed=0):
    rng = np.random.default_rng(seed)
    eval_dates = list(pd.bdate_range("2020-01-31", periods=n_dates, freq="BME"))
    names = [f"N{i}" for i in range(n_names)]
    # a single informative factor that drives next-period return + noise
    f = pd.DataFrame(rng.normal(size=(n_dates, n_names)), index=eval_dates, columns=names)
    close = pd.DataFrame(1.0, index=eval_dates, columns=names)
    for i in range(n_dates - 1):
        ret = 0.05 * f.iloc[i] + rng.normal(0, 0.01, n_names)
        close.iloc[i + 1] = close.iloc[i] * (1 + ret)
    return {"feat": f}, close, eval_dates, names


def test_build_dataset_target_is_cross_sectionally_demeaned():
    factors, close, eval_dates, _ = _panels()
    data, cols = build_dataset(factors, close, eval_dates)
    assert cols == ["feat"]
    # each signal date's forward-return target sums to ~0 (cross-sectional demean)
    per_date = data.groupby("date")["fwd_ret"].mean()
    assert np.allclose(per_date.to_numpy(), 0.0, atol=1e-9)


def test_walk_forward_is_out_of_sample_and_recovers_signal():
    factors, close, eval_dates, _ = _panels()
    data, cols = build_dataset(factors, close, eval_dates)
    signal = walk_forward_predict(data, cols, min_train=3, window=3)
    # no prediction before min_train dates have elapsed (strict no-look-ahead)
    assert signal.index.min() >= eval_dates[3]
    # the model should rank the informative factor the right way: its OOS predictions
    # correlate positively with the realized next-period return.
    t = signal.index[0]
    nxt = eval_dates[eval_dates.index(t) + 1]
    realized = (close.loc[nxt] / close.loc[t] - 1.0).reindex(signal.columns)
    corr = pd.Series(signal.loc[t]).corr(realized)
    assert corr > 0.2
