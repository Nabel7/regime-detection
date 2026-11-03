"""Tests for the walk-forward backtester and backtest metrics."""

import numpy as np
import pandas as pd
import pytest

from regime_detection import (
    GaussianHMMModel,
    GaussianHSMMModel,
    RegimeDetectionPipeline,
)
from regime_detection.backtest import (
    WalkForwardBacktester,
    adjusted_rand_index,
    best_match_accuracy,
    stability_metrics,
)
from regime_detection.data import make_regime_data


def _factory(n_regimes=3):
    return lambda: RegimeDetectionPipeline(
        model=GaussianHMMModel(
            n_regimes=n_regimes, covariance_type="diag", n_init=4, random_state=0
        ),
        min_duration=2,
    )


def test_walk_forward_runs_and_aligns():
    df, _ = make_regime_data(n_obs=180, n_regimes=3, seed=5)
    bt = WalkForwardBacktester(_factory(), train_size=100, test_size=6, expanding=True)
    res = bt.run(df)
    # OOS labels start after the initial training window and cover the rest.
    assert res.oos_labels.index.min() == df.index[100]
    assert res.oos_labels.index.max() == df.index[-1]
    assert len(res.oos_labels) == len(df) - 100
    assert res.oos_proba.shape[0] == len(res.oos_labels)


def test_walk_forward_no_lookahead(monkeypatch):
    """The pipeline must only ever see training data up to the fold origin.

    We wrap detect/fit to record the max timestamp each train window contains
    and assert it never reaches into the corresponding test block.
    """
    df, _ = make_regime_data(n_obs=150, n_regimes=2, seed=6)
    seen_train_max = []

    orig_fit_detect = RegimeDetectionPipeline.fit_detect

    def spy_fit_detect(self, X, **kw):
        seen_train_max.append(X.index.max())
        return orig_fit_detect(self, X, **kw)

    monkeypatch.setattr(RegimeDetectionPipeline, "fit_detect", spy_fit_detect)

    bt = WalkForwardBacktester(_factory(2), train_size=80, test_size=10, refit_every=1)
    bt.run(df)

    # Every training window max must be strictly before the first OOS label
    # it is used to produce. With test_size=10 and refit_every=1, train max at
    # fold k is df.index[80 + 10k - 1]; OOS predictions begin at index 80+10k.
    for k, tmax in enumerate(seen_train_max):
        first_oos = df.index[80 + 10 * k]
        assert tmax < first_oos


def test_rolling_vs_expanding():
    df, _ = make_regime_data(n_obs=160, n_regimes=2, seed=7)
    exp = WalkForwardBacktester(_factory(2), train_size=90, test_size=10, expanding=True).run(df)
    roll = WalkForwardBacktester(_factory(2), train_size=90, test_size=10, expanding=False).run(df)
    assert len(exp.oos_labels) == len(roll.oos_labels)


def test_backtest_recovers_regimes_reasonably():
    df, truth = make_regime_data(n_obs=240, n_regimes=3, seed=7, persistence=0.95)
    bt = WalkForwardBacktester(_factory(3), train_size=120, test_size=6, refit_every=2)
    res = bt.run(df)
    common = truth.reindex(res.oos_labels.index)
    acc = best_match_accuracy(common, res.oos_labels)
    ari = adjusted_rand_index(common, res.oos_labels)
    # Loose thresholds: this guards against regressions, not exact values.
    assert acc > 0.7
    assert ari > 0.4


def test_insufficient_data_raises():
    df, _ = make_regime_data(n_obs=20, seed=1)
    bt = WalkForwardBacktester(_factory(2), train_size=18, test_size=6)
    with pytest.raises(ValueError):
        bt.run(df)


def test_stability_metrics_keys():
    df, _ = make_regime_data(n_obs=120, seed=2)
    bt = WalkForwardBacktester(_factory(2), train_size=80, test_size=8)
    res = bt.run(df)
    sm = stability_metrics(res.oos_labels)
    assert {"n_transitions", "transition_rate", "whipsaw_rate"}.issubset(sm)


# ---------------------------------------------------------------------- #
# Causal (filtered) walk-forward mode
# ---------------------------------------------------------------------- #


def _hsmm_factory(n_regimes=3):
    return lambda: RegimeDetectionPipeline(
        model=GaussianHSMMModel(n_regimes=n_regimes, n_init=3, random_state=0),
    )


def test_causal_backtest_runs_and_covers_oos():
    df, _ = make_regime_data(n_obs=180, n_regimes=3, seed=5, noise=0.4)
    bt = WalkForwardBacktester(
        _hsmm_factory(), train_size=100, test_size=6, refit_every=2, causal=True
    )
    res = bt.run(df)
    assert res.diagnostics["causal"] is True
    assert len(res.oos_labels) == 180 - 100
    np.testing.assert_allclose(res.oos_proba.to_numpy().sum(axis=1), 1.0, atol=1e-6)


def test_causal_backtest_is_actually_causal():
    # OOS labels from the causal mode must be reproducible from a strict prefix:
    # truncating the series after an OOS date cannot change that date's label.
    df, _ = make_regime_data(n_obs=160, n_regimes=3, seed=7, noise=0.4)
    bt = WalkForwardBacktester(
        _hsmm_factory(), train_size=100, test_size=4, refit_every=1, causal=True
    )
    full = bt.run(df).oos_labels
    cut = df.index[130]  # truncate mid-OOS
    trimmed = bt.run(df.loc[:cut]).oos_labels
    common = full.index.intersection(trimmed.index)
    assert len(common) > 0
    pd.testing.assert_series_equal(full.loc[common], trimmed.loc[common])


def test_causal_mode_rejects_smoother_only_model():
    # A plain HMM has no filter_proba; causal mode must fail loudly, not silently
    # fall back to a look-ahead smoother.
    df, _ = make_regime_data(n_obs=140, n_regimes=2, seed=1)
    bt = WalkForwardBacktester(_factory(2), train_size=100, test_size=6, causal=True)
    with pytest.raises(TypeError):
        bt.run(df)
