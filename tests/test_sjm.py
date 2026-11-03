"""Tests for SparseJumpModel and its BaseRegimeModel contract."""

import numpy as np
import pandas as pd
import pytest

from regime_detection import RegimeDetectionPipeline, SparseJumpModel
from regime_detection.backtest import WalkForwardBacktester
from regime_detection.data import make_regime_data

# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _small_df(n_obs=120, n_regimes=2, seed=42):
    df, true_labels = make_regime_data(n_obs=n_obs, n_regimes=n_regimes, seed=seed)
    return df, true_labels


def _fitted(n_regimes=2, **kwargs):
    df, _ = _small_df()
    m = SparseJumpModel(n_regimes=n_regimes, jump_penalty=8.0, n_init=3, **kwargs)
    return m.fit(df), df


# ---------------------------------------------------------------------- #
# Basic contract
# ---------------------------------------------------------------------- #


def test_fit_predict_shapes():
    m, df = _fitted()
    labels = m.predict(df)
    proba = m.predict_proba(df)
    assert isinstance(labels, pd.Series)
    assert isinstance(proba, pd.DataFrame)
    assert labels.shape[0] == len(df)
    assert proba.shape == (len(df), 2)


def test_proba_sums_to_one():
    m, df = _fitted()
    proba = m.predict_proba(df)
    np.testing.assert_allclose(proba.to_numpy().sum(axis=1), 1.0, atol=1e-6)


def test_labels_are_valid_regime_indices():
    m, df = _fitted()
    labels = m.predict(df)
    assert set(labels.unique()).issubset({0, 1})


def test_predict_before_fit_raises():
    df, _ = _small_df()
    with pytest.raises(RuntimeError):
        SparseJumpModel().predict(df)


def test_reproducibility():
    df, _ = _small_df()
    a = SparseJumpModel(jump_penalty=8.0, n_init=3, random_state=0).fit(df).predict(df)
    b = SparseJumpModel(jump_penalty=8.0, n_init=3, random_state=0).fit(df).predict(df)
    pd.testing.assert_series_equal(a, b)


# ---------------------------------------------------------------------- #
# Transition matrix and stationary distribution
# ---------------------------------------------------------------------- #


def test_transition_matrix_is_stochastic():
    m, df = _fitted()
    T = m.transition_matrix
    assert T.shape == (2, 2)
    np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-6)
    assert (T >= 0).all()


def test_stationary_distribution_invariant():
    m, df = _fitted()
    pi = m.stationary_distribution
    np.testing.assert_allclose(pi @ m.transition_matrix, pi, atol=1e-6)
    np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-8)


# ---------------------------------------------------------------------- #
# State ordering
# ---------------------------------------------------------------------- #


def test_state_ordering_mean():
    # With sort_states_by="mean", regime 0 should have a lower average centroid
    # than regime 1.  Use clearly separated synthetic data to make this robust.
    df, _ = make_regime_data(n_obs=200, n_regimes=2, seed=1, noise=0.3)
    m = SparseJumpModel(n_regimes=2, jump_penalty=8.0, n_init=3, sort_states_by="mean").fit(df)
    means = m.centers.mean(axis=1)
    assert means[0] <= means[1], f"Expected ascending centroids, got {means}"


def test_sort_states_by_none_does_not_crash():
    m, df = _fitted(sort_states_by="none")
    _ = m.predict(df)


# ---------------------------------------------------------------------- #
# SJM-specific properties
# ---------------------------------------------------------------------- #


def test_feat_weights_shape():
    df, _ = _small_df()
    m = SparseJumpModel(jump_penalty=8.0, n_init=3).fit(df)
    w = m.feat_weights
    assert w.shape == (df.shape[1],)
    assert (w >= 0).all()


def test_centers_shape():
    m, df = _fitted()
    assert m.centers.shape == (2, df.shape[1])


def test_objective_value_is_finite():
    m, df = _fitted()
    assert np.isfinite(m.objective_value)
    assert m.objective_value >= 0


def test_fit_result_stored():
    m, df = _fitted()
    assert m.fit_result_ is not None
    assert m.fit_result_.n_regimes == 2
    assert "objective_value" in m.fit_result_.extra
    assert "feat_weights" in m.fit_result_.extra


# ---------------------------------------------------------------------- #
# online_predict flag
# ---------------------------------------------------------------------- #


def test_online_predict_shapes():
    df, _ = _small_df()
    m = SparseJumpModel(jump_penalty=8.0, n_init=3, online_predict=True).fit(df)
    proba = m.predict_proba(df)
    assert proba.shape == (len(df), 2)
    np.testing.assert_allclose(proba.to_numpy().sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------- #
# Forecast
# ---------------------------------------------------------------------- #


def test_forecast_rows_are_distributions():
    m, df = _fitted()
    f = m.forecast(steps=6, current_state=0)
    assert f.shape == (6, 2)
    np.testing.assert_allclose(f.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------- #
# Persistence (save / load)
# ---------------------------------------------------------------------- #


def test_save_load_roundtrip(tmp_path):
    m, df = _fitted()
    path = tmp_path / "sjm.pkl"
    m.save(path)
    m2 = SparseJumpModel.load(path)
    pd.testing.assert_series_equal(m.predict(df), m2.predict(df))


# ---------------------------------------------------------------------- #
# Pipeline integration
# ---------------------------------------------------------------------- #


def test_pipeline_fit_detect():
    df, _ = _small_df()
    pipe = RegimeDetectionPipeline(
        model=SparseJumpModel(n_regimes=2, jump_penalty=8.0, n_init=3),
        min_duration=2,
    )
    result = pipe.fit_detect(df)
    assert len(result.smoothed_labels) == len(df)
    assert result.proba.shape == (len(df), 2)


def test_pipeline_detect_oos():
    df, _ = _small_df(n_obs=150)
    train, test = df.iloc[:120], df.iloc[120:]
    pipe = RegimeDetectionPipeline(
        model=SparseJumpModel(n_regimes=2, jump_penalty=8.0, n_init=3),
    )
    pipe.fit_detect(train)
    result = pipe.detect(test)
    assert len(result.smoothed_labels) == len(test)


# ---------------------------------------------------------------------- #
# Walk-forward backtester integration
# ---------------------------------------------------------------------- #


def test_walkforward_oos_labels_cover_test_period():
    df, _ = _small_df(n_obs=180)

    def factory():
        return RegimeDetectionPipeline(
            model=SparseJumpModel(n_regimes=2, jump_penalty=8.0, n_init=3),
        )

    bt = WalkForwardBacktester(factory, train_size=100, test_size=6, refit_every=2)
    wf = bt.run(df)
    assert wf.n_folds > 0
    assert len(wf.oos_labels) > 0
    assert set(wf.oos_labels.unique()).issubset({0, 1})


def test_walkforward_no_lookahead(tmp_path):
    # The test asserts that each OOS label was produced without seeing future
    # training data — verified by checking that train indices never overlap
    # with test indices within a fold.
    df, _ = _small_df(n_obs=150)
    seen_train_indices: list[pd.Index] = []
    seen_test_indices: list[pd.Index] = []

    original_factory_calls: list[int] = [0]

    def factory():
        original_factory_calls[0] += 1

        class _Spy(RegimeDetectionPipeline):
            def fit_detect(self, X, **kw):
                seen_train_indices.append(X.index)
                return super().fit_detect(X, **kw)

            def detect(self, X):
                seen_test_indices.append(X.index)
                return super().detect(X)

        return _Spy(model=SparseJumpModel(n_regimes=2, jump_penalty=8.0, n_init=3))

    bt = WalkForwardBacktester(factory, train_size=100, test_size=6, expanding=True)
    bt.run(df)

    for tr_idx, te_idx in zip(seen_train_indices, seen_test_indices, strict=True):
        overlap = tr_idx.intersection(te_idx)
        assert len(overlap) == 0, f"Lookahead: {len(overlap)} overlapping indices"
