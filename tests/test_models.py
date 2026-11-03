"""Tests for the Gaussian HMM model and base contract."""

import numpy as np
import pandas as pd
import pytest

from regime_detection import GaussianHMMModel
from regime_detection.data import make_regime_data


def test_fit_predict_shapes():
    df, _ = make_regime_data(n_obs=120, n_regimes=3, seed=5)
    m = GaussianHMMModel(n_regimes=3, n_init=4, random_state=0).fit(df)
    labels = m.predict(df)
    proba = m.predict_proba(df)
    assert isinstance(labels, pd.Series)
    assert isinstance(proba, pd.DataFrame)
    assert labels.shape[0] == len(df)
    assert proba.shape == (len(df), 3)
    np.testing.assert_allclose(proba.to_numpy().sum(axis=1), 1.0, atol=1e-6)


def test_reproducibility():
    df, _ = make_regime_data(n_obs=100, seed=6)
    a = GaussianHMMModel(n_regimes=2, n_init=5, random_state=42).fit(df).predict(df)
    b = GaussianHMMModel(n_regimes=2, n_init=5, random_state=42).fit(df).predict(df)
    pd.testing.assert_series_equal(a, b)


def test_state_ordering_is_canonical():
    # sort_states_by="mean" must order regimes by ascending mean feature value.
    df, _ = make_regime_data(n_obs=200, n_regimes=3, seed=8)
    m = GaussianHMMModel(n_regimes=3, n_init=6, random_state=0, sort_states_by="mean").fit(df)
    overall_means = m.means.mean(axis=1)
    assert np.all(np.diff(overall_means) >= 0)


def test_transition_matrix_rows_sum_to_one():
    df, _ = make_regime_data(n_obs=120, seed=9)
    m = GaussianHMMModel(n_regimes=2, n_init=4, random_state=0).fit(df)
    T = m.transition_matrix
    np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-6)


def test_stationary_distribution_is_invariant():
    df, _ = make_regime_data(n_obs=120, seed=10)
    m = GaussianHMMModel(n_regimes=3, n_init=4, random_state=0).fit(df)
    pi = m.stationary_distribution
    np.testing.assert_allclose(pi @ m.transition_matrix, pi, atol=1e-6)
    np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-8)


def test_forecast_rows_are_distributions():
    df, _ = make_regime_data(n_obs=120, seed=11)
    m = GaussianHMMModel(n_regimes=3, n_init=4, random_state=0).fit(df)
    f = m.forecast(steps=5, current_state=0)
    assert f.shape == (5, 3)
    np.testing.assert_allclose(f.sum(axis=1), 1.0, atol=1e-6)


def test_predict_before_fit_raises():
    df, _ = make_regime_data(n_obs=30, seed=12)
    with pytest.raises(RuntimeError):
        GaussianHMMModel(n_regimes=2).predict(df)


def test_save_load_roundtrip(tmp_path):
    df, _ = make_regime_data(n_obs=80, seed=13)
    m = GaussianHMMModel(n_regimes=2, n_init=3, random_state=0).fit(df)
    p = tmp_path / "m.pkl"
    m.save(p)
    m2 = GaussianHMMModel.load(p)
    pd.testing.assert_series_equal(m.predict(df), m2.predict(df))
