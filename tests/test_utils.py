"""Tests for smoothing utilities and transition/regime metrics."""

import numpy as np
import pandas as pd

from regime_detection.utils import (
    empirical_transition_matrix,
    expected_durations,
    majority_vote_smoothing,
    min_duration_smoothing,
    probability_smoothing,
    regime_statistics,
    run_length_encode,
    transition_count,
    transition_entropy,
)


def test_run_length_encode():
    labels = np.array([0, 0, 1, 1, 1, 0])
    runs = run_length_encode(labels)
    assert runs == [(0, 0, 2), (1, 2, 3), (0, 5, 1)]


def test_min_duration_absorbs_short_runs():
    # The single "1" at index 3 is shorter than min_duration=2 -> absorbed.
    labels = np.array([0, 0, 0, 1, 0, 0, 0])
    out = min_duration_smoothing(labels, min_duration=2)
    assert (out == 0).all()


def test_min_duration_keeps_long_runs():
    labels = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])
    out = min_duration_smoothing(labels, min_duration=3)
    np.testing.assert_array_equal(out, labels)


def test_min_duration_preserves_index():
    idx = pd.date_range("2020-01-01", periods=5, freq="MS")
    s = pd.Series([0, 1, 0, 0, 0], index=idx)
    out = min_duration_smoothing(s, min_duration=2)
    assert isinstance(out, pd.Series)
    assert (out.index == idx).all()


def test_majority_vote():
    labels = np.array([0, 1, 0, 0, 0])
    out = majority_vote_smoothing(labels, window=3)
    assert out[1] == 0  # neighbours are 0,0 -> vote 0


def test_probability_smoothing_sticky():
    P = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
    sm = probability_smoothing(P, alpha=0.2)
    # Heavy smoothing should reduce the swing in the middle row.
    assert sm[1, 1] < 0.9
    np.testing.assert_allclose(sm.sum(axis=1), 1.0, atol=1e-9)


def test_empirical_transition_matrix_known():
    labels = np.array([0, 0, 1, 0, 1, 1])
    T = empirical_transition_matrix(labels, n_regimes=2)
    # from 0: ->0 once, ->1 twice  => [1/3, 2/3]
    np.testing.assert_allclose(T[0], [1 / 3, 2 / 3], atol=1e-9)
    # from 1: ->0 once, ->1 once   => [1/2, 1/2]
    np.testing.assert_allclose(T[1], [1 / 2, 1 / 2], atol=1e-9)


def test_expected_durations():
    T = np.array([[0.9, 0.1], [0.2, 0.8]])
    d = expected_durations(T)
    np.testing.assert_allclose(d, [10.0, 5.0], atol=1e-9)


def test_transition_entropy_absorbing_is_zero():
    T = np.array([[1.0, 0.0], [0.5, 0.5]])
    ent = transition_entropy(T)
    assert ent[0] == 0.0
    assert ent[1] == 1.0  # 0.5/0.5 split = 1 bit


def test_transition_count():
    assert transition_count(np.array([0, 0, 1, 1, 0])) == 2


def test_regime_statistics():
    labels = np.array([0, 0, 1, 1, 1, 0])
    stats = regime_statistics(labels, n_regimes=2)
    assert stats.loc[0, "n_obs"] == 3
    assert stats.loc[1, "n_obs"] == 3
    assert stats.loc[1, "n_episodes"] == 1
    assert abs(stats.loc[0, "frequency"] - 0.5) < 1e-9
