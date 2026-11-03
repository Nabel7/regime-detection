"""Tests for GaussianHSMMModel and its BaseRegimeModel contract."""

import numpy as np
import pandas as pd
import pytest

from regime_detection import GaussianHSMMModel, RegimeDetectionPipeline
from regime_detection.backtest import WalkForwardBacktester
from regime_detection.data import make_regime_data

# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _small_df(n_obs=160, n_regimes=2, seed=42, noise=0.5):
    df, true_labels = make_regime_data(n_obs=n_obs, n_regimes=n_regimes, seed=seed, noise=noise)
    return df, true_labels


def _fitted(n_regimes=2, **kwargs):
    df, _ = _small_df()
    m = GaussianHSMMModel(n_regimes=n_regimes, n_init=3, **kwargs)
    return m.fit(df), df


def _fitted3(**kwargs):
    # 3-regime fit on clearly separated data, for destination-split tests.
    df, _ = make_regime_data(n_obs=240, n_regimes=3, seed=2, noise=0.4)
    m = GaussianHSMMModel(n_regimes=3, n_init=4, random_state=0, **kwargs).fit(df)
    return m, df


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
        GaussianHSMMModel().predict(df)


def test_reproducibility():
    df, _ = _small_df()
    a = GaussianHSMMModel(n_init=3, random_state=0).fit(df).predict(df)
    b = GaussianHSMMModel(n_init=3, random_state=0).fit(df).predict(df)
    pd.testing.assert_series_equal(a, b)


def test_recovers_known_regimes():
    # HSMM should recover clearly separated, persistent synthetic regimes.
    df, true = make_regime_data(n_obs=200, n_regimes=2, seed=1, noise=0.4)
    labels = GaussianHSMMModel(n_regimes=2, n_init=5, random_state=0).fit(df).predict(df)
    # Best-match accuracy over the two label permutations.
    acc = max(
        (labels.to_numpy() == true.to_numpy()).mean(),
        (labels.to_numpy() == (1 - true.to_numpy())).mean(),
    )
    assert acc > 0.85, f"Regime recovery too low: {acc:.3f}"


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


def test_effective_transition_matches_mean_durations():
    # The framework transition matrix embeds the explicit durations, so its
    # implied expected duration 1/(1 - p_ii) must equal the HSMM mean dwell.
    m, df = _fitted()
    np.testing.assert_allclose(m.expected_durations, m.mean_durations, rtol=1e-6)


# ---------------------------------------------------------------------- #
# State ordering
# ---------------------------------------------------------------------- #


def test_state_ordering_mean():
    df, _ = make_regime_data(n_obs=200, n_regimes=2, seed=1, noise=0.3)
    m = GaussianHSMMModel(n_regimes=2, n_init=3, sort_states_by="mean").fit(df)
    means = m.means.mean(axis=1)
    assert means[0] <= means[1], f"Expected ascending means, got {means}"


def test_sort_states_by_none_does_not_crash():
    m, df = _fitted(sort_states_by="none")
    _ = m.predict(df)


# ---------------------------------------------------------------------- #
# HSMM-specific: explicit duration model
# ---------------------------------------------------------------------- #


def test_duration_distributions_are_valid_pmfs():
    m, df = _fitted()
    dur = m.duration_distributions
    assert dur.shape[0] == 2
    np.testing.assert_allclose(dur.sum(axis=1), 1.0, atol=1e-6)
    assert (dur >= 0).all()


def test_mean_durations_positive():
    m, df = _fitted()
    md = m.mean_durations
    assert md.shape == (2,)
    assert (md >= 1.0).all()


def test_embedded_transition_zero_diagonal():
    m, df = _fitted()
    A = m.embedded_transition_matrix
    np.testing.assert_allclose(np.diag(A), 0.0, atol=1e-8)
    np.testing.assert_allclose(A.sum(axis=1), 1.0, atol=1e-6)


def test_nonparametric_duration():
    df, _ = _small_df()
    m = GaussianHSMMModel(n_regimes=2, duration_dist="nonparametric", n_init=2).fit(df)
    np.testing.assert_allclose(m.duration_distributions.sum(axis=1), 1.0, atol=1e-6)


def test_full_covariance():
    df, _ = _small_df()
    m = GaussianHSMMModel(n_regimes=2, covariance_type="full", n_init=2).fit(df)
    assert m.covariances.shape == (2, df.shape[1], df.shape[1])


def test_fit_result_stored():
    m, df = _fitted()
    assert m.fit_result_ is not None
    assert m.fit_result_.n_regimes == 2
    assert "mean_durations" in m.fit_result_.extra
    assert "max_duration" in m.fit_result_.extra


# ---------------------------------------------------------------------- #
# HSMM-specific: duration-conditional dynamics (hazard / switch curves)
# ---------------------------------------------------------------------- #


def test_hazard_curves_shape_and_bounds():
    m, df = _fitted()
    haz = m.hazard_curves
    assert haz.shape == (2, m.fit_result_.extra["max_duration"])
    assert (haz >= -1e-12).all() and (haz <= 1.0 + 1e-9).all()
    # Hazard must reach 1 at the support edge: you cannot outlast max_duration.
    np.testing.assert_allclose(haz[:, -1], 1.0, atol=1e-6)


def test_survival_curves_monotone_decreasing():
    m, df = _fitted()
    surv = m.survival_curves
    assert surv.shape == m.hazard_curves.shape
    assert (surv >= -1e-12).all() and (surv <= 1.0 + 1e-9).all()
    assert (np.diff(surv, axis=1) <= 1e-9).all(), "survival must be non-increasing"


def test_duration_conditional_transition_is_stochastic():
    m, df = _fitted()
    for elapsed in (1, 5, 20):
        T = m.duration_conditional_transition(elapsed)
        assert T.shape == (2, 2)
        np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-6)
        assert (T >= -1e-12).all()


def test_hazard_rises_with_dwell_for_peaked_durations():
    # The defining HSMM property: for a peaked (Poisson) duration the exit
    # hazard is *not* constant — it increases with elapsed dwell, unlike the
    # memoryless geometric hazard of a plain HMM.
    df, _ = make_regime_data(n_obs=240, n_regimes=2, seed=3, noise=0.4)
    m = GaussianHSMMModel(n_regimes=2, n_init=5, random_state=0).fit(df)
    haz = m.hazard_curves
    # Compare hazard early vs near the mean dwell for each regime.
    for k in range(2):
        mean_d = int(round(m.mean_durations[k]))
        early = max(1, mean_d // 4)
        late = min(haz.shape[1], mean_d)
        assert haz[k, late - 1] > haz[k, early - 1], (
            f"regime {k}: hazard should rise with dwell "
            f"({haz[k, early - 1]:.3f} -> {haz[k, late - 1]:.3f})"
        )


def test_switch_probability_curve_matches_conditional_transition():
    m, df = _fitted()
    curve = m.switch_probability_curve(0)  # (D, K)
    assert curve.shape == (m.hazard_curves.shape[1], 2)
    np.testing.assert_allclose(curve.sum(axis=1), 1.0, atol=1e-6)
    # Row d of the curve must equal row 0 of the elapsed-d transition matrix.
    for d in (1, 3, 10):
        T = m.duration_conditional_transition(d)
        np.testing.assert_allclose(curve[d - 1], T[0], atol=1e-9)


def test_expected_remaining_duration_shrinks_with_dwell():
    m, df = _fitted()
    r_early = m.expected_remaining_duration(1)
    r_late = m.expected_remaining_duration(int(m.mean_durations.max()))
    assert (r_late <= r_early + 1e-9).all()
    assert (r_early >= -1e-9).all()


def test_duration_conditional_transition_per_regime_elapsed():
    m, df = _fitted()
    T = m.duration_conditional_transition(np.array([2, 15]))
    np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-6)


def test_duration_conditional_transition_validates_elapsed():
    m, df = _fitted()
    with pytest.raises(ValueError):
        m.duration_conditional_transition(0)  # dwell must be >= 1
    with pytest.raises(ValueError):
        m.duration_conditional_transition([1, 2, 3])  # wrong length


# ---------------------------------------------------------------------- #
# Causal (filtered) inference — no look-ahead bias
# ---------------------------------------------------------------------- #


def test_filter_proba_shape_and_normalised():
    m, df = _fitted()
    filt = m.filter_proba(df)
    assert filt.shape == (len(df), 2)
    np.testing.assert_allclose(filt.to_numpy().sum(axis=1), 1.0, atol=1e-6)


def test_filter_is_causal():
    # THE defining property: the filtered posterior at time t must not change
    # when future observations are appended. Filtering a prefix X[:t] must
    # equal the first t rows of filtering the whole series.
    df, _ = _small_df(n_obs=180)
    m = GaussianHSMMModel(n_regimes=2, n_init=3, random_state=0).fit(df)
    full = m.filter_proba(df).to_numpy()
    for t in (30, 90, 150):
        prefix = m.filter_proba(df.iloc[:t]).to_numpy()
        np.testing.assert_allclose(prefix, full[:t], atol=1e-9), t


def test_smoothed_differs_from_filtered():
    # Sanity that smoothing genuinely uses the future: the two must diverge
    # somewhere (otherwise beta would be doing nothing).
    df, _ = _small_df(n_obs=180)
    m = GaussianHSMMModel(n_regimes=2, n_init=3, random_state=0).fit(df)
    smoothed = m.predict_proba(df).to_numpy()
    filtered = m.filter_proba(df).to_numpy()
    assert np.abs(smoothed - filtered).max() > 1e-3


def test_filter_predict_labels_valid():
    m, df = _fitted()
    labels = m.filter_predict(df)
    assert isinstance(labels, pd.Series)
    assert set(labels.unique()).issubset({0, 1})


def test_filtered_next_regime_is_distribution_and_causal():
    df, _ = _small_df(n_obs=160)
    m = GaussianHSMMModel(n_regimes=2, n_init=3, random_state=0).fit(df)
    pred = m.filtered_next_regime(df)
    assert pred.shape == (len(df), 2)
    np.testing.assert_allclose(pred.to_numpy().sum(axis=1), 1.0, atol=1e-6)
    assert (pred.to_numpy() >= -1e-12).all()
    # Causal: one-step forecast at t uses only data through t.
    full = pred.to_numpy()
    for t in (40, 100):
        prefix = m.filtered_next_regime(df.iloc[:t]).to_numpy()
        np.testing.assert_allclose(prefix, full[:t], atol=1e-9), t


# ---------------------------------------------------------------------- #
# Directional (momentum-tilted) destination split
# ---------------------------------------------------------------------- #


def test_directional_split_zero_sensitivity_recovers_embedded():
    m, df = _fitted(n_regimes=2)
    for i in range(2):
        split = m.directional_split(i, np.ones(df.shape[1]), sensitivity=0.0)
        np.testing.assert_allclose(split, m.embedded_transition_matrix[i], atol=1e-9)


def test_directional_split_is_distribution_over_destinations():
    m, df = _fitted3()
    split = m.directional_split(1, np.ones(df.shape[1]), sensitivity=1.0)
    np.testing.assert_allclose(split.sum(), 1.0, atol=1e-9)
    assert split[1] == 0.0  # no mass on the source regime itself


def test_directional_split_tilts_with_momentum_sign():
    # A 3-regime model ordered by mean: 0=low, 2=high. Momentum "up" (positive
    # in the direction of higher means) must raise the share going to regime 2.
    m, df = _fitted3()
    # Build an up-momentum aligned with the low->high mean axis.
    axis = m.means[2] - m.means[0]
    up = m.directional_split(1, axis, sensitivity=1.0)
    down = m.directional_split(1, -axis, sensitivity=1.0)
    assert up[2] > up[0], "up-momentum should favour the higher regime"
    assert down[0] > down[2], "down-momentum should favour the lower regime"


def test_directional_next_regime_zero_sensitivity_matches_filtered():
    m, df = _fitted3()
    a = m.filtered_next_regime(df).to_numpy()
    b = m.directional_next_regime(df, sensitivity=0.0).to_numpy()
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_directional_next_regime_causal_and_normalised():
    df, _ = make_regime_data(n_obs=180, n_regimes=3, seed=5, noise=0.4)
    m = GaussianHSMMModel(n_regimes=3, n_init=3, random_state=0).fit(df)
    pred = m.directional_next_regime(df, momentum_window=3, sensitivity=1.5)
    np.testing.assert_allclose(pred.to_numpy().sum(axis=1), 1.0, atol=1e-6)
    assert (pred.to_numpy() >= -1e-12).all()
    full = pred.to_numpy()
    prefix = m.directional_next_regime(df.iloc[:90], sensitivity=1.5).to_numpy()
    np.testing.assert_allclose(prefix, full[:90], atol=1e-9)


# ---------------------------------------------------------------------- #
# Forecast
# ---------------------------------------------------------------------- #


def test_forecast_rows_are_distributions():
    m, df = _fitted()
    f = m.forecast(steps=6, current_state=0)
    assert f.shape == (6, 2)
    np.testing.assert_allclose(f.sum(axis=1), 1.0, atol=1e-6)


def test_duration_aware_forecast_rows_are_distributions():
    m, df = _fitted3()
    f = m.forecast(steps=24, current_state=1, elapsed=3)
    assert f.shape == (24, 3)
    np.testing.assert_allclose(f.sum(axis=1), 1.0, atol=1e-9)
    assert (f >= -1e-12).all()


def test_duration_aware_forecast_first_step_matches_conditional_transition():
    m, df = _fitted3()
    for state in range(3):
        for elapsed in (1, 4, 12):
            f = m.forecast(steps=1, current_state=state, elapsed=elapsed)
            T = m.duration_conditional_transition(elapsed)
            np.testing.assert_allclose(f[0], T[state], atol=1e-9)


def test_duration_aware_forecast_depends_on_elapsed():
    # The point of the elapsed argument: a regime that has run longer is more
    # likely to have ended a few steps ahead (peaked Poisson durations).
    df, _ = make_regime_data(n_obs=240, n_regimes=2, seed=3, noise=0.4)
    m = GaussianHSMMModel(n_regimes=2, n_init=5, random_state=0).fit(df)
    mean_d = int(round(m.mean_durations[0]))
    early = m.forecast(steps=3, current_state=0, elapsed=1)
    late = m.forecast(steps=3, current_state=0, elapsed=mean_d)
    assert late[-1, 0] < early[-1, 0]


def test_forecast_without_elapsed_unchanged():
    m, df = _fitted()
    f = m.forecast(steps=4, current_state=1)
    T = m.transition_matrix
    expected = np.eye(2)[1] @ np.linalg.matrix_power(T, 4)
    np.testing.assert_allclose(f[-1], expected, atol=1e-9)


def test_duration_aware_forecast_validates_inputs():
    m, df = _fitted()
    with pytest.raises(ValueError):
        m.forecast(steps=2, elapsed=3)  # needs current_state
    with pytest.raises(ValueError):
        m.forecast(steps=2, current_state=0, elapsed=0)  # dwell must be >= 1
    with pytest.raises(ValueError):
        m.forecast(steps=2, current_state=5, elapsed=1)


# ---------------------------------------------------------------------- #
# Persistence (save / load)
# ---------------------------------------------------------------------- #


def test_save_load_roundtrip(tmp_path):
    m, df = _fitted()
    path = tmp_path / "hsmm.pkl"
    m.save(path)
    m2 = GaussianHSMMModel.load(path)
    pd.testing.assert_series_equal(m.predict(df), m2.predict(df))


# ---------------------------------------------------------------------- #
# Pipeline integration
# ---------------------------------------------------------------------- #


def test_pipeline_fit_detect():
    df, _ = _small_df()
    pipe = RegimeDetectionPipeline(
        model=GaussianHSMMModel(n_regimes=2, n_init=3),
        min_duration=2,
    )
    result = pipe.fit_detect(df)
    assert len(result.smoothed_labels) == len(df)
    assert result.proba.shape == (len(df), 2)


def test_pipeline_names_hsmm_regimes_by_feature():
    df, _ = _small_df()
    pipe = RegimeDetectionPipeline(
        model=GaussianHSMMModel(n_regimes=2, n_init=3),
        label_feature=df.columns[0],
        label_names=("low", "high"),
    )
    res = pipe.fit_detect(df)
    assert set(res.regime_names.values()) == {"low", "high"}
    means = res.model.state_means[:, 0]
    assert res.regime_names[int(np.argmin(means))] == "low"


def test_pipeline_detect_oos():
    df, _ = _small_df(n_obs=180)
    train, test = df.iloc[:130], df.iloc[130:]
    pipe = RegimeDetectionPipeline(model=GaussianHSMMModel(n_regimes=2, n_init=3))
    pipe.fit_detect(train)
    result = pipe.detect(test)
    assert len(result.smoothed_labels) == len(test)


# ---------------------------------------------------------------------- #
# Walk-forward backtester integration
# ---------------------------------------------------------------------- #


def test_walkforward_oos_labels_cover_test_period():
    df, _ = _small_df(n_obs=200)

    def factory():
        return RegimeDetectionPipeline(model=GaussianHSMMModel(n_regimes=2, n_init=2))

    bt = WalkForwardBacktester(factory, train_size=120, test_size=6, refit_every=2)
    wf = bt.run(df)
    assert wf.n_folds > 0
    assert len(wf.oos_labels) > 0
    assert set(wf.oos_labels.unique()).issubset({0, 1})
