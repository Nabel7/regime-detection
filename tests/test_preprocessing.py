"""Tests for the leakage-safe preprocessor."""

import numpy as np
import pandas as pd

from regime_detection.data import Preprocessor, make_regime_data


def test_zscore_fit_on_train_only():
    df, _ = make_regime_data(n_obs=100, seed=3)
    train, test = df.iloc[:60], df.iloc[60:]
    pre = Preprocessor(impute="ffill", scale="zscore")
    pre.fit(train)

    # Train transform should be ~zero mean / unit std by construction.
    zt = pre.transform(train)
    assert np.allclose(zt.mean().to_numpy(), 0.0, atol=1e-8)
    assert np.allclose(zt.std(ddof=0).to_numpy(), 1.0, atol=1e-8)

    # Test transform uses TRAIN stats => generally not exactly zero mean.
    ztest = pre.transform(test)
    assert ztest.shape == test.shape
    assert not np.allclose(ztest.mean().to_numpy(), 0.0, atol=1e-8)


def test_scale_none_passthrough_for_zscored_inputs():
    df, _ = make_regime_data(n_obs=40, seed=4)
    pre = Preprocessor(impute="ffill", scale="none")
    out = pre.fit_transform(df)
    np.testing.assert_allclose(out.to_numpy(), df.to_numpy(), atol=1e-9)


def test_ffill_imputation():
    idx = pd.date_range("2020-01-01", periods=6, freq="MS")
    df = pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0, 5.0, 6.0]}, index=idx)
    pre = Preprocessor(impute="ffill", scale="none")
    out = pre.fit_transform(df)
    assert not out.isna().any().any()
    assert out["a"].iloc[1] == 1.0  # forward filled


def test_clip_sigma_winsorises():
    idx = pd.date_range("2020-01-01", periods=50, freq="MS")
    vals = np.random.default_rng(0).normal(size=50)
    vals[0] = 50.0  # outlier
    df = pd.DataFrame({"a": vals}, index=idx)
    pre = Preprocessor(impute="none", scale="zscore", clip_sigma=3.0)
    out = pre.fit_transform(df)
    assert out["a"].abs().max() <= 3.0 + 1e-9
