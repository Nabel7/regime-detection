"""Tests for data loading, validation and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from regime_detection.data import (
    DataValidationError,
    align_frequencies,
    load_csv,
    make_regime_data,
    validate_frame,
)


def test_validate_frame_ok():
    df, _ = make_regime_data(n_obs=50, seed=1)
    report = validate_frame(df)
    assert report.n_obs == 50
    assert report.n_features == 5
    assert report.inferred_freq is not None


def test_validate_requires_datetime_index():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(DataValidationError):
        validate_frame(df)


def test_validate_rejects_duplicate_index():
    idx = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01"])
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
    with pytest.raises(DataValidationError):
        validate_frame(df)


def test_validate_rejects_too_much_missing():
    idx = pd.date_range("2020-01-01", periods=10, freq="MS")
    df = pd.DataFrame({"a": [1.0] + [np.nan] * 9}, index=idx)
    with pytest.raises(DataValidationError):
        validate_frame(df, max_missing_frac=0.5)


def test_load_csv_roundtrip(tmp_path):
    df, _ = make_regime_data(n_obs=24, seed=2)
    p = tmp_path / "macro.csv"
    df.to_csv(p, index_label="date")
    loaded = load_csv(p, date_col="date", freq="MS")
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert list(loaded.columns) == list(df.columns)
    np.testing.assert_allclose(loaded.to_numpy(), df.to_numpy(), rtol=1e-6)


def test_align_frequencies_mixed_cadence():
    monthly = pd.DataFrame(
        {"pmi": np.arange(12.0)},
        index=pd.date_range("2020-01-01", periods=12, freq="MS"),
    )
    quarterly = pd.DataFrame(
        {"growth": [1.0, 2.0, 3.0, 4.0]},
        index=pd.date_range("2020-01-01", periods=4, freq="QS"),
    )
    aligned = align_frequencies({"m": monthly, "q": quarterly}, target_freq="MS")
    assert len(aligned) == 12
    # Quarterly value should be forward-filled across the quarter.
    assert aligned["q__growth"].iloc[1] == aligned["q__growth"].iloc[0]
