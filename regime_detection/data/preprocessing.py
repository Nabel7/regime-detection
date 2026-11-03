"""Preprocessing transformers: scaling and missing-data handling.

The key design point here is **no look-ahead**. In a walk-forward setting,
statistics used to scale or impute must be estimated on the training window
only and then applied to the test window. Every transformer below follows
the scikit-learn ``fit`` / ``transform`` split so the backtester can fit on
train and transform test without leakage.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


class Preprocessor:
    """Configurable, leakage-safe preprocessing pipeline.

    Order of operations on ``transform``: impute missing values, then scale.
    Statistics (means, std, fill values) are learned in ``fit`` only.

    Parameters
    ----------
    impute : {"ffill", "mean", "median", "interpolate", "drop", "none"}, default "ffill"
        Strategy for missing values. ``"ffill"`` suits publication-lagged
        macro data; ``"drop"`` removes rows with any NaN.
    scale : {"zscore", "minmax", "none"}, default "zscore"
        Feature scaling. Use ``"none"`` if inputs are already standardised
        (e.g. the ``*_z`` columns in this project).
    clip_sigma : float, optional
        If set, winsorise scaled features to +/- this many standard deviations
        (robustness against macro outliers / data spikes).
    """

    def __init__(
        self,
        impute: Literal["ffill", "mean", "median", "interpolate", "drop", "none"] = "ffill",
        scale: Literal["zscore", "minmax", "none"] = "zscore",
        clip_sigma: float | None = None,
    ) -> None:
        self.impute = impute
        self.scale = scale
        self.clip_sigma = clip_sigma
        self._columns: list[str] | None = None
        self._fill: pd.Series | None = None
        self._center: np.ndarray | None = None
        self._scale_: np.ndarray | None = None
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> Preprocessor:
        df = self._as_frame(X)
        self._columns = list(df.columns)

        # Learn imputation fill values on the training window only.
        if self.impute in ("mean",):
            self._fill = df.mean()
        elif self.impute in ("median",):
            self._fill = df.median()
        else:
            self._fill = None

        imputed = self._apply_impute(df)

        if self.scale == "zscore":
            self._center = imputed.mean().to_numpy()
            std = imputed.std(ddof=0).to_numpy()
            self._scale_ = np.where(std == 0, 1.0, std)
        elif self.scale == "minmax":
            self._center = imputed.min().to_numpy()
            rng = (imputed.max() - imputed.min()).to_numpy()
            self._scale_ = np.where(rng == 0, 1.0, rng)
        elif self.scale == "none":
            self._center = np.zeros(imputed.shape[1])
            self._scale_ = np.ones(imputed.shape[1])
        else:
            raise ValueError(f"Unknown scale={self.scale!r}.")

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Preprocessor is not fitted. Call .fit(X) first.")
        df = self._as_frame(X)
        if list(df.columns) != self._columns:
            df = df[self._columns]  # enforce training column order
        imputed = self._apply_impute(df)
        scaled = (imputed.to_numpy() - self._center) / self._scale_
        if self.clip_sigma is not None and self.scale == "zscore":
            scaled = np.clip(scaled, -self.clip_sigma, self.clip_sigma)
        return pd.DataFrame(scaled, index=imputed.index, columns=self._columns)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------ #
    def _apply_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.impute == "ffill":
            return df.ffill().bfill()
        if self.impute == "interpolate":
            return df.interpolate(method="time").ffill().bfill()
        if self.impute in ("mean", "median"):
            return df.fillna(self._fill)
        if self.impute == "drop":
            return df.dropna(axis=0, how="any")
        if self.impute == "none":
            return df
        raise ValueError(f"Unknown impute={self.impute!r}.")

    @staticmethod
    def _as_frame(X: pd.DataFrame | pd.Series) -> pd.DataFrame:
        if isinstance(X, pd.Series):
            return X.to_frame()
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Preprocessor expects a pandas DataFrame/Series.")
        return X.copy()
