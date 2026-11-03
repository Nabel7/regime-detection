"""End-to-end regime detection pipeline.

Bundles the three steps of regime detection — preprocess, fit/detect, smooth
for persistence — behind one object with a clean fit/predict surface, while
leaving each component swappable. The backtester drives this same object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data.preprocessing import Preprocessor
from .models.base import BaseRegimeModel
from .models.hmm import GaussianHMMModel
from .utils.metrics import (
    empirical_transition_matrix,
    label_regimes_by_feature,
    regime_statistics,
)
from .utils.smoothing import min_duration_smoothing, probability_smoothing


@dataclass
class RegimeResult:
    """Output bundle from a pipeline ``fit_detect`` call."""

    labels: pd.Series
    proba: pd.DataFrame
    smoothed_labels: pd.Series
    regime_names: dict[int, str]
    statistics: pd.DataFrame
    model: BaseRegimeModel


class RegimeDetectionPipeline:
    """Compose preprocessing, a regime model and persistence smoothing.

    Parameters
    ----------
    model : BaseRegimeModel, optional
        Any model implementing the base contract. Defaults to a 2-regime
        Gaussian HMM.
    preprocessor : Preprocessor, optional
        Defaults to forward-fill imputation with no scaling (suits already
        z-scored macro inputs). Pass ``scale="zscore"`` for raw series.
    prob_smooth_alpha : float, optional
        If set, exponentially smooth posteriors before argmax.
    min_duration : int, default 1
        Minimum regime persistence enforced on the final label path.
    label_feature : int or str, optional
        Feature used to name regimes (e.g. PMI -> contraction/expansion).
    label_names : tuple[str, ...], optional
        Names ordered from lowest to highest ``label_feature`` mean.
    """

    def __init__(
        self,
        model: BaseRegimeModel | None = None,
        preprocessor: Preprocessor | None = None,
        prob_smooth_alpha: float | None = None,
        min_duration: int = 1,
        label_feature: int | str | None = None,
        label_names: tuple[str, ...] | None = None,
    ) -> None:
        self.model = model if model is not None else GaussianHMMModel(n_regimes=2)
        self.preprocessor = preprocessor if preprocessor is not None else Preprocessor(scale="none")
        self.prob_smooth_alpha = prob_smooth_alpha
        self.min_duration = min_duration
        self.label_feature = label_feature
        self.label_names = label_names
        self._feature_names: list[str] | None = None

    # ------------------------------------------------------------------ #
    def fit_detect(self, X: pd.DataFrame, **fit_kwargs: Any) -> RegimeResult:
        """Fit on ``X`` and return regimes for the same (in-sample) window."""
        self._feature_names = list(X.columns)
        Xp = self.preprocessor.fit_transform(X)
        self.model.fit(Xp, **fit_kwargs)
        return self._detect(Xp, X)

    def detect(self, X: pd.DataFrame) -> RegimeResult:
        """Apply an already-fitted pipeline to new data (out-of-sample)."""
        Xp = self.preprocessor.transform(X)
        return self._detect(Xp, X)

    def filter_detect(self, X_window: pd.DataFrame) -> RegimeResult:
        """Causal (filtered) detection over a window — no look-ahead.

        Uses the model's forward filter instead of the forward–backward
        smoother, so the label at each timestamp depends only on observations
        up to and including it. Pass a window that begins at (or before) the
        training start so the filter is warm-started across the train→test
        boundary; the caller then slices out the out-of-sample rows.

        Requires a model exposing ``filter_proba`` (e.g. ``GaussianHSMMModel``);
        the smoother-only ``GaussianHMMModel``/``SparseJumpModel`` will raise.
        """
        if not hasattr(self.model, "filter_proba"):
            raise TypeError(
                f"{type(self.model).__name__} has no causal filter_proba; "
                "causal detection requires an explicit-duration HSMM."
            )
        Xp = self.preprocessor.transform(X_window)
        return self._detect(Xp, X_window, causal=True)

    def _detect(self, Xp: pd.DataFrame, X_raw: pd.DataFrame, causal: bool = False) -> RegimeResult:
        proba = self.model.filter_proba(Xp) if causal else self.model.predict_proba(Xp)
        if self.prob_smooth_alpha is not None:
            proba = probability_smoothing(proba, alpha=self.prob_smooth_alpha)

        labels = pd.Series(proba.to_numpy().argmax(axis=1), index=proba.index, name="regime")
        smoothed = labels
        # min-duration smoothing rewrites short runs using their *future* — not
        # causal — so it is skipped when detection is filtered (real-time).
        if self.min_duration > 1 and not causal:
            smoothed = min_duration_smoothing(labels, min_duration=self.min_duration)

        regime_names = self._build_names()
        stats = regime_statistics(
            smoothed, n_regimes=self.model.n_regimes, regime_names=regime_names
        )
        return RegimeResult(
            labels=labels,
            proba=proba,
            smoothed_labels=smoothed,
            regime_names=regime_names,
            statistics=stats,
            model=self.model,
        )

    # ------------------------------------------------------------------ #
    def _build_names(self) -> dict[int, str]:
        if self.label_feature is None or self.label_names is None:
            return {k: f"regime_{k}" for k in range(self.model.n_regimes)}
        feat_idx = self.label_feature
        if isinstance(feat_idx, str):
            if self._feature_names is None or feat_idx not in self._feature_names:
                return {k: f"regime_{k}" for k in range(self.model.n_regimes)}
            feat_idx = self._feature_names.index(feat_idx)
        return label_regimes_by_feature(
            self.model.state_means, feature_index=feat_idx, names=self.label_names
        )

    def empirical_transitions(self, result: RegimeResult) -> np.ndarray:
        """Realised transition matrix from a result's smoothed path."""
        return empirical_transition_matrix(result.smoothed_labels, n_regimes=self.model.n_regimes)
