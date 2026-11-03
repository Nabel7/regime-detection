"""Smoothing, transition/entropy metrics and plotting utilities."""

from .metrics import (
    empirical_transition_matrix,
    expected_durations,
    label_regimes_by_feature,
    regime_statistics,
    stationary_distribution,
    transition_count,
    transition_entropy,
    transition_rate,
)
from .smoothing import (
    majority_vote_smoothing,
    min_duration_smoothing,
    probability_smoothing,
    run_length_encode,
    smoothed_labels_from_proba,
)

__all__ = [
    "majority_vote_smoothing",
    "min_duration_smoothing",
    "probability_smoothing",
    "run_length_encode",
    "smoothed_labels_from_proba",
    "empirical_transition_matrix",
    "expected_durations",
    "label_regimes_by_feature",
    "regime_statistics",
    "stationary_distribution",
    "transition_count",
    "transition_entropy",
    "transition_rate",
]
