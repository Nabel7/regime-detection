"""Transition analysis, entropy and regime statistics.

These operate on a realised label sequence (e.g. the smoothed regime path),
independent of which model produced it — so they apply equally to every
model in the package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .smoothing import _to_labels, run_length_encode


def empirical_transition_matrix(
    labels: np.ndarray | pd.Series, n_regimes: int | None = None
) -> np.ndarray:
    """Count-based transition matrix from a realised label path.

    Row-normalised: entry ``[i, j]`` is the empirical P(next=j | curr=i).
    Differs from a model's ``transmat_`` — this is what *actually* happened
    in the data, useful for validating the model against realised dynamics.
    """
    arr, _ = _to_labels(labels)
    k = n_regimes if n_regimes is not None else int(arr.max()) + 1
    counts = np.zeros((k, k), dtype=float)
    for a, b in zip(arr[:-1], arr[1:], strict=True):
        counts[a, b] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.where(row_sums > 0, counts / row_sums, 0.0)
    return T


def stationary_distribution(T: np.ndarray) -> np.ndarray:
    """Stationary distribution of a row-stochastic transition matrix."""
    vals, vecs = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(vals - 1.0))
    pi = np.real(vecs[:, idx])
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


def expected_durations(T: np.ndarray) -> np.ndarray:
    """Expected dwell time per regime from the diagonal: 1 / (1 - p_ii)."""
    diag = np.clip(np.diag(T), None, 1 - 1e-12)
    return 1.0 / (1.0 - diag)


def transition_entropy(T: np.ndarray) -> np.ndarray:
    """Per-regime Shannon entropy (bits) of the outgoing transition row.

    High entropy => an unstable regime that scatters to many others; low
    entropy (near 0) => a sticky / absorbing regime.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(T > 0, np.log2(T), 0.0)
    return -(T * logs).sum(axis=1)


def regime_statistics(
    labels: np.ndarray | pd.Series,
    n_regimes: int | None = None,
    regime_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Per-regime summary: count, frequency, run statistics.

    Returns one row per regime with: number of observations, share of
    sample, number of distinct episodes (runs), and mean/median/max run
    length in periods.
    """
    arr, _ = _to_labels(labels)
    k = n_regimes if n_regimes is not None else int(arr.max()) + 1
    runs = run_length_encode(arr)

    rows = []
    for r in range(k):
        mask = arr == r
        run_lengths = [length for val, _, length in runs if val == r]
        rows.append(
            {
                "regime": r,
                "name": (regime_names or {}).get(r, f"regime_{r}"),
                "n_obs": int(mask.sum()),
                "frequency": float(mask.mean()),
                "n_episodes": len(run_lengths),
                "mean_duration": float(np.mean(run_lengths)) if run_lengths else 0.0,
                "median_duration": float(np.median(run_lengths)) if run_lengths else 0.0,
                "max_duration": int(np.max(run_lengths)) if run_lengths else 0,
            }
        )
    return pd.DataFrame(rows).set_index("regime")


def label_regimes_by_feature(
    means: np.ndarray,
    feature_index: int = 0,
    names: tuple[str, ...] = ("contraction", "expansion"),
) -> dict[int, str]:
    """Map regime ids to human labels by ranking a feature's per-state mean.

    Example: rank the PMI mean across regimes and call the lowest
    "contraction" and the highest "expansion". For >2 regimes pass a longer
    ``names`` tuple ordered from lowest to highest feature value.
    """
    order = np.argsort(means[:, feature_index])
    if len(names) != len(order):
        raise ValueError(f"Got {len(order)} regimes but {len(names)} names; lengths must match.")
    return {int(regime): names[rank] for rank, regime in enumerate(order)}


def transition_count(labels: np.ndarray | pd.Series) -> int:
    """Total number of regime switches in the path."""
    arr, _ = _to_labels(labels)
    return int((np.diff(arr) != 0).sum())


def transition_rate(labels: np.ndarray | pd.Series) -> float:
    """Fraction of consecutive periods that are regime switches."""
    arr, _ = _to_labels(labels)
    if len(arr) < 2:
        return 0.0
    return transition_count(arr) / (len(arr) - 1)
