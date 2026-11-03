"""Regime persistence smoothing.

Raw argmax regime sequences flicker: a one-period blip into another regime
is usually noise, not a genuine switch. These functions impose minimum
dwell time / persistence the way a desk would want it — either on the hard
label sequence or on the posterior probabilities before the argmax.

All functions accept and return pandas objects when given them, preserving
the index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_labels(labels: np.ndarray | pd.Series) -> tuple[np.ndarray, pd.Index | None]:
    if isinstance(labels, pd.Series):
        return labels.to_numpy().astype(int), labels.index
    return np.asarray(labels).astype(int), None


def _wrap(arr: np.ndarray, index: pd.Index | None, name: str = "regime"):
    return pd.Series(arr, index=index, name=name) if index is not None else arr


def run_length_encode(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Encode a label sequence as ``(value, start, length)`` runs."""
    runs: list[tuple[int, int, int]] = []
    if len(labels) == 0:
        return runs
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start]:
            runs.append((int(labels[start]), start, i - start))
            start = i
    runs.append((int(labels[start]), start, len(labels) - start))
    return runs


def min_duration_smoothing(
    labels: np.ndarray | pd.Series, min_duration: int = 3
) -> np.ndarray | pd.Series:
    """Absorb regime runs shorter than ``min_duration`` into their neighbour.

    A short run is reassigned to the label of the *preceding* run (or the
    following run if it occurs at the very start). Applied iteratively until
    no sub-threshold run remains, which keeps the result internally
    consistent when several short runs are adjacent.

    Parameters
    ----------
    labels : array-like of int
    min_duration : int, default 3
        Minimum allowed regime persistence, in periods.
    """
    arr, index = _to_labels(labels)
    if min_duration <= 1 or len(arr) == 0:
        return _wrap(arr.copy(), index)

    out = arr.copy()
    changed = True
    while changed:
        changed = False
        runs = run_length_encode(out)
        for ri, (_val, start, length) in enumerate(runs):
            if length >= min_duration:
                continue
            if ri > 0:
                fill = runs[ri - 1][0]
            elif ri + 1 < len(runs):
                fill = runs[ri + 1][0]
            else:
                continue  # single run spanning everything; leave as-is
            out[start : start + length] = fill
            changed = True
            break  # re-encode after each merge to stay consistent
    return _wrap(out, index)


def majority_vote_smoothing(
    labels: np.ndarray | pd.Series, window: int = 3
) -> np.ndarray | pd.Series:
    """Centred rolling majority vote (window must be odd).

    Cheaper and less aggressive than :func:`min_duration_smoothing`; useful
    as a first pass or when you want a fixed-window filter.
    """
    if window % 2 == 0:
        raise ValueError("window must be odd for a centred majority vote.")
    arr, index = _to_labels(labels)
    n = len(arr)
    half = window // 2
    out = arr.copy()
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        vals, counts = np.unique(arr[lo:hi], return_counts=True)
        out[i] = int(vals[np.argmax(counts)])
    return _wrap(out, index)


def probability_smoothing(
    proba: np.ndarray | pd.DataFrame, alpha: float = 0.3
) -> np.ndarray | pd.DataFrame:
    """Exponential smoothing of posterior probabilities before argmax.

    ``p_t <- alpha * p_t + (1 - alpha) * p_{t-1}``, renormalised. Smaller
    ``alpha`` => stickier regimes. Operating on probabilities (rather than
    hard labels) preserves uncertainty information and tends to switch only
    when the evidence is sustained.

    Parameters
    ----------
    proba : array of shape (n_obs, n_regimes)
    alpha : float in (0, 1], default 0.3
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")
    is_df = isinstance(proba, pd.DataFrame)
    index = proba.index if is_df else None
    cols = proba.columns if is_df else None
    P = (proba.to_numpy() if is_df else np.asarray(proba)).astype(float)

    out = np.empty_like(P)
    out[0] = P[0]
    for t in range(1, len(P)):
        sm = alpha * P[t] + (1 - alpha) * out[t - 1]
        out[t] = sm / sm.sum()
    if is_df:
        return pd.DataFrame(out, index=index, columns=cols)
    return out


def smoothed_labels_from_proba(
    proba: np.ndarray | pd.DataFrame,
    alpha: float = 0.3,
    min_duration: int = 1,
) -> np.ndarray | pd.Series:
    """Convenience: probability-smooth, argmax, then min-duration smooth."""
    sm = probability_smoothing(proba, alpha=alpha)
    if isinstance(sm, pd.DataFrame):
        labels = sm.to_numpy().argmax(axis=1)
        labels = pd.Series(labels, index=sm.index, name="regime")
    else:
        labels = sm.argmax(axis=1)
    if min_duration > 1:
        labels = min_duration_smoothing(labels, min_duration=min_duration)
    return labels
