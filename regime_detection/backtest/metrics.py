"""Regime backtest metrics.

A subtlety worth stating plainly: HMM regimes are *unsupervised*, so raw
"accuracy" against ground-truth labels is meaningless — regime 2 from the
model may correspond to ground-truth regime 0. We therefore:

* when ground truth exists, solve the optimal label assignment (Hungarian
  algorithm) before scoring accuracy, and also report assignment-free
  agreement (ARI, NMI);
* when it does not (the usual case for live macro regimes), report
  *stability / persistence* diagnostics that measure whether the detected
  path is well-behaved out-of-sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.metrics import transition_count, transition_rate
from ..utils.smoothing import _to_labels, run_length_encode


# --------------------------------------------------------------------- #
# Supervised-style metrics (require ground-truth regime labels)
# --------------------------------------------------------------------- #
def best_match_accuracy(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    return_mapping: bool = False,
):
    """Accuracy under the optimal regime-label permutation (Hungarian).

    Handles the label-switching problem so that detected regimes are scored
    against ground truth up to a relabelling.
    """
    yt, _ = _to_labels(y_true)
    yp, _ = _to_labels(y_pred)
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have equal length.")

    k = int(max(yt.max(), yp.max())) + 1
    cost = np.zeros((k, k))
    for t, p in zip(yt, yp, strict=True):
        cost[t, p] += 1  # confusion counts (to be maximised)

    try:
        from scipy.optimize import linear_sum_assignment

        row, col = linear_sum_assignment(-cost)
        mapping = {int(c): int(r) for r, c in zip(row, col, strict=True)}
    except ImportError:  # greedy fallback if scipy is unavailable
        mapping = {}
        used = set()
        for p in np.argsort(-cost.max(axis=0)):
            best_t = int(np.argmax([cost[t, p] if t not in used else -1 for t in range(k)]))
            mapping[int(p)] = best_t
            used.add(best_t)

    remapped = np.array([mapping.get(int(p), int(p)) for p in yp])
    acc = float((remapped == yt).mean())
    if return_mapping:
        return acc, mapping
    return acc


def adjusted_rand_index(y_true, y_pred) -> float:
    """Adjusted Rand Index — agreement invariant to label permutation."""
    from sklearn.metrics import adjusted_rand_score

    yt, _ = _to_labels(y_true)
    yp, _ = _to_labels(y_pred)
    return float(adjusted_rand_score(yt, yp))


def normalized_mutual_info(y_true, y_pred) -> float:
    """Normalised Mutual Information between true and detected regimes."""
    from sklearn.metrics import normalized_mutual_info_score

    yt, _ = _to_labels(y_true)
    yp, _ = _to_labels(y_pred)
    return float(normalized_mutual_info_score(yt, yp))


# --------------------------------------------------------------------- #
# Unsupervised stability / persistence diagnostics (no ground truth)
# --------------------------------------------------------------------- #
def stability_metrics(labels: np.ndarray | pd.Series) -> dict[str, float]:
    """Persistence diagnostics for an out-of-sample regime path.

    Returns a dict with: number of switches, switch rate, mean/median run
    length, and a "whipsaw rate" (share of runs shorter than 2 periods) —
    a high whipsaw rate signals an unstable, untradeable regime signal.
    """
    arr, _ = _to_labels(labels)
    runs = run_length_encode(arr)
    lengths = np.array([length for _, _, length in runs]) if runs else np.array([0])
    return {
        "n_transitions": float(transition_count(arr)),
        "transition_rate": float(transition_rate(arr)),
        "n_episodes": float(len(runs)),
        "mean_run_length": float(lengths.mean()),
        "median_run_length": float(np.median(lengths)),
        "min_run_length": float(lengths.min()),
        "whipsaw_rate": float((lengths < 2).mean()),
    }


def proba_confidence(proba: np.ndarray | pd.DataFrame) -> dict[str, float]:
    """Confidence diagnostics from posterior probabilities.

    ``mean_max_proba`` near 1 means the model is decisive; a high
    ``mean_entropy`` means it is frequently uncertain which regime holds.
    """
    P = proba.to_numpy() if isinstance(proba, pd.DataFrame) else np.asarray(proba)
    max_p = P.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(P * np.where(P > 0, np.log2(P), 0.0)).sum(axis=1)
    return {
        "mean_max_proba": float(max_p.mean()),
        "min_max_proba": float(max_p.min()),
        "mean_entropy": float(ent.mean()),
    }


def regime_forward_consistency(
    labels: np.ndarray | pd.Series,
    target: pd.Series,
    horizon: int = 1,
) -> pd.DataFrame:
    """Mean forward change in ``target`` conditioned on the current regime.

    A practical sanity check: if regimes are economically meaningful, the
    distribution of next-``horizon`` moves in a target series (e.g. equity
    returns) should differ across regimes. Returns per-regime mean, std and
    count of the forward change.
    """
    arr, idx = _to_labels(labels)
    if idx is None:
        idx = target.index
    tgt = target.reindex(idx)
    fwd = tgt.shift(-horizon) - tgt
    df = pd.DataFrame({"regime": arr, "fwd": fwd.to_numpy()}).dropna()
    return df.groupby("regime")["fwd"].agg(["mean", "std", "count"])
