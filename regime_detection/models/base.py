"""Abstract base class defining the regime-model contract.

Every regime model in this package (HMM, HSMM, sparse jump model) and any
future variant such as Markov-switching or BOCPD implements this interface.
The backtesting engine and pipeline depend *only* on this contract, never
on a concrete model, which is what makes implementations swappable.

Data contract
-------------
``fit`` / ``predict`` / ``predict_proba`` accept either a 2-D ``numpy``
array of shape ``(n_obs, n_features)`` or a ``pandas.DataFrame`` with a
``DatetimeIndex``.  When a DataFrame is passed, label/probability outputs
are returned index-aligned as ``pandas`` objects; when an array is passed,
plain ``numpy`` arrays are returned.  Concrete subclasses get this
alignment for free via the helpers in this module.
"""

from __future__ import annotations

import abc
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FitResult:
    """Lightweight diagnostics returned by :meth:`BaseRegimeModel.fit`."""

    log_likelihood: float
    n_iter: int
    converged: bool
    n_regimes: int
    extra: dict[str, Any] = field(default_factory=dict)


def as_array(X: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, pd.Index | None]:
    """Coerce input to a 2-D float array, returning the original index if any.

    Returns
    -------
    values : np.ndarray of shape (n_obs, n_features)
    index : pandas.Index or None
        The DataFrame index if ``X`` was a DataFrame, else ``None``. Used to
        re-attach alignment to outputs.
    """
    if isinstance(X, pd.DataFrame):
        index = X.index
        values = X.to_numpy(dtype=float)
    elif isinstance(X, pd.Series):
        index = X.index
        values = X.to_numpy(dtype=float).reshape(-1, 1)
    else:
        index = None
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(f"Expected 2-D input (n_obs, n_features), got shape {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("Input contains NaN or inf. Run preprocessing (impute/scale) first.")
    return values, index


def attach_index(
    arr: np.ndarray, index: pd.Index | None, columns: Any = None
) -> np.ndarray | pd.Series | pd.DataFrame:
    """Re-attach a pandas index to a model output, if one was supplied."""
    if index is None:
        return arr
    if arr.ndim == 1:
        return pd.Series(arr, index=index, name="regime")
    return pd.DataFrame(arr, index=index, columns=columns)


class BaseRegimeModel(abc.ABC):
    """Common interface for all regime-detection models.

    Subclasses must implement :meth:`_fit`, :meth:`_predict_proba` and
    :meth:`_transition_matrix`. They operate purely on numpy arrays; the
    public methods handle pandas alignment, validation and state ordering.

    Parameters
    ----------
    n_regimes : int
        Number of latent regimes to model.
    sort_states_by : {"mean", "variance", "none"} or int, default "mean"
        Canonical ordering applied to fitted states so that regime indices
        are stable and interpretable across refits (HMM/EM assigns state
        labels arbitrarily). ``"mean"``/``"variance"`` sort by the per-state
        mean/variance averaged across features; an ``int`` sorts by that
        feature column's per-state mean; ``"none"`` keeps the raw EM order.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        sort_states_by: str | int = "mean",
    ) -> None:
        if n_regimes < 1:
            raise ValueError("n_regimes must be >= 1.")
        self.n_regimes = n_regimes
        self.sort_states_by = sort_states_by
        self._fitted = False
        self._state_order: np.ndarray | None = None
        self.fit_result_: FitResult | None = None

    # ------------------------------------------------------------------ #
    # Methods subclasses must implement (numpy in / numpy out)
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def _fit(self, X: np.ndarray) -> FitResult:
        """Fit the model on a (n_obs, n_features) array."""

    @abc.abstractmethod
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior state probabilities, shape (n_obs, n_regimes)."""

    @abc.abstractmethod
    def _transition_matrix(self) -> np.ndarray:
        """Return the (n_regimes, n_regimes) state transition matrix."""

    @abc.abstractmethod
    def _state_means(self) -> np.ndarray:
        """Per-state feature means, shape (n_regimes, n_features).

        Used for canonical state ordering. Models without an explicit mean
        (e.g. a future BOCPD wrapper) may return zeros and set
        ``sort_states_by="none"``.
        """

    # ------------------------------------------------------------------ #
    # Public API (the contract the rest of the package depends on)
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray | pd.DataFrame, **kwargs: Any) -> BaseRegimeModel:
        values, _ = as_array(X)
        result = self._fit(values, **kwargs)
        self.fit_result_ = result
        self._compute_state_order(values)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        self._check_fitted()
        values, index = as_array(X)
        proba = self._predict_proba(values)
        proba = self._reorder_columns(proba)
        cols = [f"regime_{k}" for k in range(self.n_regimes)]
        return attach_index(proba, index, columns=cols)

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.Series:
        self._check_fitted()
        values, index = as_array(X)
        proba = self._reorder_columns(self._predict_proba(values))
        labels = proba.argmax(axis=1).astype(int)
        return attach_index(labels, index)

    def forecast(self, steps: int = 1, current_state: int | None = None) -> np.ndarray:
        """Forecast the regime distribution ``steps`` periods ahead.

        Parameters
        ----------
        steps : int
            Forecast horizon in periods.
        current_state : int, optional
            If given, condition on a known current regime (a one-hot start
            distribution). Otherwise the model's stationary distribution is
            used as the starting point.

        Returns
        -------
        np.ndarray of shape (steps, n_regimes)
            Row ``h`` is the regime probability vector ``h + 1`` steps ahead.
        """
        self._check_fitted()
        if steps < 1:
            raise ValueError("steps must be >= 1.")
        T = self.transition_matrix
        if current_state is not None:
            dist = np.zeros(self.n_regimes)
            dist[current_state] = 1.0
        else:
            dist = self.stationary_distribution
        out = np.empty((steps, self.n_regimes))
        for h in range(steps):
            dist = dist @ T
            out[h] = dist
        return out

    # ------------------------------------------------------------------ #
    # Derived quantities
    # ------------------------------------------------------------------ #
    @property
    def transition_matrix(self) -> np.ndarray:
        self._check_fitted()
        return self._reorder_matrix(self._transition_matrix())

    @property
    def state_means(self) -> np.ndarray:
        """Per-state feature means (n_regimes, n_features), canonical order."""
        self._check_fitted()
        means = np.asarray(self._state_means())
        return means[self._state_order] if self._state_order is not None else means

    @property
    def stationary_distribution(self) -> np.ndarray:
        """Stationary distribution (left eigenvector of T for eigenvalue 1)."""
        T = self.transition_matrix
        vals, vecs = np.linalg.eig(T.T)
        idx = np.argmin(np.abs(vals - 1.0))
        pi = np.real(vecs[:, idx])
        pi = pi / pi.sum()
        return np.clip(pi, 0.0, None) / np.clip(pi, 0.0, None).sum()

    @property
    def expected_durations(self) -> np.ndarray:
        """Expected dwell time per regime: 1 / (1 - p_ii)."""
        diag = np.clip(np.diag(self.transition_matrix), None, 1 - 1e-12)
        return 1.0 / (1.0 - diag)

    # ------------------------------------------------------------------ #
    # State ordering (solves the label-switching problem across refits)
    # ------------------------------------------------------------------ #
    def _compute_state_order(self, X: np.ndarray) -> None:
        if self.sort_states_by == "none" or self.n_regimes == 1:
            self._state_order = np.arange(self.n_regimes)
            return
        means = self._state_means()
        if self.sort_states_by == "mean":
            key = means.mean(axis=1)
        elif self.sort_states_by == "variance":
            # Fall back to dispersion of state means across features.
            key = means.var(axis=1)
        elif isinstance(self.sort_states_by, int):
            key = means[:, self.sort_states_by]
        else:
            raise ValueError(f"Unknown sort_states_by={self.sort_states_by!r}.")
        self._state_order = np.argsort(key)

    def _reorder_columns(self, proba: np.ndarray) -> np.ndarray:
        if self._state_order is None:
            return proba
        return proba[:, self._state_order]

    def _reorder_matrix(self, T: np.ndarray) -> np.ndarray:
        if self._state_order is None:
            return T
        order = self._state_order
        return T[np.ix_(order, order)]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str | Path) -> BaseRegimeModel:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, BaseRegimeModel):
            raise TypeError("Loaded object is not a BaseRegimeModel.")
        return obj

    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call .fit(X) first.")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self._fitted else "unfitted"
        return f"{type(self).__name__}(n_regimes={self.n_regimes}, {state})"
