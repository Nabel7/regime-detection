"""Standard Gaussian Hidden Markov Model for regime detection.

Wraps :class:`hmmlearn.hmm.GaussianHMM` behind the package's
:class:`~regime_detection.models.base.BaseRegimeModel` contract, adding the
things a production setup actually needs:

* **Random restarts.** EM is non-convex and seed-sensitive; we fit
  ``n_init`` times and keep the highest-likelihood solution.
* **Canonical state ordering.** Inherited from the base class so regime
  indices stay interpretable across walk-forward refits.
* **Reproducibility.** A single ``random_state`` controls all restarts.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .base import BaseRegimeModel, FitResult

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "hmmlearn is required for the HMM model. Install with `pip install hmmlearn`."
    ) from exc


class GaussianHMMModel(BaseRegimeModel):
    """Gaussian-emission HMM regime detector.

    Parameters
    ----------
    n_regimes : int, default 2
        Number of latent regimes (HMM hidden states).
    covariance_type : {"full", "diag", "tied", "spherical"}, default "full"
        Covariance structure of the Gaussian emissions. ``"full"`` is most
        expressive; ``"diag"`` is a robust default when observations are
        few relative to features (common with monthly macro data).
    n_init : int, default 10
        Number of random restarts. The best log-likelihood is kept.
    n_iter : int, default 200
        Max EM iterations per restart.
    tol : float, default 1e-4
        EM convergence tolerance.
    random_state : int, default 0
        Seed controlling restart initialisations (reproducibility).
    sort_states_by : str or int, default "mean"
        See :class:`BaseRegimeModel`.
    min_covar : float, default 1e-3
        Floor added to the diagonal of covariance matrices for numerical
        stability — important for short macro samples.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        covariance_type: str = "full",
        n_init: int = 10,
        n_iter: int = 200,
        tol: float = 1e-4,
        random_state: int = 0,
        sort_states_by: str | int = "mean",
        min_covar: float = 1e-3,
    ) -> None:
        super().__init__(n_regimes=n_regimes, sort_states_by=sort_states_by)
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.min_covar = min_covar
        self.model_: GaussianHMM | None = None

    def _build(self, seed: int) -> GaussianHMM:
        return GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=seed,
            min_covar=self.min_covar,
            init_params="stmc",
        )

    def _fit(self, X: np.ndarray, lengths: list[int] | None = None) -> FitResult:
        """Fit with random restarts, keeping the best log-likelihood.

        Parameters
        ----------
        lengths : list[int], optional
            Sequence lengths for fitting multiple independent series at once
            (hmmlearn convention). ``sum(lengths)`` must equal ``len(X)``.
        """
        best_model: GaussianHMM | None = None
        best_ll = -np.inf
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31 - 1, size=self.n_init)

        for seed in seeds:
            model = self._build(int(seed))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model.fit(X, lengths=lengths)
                    ll = model.score(X, lengths=lengths)
                except (ValueError, np.linalg.LinAlgError):
                    continue
            if np.isfinite(ll) and ll > best_ll:
                best_ll = ll
                best_model = model

        if best_model is None:
            raise RuntimeError(
                "All EM restarts failed to converge. Check for degenerate "
                "features (constant columns), too many regimes for the sample "
                "size, or try covariance_type='diag'."
            )

        self.model_ = best_model
        return FitResult(
            log_likelihood=float(best_ll),
            n_iter=int(getattr(best_model.monitor_, "iter", self.n_iter)),
            converged=bool(best_model.monitor_.converged),
            n_regimes=self.n_regimes,
            extra={"n_features": X.shape[1], "n_obs": X.shape[0]},
        )

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model_ is not None
        return self.model_.predict_proba(X)

    def _transition_matrix(self) -> np.ndarray:
        assert self.model_ is not None
        return np.asarray(self.model_.transmat_)

    def _state_means(self) -> np.ndarray:
        assert self.model_ is not None
        return np.asarray(self.model_.means_)

    @property
    def covariances(self) -> np.ndarray:
        """Per-state covariance matrices, reordered to canonical state order."""
        self._check_fitted()
        assert self.model_ is not None
        cov = self.model_.covars_
        if self._state_order is not None:
            cov = cov[self._state_order]
        return np.asarray(cov)

    @property
    def means(self) -> np.ndarray:
        """Per-state feature means, reordered to canonical state order."""
        self._check_fitted()
        means = self._state_means()
        if self._state_order is not None:
            means = means[self._state_order]
        return np.asarray(means)

    def score(self, X: np.ndarray | Any) -> float:
        """Log-likelihood of ``X`` under the fitted model (for model selection)."""
        from .base import as_array

        self._check_fitted()
        assert self.model_ is not None
        values, _ = as_array(X)
        return float(self.model_.score(values))
