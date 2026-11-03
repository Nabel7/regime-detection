"""Sparse Jump Model (SJM) for regime detection.

Wraps :class:`jumpmodels.sparse_jump.SparseJumpModel` behind the package's
:class:`~regime_detection.models.base.BaseRegimeModel` contract.

The SJM replaces EM/likelihood maximisation with a penalised clustering
objective: minimise within-cluster dispersion plus a jump penalty that
charges for every state transition.  This gives two properties the Gaussian
HMM lacks:

* **Explicit stability control** — the jump penalty λ is interpretable and
  directly governs how often the model is allowed to switch regimes.
* **Likelihood-free** — no parametric emission assumption, so the model does
  not break when active returns violate Gaussianity.

Intended use: per-factor regime inference on *active* returns (factor minus
market).  Active returns are decorrelated from the shared market beta, so each
factor's regime signal is cleaner and the two-state (bear/bull) setup is
natural.

Reference: Shu & Mulvey (2024), "Dynamic Factor Allocation Leveraging
Regime-Switching Signals", arXiv:2410.14841.
"""

from __future__ import annotations

import numpy as np

from .base import BaseRegimeModel, FitResult

try:
    from jumpmodels.sparse_jump import SparseJumpModel as _SJM
except ImportError as exc:
    raise ImportError(
        "jumpmodels is required for SparseJumpModel. Install with `pip install jumpmodels`."
    ) from exc


class SparseJumpModel(BaseRegimeModel):
    """Sparse Jump Model regime detector.

    Parameters
    ----------
    n_regimes : int, default 2
        Number of latent regimes (K states).  Two states (bear/bull) is the
        natural choice for per-factor active-return inference.
    jump_penalty : float, default 16.0
        Transition penalty λ.  Higher values produce more persistent regimes.
        The library scales this internally by ``1 / sqrt(n_features)`` so the
        effective penalty is comparable across different feature dimensions.
        Shu & Mulvey (2024) search λ ∈ {4, 8, 16, 32}; 16 is a reasonable
        starting point.
    max_feats : float, default 100.0
        Sparse feature selection threshold (the κ² parameter in the paper).
        Represents the effective number of features selected.  Setting this
        above the actual feature count disables selection — all features are
        used with equal initial weight, then refined by the lasso step.
    cont : bool, default False
        If True, use the continuous jump model (soft assignments on the
        probability simplex).  The discrete model is standard.
    n_init : int, default 10
        Number of random initialisations for the inner JumpModel.  Kept for
        interface parity with GaussianHMMModel; the best objective is kept.
    max_iter : int, default 30
        Outer coordinate-descent iterations for the feature-weight optimisation
        (SJM-specific).  Irrelevant when ``max_feats`` disables selection.
    random_state : int, default 0
        Seed for reproducibility across random initialisations.
    sort_states_by : {"mean", "variance", "none"} or int, default "mean"
        Canonical state ordering applied after fitting.  ``"mean"`` sorts by
        average centroid value across features — for active-return features
        this places the bear (negative) regime at index 0 and bull at index 1.
        Inherited from :class:`BaseRegimeModel`.
    online_predict : bool, default False
        If True, :meth:`predict_proba` uses the causal online E-step (each
        observation is classified using only prior-in-window data).  If False
        (default), the standard Viterbi E-step runs over the full input window,
        which is appropriate when the walk-forward backtester already separates
        train and test data at the fold level.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        jump_penalty: float = 16.0,
        max_feats: float = 100.0,
        cont: bool = False,
        n_init: int = 10,
        max_iter: int = 30,
        random_state: int = 0,
        sort_states_by: str | int = "mean",
        online_predict: bool = False,
    ) -> None:
        super().__init__(n_regimes=n_regimes, sort_states_by=sort_states_by)
        self.jump_penalty = jump_penalty
        self.max_feats = max_feats
        self.cont = cont
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.online_predict = online_predict
        self._model: _SJM | None = None

    # ------------------------------------------------------------------ #
    # BaseRegimeModel interface
    # ------------------------------------------------------------------ #

    def _fit(self, X: np.ndarray) -> FitResult:
        model = _SJM(
            n_components=self.n_regimes,
            jump_penalty=self.jump_penalty,
            max_feats=self.max_feats,
            cont=self.cont,
            n_init_jm=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        model.fit(X)
        self._model = model

        n_features = X.shape[1]
        n_obs = X.shape[0]
        obj = float(model.jm_ins.val_)

        return FitResult(
            # SJM has no log-likelihood; store negative objective (higher = better)
            # so the FitResult convention (higher = better) is preserved.
            log_likelihood=-obj,
            n_iter=0,  # not exposed by the library
            converged=True,
            n_regimes=self.n_regimes,
            extra={
                "n_features": n_features,
                "n_obs": n_obs,
                "objective_value": obj,
                "jump_penalty": self.jump_penalty,
                "feat_weights": np.asarray(model.feat_weights),
            },
        )

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None
        if self.online_predict:
            result = self._model.predict_proba_online(X)
        else:
            result = self._model.predict_proba(X)
        return np.asarray(result)

    def _transition_matrix(self) -> np.ndarray:
        assert self._model is not None
        # transmat_ is on the inner JumpModel, computed from its fitted labels.
        return np.asarray(self._model.jm_ins.transmat_)

    def _state_means(self) -> np.ndarray:
        assert self._model is not None
        # centers_ are the weighted cluster centroids (n_regimes, n_features).
        return np.asarray(self._model.centers_)

    # ------------------------------------------------------------------ #
    # SJM-specific properties
    # ------------------------------------------------------------------ #

    @property
    def feat_weights(self) -> np.ndarray:
        """Feature weights optimised by the sparse penalty (shape: n_features).

        Values near zero indicate features that contributed little to the
        regime separation; values near 1 indicate the most discriminative
        features.  Uniform when ``max_feats`` exceeds the feature count.
        """
        self._check_fitted()
        assert self._model is not None
        return np.asarray(self._model.feat_weights)

    @property
    def centers(self) -> np.ndarray:
        """Per-state centroids in the weighted feature space (n_regimes, n_features),
        reordered to canonical state order."""
        self._check_fitted()
        assert self._model is not None
        centers = self._model.centers_
        if self._state_order is not None:
            centers = centers[self._state_order]
        return np.asarray(centers)

    @property
    def objective_value(self) -> float:
        """Optimal value of the SJM loss (within-cluster dispersion + jump cost)."""
        self._check_fitted()
        assert self._model is not None
        return float(self._model.jm_ins.val_)
