"""Gaussian Hidden Semi-Markov Model (HSMM / SHMM) for regime detection.

A standard HMM ties each regime's dwell time to the geometric distribution
implied by its self-transition probability ``p_ii`` — the only shape it can
express is memoryless decay (``expected_duration = 1 / (1 - p_ii)``).  Macro
regimes rarely behave that way: expansions and contractions have a
characteristic *length*, and the hazard of leaving usually rises with time
spent in the regime.  A **Hidden Semi-Markov Model** removes this constraint
by modelling each regime's duration with an *explicit* distribution, so the
model can learn "this regime typically lasts ~9 months" rather than being
forced into geometric decay.

This is the classic **explicit-duration HMM** (EDHMM).  It is fitted here from
scratch with EM — no external HSMM library — using a scaled forward-backward
recursion on the *residual-time* expanded state space ``(regime, remaining
duration)``.  That representation makes the semi-Markov dynamics an ordinary
HMM on ``K * D`` states with tied emissions, which keeps the recursion
transparent and O(T * K * D) per iteration.

Design choices, matching the rest of the package
------------------------------------------------
* **Explicit durations.** Each regime ``i`` carries a duration pmf ``p_i(d)``
  over ``d = 1 .. max_duration``.  ``duration_dist="poisson"`` (default) fits a
  one-parameter (mean dwell time) shape — parsimonious and robust on short
  macro samples; ``"nonparametric"`` fits the full pmf when data are ample.
* **Random restarts.** EM is non-convex; the model fits ``n_init`` times from
  distinct k-means initialisations and keeps the best log-likelihood.
* **Canonical state ordering.** Inherited from :class:`BaseRegimeModel`, so
  regime indices stay interpretable across walk-forward refits.
* **Framework transition matrix.** The semi-Markov process has no single
  Markov transition matrix; :meth:`_transition_matrix` returns the
  *duration-embedded effective chain* whose diagonal reproduces each regime's
  mean dwell time, so ``transition_matrix``, ``stationary_distribution``,
  ``expected_durations`` and ``forecast`` all behave sensibly.  The raw
  embedded (zero-diagonal) transitions and the explicit duration pmfs are
  exposed as HSMM-specific properties, and ``forecast(..., elapsed=d)``
  propagates the full duration model instead of the mean dwell time.

References
----------
* Ferguson, J. D. (1980). "Variable duration models for speech."
* Rabiner, L. R. (1989). "A tutorial on hidden Markov models...", §III
  (explicit duration / hidden semi-Markov models).  Proc. IEEE 77(2).
* Yu, S-Z. (2010). "Hidden semi-Markov models." Artificial Intelligence 174.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseRegimeModel, FitResult, as_array, attach_index

try:
    from scipy.stats import poisson as _poisson
    from sklearn.cluster import KMeans
except ImportError as exc:  # pragma: no cover
    raise ImportError("scipy and scikit-learn are required for GaussianHSMMModel.") from exc

_LOG_2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------- #
# Gaussian emission log-density
# ---------------------------------------------------------------------- #
def _log_gaussian(
    X: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    covariance_type: str,
    min_covar: float,
) -> np.ndarray:
    """Per-state Gaussian log-density, shape ``(n_obs, n_regimes)``."""
    n_obs, n_feat = X.shape
    n_states = means.shape[0]
    logB = np.empty((n_obs, n_states))
    if covariance_type == "diag":
        for k in range(n_states):
            var = covars[k] + min_covar
            diff = X - means[k]
            logB[:, k] = -0.5 * (
                n_feat * _LOG_2PI + np.log(var).sum() + ((diff * diff) / var).sum(axis=1)
            )
    elif covariance_type == "full":
        eye = min_covar * np.eye(n_feat)
        for k in range(n_states):
            cov = covars[k] + eye
            chol = np.linalg.cholesky(cov)
            logdet = 2.0 * np.log(np.diag(chol)).sum()
            solved = np.linalg.solve(chol, (X - means[k]).T)  # (n_feat, n_obs)
            maha = (solved * solved).sum(axis=0)
            logB[:, k] = -0.5 * (n_feat * _LOG_2PI + logdet + maha)
    else:
        raise ValueError(f"Unknown covariance_type={covariance_type!r}.")
    return logB


class GaussianHSMMModel(BaseRegimeModel):
    """Gaussian explicit-duration Hidden Semi-Markov Model.

    Parameters
    ----------
    n_regimes : int, default 2
        Number of latent regimes (K).
    covariance_type : {"diag", "full"}, default "diag"
        Covariance structure of the Gaussian emissions.  ``"diag"`` is the
        robust default for short, wide macro panels; ``"full"`` captures
        cross-feature correlation when observations are plentiful.
    duration_dist : {"poisson", "nonparametric"}, default "poisson"
        Explicit duration model.  ``"poisson"`` fits one parameter per regime
        (the mean dwell time) and is the safer choice on limited macro data;
        ``"nonparametric"`` fits the full duration pmf and is more flexible
        when the sample is large.
    max_duration : int, optional
        Support ``D`` of the duration distribution (max dwell time in periods).
        Defaults to ``min(n_obs, max(10, n_obs // 4))`` at fit time — large
        enough to cover realistic regime lengths without wasting compute.
    n_init : int, default 10
        Number of k-means-seeded EM restarts; the best log-likelihood is kept.
    n_iter : int, default 100
        Max EM iterations per restart.
    tol : float, default 1e-4
        EM convergence tolerance on the per-observation log-likelihood.
    random_state : int, default 0
        Seed controlling restart initialisations (reproducibility).
    sort_states_by : {"mean", "variance", "none"} or int, default "mean"
        Canonical state ordering (see :class:`BaseRegimeModel`).
    min_covar : float, default 1e-3
        Floor added to emission variances for numerical stability on short
        macro samples.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        covariance_type: str = "diag",
        duration_dist: str = "poisson",
        max_duration: int | None = None,
        n_init: int = 10,
        n_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 0,
        sort_states_by: str | int = "mean",
        min_covar: float = 1e-3,
    ) -> None:
        super().__init__(n_regimes=n_regimes, sort_states_by=sort_states_by)
        if covariance_type not in ("diag", "full"):
            raise ValueError("covariance_type must be 'diag' or 'full'.")
        if duration_dist not in ("poisson", "nonparametric"):
            raise ValueError("duration_dist must be 'poisson' or 'nonparametric'.")
        self.covariance_type = covariance_type
        self.duration_dist = duration_dist
        self.max_duration = max_duration
        self.n_init = n_init
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.min_covar = min_covar

        # Fitted parameters (canonical, unordered — ordering applied on access).
        self.startprob_: np.ndarray | None = None  # (K,)
        self.transmat_: np.ndarray | None = None  # (K, K) embedded, zero diag
        self.durations_: np.ndarray | None = None  # (K, D) duration pmf
        self.means_: np.ndarray | None = None  # (K, n_features)
        self.covars_: np.ndarray | None = None  # (K, n_features[, n_features])
        self._D: int | None = None

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def _resolve_max_duration(self, n_obs: int) -> int:
        if self.max_duration is not None:
            return int(min(self.max_duration, n_obs))
        return int(min(n_obs, max(10, n_obs // 4)))

    def _fit(self, X: np.ndarray) -> FitResult:
        n_obs, n_feat = X.shape
        K = self.n_regimes
        D = self._resolve_max_duration(n_obs)
        self._D = D

        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31 - 1, size=self.n_init)

        best: dict[str, Any] | None = None
        best_ll = -np.inf
        for seed in seeds:
            try:
                params = self._fit_once(X, D, int(seed))
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                continue
            if np.isfinite(params["loglik"]) and params["loglik"] > best_ll:
                best_ll = params["loglik"]
                best = params

        if best is None:
            raise RuntimeError(
                "All EM restarts failed. Try covariance_type='diag', fewer "
                "regimes, or a smaller max_duration for this sample size."
            )

        self.startprob_ = best["startprob"]
        self.transmat_ = best["transmat"]
        self.durations_ = best["durations"]
        self.means_ = best["means"]
        self.covars_ = best["covars"]

        return FitResult(
            log_likelihood=float(best["loglik"]),
            n_iter=int(best["n_iter"]),
            converged=bool(best["converged"]),
            n_regimes=K,
            extra={
                "n_features": n_feat,
                "n_obs": n_obs,
                "max_duration": D,
                "duration_dist": self.duration_dist,
                "mean_durations": self._mean_durations_from(best["durations"]),
            },
        )

    # -- single EM run --------------------------------------------------- #
    def _init_params(self, X: np.ndarray, D: int, seed: int):
        """K-means initialisation of emissions, transitions and durations."""
        K, n_feat = self.n_regimes, X.shape[1]
        if K == 1:
            labels = np.zeros(X.shape[0], dtype=int)
            means = X.mean(axis=0, keepdims=True)
        else:
            km = KMeans(n_clusters=K, n_init=1, random_state=seed).fit(X)
            labels = km.labels_
            means = km.cluster_centers_.copy()

        # Emission covariances from cluster scatter.
        if self.covariance_type == "diag":
            covars = np.ones((K, n_feat))
            for k in range(K):
                mask = labels == k
                if mask.sum() > 1:
                    covars[k] = X[mask].var(axis=0) + self.min_covar
        else:
            covars = np.array([np.eye(n_feat) for _ in range(K)])
            for k in range(K):
                mask = labels == k
                if mask.sum() > 1:
                    covars[k] = np.cov(X[mask].T) + self.min_covar * np.eye(n_feat)

        # Embedded transitions and durations from the run structure of labels.
        transmat = np.zeros((K, K))
        run_lengths: list[list[int]] = [[] for _ in range(K)]
        prev, run = labels[0], 1
        for t in range(1, len(labels)):
            if labels[t] == prev:
                run += 1
            else:
                transmat[prev, labels[t]] += 1
                run_lengths[prev].append(run)
                prev, run = labels[t], 1
        run_lengths[prev].append(run)

        for i in range(K):
            row = transmat[i].copy()
            row[i] = 0.0
            if row.sum() == 0:  # never observed leaving -> uniform over others
                if K > 1:
                    row[:] = 1.0
                    row[i] = 0.0
            transmat[i] = row / row.sum() if row.sum() > 0 else self._unit_row(i, K)

        startprob = np.full(K, 1.0 / K)
        durations = np.empty((K, D))
        for i in range(K):
            mu = float(np.mean(run_lengths[i])) if run_lengths[i] else 3.0
            durations[i] = self._duration_pmf(mu, np.zeros(D), D, init=True)
        return startprob, transmat, durations, means, covars

    @staticmethod
    def _unit_row(i: int, K: int) -> np.ndarray:
        row = np.zeros(K)
        if K > 1:
            row[:] = 1.0 / (K - 1)
            row[i] = 0.0
        else:
            row[i] = 1.0
        return row

    def _duration_pmf(
        self, mu: float, counts: np.ndarray, D: int, init: bool = False
    ) -> np.ndarray:
        """Build a duration pmf over d = 1..D from a mean (poisson) or counts."""
        d = np.arange(1, D + 1)
        if self.duration_dist == "poisson" or init:
            lam = max(mu, 1e-3)
            pmf = _poisson.pmf(d, lam)
            if not np.isfinite(pmf).all() or pmf.sum() == 0:
                pmf = np.ones(D)
        else:  # nonparametric: smoothed expected counts
            pmf = counts + 1e-6
        return pmf / pmf.sum()

    def _fit_once(self, X: np.ndarray, D: int, seed: int) -> dict[str, Any]:
        K, n_obs = self.n_regimes, X.shape[0]
        startprob, transmat, durations, means, covars = self._init_params(X, D, seed)

        prev_ll = -np.inf
        converged = False
        it = 0
        for it in range(1, self.n_iter + 1):  # noqa: B007 - read after the loop
            logB = _log_gaussian(X, means, covars, self.covariance_type, self.min_covar)
            fb = self._forward_backward(logB, startprob, transmat, durations)
            ll = fb["loglik"]

            # --- M-step ------------------------------------------------- #
            gamma = fb["gamma"]  # (T, K) regime posteriors
            Nk = gamma.sum(axis=0)
            Nk_safe = np.where(Nk > 1e-12, Nk, 1e-12)

            means = (gamma.T @ X) / Nk_safe[:, None]
            if self.covariance_type == "diag":
                covars = np.empty_like(means)
                for k in range(K):
                    diff = X - means[k]
                    covars[k] = (gamma[:, k, None] * diff * diff).sum(axis=0) / Nk_safe[k]
                covars = np.maximum(covars, self.min_covar)
            else:
                covars = np.empty((K, X.shape[1], X.shape[1]))
                for k in range(K):
                    diff = X - means[k]
                    covars[k] = (gamma[:, k, None, None] * diff[:, :, None] * diff[:, None, :]).sum(
                        0
                    ) / Nk_safe[k] + self.min_covar * np.eye(X.shape[1])

            startprob = gamma[0] / gamma[0].sum()
            transmat = self._update_transmat(fb, transmat, durations, K)
            durations = self._update_durations(fb, transmat, durations, D, K)

            if np.isfinite(ll) and abs(ll - prev_ll) < self.tol * n_obs:
                converged = True
                prev_ll = ll
                break
            prev_ll = ll

        # Final log-likelihood under the converged parameters.
        logB = _log_gaussian(X, means, covars, self.covariance_type, self.min_covar)
        final_ll = self._forward_backward(logB, startprob, transmat, durations)["loglik"]
        return {
            "startprob": startprob,
            "transmat": transmat,
            "durations": durations,
            "means": means,
            "covars": covars,
            "loglik": float(final_ll),
            "n_iter": it,
            "converged": converged,
        }

    # ------------------------------------------------------------------ #
    # Scaled forward-backward on the residual-time expanded state space
    # ------------------------------------------------------------------ #
    def _forward_backward(
        self,
        logB: np.ndarray,
        startprob: np.ndarray,
        transmat: np.ndarray,
        durations: np.ndarray,
    ) -> dict[str, Any]:
        """Rabiner-scaled forward-backward for the explicit-duration HMM.

        The expanded state is ``(regime i, remaining duration r)`` with
        ``r = 1..D`` (index ``r-1``).  While ``r > 1`` the state decrements
        deterministically emitting from regime ``i``; at ``r = 1`` it jumps to a
        new regime ``j`` (embedded ``transmat``) drawing a fresh duration
        ``p_j(d)``.  Returns regime posteriors plus the sufficient statistics
        the M-step needs.
        """
        T, K = logB.shape
        D = durations.shape[1]

        # Per-timestep emission scaling (numerical stability); folded into loglik.
        m = logB.max(axis=1)
        B = np.exp(logB - m[:, None])  # (T, K)

        alpha = np.zeros((T, K, D))
        c = np.zeros(T)
        a0 = startprob[:, None] * durations * B[0][:, None]  # (K, D)
        c[0] = a0.sum()
        alpha[0] = a0 / c[0]
        for t in range(1, T):
            u = alpha[t - 1][:, 0]  # alpha_{t-1}(i, r=1)
            S = u @ transmat  # entries into each j
            carry = np.zeros((K, D))
            carry[:, : D - 1] = alpha[t - 1][:, 1:]  # (i,r) <- (i, r+1)
            a_un = B[t][:, None] * (carry + durations * S[:, None])
            c[t] = a_un.sum()
            alpha[t] = a_un / c[t]

        loglik = float(np.log(c).sum() + m.sum())

        beta = np.zeros((T, K, D))
        beta[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            R = B[t + 1] * (durations * beta[t + 1]).sum(axis=1)  # (K,)
            beta_raw = np.zeros((K, D))
            beta_raw[:, 1:] = B[t + 1][:, None] * beta[t + 1][:, : D - 1]
            beta_raw[:, 0] = transmat @ R
            beta[t] = beta_raw / c[t + 1]

        gamma_exp = alpha * beta
        gamma_exp /= np.clip(gamma_exp.sum(axis=(1, 2), keepdims=True), 1e-300, None)
        gamma = gamma_exp.sum(axis=2)  # (T, K)

        return {
            "loglik": loglik,
            "gamma": gamma,
            "gamma_exp": gamma_exp,
            "alpha": alpha,
            "beta": beta,
            "c": c,
            "B": B,
        }

    def _update_transmat(self, fb, transmat_old, durations, K):
        """Re-estimate embedded transitions from expected regime-jump counts."""
        if K == 1:
            return np.array([[1.0]])
        alpha, beta, c, B = fb["alpha"], fb["beta"], fb["c"], fb["B"]
        # R_j(t+1) = B[t+1, j] * sum_d p_j(d) beta[t+1, j, d]: mass of a fresh
        # segment of regime j starting at t+1, marginalised over its duration.
        R = B[1:] * (durations[None] * beta[1:]).sum(axis=2)  # (T-1, K)
        src = alpha[:-1, :, 0] / c[1:, None]  # (T-1, K)
        M = src.T @ R  # (K, K)
        trans_num = transmat_old * M
        np.fill_diagonal(trans_num, 0.0)
        row = trans_num.sum(axis=1)
        transmat = np.empty((K, K))
        for i in range(K):
            transmat[i] = trans_num[i] / row[i] if row[i] > 1e-12 else self._unit_row(i, K)
        return transmat

    # ------------------------------------------------------------------ #
    # Prediction / contract methods
    # ------------------------------------------------------------------ #
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.means_ is not None
        logB = _log_gaussian(X, self.means_, self.covars_, self.covariance_type, self.min_covar)
        fb = self._forward_backward(logB, self.startprob_, self.transmat_, self.durations_)
        gamma = fb["gamma"]
        return gamma / np.clip(gamma.sum(axis=1, keepdims=True), 1e-300, None)

    # ------------------------------------------------------------------ #
    # Causal (real-time / filtered) inference — no look-ahead
    #
    # ``predict`` / ``predict_proba`` return the *smoothed* posterior
    # ``gamma_t = alpha_t * beta_t``.  The backward pass ``beta_t`` propagates
    # information from observations *after* t, so smoothed labels at time t are
    # informed by the future — fine for a historical study, invalid for a
    # live signal.  The methods below use the *forward* variable ``alpha_t``
    # only, which depends solely on observations ``0..t``.  Running them on a
    # window ``X[:t]`` yields exactly the same row t as running on the full
    # ``X`` — the defining property of a causal estimator.
    # ------------------------------------------------------------------ #
    def _forward_filter(self, X: np.ndarray) -> np.ndarray:
        """Scaled forward pass only. Returns alpha, shape ``(T, K, D)``.

        Each ``alpha[t]`` is normalised over the expanded ``(regime, remaining
        duration)`` state and depends only on observations up to and including
        ``t`` — the online filter.
        """
        assert self.means_ is not None
        logB = _log_gaussian(X, self.means_, self.covars_, self.covariance_type, self.min_covar)
        T, K = logB.shape
        D = self.durations_.shape[1]
        m = logB.max(axis=1)
        B = np.exp(logB - m[:, None])

        alpha = np.zeros((T, K, D))
        a0 = self.startprob_[:, None] * self.durations_ * B[0][:, None]
        alpha[0] = a0 / a0.sum()
        for t in range(1, T):
            u = alpha[t - 1][:, 0]  # segments ending at t-1
            S = u @ self.transmat_  # entries into each regime
            carry = np.zeros((K, D))
            carry[:, : D - 1] = alpha[t - 1][:, 1:]  # remaining-duration decrement
            a_un = B[t][:, None] * (carry + self.durations_ * S[:, None])
            alpha[t] = a_un / a_un.sum()
        return alpha

    def filter_proba(self, X: np.ndarray | Any) -> np.ndarray | Any:
        """Causal filtered regime posterior ``P(regime_t | obs_{0..t})``.

        The real-time counterpart of :meth:`predict_proba`.  Uses only past and
        present observations, so it carries **no look-ahead bias** and is the
        correct object for a live regime signal.  Returned in canonical state
        order, index-aligned when a DataFrame is passed.
        """
        self._check_fitted()
        values, index = as_array(X)
        alpha = self._forward_filter(values)
        filt = alpha.sum(axis=2)  # marginalise duration
        filt = filt / np.clip(filt.sum(axis=1, keepdims=True), 1e-300, None)
        filt = self._reorder_columns(filt)
        cols = [f"regime_{k}" for k in range(self.n_regimes)]
        return attach_index(filt, index, columns=cols)

    def filter_predict(self, X: np.ndarray | Any) -> np.ndarray | Any:
        """Causal filtered hard labels — real-time counterpart of :meth:`predict`."""
        self._check_fitted()
        values, index = as_array(X)
        filt = self._reorder_columns(self._forward_filter(values).sum(axis=2))
        labels = filt.argmax(axis=1).astype(int)
        return attach_index(labels, index)

    def filtered_next_regime(self, X: np.ndarray | Any) -> np.ndarray | Any:
        """Causal one-step-ahead forecast ``P(regime_{t+1} | obs_{0..t})``.

        The model-native, duration-aware "probability of switching next period"
        — with **zero look-ahead**.  It propagates the filtered forward state
        one step through the explicit-duration dynamics: mass in segments that
        are ending (residual duration 1) rotates via the embedded jump matrix,
        the rest ages by one period.  Unlike
        :meth:`duration_conditional_transition` — which conditions on a single
        *hard* elapsed-dwell number — this integrates over the model's full
        *soft* belief about how long the current segment has already run, so it
        is the cleaner live signal.  Row ``t`` is the distribution for ``t+1``
        given data through ``t``; returned in canonical state order.
        """
        self._check_fitted()
        values, index = as_array(X)
        alpha = self._forward_filter(values)
        T, K, D = alpha.shape
        pred = np.zeros((T, K))
        for t in range(T):
            p = np.zeros((K, D))
            p[:, : D - 1] += alpha[t][:, 1:]  # age surviving segments
            S = alpha[t][:, 0] @ self.transmat_  # rotate ending segments
            p += self.durations_ * S[:, None]
            pred[t] = p.sum(axis=1)
        pred = self._reorder_columns(pred)
        cols = [f"regime_{k}" for k in range(self.n_regimes)]
        return attach_index(pred, index, columns=cols)

    # ------------------------------------------------------------------ #
    # Directional (momentum-tilted) destination — breaking the frozen split
    #
    # A vanilla HSMM factorises switching into *timing* (the duration hazard)
    # and *destination* (the embedded jump matrix ``A``).  ``A`` is
    # unconditional, so the destination split when leaving a regime is frozen
    # — leaving "Neutral" always splits ~53/47 to Contraction/Expansion no
    # matter which way the economy is actually drifting.  That is wrong: if
    # momentum points up we should lean toward Expansion.
    #
    # We keep the HSMM's timing untouched and only tilt the *destination*
    # split by how the recent observation momentum aligns with each candidate
    # regime's mean direction.  ``A`` acts as a prior; the trajectory reweights
    # it.  ``sensitivity=0`` recovers the original frozen behaviour.  This is a
    # deliberately transparent overlay — the fully principled alternative is a
    # covariate-dependent transition HSMM, but monthly macro data has too few
    # regime exits to fit one reliably.
    # ------------------------------------------------------------------ #
    def _tilt_weights(
        self, from_regime: int, momentum: np.ndarray, sensitivity: float
    ) -> np.ndarray:
        """Momentum-tilted destination distribution over ``j != from_regime``.

        Works in *raw* (unordered) state space; caller reorders.  ``momentum``
        is a length-``n_features`` vector (recent change in the observations).
        """
        A = np.asarray(self.transmat_)  # (K,K) raw, zero diagonal
        means = np.asarray(self.means_)  # (K, n_feat) raw
        K = self.n_regimes
        logw = np.full(K, -np.inf)
        for j in range(K):
            if j == from_regime or A[from_regime, j] <= 0:
                continue
            direction = means[j] - means[from_regime]
            dnorm = np.linalg.norm(direction)
            # Projection of momentum onto the i->j axis (unit direction). Units
            # of feature-z per momentum window; positive == drifting toward j.
            align = float(momentum @ direction / dnorm) if dnorm > 1e-12 else 0.0
            logw[j] = np.log(A[from_regime, j]) + sensitivity * align
        m = logw.max()
        if not np.isfinite(m):  # no valid destination (K==1)
            out = np.zeros(K)
            return out
        w = np.exp(logw - m)
        return w / w.sum()

    def directional_split(
        self,
        from_regime: int,
        momentum: np.ndarray,
        sensitivity: float = 1.0,
    ) -> np.ndarray:
        """Destination split when leaving ``from_regime`` given a momentum vector.

        Illustrative, data-free: pass a hypothetical momentum (change in the
        feature vector) and see how the exit destination re-weights.  Returns a
        length-``K`` distribution over destinations (the ``from_regime`` entry is
        0), in canonical state order.  With ``sensitivity=0`` this equals the
        frozen embedded split; positive momentum toward a regime raises its
        share.

        Parameters
        ----------
        from_regime : int
            Source regime index in canonical order.
        momentum : array-like, shape (n_features,)
            Recent change in the (same-space) observations, e.g. ``x_t - x_{t-k}``.
        sensitivity : float, default 1.0
            Tilt strength. 0 = frozen embedded split; larger = more directional.
        """
        self._check_fitted()
        momentum = np.asarray(momentum, dtype=float).ravel()
        if self._state_order is not None:
            raw_from = int(np.asarray(self._state_order)[from_regime])
        else:
            raw_from = int(from_regime)
        w = self._tilt_weights(raw_from, momentum, sensitivity)
        return w[self._state_order] if self._state_order is not None else w

    def directional_next_regime(
        self,
        X: np.ndarray | Any,
        momentum_window: int = 3,
        sensitivity: float = 1.0,
    ) -> np.ndarray | Any:
        """Causal one-step forecast with a momentum-tilted destination split.

        Identical to :meth:`filtered_next_regime` in its *timing* (same duration
        hazard, same causal filter), but the mass that rotates out of a regime
        is distributed by :meth:`_tilt_weights` — using the realised momentum
        ``x_t - x_{t-momentum_window}`` — instead of the frozen embedded matrix.
        The result answers "given where we are *and which way we are heading*,
        where do we go next period?".  Row ``t`` uses only data through ``t``.

        Parameters
        ----------
        X : array or DataFrame
            Observations in the model's feature space.
        momentum_window : int, default 3
            Lookback (periods) for the momentum ``x_t - x_{t-window}``.
        sensitivity : float, default 1.0
            Destination tilt strength (0 recovers :meth:`filtered_next_regime`).
        """
        self._check_fitted()
        values, index = as_array(X)
        alpha = self._forward_filter(values)
        T, K, D = alpha.shape
        durations = np.asarray(self.durations_)
        pred = np.zeros((T, K))
        for t in range(T):
            if t >= momentum_window:
                mom = values[t] - values[t - momentum_window]
            else:
                mom = np.zeros(values.shape[1])
            # Per-source tilted transition (raw space); rows for regimes with no
            # ending mass are irrelevant to the propagation below.
            A_t = np.zeros((K, K))
            for i in range(K):
                A_t[i] = self._tilt_weights(i, mom, sensitivity)
            p = np.zeros((K, D))
            p[:, : D - 1] += alpha[t][:, 1:]  # timing: age survivors
            S = alpha[t][:, 0] @ A_t  # direction: tilted rotation
            p += durations * S[:, None]
            pred[t] = p.sum(axis=1)
        pred = self._reorder_columns(pred)
        cols = [f"regime_{k}" for k in range(self.n_regimes)]
        return attach_index(pred, index, columns=cols)

    def _transition_matrix(self) -> np.ndarray:
        """Duration-embedded *effective* Markov transition matrix.

        Combines the embedded (zero-diagonal) regime transitions with each
        regime's mean dwell time ``E_i`` so that ``p_ii = 1 - 1/E_i`` and
        ``p_ij = a_ij / E_i``.  The result is row-stochastic and its implied
        expected duration ``1/(1 - p_ii)`` equals the explicit ``E_i``.
        """
        assert self.transmat_ is not None
        K = self.n_regimes
        E = self._mean_durations_from(self.durations_)
        E = np.clip(E, 1.0 + 1e-9, None)
        T = np.empty((K, K))
        for i in range(K):
            stay = 1.0 - 1.0 / E[i]
            T[i] = self.transmat_[i] * (1.0 / E[i])
            T[i, i] = stay
        return T

    def _state_means(self) -> np.ndarray:
        assert self.means_ is not None
        return np.asarray(self.means_)

    # ------------------------------------------------------------------ #
    # Duration helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mean_durations_from(durations: np.ndarray) -> np.ndarray:
        d = np.arange(1, durations.shape[1] + 1)
        return (durations * d).sum(axis=1)

    def _update_durations(self, fb, transmat, durations_old, D, K):
        alpha, beta, c, B = fb["alpha"], fb["beta"], fb["c"], fb["B"]
        gamma_exp = fb["gamma_exp"]
        T = alpha.shape[0]
        dur_num = gamma_exp[0].copy()  # initial-segment entries
        for t in range(T - 1):
            S = alpha[t][:, 0] @ transmat  # (K,) entries into j
            g = S * B[t + 1] / c[t + 1]  # (K,)
            dur_num += durations_old * beta[t + 1] * g[:, None]
        dur_new = np.empty((K, D))
        for i in range(K):
            mu = self._mean_from_counts(dur_num[i])
            dur_new[i] = self._duration_pmf(mu, dur_num[i], D)
        return dur_new

    @staticmethod
    def _mean_from_counts(counts: np.ndarray) -> float:
        d = np.arange(1, counts.shape[0] + 1)
        total = counts.sum()
        return float((counts * d).sum() / total) if total > 1e-12 else 3.0

    # ------------------------------------------------------------------ #
    # HSMM-specific public properties (reordered to canonical state order)
    # ------------------------------------------------------------------ #
    @property
    def means(self) -> np.ndarray:
        """Per-regime emission means (K, n_features), canonical order."""
        self._check_fitted()
        means = np.asarray(self.means_)
        return means[self._state_order] if self._state_order is not None else means

    @property
    def covariances(self) -> np.ndarray:
        """Per-regime emission covariances, canonical order."""
        self._check_fitted()
        cov = np.asarray(self.covars_)
        return cov[self._state_order] if self._state_order is not None else cov

    @property
    def duration_distributions(self) -> np.ndarray:
        """Explicit duration pmfs (K, max_duration), canonical order.

        Row ``i`` is ``p_i(d)`` for ``d = 1 .. max_duration`` — the probability
        that a visit to regime ``i`` lasts exactly ``d`` periods.
        """
        self._check_fitted()
        dur = np.asarray(self.durations_)
        return dur[self._state_order] if self._state_order is not None else dur

    @property
    def mean_durations(self) -> np.ndarray:
        """Expected dwell time per regime from the explicit duration model."""
        self._check_fitted()
        return self._mean_durations_from(self.duration_distributions)

    @property
    def embedded_transition_matrix(self) -> np.ndarray:
        """Raw embedded (zero-diagonal) regime-to-regime transitions.

        These are the semi-Markov jump probabilities *conditional on leaving*
        the current regime — distinct from the duration-embedded effective
        chain returned by :attr:`transition_matrix`.
        """
        self._check_fitted()
        return self._reorder_matrix(np.asarray(self.transmat_))

    # ------------------------------------------------------------------ #
    # Duration-conditional dynamics (the reason to prefer an HSMM)
    #
    # A plain HMM has a memoryless hazard: the probability of leaving a
    # regime next period is the constant ``1 - p_ii`` no matter how long you
    # have already been there.  The whole point of the explicit-duration
    # model is that the hazard *rises with elapsed dwell time*.  These
    # methods turn the learned duration pmf into the object a practitioner
    # actually wants live: "given we have sat in this regime for ``d``
    # periods, what is the probability we switch out — and into which
    # regime — next period?"
    # ------------------------------------------------------------------ #
    @staticmethod
    def _hazard_from_durations(durations: np.ndarray) -> np.ndarray:
        """Discrete hazard ``h_i(d) = P(D = d | D >= d)`` from a duration pmf.

        ``durations`` has shape ``(K, D)`` with column ``d-1`` holding
        ``p_i(d)``.  The hazard at elapsed time ``d`` is the probability the
        current segment *ends now* given it has survived to ``d`` — it climbs
        toward 1.0 as ``d`` approaches the support edge, which is exactly the
        rising exit pressure a geometric HMM cannot represent.
        """
        # tail[:, d-1] = P(D >= d) = sum_{k >= d} p_i(k)  (inclusive rev-cumsum)
        tail = np.cumsum(durations[:, ::-1], axis=1)[:, ::-1]
        return np.where(tail > 1e-300, durations / tail, 0.0)

    @property
    def hazard_curves(self) -> np.ndarray:
        """Exit hazard ``h_i(d)`` per regime, shape ``(K, max_duration)``.

        Row ``i``, column ``d-1`` is the probability that a visit to regime
        ``i`` ends at exactly period ``d`` *given* it has lasted ``d`` periods
        — i.e. the conditional probability of switching out next period after
        ``d`` periods of dwell.  Monotone-ish increasing for peaked (Poisson)
        durations; flat only in the degenerate geometric case.
        """
        self._check_fitted()
        haz = self._hazard_from_durations(np.asarray(self.durations_))
        return haz[self._state_order] if self._state_order is not None else haz

    @property
    def survival_curves(self) -> np.ndarray:
        """Survival ``S_i(d) = P(D > d)`` per regime, shape ``(K, max_duration)``.

        Probability that a visit to regime ``i`` is *still ongoing* after ``d``
        periods.  Column ``d-1`` holds ``S_i(d)``; the curve starts near 1 and
        decays to 0 at the support edge.
        """
        self._check_fitted()
        dur = np.asarray(self.durations_)
        surv = 1.0 - np.cumsum(dur, axis=1)  # 1 - P(D <= d) = P(D > d)
        surv = np.clip(surv, 0.0, 1.0)
        return surv[self._state_order] if self._state_order is not None else surv

    def _elapsed_vector(self, elapsed: int | np.ndarray, D: int) -> np.ndarray:
        """Broadcast ``elapsed`` to a per-regime int index into ``1..D``."""
        K = self.n_regimes
        e = np.atleast_1d(np.asarray(elapsed))
        if e.size == 1:
            e = np.full(K, int(e.item()))
        elif e.size != K:
            raise ValueError(
                f"elapsed must be a scalar or length-{K} (one per regime), got size {e.size}."
            )
        if (e < 1).any():
            raise ValueError("elapsed dwell time must be >= 1 period.")
        return np.clip(e.astype(int) - 1, 0, D - 1)  # column indices

    def duration_conditional_transition(self, elapsed: int | np.ndarray) -> np.ndarray:
        """Next-period transition matrix given elapsed dwell time in each regime.

        This is the headline HSMM output.  Given we have already spent
        ``elapsed`` periods in a regime, the probability of switching out next
        period is the hazard ``h_i(elapsed)``; conditional on switching, the
        destination is drawn from the embedded jump matrix ``A``.  So

            ``T[i, i] = 1 - h_i(elapsed)``                  (stay)
            ``T[i, j] = h_i(elapsed) * A[i, j]``   (j != i, switch to j)

        Contrast :attr:`transition_matrix`, which bakes in the *mean* dwell
        time and is therefore duration-blind.  Here the same regime becomes
        progressively more likely to break the longer it has run.

        Parameters
        ----------
        elapsed : int or array-like of shape (K,)
            Periods already spent in the regime.  A scalar applies the same
            elapsed time to every regime; a length-K array sets it per regime.

        Returns
        -------
        np.ndarray of shape (K, K), row-stochastic, in canonical state order.

        Examples
        --------
        >>> # P(switch out of "neutral" after 30 months, and into what)
        >>> T = model.duration_conditional_transition(30)
        >>> T[neutral]            # row: [P->contraction, P(stay), P->expansion]
        """
        self._check_fitted()
        haz = self.hazard_curves  # (K, D), canonical order
        A = self.embedded_transition_matrix  # (K, K), canonical, zero diagonal
        K, D = haz.shape
        e_idx = self._elapsed_vector(elapsed, D)
        h = haz[np.arange(K), e_idx]  # (K,) exit hazard at elapsed dwell
        T = A * h[:, None]
        T[np.diag_indices(K)] = 1.0 - h
        return T

    def switch_probability_curve(self, regime: int) -> np.ndarray:
        """Per-destination switch probabilities vs elapsed dwell for one regime.

        Returns an array of shape ``(max_duration, K)`` whose row ``d-1`` is the
        next-period distribution ``[P(->0), ..., P(->K-1)]`` given the regime has
        already lasted ``d`` periods (the diagonal entry is the stay
        probability).  This is the curve to plot when asking "how does the odds
        of rotating from *neutral* into *contraction* vs *expansion* evolve the
        longer we sit in neutral?".

        Parameters
        ----------
        regime : int
            Regime index in canonical order (0 = lowest mean, ...).
        """
        self._check_fitted()
        haz = self.hazard_curves  # (K, D)
        A = self.embedded_transition_matrix  # (K, K)
        K, D = haz.shape
        if not 0 <= regime < K:
            raise ValueError(f"regime must be in [0, {K}), got {regime}.")
        h = haz[regime]  # (D,) hazard vs elapsed
        curve = h[:, None] * A[regime][None, :]  # (D, K) switch mass split by dest
        curve[:, regime] = 1.0 - h  # stay probability on the diagonal
        return curve

    def expected_remaining_duration(self, elapsed: int | np.ndarray) -> np.ndarray:
        """Expected *additional* periods in each regime given elapsed dwell.

        ``E[D - elapsed | D >= elapsed]`` per regime — the mean residual life.
        For a memoryless (geometric) process this is constant; for a peaked
        duration model it *shrinks* as ``elapsed`` grows, quantifying "we are
        overdue to rotate."  Returns shape ``(K,)`` in canonical order.
        """
        self._check_fitted()
        dur = self.duration_distributions  # (K, D), canonical
        K, D = dur.shape
        e_idx = self._elapsed_vector(elapsed, D)
        d = np.arange(1, D + 1)
        out = np.empty(K)
        for i in range(K):
            e = e_idx[i] + 1  # elapsed periods (>=1)
            mask = d >= e
            tail = dur[i, mask].sum()
            if tail <= 1e-300:
                out[i] = 0.0
            else:
                out[i] = ((d[mask] - e) * dur[i, mask]).sum() / tail
        return out

    def forecast(
        self,
        steps: int = 1,
        current_state: int | None = None,
        elapsed: int | None = None,
    ) -> np.ndarray:
        """Forecast the regime distribution ``steps`` periods ahead.

        Without ``elapsed`` this is the base-class forecast on the
        duration-embedded chain (:attr:`transition_matrix`), which is
        duration-blind: it uses each regime's *mean* dwell time, so it gives
        the same answer whether the regime started last period or long ago.

        Pass ``elapsed`` (with ``current_state``) to forecast on the
        ``(regime, elapsed dwell)`` state space instead.  Each step a segment
        ends with hazard ``h_i(d)`` and, if it ends, jumps via the embedded
        matrix ``A`` into a fresh segment with ``d = 1``.  Step one therefore
        equals row ``current_state`` of
        :meth:`duration_conditional_transition`, and later steps carry the
        rising exit pressure forward rather than resetting it.

        Parameters
        ----------
        steps : int
            Forecast horizon in periods.
        current_state : int, optional
            Known current regime (canonical order). Required with ``elapsed``.
        elapsed : int, optional
            Periods already spent in ``current_state`` (>= 1).

        Returns
        -------
        np.ndarray of shape (steps, n_regimes)
            Row ``h`` is the regime probability vector ``h + 1`` steps ahead.
        """
        if elapsed is None:
            return super().forecast(steps=steps, current_state=current_state)
        self._check_fitted()
        if steps < 1:
            raise ValueError("steps must be >= 1.")
        if current_state is None:
            raise ValueError("elapsed requires current_state.")
        K = self.n_regimes
        if not 0 <= current_state < K:
            raise ValueError(f"current_state must be in [0, {K}), got {current_state}.")

        dur = self.duration_distributions  # (K, D), canonical
        A = self.embedded_transition_matrix  # (K, K), canonical, zero diagonal
        D = dur.shape[1]
        # A segment cannot outlive the duration support, so where no mass
        # remains (and at the last support point) the segment must end.
        tail = np.cumsum(dur[:, ::-1], axis=1)[:, ::-1]
        haz = np.where(tail > 1e-12, dur / np.maximum(tail, 1e-300), 1.0)
        haz[:, -1] = 1.0

        dist = np.zeros((K, D))  # mass over (regime, dwell - 1)
        dist[current_state, self._elapsed_vector(elapsed, D)[current_state]] = 1.0
        out = np.empty((steps, K))
        for h in range(steps):
            leaving = dist * haz
            nxt = np.zeros_like(dist)
            nxt[:, 1:] = (dist - leaving)[:, :-1]  # survivors age by one period
            nxt[:, 0] = leaving.sum(axis=1) @ A  # exits start a new segment
            dist = nxt
            out[h] = dist.sum(axis=1)
        return out
