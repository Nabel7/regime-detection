"""Synthetic regime-switching macro data.

Generates a labelled dataset with a known hidden-state path so examples and
tests are self-contained and reproducible — no external data needed. The
default features stand in for a z-scored macro panel (growth, inflation,
inflation expectations, commodity and equity returns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_FEATURES = ["growth_z", "inflation_z", "infl_expect_z", "commodity_ret_z", "equity_ret_z"]

# Per-regime feature means for a 3-regime macro world:
#   0 = contraction/risk-off, 1 = neutral, 2 = expansion/risk-on
_REGIME_MEANS = {
    0: np.array([-1.2, -0.3, -0.6, -0.9, -1.1]),
    1: np.array([0.0, 0.1, 0.1, 0.0, 0.0]),
    2: np.array([1.1, 0.6, 0.7, 0.8, 1.0]),
}


def make_regime_data(
    n_obs: int = 240,
    n_regimes: int = 3,
    freq: str = "MS",
    start: str = "2005-01-01",
    persistence: float = 0.94,
    noise: float = 0.55,
    seed: int = 7,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a regime-switching macro panel and its true regime path.

    Parameters
    ----------
    n_obs : int
        Number of periods.
    n_regimes : int, default 3
        Must be <= 3 with the built-in means (else means are interpolated).
    persistence : float
        Diagonal of the (symmetric) transition matrix — higher = stickier.
    noise : float
        Std of the Gaussian emission noise around regime means.
    seed : int
        Reproducibility.

    Returns
    -------
    (df, true_labels) : (DataFrame, Series)
        ``df`` has a DatetimeIndex and the macro feature columns; ``true_labels``
        is the hidden regime path (for backtest scoring).
    """
    rng = np.random.default_rng(seed)
    features = features or DEFAULT_FEATURES[:]
    n_feat = len(features)

    # Build a persistent transition matrix.
    off = (1.0 - persistence) / (n_regimes - 1)
    T = np.full((n_regimes, n_regimes), off)
    np.fill_diagonal(T, persistence)

    # Regime means: use built-ins where available, else spread linearly.
    means = np.zeros((n_regimes, n_feat))
    for r in range(n_regimes):
        if r in _REGIME_MEANS:
            base = _REGIME_MEANS[r][:n_feat]
        else:
            base = np.linspace(-1, 1, n_feat) * (r - (n_regimes - 1) / 2)
        means[r] = base

    # Simulate the hidden Markov chain.
    states = np.empty(n_obs, dtype=int)
    states[0] = rng.integers(n_regimes)
    for t in range(1, n_obs):
        states[t] = rng.choice(n_regimes, p=T[states[t - 1]])

    # Emit observations.
    obs = means[states] + rng.normal(0, noise, size=(n_obs, n_feat))

    idx = pd.date_range(start=start, periods=n_obs, freq=freq)
    df = pd.DataFrame(obs, index=idx, columns=features)
    true_labels = pd.Series(states, index=idx, name="true_regime")
    return df, true_labels
