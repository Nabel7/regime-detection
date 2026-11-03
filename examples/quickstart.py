"""Quickstart on synthetic data: HMM vs HSMM, walk-forward, duration-aware forecasts.

Runs offline in a few seconds:

    python examples/quickstart.py
"""

import logging

import numpy as np

from regime_detection import GaussianHMMModel, GaussianHSMMModel, RegimeDetectionPipeline
from regime_detection.backtest import (
    WalkForwardBacktester,
    best_match_accuracy,
    stability_metrics,
)
from regime_detection.data import make_regime_data

np.set_printoptions(precision=3, suppress=True)
logging.getLogger("hmmlearn").setLevel(logging.ERROR)  # silence EM convergence chatter

# A 3-regime monthly panel with a known regime path, so recovery can be scored.
df, truth = make_regime_data(n_obs=300, n_regimes=3, seed=7)
print(f"Panel: {df.shape[0]} months x {df.shape[1]} features\n")

# --- 1. In-sample detection --------------------------------------------------
for model in (
    GaussianHMMModel(n_regimes=3, n_init=8, random_state=0),
    GaussianHSMMModel(n_regimes=3, n_init=4, random_state=0),
):
    pipe = RegimeDetectionPipeline(model=model, min_duration=2)
    result = pipe.fit_detect(df)
    acc = best_match_accuracy(truth, result.smoothed_labels)
    print(f"{type(model).__name__:<18} in-sample accuracy {acc:.1%}")

# --- 2. Walk-forward, out-of-sample -------------------------------------------
bt = WalkForwardBacktester(
    lambda: RegimeDetectionPipeline(
        model=GaussianHMMModel(n_regimes=3, n_init=4, random_state=0), min_duration=2
    ),
    train_size=120,
    test_size=6,
    refit_every=2,
)
wf = bt.run(df)
oos_acc = best_match_accuracy(truth.reindex(wf.oos_labels.index), wf.oos_labels)
print(f"\nWalk-forward OOS accuracy {oos_acc:.1%}")
print("OOS stability:", {k: round(v, 3) for k, v in stability_metrics(wf.oos_labels).items()})

# --- 3. What the HSMM adds: hazards that rise with time in the regime ---------
hsmm = GaussianHSMMModel(n_regimes=3, n_init=4, random_state=0).fit(df)
print(f"\nHSMM mean regime durations (months): {hsmm.mean_durations}")

regime = 1
mean_d = int(round(hsmm.mean_durations[regime]))
for elapsed in (1, mean_d):
    h = hsmm.hazard_curves[regime, elapsed - 1]
    print(f"P(regime {regime} ends next month | {elapsed:>2} months in) = {h:.3f}")

# A plain HMM forecast forgets how long we have been in the regime; passing
# `elapsed` propagates the full duration model instead.
blind = hsmm.forecast(steps=6, current_state=regime)
for elapsed in (1, mean_d):
    aware = hsmm.forecast(steps=6, current_state=regime, elapsed=elapsed)
    print(
        f"P(still in regime {regime} in 6 months | {elapsed:>2} months in): "
        f"{aware[-1, regime]:.3f}  (duration-blind: {blind[-1, regime]:.3f})"
    )
