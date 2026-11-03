"""Walk-forward validation engine.

Produces genuinely out-of-sample regime labels by repeatedly fitting on a
trailing window and predicting the next block, then stepping forward. This
mirrors how regime detection runs in production (periodic re-estimation) and
is the honest way to assess it — an HMM fit on the full history and then
"predicted" in-sample will always look better than it deserves.

Two leakage guards are built in:

* The preprocessor is re-fit on each training window only (train-only
  scaling/imputation), then applied to the test block.
* Regime ids are kept comparable across refits via the model's canonical
  state ordering (``sort_states_by``), so a regime means the same thing in
  every fold.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from ..pipeline import RegimeDetectionPipeline


@dataclass
class WalkForwardResult:
    """Aggregated output of a walk-forward run."""

    oos_labels: pd.Series
    oos_proba: pd.DataFrame
    fold_boundaries: list[tuple[pd.Timestamp, pd.Timestamp]]
    n_folds: int
    diagnostics: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Join OOS labels and probabilities into one tidy DataFrame."""
        out = self.oos_proba.copy()
        out.insert(0, "regime", self.oos_labels)
        return out


class WalkForwardBacktester:
    """Expanding- or rolling-window walk-forward regime backtester.

    Parameters
    ----------
    pipeline_factory : callable () -> RegimeDetectionPipeline
        Factory returning a *fresh* pipeline for each refit. A factory (not a
        single instance) guarantees no state leaks between folds.
    train_size : int
        Initial training window length (number of observations).
    test_size : int, default 1
        Number of observations predicted per fold before refitting.
    step : int, optional
        Advance between folds. Defaults to ``test_size`` (non-overlapping).
    expanding : bool, default True
        If True, the training window grows (expanding origin). If False, it
        slides with fixed length ``train_size`` (rolling origin).
    refit_every : int, default 1
        Refit the model every ``refit_every`` folds; in between, reuse the
        last fitted pipeline (cheaper, models slower-moving re-estimation
        cadence as used on a real desk).
    causal : bool, default False
        If True, produce out-of-sample labels with the model's causal
        **filter** rather than the forward–backward **smoother**. This removes
        the second source of look-ahead (state inference) on top of the
        trailing-window refit that already removes the first (parameters). The
        filter is warm-started by running it over the whole
        ``[train … test]`` context window and slicing out the test rows, so no
        future observation ever informs an out-of-sample label. Requires a
        model exposing ``filter_proba`` (e.g. ``GaussianHSMMModel``).
    """

    def __init__(
        self,
        pipeline_factory: Callable[[], RegimeDetectionPipeline],
        train_size: int,
        test_size: int = 1,
        step: int | None = None,
        expanding: bool = True,
        refit_every: int = 1,
        causal: bool = False,
    ) -> None:
        if train_size < 2:
            raise ValueError("train_size must be >= 2.")
        self.pipeline_factory = pipeline_factory
        self.train_size = train_size
        self.test_size = test_size
        self.step = step if step is not None else test_size
        self.expanding = expanding
        self.refit_every = max(1, refit_every)
        self.causal = causal

    def run(self, X: pd.DataFrame, verbose: bool = False) -> WalkForwardResult:
        n = len(X)
        if self.train_size + self.test_size > n:
            raise ValueError(
                f"Not enough data: need >= train_size + test_size = "
                f"{self.train_size + self.test_size}, got {n}."
            )

        all_labels: list[pd.Series] = []
        all_proba: list[pd.DataFrame] = []
        boundaries: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        pipeline: RegimeDetectionPipeline | None = None
        fold = 0
        train_start = 0
        test_start = self.train_size

        while test_start < n:
            test_end = min(test_start + self.test_size, n)
            tr_lo = train_start if not self.expanding else 0
            X_train = X.iloc[tr_lo:test_start]
            X_test = X.iloc[test_start:test_end]

            if pipeline is None or (fold % self.refit_every == 0):
                pipeline = self.pipeline_factory()
                pipeline.fit_detect(X_train)
                if verbose:
                    print(
                        f"[fold {fold}] refit on {X_train.index[0].date()}"
                        f"..{X_train.index[-1].date()} (n={len(X_train)})"
                    )

            if self.causal:
                # Filter over the full context window (train history + test) so
                # the online state is warm-started, then keep only the test rows.
                context = X.iloc[tr_lo:test_end]
                result = pipeline.filter_detect(context)
                fold_labels = result.smoothed_labels.loc[X_test.index]
                fold_proba = result.proba.loc[X_test.index]
            else:
                result = pipeline.detect(X_test)
                fold_labels = result.smoothed_labels
                fold_proba = result.proba
            all_labels.append(fold_labels)
            all_proba.append(fold_proba)
            boundaries.append((X_test.index[0], X_test.index[-1]))

            fold += 1
            test_start += self.step
            if not self.expanding:
                train_start += self.step

        oos_labels = pd.concat(all_labels)
        oos_labels = oos_labels[~oos_labels.index.duplicated(keep="last")]
        oos_proba = pd.concat(all_proba)
        oos_proba = oos_proba[~oos_proba.index.duplicated(keep="last")]

        return WalkForwardResult(
            oos_labels=oos_labels.rename("regime"),
            oos_proba=oos_proba,
            fold_boundaries=boundaries,
            n_folds=fold,
            diagnostics={
                "expanding": self.expanding,
                "train_size": self.train_size,
                "test_size": self.test_size,
                "refit_every": self.refit_every,
                "causal": self.causal,
                "n_oos_obs": len(oos_labels),
            },
        )
