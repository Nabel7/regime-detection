# regime-detection

[![CI](https://github.com/Nabel7/regime-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Nabel7/regime-detection/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Market and macro **regime detection** in Python, with three models behind one interface:

| Model | What it is |
|---|---|
| `GaussianHMMModel` | Gaussian hidden Markov model (wraps `hmmlearn`), with random restarts and stable state ordering |
| `GaussianHSMMModel` | **Explicit-duration hidden semi-Markov model**, fitted from scratch with EM. Each regime has its own duration distribution, so the chance of leaving can rise the longer a regime lasts |
| `SparseJumpModel` | Sparse jump model (wraps `jumpmodels`), with built-in feature selection |

Around them: leakage-safe preprocessing, persistence smoothing, a walk-forward backtester, and evaluation metrics for unsupervised regimes.

## Why an HSMM

A standard HMM is memoryless: the probability of leaving a regime next month is the same whether the regime started last month or five years ago. Real regimes don't behave like that. The HSMM learns a duration distribution per regime, which gives you:

- `hazard_curves`: P(regime ends next period | it has lasted *d* periods)
- `duration_conditional_transition(elapsed)`: the next-period transition matrix given time already spent in each regime
- `expected_remaining_duration(elapsed)`: the expected time left
- `forecast(steps, current_state, elapsed=d)`: multi-step regime forecasts that carry the duration model forward

From [`examples/quickstart.py`](examples/quickstart.py) (synthetic data):

```
P(still in regime 1 in 6 months |  1 months in): 0.999  (duration-blind: 0.753)
P(still in regime 1 in 6 months | 18 months in): 0.216  (duration-blind: 0.753)
```

The duration-blind forecast gives the same answer in both cases. The duration-aware one doesn't.

## Install

```bash
git clone https://github.com/Nabel7/regime-detection.git
cd regime-detection
pip install -e ".[viz]"      # or ".[dev]" for tests, ruff and pre-commit
```

Python 3.10+.

## Quickstart

```python
from regime_detection import GaussianHSMMModel, RegimeDetectionPipeline
from regime_detection.data import Preprocessor, make_regime_data

df, truth = make_regime_data(n_obs=300, n_regimes=3)  # or your own DataFrame

pipe = RegimeDetectionPipeline(
    model=GaussianHSMMModel(n_regimes=3, n_init=4),
    preprocessor=Preprocessor(scale="zscore"),
    min_duration=2,  # absorb one-period flickers
    label_feature=df.columns[0],  # name regimes by this feature's mean
    label_names=("low", "mid", "high"),
)
result = pipe.fit_detect(df)

result.smoothed_labels  # regime path (pandas Series)
result.proba  # posterior probabilities (DataFrame)
result.statistics  # frequency, episodes and durations per regime
result.model.hazard_curves  # HSMM exit hazard vs time in regime
```

Walk-forward, out-of-sample:

```python
from regime_detection import GaussianHMMModel
from regime_detection.backtest import WalkForwardBacktester, stability_metrics

bt = WalkForwardBacktester(
    lambda: RegimeDetectionPipeline(model=GaussianHMMModel(n_regimes=3), min_duration=2),
    train_size=120,
    test_size=6,
    refit_every=2,
)
wf = bt.run(df)
wf.oos_labels  # out-of-sample regimes only
stability_metrics(wf.oos_labels)  # switch rate, whipsaw rate, run lengths
```

For causal (real-time) labels with the HSMM, use `filter_proba` or `pipeline.filter_detect`, which use only past data.

## Examples

| Script | Data | What it shows |
|---|---|---|
| [`examples/quickstart.py`](examples/quickstart.py) | synthetic, offline | HMM vs HSMM recovery, walk-forward, hazards, duration-aware forecasts |
| [`examples/fred_macro_regimes.py`](examples/fred_macro_regimes.py) | public [FRED](https://fred.stlouisfed.org) series, no API key | 3-regime HSMM on the US yield-curve slope, VIX and industrial production since 1990 |

The FRED example uses the default Poisson duration model, which is narrow. Once a regime has run well past its mean, the hazard climbs towards 1 very quickly. Treat its output as a demonstration of the mechanics, not a forecast. `duration_dist="nonparametric"` gives a more flexible fit when you have enough data.

## Architecture

```
regime_detection/
├── models/
│   ├── base.py          # BaseRegimeModel: the contract everything depends on
│   ├── hmm.py           # GaussianHMMModel
│   ├── hsmm.py          # GaussianHSMMModel (explicit-duration HSMM, EM from scratch)
│   └── sjm.py           # SparseJumpModel
├── data/
│   ├── loaders.py       # load_csv, validate_frame, align_frequencies
│   ├── preprocessing.py # Preprocessor (impute + scale, fitted on training data only)
│   └── synthetic.py     # make_regime_data (labelled synthetic panel)
├── utils/
│   ├── smoothing.py     # min-duration, majority-vote, probability smoothing
│   ├── metrics.py       # transition matrices, stationary distribution, entropy, durations
│   └── viz.py           # regime, probability and transition-matrix plots
├── backtest/
│   ├── engine.py        # WalkForwardBacktester (expanding/rolling, refit cadence)
│   └── metrics.py       # best-match accuracy, ARI/NMI, stability, confidence
└── pipeline.py          # RegimeDetectionPipeline (preprocess -> detect -> smooth)
```

The pipeline and backtester depend only on `BaseRegimeModel`, so models are interchangeable. To add one, subclass it and implement four NumPy-in, NumPy-out methods:

```python
from regime_detection.models.base import BaseRegimeModel, FitResult


class MyModel(BaseRegimeModel):
    def _fit(self, X): ...  # -> FitResult
    def _predict_proba(self, X): ...  # -> (n_obs, n_regimes)
    def _transition_matrix(self): ...  # -> (n_regimes, n_regimes)
    def _state_means(self): ...  # -> (n_regimes, n_features)
```

## Design choices

- **Random restarts.** EM is non-convex, so models fit `n_init` times and keep the best likelihood. Set `random_state` for reproducibility.
- **Stable state ordering.** EM labels states arbitrarily. `sort_states_by` orders them by mean or variance, so regime 0 means the same thing across walk-forward refits.
- **No look-ahead.** The preprocessor learns its statistics on each training window only, and the backtester builds a fresh pipeline per refit. A test checks this.
- **Honest evaluation.** Regimes are unsupervised, so accuracy against a known path uses the best label permutation (Hungarian matching), alongside permutation-invariant ARI and NMI. Without ground truth, stability and confidence diagnostics show whether the out-of-sample signal is well behaved.

## Development

```bash
pip install -e ".[dev]"
pre-commit install    # ruff lint + format on every commit
pytest -q             # 104 tests
```

CI runs ruff and the test suite on Python 3.10, 3.11 and 3.12.

## References

- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proc. IEEE* 77(2).
- Yu, S.-Z. (2010). Hidden semi-Markov models. *Artificial Intelligence* 174(2).
- Nystrup, P., Kolm, P. N. and Lindström, E. (2021). Feature selection in jump models. *Expert Systems with Applications* 184.

## License

MIT
