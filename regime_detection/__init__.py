"""regime_detection: modular HMM-based macroeconomic regime detection.

Quickstart
----------
>>> from regime_detection import GaussianHMMModel, RegimeDetectionPipeline
>>> from regime_detection.data import load_csv
>>> df = load_csv("macro.csv", date_col="date", freq="MS")
>>> pipe = RegimeDetectionPipeline(
...     model=GaussianHMMModel(n_regimes=3, covariance_type="diag"),
...     min_duration=3,
... )
>>> result = pipe.fit_detect(df)
>>> result.smoothed_labels.tail()
"""

from .models import (
    BaseRegimeModel,
    GaussianHMMModel,
    GaussianHSMMModel,
    SparseJumpModel,
)
from .pipeline import RegimeDetectionPipeline, RegimeResult

__version__ = "0.1.0"

__all__ = [
    "BaseRegimeModel",
    "GaussianHMMModel",
    "GaussianHSMMModel",
    "SparseJumpModel",
    "RegimeDetectionPipeline",
    "RegimeResult",
    "__version__",
]
