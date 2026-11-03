"""Macro regimes on public FRED data with an explicit-duration HSMM.

Downloads three public series from FRED (no API key), builds a monthly panel,
fits a 3-regime HSMM and asks: given how long the current regime has lasted,
how likely is it to still be in place in six months?

    python examples/fred_macro_regimes.py

Series (FRED IDs):
    T10Y2Y  10-year minus 2-year Treasury spread (yield-curve slope)
    VIXCLS  CBOE VIX (equity implied volatility)
    INDPRO  US industrial production index (used as year-on-year growth)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from regime_detection import GaussianHSMMModel, RegimeDetectionPipeline
from regime_detection.data import Preprocessor

logging.getLogger("hmmlearn").setLevel(logging.ERROR)

DATA_DIR = Path(__file__).parent / "data"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def fred(series_id: str) -> pd.Series:
    """Fetch one FRED series, cached under examples/data/ (git-ignored)."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{series_id}.csv"
    if not path.exists():
        pd.read_csv(FRED_URL.format(series_id)).to_csv(path, index=False)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return pd.to_numeric(df.iloc[:, 0], errors="coerce")


def build_panel(start: str = "1990-01-01") -> pd.DataFrame:
    monthly = lambda s: s.resample("MS").mean()  # noqa: E731
    panel = pd.DataFrame(
        {
            "curve_slope": monthly(fred("T10Y2Y")),
            "log_vix": np.log(monthly(fred("VIXCLS"))),
            # Industrial production is published about two weeks after the
            # month ends, so lag it one month to avoid look-ahead.
            "indpro_yoy": fred("INDPRO").pct_change(12, fill_method=None).shift(1) * 100,
        }
    )
    return panel.loc[start:].dropna()


def current_run_length(labels: pd.Series) -> int:
    """Number of consecutive periods the last regime has lasted."""
    values = labels.to_numpy()
    changes = np.flatnonzero(values != values[-1])
    return len(values) if changes.size == 0 else len(values) - 1 - changes[-1]


if __name__ == "__main__":
    df = build_panel()
    print(f"Panel: {df.index[0]:%b %Y} to {df.index[-1]:%b %Y}, {len(df)} months\n")

    pipe = RegimeDetectionPipeline(
        model=GaussianHSMMModel(n_regimes=3, n_init=6, max_duration=120, random_state=0),
        preprocessor=Preprocessor(scale="zscore"),
        min_duration=2,
        label_feature="log_vix",
        label_names=("calm", "normal", "stressed"),
    )
    result = pipe.fit_detect(df)
    hsmm = result.model
    names = result.regime_names

    print("Average conditions by regime:")
    summary = df.groupby(result.smoothed_labels.map(names)).mean()
    summary["log_vix"] = np.exp(summary["log_vix"])
    print(summary.rename(columns={"log_vix": "vix"}).round(2), "\n")

    print("Mean duration (months):")
    for k, d in enumerate(hsmm.mean_durations):
        print(f"  {names[k]:<9} {d:5.1f}")

    current = int(result.smoothed_labels.iloc[-1])
    elapsed = current_run_length(result.smoothed_labels)
    aware = hsmm.forecast(steps=6, current_state=current, elapsed=elapsed)
    blind = hsmm.forecast(steps=6, current_state=current)
    print(f"\nCurrent regime: {names[current]}, for the last {elapsed} months")
    print(f"P(still {names[current]} in 6 months), using elapsed time: {aware[-1, current]:.2f}")
    print(f"P(still {names[current]} in 6 months), duration-blind:     {blind[-1, current]:.2f}")
