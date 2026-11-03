"""Plotting utilities.

matplotlib/seaborn are imported lazily inside each function so the core
package has no hard plotting dependency at import time (useful in headless
production jobs that only need fit/predict).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_DEFAULT_COLORS = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
]


def _regime_palette(n: int) -> list[str]:
    if n <= len(_DEFAULT_COLORS):
        return _DEFAULT_COLORS[:n]
    import matplotlib.cm as cm

    return [cm.tab20(i % 20) for i in range(n)]


def plot_regimes(
    series: pd.Series,
    labels: pd.Series,
    *,
    n_regimes: int | None = None,
    regime_names: dict[int, str] | None = None,
    title: str = "Regime detection",
    ax=None,
):
    """Plot a reference series with shaded regime backgrounds.

    Parameters
    ----------
    series : pandas.Series
        Reference series to draw (e.g. PMI level or an equity index).
    labels : pandas.Series
        Regime labels, aligned to ``series.index``.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4.5))

    labels = labels.reindex(series.index).ffill()
    k = n_regimes if n_regimes is not None else int(labels.max()) + 1
    colors = _regime_palette(k)

    idx = series.index
    lab = labels.to_numpy()
    start = 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[start]:
            r = int(lab[start])
            ax.axvspan(
                idx[start],
                idx[min(i, len(idx) - 1)],
                color=colors[r % len(colors)],
                alpha=0.18,
                lw=0,
            )
            start = i

    ax.plot(idx, series.to_numpy(), color="black", lw=1.3)
    ax.set_title(title)
    ax.margins(x=0)

    from matplotlib.patches import Patch

    handles = [
        Patch(
            facecolor=colors[r % len(colors)],
            alpha=0.5,
            label=(regime_names or {}).get(r, f"regime {r}"),
        )
        for r in range(k)
    ]
    ax.legend(handles=handles, loc="upper left", ncol=k, fontsize=9, framealpha=0.9)
    return ax


def plot_regime_probabilities(proba: pd.DataFrame, *, title="Regime probabilities", ax=None):
    """Stacked area plot of posterior regime probabilities over time."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.5))
    colors = _regime_palette(proba.shape[1])
    ax.stackplot(
        proba.index, proba.to_numpy().T, colors=colors, alpha=0.85, labels=list(proba.columns)
    )
    ax.set_ylim(0, 1)
    ax.margins(x=0)
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=proba.shape[1], fontsize=9)
    return ax


def plot_transition_matrix(T: np.ndarray, *, regime_names=None, title="Transition matrix", ax=None):
    """Heatmap of a transition matrix with probabilities annotated."""
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        have_sns = True
    except ImportError:
        have_sns = False

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    k = T.shape[0]
    names = [(regime_names or {}).get(i, str(i)) for i in range(k)]

    if have_sns:
        sns.heatmap(
            T,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            xticklabels=names,
            yticklabels=names,
            ax=ax,
            cbar=True,
        )
    else:
        im = ax.imshow(T, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(k), names)
        ax.set_yticks(range(k), names)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{T[i, j]:.2f}", ha="center", va="center")
        ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel("to regime")
    ax.set_ylabel("from regime")
    ax.set_title(title)
    return ax
