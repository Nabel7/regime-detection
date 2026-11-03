"""Flexible ingestion and validation for macroeconomic time series.

The loaders here enforce a single **data contract** so any macro series —
employment, PMI, CPI, inflation expectations, commodity/equity returns —
flows through the rest of the package unchanged:

* a ``pandas.DataFrame`` indexed by a sorted, unique ``DatetimeIndex``;
* numeric feature columns only;
* an explicit, declared frequency (monthly by default).

Mixed-cadence inputs (e.g. monthly PMI alongside quarterly growth) are
aligned onto a common grid by :func:`align_frequencies`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class DataValidationError(ValueError):
    """Raised when an input series violates the data contract."""


@dataclass
class ValidationReport:
    """Summary of a validation pass (returned by :func:`validate_frame`)."""

    n_obs: int
    n_features: int
    start: pd.Timestamp
    end: pd.Timestamp
    inferred_freq: str | None
    missing_per_column: dict[str, int]
    warnings: list[str]

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lines = [
            f"ValidationReport: {self.n_obs} obs x {self.n_features} features",
            f"  span      : {self.start.date()} -> {self.end.date()}",
            f"  freq      : {self.inferred_freq}",
            f"  missing   : {self.missing_per_column}",
        ]
        if self.warnings:
            lines.append("  warnings  :")
            lines.extend(f"    - {w}" for w in self.warnings)
        return "\n".join(lines)


def validate_frame(
    df: pd.DataFrame,
    *,
    require_numeric: bool = True,
    max_missing_frac: float = 0.5,
    allow_duplicates: bool = False,
) -> ValidationReport:
    """Validate a DataFrame against the package data contract.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have a ``DatetimeIndex``.
    require_numeric : bool, default True
        If True, all columns must be numeric.
    max_missing_frac : float, default 0.5
        Columns missing more than this fraction of observations raise.
    allow_duplicates : bool, default False
        If False, duplicate index timestamps raise.

    Returns
    -------
    ValidationReport

    Raises
    ------
    DataValidationError
        On any hard contract violation.
    """
    warnings_: list[str] = []

    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError(
            "Index must be a DatetimeIndex. Use load_csv(..., date_col=...) "
            "or df.set_index(pd.to_datetime(df[col]))."
        )
    if df.shape[1] == 0:
        raise DataValidationError("DataFrame has no feature columns.")
    if df.shape[0] < 2:
        raise DataValidationError("Need at least 2 observations.")

    if not df.index.is_monotonic_increasing:
        warnings_.append("Index was not sorted; sort before fitting.")
    if df.index.has_duplicates and not allow_duplicates:
        dupes = df.index[df.index.duplicated()].unique()
        raise DataValidationError(f"Duplicate timestamps in index: {list(dupes[:5])}...")

    if require_numeric:
        non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise DataValidationError(f"Non-numeric columns: {non_numeric}")

    missing = {c: int(df[c].isna().sum()) for c in df.columns}
    for col, n_missing in missing.items():
        frac = n_missing / len(df)
        if frac > max_missing_frac:
            raise DataValidationError(
                f"Column '{col}' is {frac:.0%} missing (> {max_missing_frac:.0%})."
            )
    constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant:
        warnings_.append(f"Constant/degenerate columns (will break covariance): {constant}")

    inferred = pd.infer_freq(df.index) if df.index.is_monotonic_increasing else None
    if inferred is None:
        warnings_.append("Could not infer a regular frequency; cadence may be irregular.")

    return ValidationReport(
        n_obs=df.shape[0],
        n_features=df.shape[1],
        start=df.index.min(),
        end=df.index.max(),
        inferred_freq=inferred,
        missing_per_column=missing,
        warnings=warnings_,
    )


def load_csv(
    path: str | Path,
    *,
    date_col: str | int = 0,
    feature_cols: Sequence[str] | None = None,
    freq: str | None = "MS",
    parse_dates: bool = True,
    sort: bool = True,
    validate: bool = True,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load a macro time series from CSV into a contract-compliant DataFrame.

    Parameters
    ----------
    path : str or Path
    date_col : str or int, default 0
        Column to parse as the datetime index.
    feature_cols : sequence of str, optional
        Subset of columns to keep as features. Defaults to all non-date columns.
    freq : str, optional, default "MS"
        Target pandas frequency to conform the index to (``"MS"`` = month
        start). Pass ``None`` to leave the native cadence untouched.
    parse_dates, sort, validate : bool
        Self-explanatory toggles.
    **read_csv_kwargs
        Forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
    """
    df = pd.read_csv(path, **read_csv_kwargs)
    col = df.columns[date_col] if isinstance(date_col, int) else date_col
    if parse_dates:
        df[col] = pd.to_datetime(df[col])
    df = df.set_index(col)

    if feature_cols is not None:
        df = df[list(feature_cols)]

    if sort:
        df = df.sort_index()
    if freq is not None:
        df = df.asfreq(freq)
    if validate:
        report = validate_frame(df)
        for w in report.warnings:
            import warnings as _w

            _w.warn(f"[load_csv] {w}", stacklevel=2)
    return df


def align_frequencies(
    frames: dict[str, pd.DataFrame] | Iterable[pd.DataFrame],
    *,
    target_freq: str = "MS",
    method: str = "ffill",
    how: str = "outer",
) -> pd.DataFrame:
    """Align mixed-cadence series onto a common frequency grid.

    Typical use: monthly PMI/CPI alongside quarterly growth proxies. Lower
    frequency series are upsampled and forward-filled (the standard,
    publication-lag-respecting choice for macro nowcasting).

    Parameters
    ----------
    frames : dict[str, DataFrame] or iterable of DataFrame
        If a dict, keys are used to namespace overlapping column names.
    target_freq : str, default "MS"
    method : {"ffill", "interpolate", "none"}, default "ffill"
    how : {"outer", "inner"}, default "outer"
        Join policy across the source indices.

    Returns
    -------
    pandas.DataFrame on the target frequency.
    """
    if isinstance(frames, dict):
        named = {k: v.copy() for k, v in frames.items()}
        renamed = []
        for name, fr in named.items():
            fr = fr.copy()
            fr.columns = [f"{name}__{c}" if len(named) > 1 else c for c in fr.columns]
            renamed.append(fr)
        frame_list = renamed
    else:
        frame_list = [f.copy() for f in frames]

    combined = pd.concat(frame_list, axis=1, join=how).sort_index()
    grid = pd.date_range(combined.index.min(), combined.index.max(), freq=target_freq)
    combined = combined.reindex(combined.index.union(grid)).sort_index()

    if method == "ffill":
        combined = combined.ffill()
    elif method == "interpolate":
        combined = combined.interpolate(method="time")
    elif method != "none":
        raise ValueError(f"Unknown method={method!r}.")

    return combined.reindex(grid)


class MacroDataLoader:
    """Convenience facade bundling load + align + validate.

    Examples
    --------
    >>> loader = MacroDataLoader(freq="MS")
    >>> df = loader.from_csv("macro.csv", date_col="date")
    >>> report = loader.last_report
    """

    def __init__(self, freq: str = "MS") -> None:
        self.freq = freq
        self.last_report: ValidationReport | None = None

    def from_csv(self, path: str | Path, **kwargs) -> pd.DataFrame:
        kwargs.setdefault("freq", self.freq)
        df = load_csv(path, validate=False, **kwargs)
        self.last_report = validate_frame(df)
        return df

    def from_frame(self, df: pd.DataFrame, *, conform_freq: bool = True) -> pd.DataFrame:
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataValidationError("DataFrame must already have a DatetimeIndex.")
        df = df.sort_index()
        if conform_freq:
            df = df.asfreq(self.freq)
        self.last_report = validate_frame(df)
        return df
