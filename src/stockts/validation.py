"""Data-validation checks run in CI before forecasts are trusted.

Each check returns a list of human-readable problems; an empty list means the
frame passed.  :func:`validate_history` aggregates them and
:func:`assert_valid` raises when anything failed (used by the CI entrypoint).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["ValidationReport", "validate_history", "assert_valid"]


@dataclass
class ValidationReport:
    problems: list[str] = field(default_factory=list)
    n_tickers: int = 0
    n_rows: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems

    def __str__(self) -> str:
        head = f"{self.n_tickers} tickers x {self.n_rows} rows"
        if self.ok:
            return f"PASS - {head}: no problems found"
        lines = "\n".join(f"  - {p}" for p in self.problems)
        return f"FAIL - {head}:\n{lines}"


def validate_history(
    history: pd.DataFrame,
    min_rows: int = 30,
    max_nan_fraction: float = 0.5,
    max_daily_move: float = 0.75,
) -> ValidationReport:
    """Validate a wide price-history frame.

    Checks performed:
      * the frame is non-empty and has a sorted, unique, datetime-like index;
      * enough rows are present;
      * no all-NaN or excessively-NaN columns;
      * no non-positive prices;
      * no implausibly large single-day moves (likely bad ticks).
    """
    report = ValidationReport(
        n_tickers=history.shape[1], n_rows=history.shape[0]
    )
    p = report.problems

    if history.empty:
        p.append("history is empty")
        return report

    if history.shape[0] < min_rows:
        p.append(f"only {history.shape[0]} rows (< {min_rows})")

    if not history.index.is_monotonic_increasing:
        p.append("index is not sorted ascending")
    if history.index.has_duplicates:
        p.append("index has duplicate dates")

    for col in history.columns:
        s = history[col]
        nan_frac = float(s.isna().mean())
        if nan_frac >= 1.0:
            p.append(f"{col}: all values are NaN")
            continue
        if nan_frac > max_nan_fraction:
            p.append(f"{col}: {nan_frac:.0%} NaN (> {max_nan_fraction:.0%})")

        vals = s.dropna()
        if (vals <= 0).any():
            p.append(f"{col}: contains non-positive prices")
        if not np.isfinite(vals.to_numpy()).all():
            p.append(f"{col}: contains non-finite values")

        if vals.size > 1:
            moves = vals.pct_change().abs().dropna()
            if (moves > max_daily_move).any():
                worst = float(moves.max())
                p.append(
                    f"{col}: implausible daily move {worst:.0%} (> {max_daily_move:.0%})"
                )

    return report


def assert_valid(history: pd.DataFrame, **kwargs) -> ValidationReport:
    """Validate and raise ``ValueError`` if any problem was found."""
    report = validate_history(history, **kwargs)
    if not report.ok:
        raise ValueError(str(report))
    return report
