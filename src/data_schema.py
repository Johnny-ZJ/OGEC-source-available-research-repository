from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import pandas as pd


@dataclass(frozen=True)
class SchemaReport:
    n_rows: int
    n_columns: int
    missing_required: tuple[str, ...]
    present_required: tuple[str, ...]


def check_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
) -> SchemaReport:
    required = list(required_columns)
    missing = tuple(c for c in required if c not in df.columns)
    present = tuple(c for c in required if c in df.columns)
    return SchemaReport(
        n_rows=len(df),
        n_columns=len(df.columns),
        missing_required=missing,
        present_required=present,
    )


def assert_no_duplicate_subject_rows(
    df: pd.DataFrame,
    subject_column: str,
) -> None:
    """Optional guard for subject-level tables.

    Do not use this check when a source file legitimately stores multiple rows
    per participant.
    """
    if df[subject_column].duplicated().any():
        raise ValueError(
            f"Duplicate values found in subject column '{subject_column}'. "
            "Confirm whether the input is truly a one-row-per-subject table."
        )
