"""Load an uploaded CSV and check it, in plain language, before anything uses it.

The checks cover what the rest of the code relies on. Each problem comes with a
suggested fix as text; nothing in the uploaded data is changed.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from churn_platform.cleaning import CHURN, TOTAL_CHARGES, YES_NO_COLUMNS
from churn_platform.preprocessing import (
    CATEGORICAL_COLUMNS,
    FLAG_COLUMNS,
    NUMERIC_COLUMNS,
)
from churn_platform.scoring import ID_COLUMN

MAX_FILE_MB = 200
# Starting thresholds from the plan: below these, warn rather than refuse.
MIN_ROWS = 500
MIN_CHURNERS = 50
TRAIN_MODE = "train"
SCORE_MODE = "score"
YES_NO = ("Yes", "No")
REQUIRED_FEATURES = (*CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS, *FLAG_COLUMNS)
_PLAIN_NUMBER_COLUMNS = ("tenure", "MonthlyCharges")


@dataclass(frozen=True)
class Issue:
    """One problem found in a file: what is wrong, and how to fix it."""

    message: str
    fix: str


@dataclass(frozen=True)
class ValidationReport:
    """The result of checking a table: its mode, blocking errors and warnings."""

    mode: str
    errors: tuple[Issue, ...]
    warnings: tuple[Issue, ...]

    @property
    def is_valid(self) -> bool:
        """True when nothing blocks training or scoring (warnings do not block)."""
        return not self.errors

    def describe(self) -> str:
        """Return the report as plain-language text, one problem and fix per line."""
        purpose = {
            TRAIN_MODE: "train a model and score its customers",
            SCORE_MODE: "score its customers with a saved model",
        }[self.mode]
        lines = [f"This file will be used to {purpose}."]
        lines += [f"PROBLEM: {i.message} FIX: {i.fix}" for i in self.errors]
        lines += [f"WARNING: {i.message} SUGGESTION: {i.fix}" for i in self.warnings]
        if self.is_valid and not self.warnings:
            lines.append("No problems found.")
        return "\n".join(lines)


class UploadError(Exception):
    """Raised when a file cannot even be read; carries the same message and fix."""

    def __init__(self, issue: Issue) -> None:
        super().__init__(f"{issue.message} {issue.fix}")
        self.issue = issue


def load_csv(path: Path, max_mb: float = MAX_FILE_MB) -> pd.DataFrame:
    """Read a CSV file into a table, or raise ``UploadError`` saying what to fix."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise UploadError(
            Issue(
                f"Files ending in '{path.suffix}' are not supported yet.",
                "Save the file as a CSV (comma-separated) and upload it again.",
            )
        )
    if not path.exists():
        raise UploadError(
            Issue("We could not find that file.", "Check the name and upload it again.")
        )
    size_mb = path.stat().st_size / 1_000_000
    if size_mb > max_mb:
        raise UploadError(
            Issue(
                f"The file is {size_mb:.0f} MB, over the {max_mb:g} MB limit.",
                "Upload a smaller file, for example with fewer rows.",
            )
        )
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise UploadError(
            Issue("The file is empty.", "Upload a file with a header row and rows.")
        ) from None
    except (pd.errors.ParserError, UnicodeDecodeError):
        raise UploadError(
            Issue(
                "We could not read the file as a table.",
                "Save it as a plain CSV with one header row and upload it again.",
            )
        ) from None


def validate_upload(
    df: pd.DataFrame, target_column: str = CHURN, id_column: str = ID_COLUMN
) -> ValidationReport:
    """Check an uploaded table and report every problem found, not just the first.

    A table with a ``target_column`` is for training and scoring ("train" mode); one
    without it is only scored ("score" mode).
    """
    mode = TRAIN_MODE if target_column in df.columns else SCORE_MODE
    if df.empty:
        empty = Issue("The file has no customer rows.", "Add one row per customer.")
        return ValidationReport(mode, (empty,), ())

    errors = [
        *_check_id(df, id_column),
        *_check_required_columns(df),
        *_check_yes_no_columns(df),
        *_check_number_columns(df),
        *_check_total_charges(df),
    ]
    warnings = list(_warn_blank_charges_after_start(df))
    if mode == TRAIN_MODE:
        target_errors, target_warnings = _check_target(df[target_column])
        errors += target_errors
        warnings += target_warnings
    return ValidationReport(mode, tuple(errors), tuple(warnings))


def _examples(values: Sequence[object]) -> str:
    """Show up to five values as text, for use inside a message."""
    shown = ", ".join(repr(v) for v in values[:5])
    return shown + (", ..." if len(values) > 5 else "")


def _check_id(df: pd.DataFrame, id_column: str) -> list[Issue]:
    if id_column not in df.columns:
        return [
            Issue(
                f"There is no '{id_column}' column to tell customers apart.",
                f"Add a column named '{id_column}' with one unique value per customer.",
            )
        ]
    repeated = int(df[id_column].duplicated().sum())
    if repeated:
        return [
            Issue(
                f"{repeated} rows repeat a {id_column} that appears earlier.",
                "Keep one row per customer: remove or merge the repeated rows.",
            )
        ]
    return []


def _check_required_columns(df: pd.DataFrame) -> list[Issue]:
    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if not missing:
        return []
    return [
        Issue(
            f"These columns are missing: {', '.join(missing)}.",
            "Add them using exactly these names, or rename your columns to match.",
        )
    ]


def _check_yes_no_columns(df: pd.DataFrame) -> list[Issue]:
    issues = []
    for column in YES_NO_COLUMNS:
        if column not in df.columns:
            continue
        other = df[column][~df[column].isin(YES_NO)]
        if not other.empty:
            found = _examples(other.fillna("(blank)").unique().tolist())
            issues.append(
                Issue(
                    f"'{column}' should only contain Yes or No, but also has: {found}.",
                    "Replace those values with Yes or No.",
                )
            )
    return issues


def _check_number_columns(df: pd.DataFrame) -> list[Issue]:
    issues = []
    for column in _PLAIN_NUMBER_COLUMNS:
        if column not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            issues.append(
                Issue(
                    f"'{column}' should be numbers, but some values contain text.",
                    "Remove words or units so every value is a plain number.",
                )
            )
        elif df[column].isna().any():
            empty = int(df[column].isna().sum())
            issues.append(
                Issue(
                    f"'{column}' has {empty} empty cells.",
                    "Fill in a number for every customer.",
                )
            )
    return issues


def _check_total_charges(df: pd.DataFrame) -> list[Issue]:
    if TOTAL_CHARGES not in df.columns or pd.api.types.is_numeric_dtype(
        df[TOTAL_CHARGES]
    ):
        return []
    text = df[TOTAL_CHARGES].dropna().str.strip()
    filled = text[text != ""]
    bad = filled[pd.to_numeric(filled, errors="coerce").isna()]
    if bad.empty:
        return []
    return [
        Issue(
            f"'{TOTAL_CHARGES}' has values that are not numbers ({len(bad)} in all), "
            f"such as {_examples(bad.unique().tolist())}.",
            "Replace them with numbers, or leave the cell blank for customers who "
            "have not been billed yet.",
        )
    ]


def _check_target(target: pd.Series) -> tuple[list[Issue], list[Issue]]:
    other = target[~target.isin(YES_NO)]
    if not other.empty:
        found = _examples(other.fillna("(blank)").unique().tolist())
        return [
            Issue(
                f"'{CHURN}' should only contain Yes or No, but also has: {found}.",
                "Recode it so Yes means the customer left and No means they stayed.",
            )
        ], []
    churners = int((target == "Yes").sum())
    if churners in (0, len(target)):
        return [
            Issue(
                f"'{CHURN}' has only one outcome, so nothing can be learned from it.",
                "Upload a file where some customers left and some stayed, or remove "
                f"the '{CHURN}' column to only score the customers.",
            )
        ], []
    warnings = []
    if len(target) < MIN_ROWS:
        warnings.append(
            Issue(
                f"The file has only {len(target)} customers (we suggest {MIN_ROWS}+).",
                "Results will be unstable; add more customers if you can.",
            )
        )
    if churners < MIN_CHURNERS:
        warnings.append(
            Issue(
                f"Only {churners} customers left (we suggest {MIN_CHURNERS}+).",
                "Scores will be unreliable; add more churned customers if you can.",
            )
        )
    return [], warnings


def _warn_blank_charges_after_start(df: pd.DataFrame) -> list[Issue]:
    if TOTAL_CHARGES not in df.columns or "tenure" not in df.columns:
        return []
    charges = df[TOTAL_CHARGES]
    if pd.api.types.is_numeric_dtype(charges):
        blank = charges.isna()
    else:
        blank = charges.isna() | (charges.str.strip() == "")
    started = pd.to_numeric(df["tenure"], errors="coerce") > 0
    count = int((blank & started).sum())
    if not count:
        return []
    return [
        Issue(
            f"{count} customers have been with you for more than 0 months but have "
            f"no {TOTAL_CHARGES}. They will be treated as billed 0.",
            "Check those rows if the missing totals are a mistake.",
        )
    ]
