"""The two things a user does with a file: train a model from it, or score it."""

from pathlib import Path

import pandas as pd

from churn_platform.drift import (
    DriftReport,
    check_drift,
    load_reference,
    save_reference,
)
from churn_platform.ingest import (
    TRAIN_MODE,
    Issue,
    ValidationReport,
    load_csv,
    validate_upload,
)
from churn_platform.persistence import load_model, save_model
from churn_platform.scoring import HIGH_SHARE, score_customers
from churn_platform.training import TrainingResult, train_baseline


class InvalidFileError(Exception):
    """Raised when a file fails validation; the message lists every problem and fix."""

    def __init__(self, report: ValidationReport) -> None:
        super().__init__(report.describe())
        self.report = report


def train_from_file(csv_path: Path, model_path: Path) -> TrainingResult:
    """Train the baseline model on a CSV that has a ``Churn`` column and save it.

    Raises ``UploadError`` if the file cannot be read, and ``InvalidFileError`` if it
    fails the checks or has no ``Churn`` column to learn from. Nothing is saved then.
    """
    table = load_csv(csv_path)
    report = validate_upload(table)
    if report.is_valid and report.mode != TRAIN_MODE:
        report = ValidationReport(
            mode=report.mode,
            errors=(
                Issue(
                    "The file has no 'Churn' column, so a model cannot learn from it.",
                    "Add a 'Churn' column (Yes or No), or use this file to score "
                    "customers with a model you already trained.",
                ),
            ),
            warnings=report.warnings,
        )
    _require_valid(report)
    result = train_baseline(table)
    save_model(result.pipeline, model_path)
    save_reference(result.drift_reference, model_path)
    return result


def score_file(
    csv_path: Path,
    model_path: Path,
    output_path: Path,
    high_share: float = HIGH_SHARE,
) -> pd.DataFrame:
    """Score every customer in a CSV with a saved model and write the result as a CSV.

    The output has one row per customer: ``customerID``, ``churn_probability`` and
    ``risk_band`` (the riskiest ``high_share`` of customers are High), and, for High
    customers, ``recommended_action`` and ``reason``.
    Raises ``UploadError`` if the file cannot be read, ``InvalidFileError`` if it fails
    the checks, and ``FileNotFoundError`` if there is no saved model. Nothing is
    written then.
    """
    table = load_csv(csv_path)
    _require_valid(validate_upload(table))
    scored = score_customers(load_model(model_path), table, high_share=high_share)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    return scored


def drift_report(csv_path: Path, model_path: Path) -> DriftReport | None:
    """Compare a CSV with the data the saved model learned from.

    Returns None when the model has no saved reference (it was saved before this check
    existed), so there is nothing to compare with. Raises ``UploadError`` if the file
    cannot be read.
    """
    reference = load_reference(model_path)
    if reference is None:
        return None
    return check_drift(reference, load_csv(csv_path))


def _require_valid(report: ValidationReport) -> None:
    if not report.is_valid:
        raise InvalidFileError(report)
