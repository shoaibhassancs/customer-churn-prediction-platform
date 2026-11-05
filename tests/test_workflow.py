"""Tests for churn_platform.workflow."""

import numpy as np
import pandas as pd
import pytest

from churn_platform.drift import reference_path
from churn_platform.ingest import UploadError
from churn_platform.persistence import load_model
from churn_platform.preprocessing import CATEGORICAL_COLUMNS
from churn_platform.workflow import (
    InvalidFileError,
    drift_report,
    score_file,
    train_from_file,
)

# The random-split warning is tested in test_splitting; here it would only add noise.
pytestmark = pytest.mark.filterwarnings("ignore:Random split")


def make_upload(n: int = 600, churners: int = 120, first_id: int = 0) -> pd.DataFrame:
    """A table that passes validation: every required column, 120 of 600 churned."""
    rng = np.random.RandomState(first_id)
    table = {"customerID": [f"C{first_id + i:05d}" for i in range(n)]}
    for column in CATEGORICAL_COLUMNS:
        table[column] = rng.choice(["A", "B", "C"], size=n)
    table["tenure"] = rng.randint(1, 72, size=n)
    table["MonthlyCharges"] = rng.uniform(20, 120, size=n).round(2)
    table["TotalCharges"] = (table["tenure"] * 50.5).astype(str)
    for column in ("Partner", "Dependents", "PhoneService", "PaperlessBilling"):
        table[column] = rng.choice(["Yes", "No"], size=n)
    table["SeniorCitizen"] = rng.randint(0, 2, size=n)
    table["Churn"] = rng.permutation(["Yes"] * churners + ["No"] * (n - churners))
    return pd.DataFrame(table)


def write_csv(table: pd.DataFrame, path):
    table.to_csv(path, index=False)
    return path


def test_train_from_file_saves_a_model_that_can_be_loaded(tmp_path) -> None:
    csv = write_csv(make_upload(), tmp_path / "train.csv")
    model_path = tmp_path / "models" / "baseline.joblib"

    result = train_from_file(csv, model_path)

    assert model_path.exists()
    assert len(result.cv_scores) == 4
    assert load_model(model_path).steps[-1][0] == "model"


def test_a_file_that_fails_the_checks_trains_nothing_and_saves_nothing(
    tmp_path,
) -> None:
    csv = write_csv(make_upload().drop(columns=["customerID"]), tmp_path / "bad.csv")
    model_path = tmp_path / "baseline.joblib"

    with pytest.raises(InvalidFileError, match="customerID") as caught:
        train_from_file(csv, model_path)

    assert "FIX:" in str(caught.value)
    assert not caught.value.report.is_valid
    assert not model_path.exists()


def test_a_file_without_churn_cannot_train_a_model(tmp_path) -> None:
    csv = write_csv(make_upload().drop(columns=["Churn"]), tmp_path / "no_churn.csv")

    with pytest.raises(InvalidFileError, match="no 'Churn' column"):
        train_from_file(csv, tmp_path / "baseline.joblib")

    assert not (tmp_path / "baseline.joblib").exists()


def test_score_file_writes_one_row_per_customer(tmp_path) -> None:
    model_path = tmp_path / "baseline.joblib"
    train_from_file(write_csv(make_upload(), tmp_path / "train.csv"), model_path)
    new_customers = make_upload(n=50, churners=10, first_id=9000).drop(columns="Churn")
    csv = write_csv(new_customers, tmp_path / "new.csv")
    output = tmp_path / "out" / "scores.csv"

    returned = score_file(csv, model_path, output)

    # Blank cells are empty text for customers with no action, not missing values.
    written = pd.read_csv(output, keep_default_na=False)
    assert written.columns.tolist() == [
        "customerID",
        "churn_probability",
        "risk_band",
        "recommended_action",
        "reason",
    ]
    assert written["customerID"].tolist() == new_customers["customerID"].tolist()
    assert written["churn_probability"].between(0, 1).all()
    pd.testing.assert_frame_equal(returned, written)


def test_score_file_marks_the_riskiest_share_as_high(tmp_path) -> None:
    model_path = tmp_path / "baseline.joblib"
    train_from_file(write_csv(make_upload(), tmp_path / "train.csv"), model_path)
    new_customers = make_upload(n=50, churners=10, first_id=9000).drop(columns="Churn")
    csv = write_csv(new_customers, tmp_path / "new.csv")

    scored = score_file(csv, model_path, tmp_path / "scores.csv", high_share=0.30)

    assert (scored["risk_band"] == "High").sum() == 15  # 30% of 50
    assert (scored["risk_band"] == "Medium").sum() == 10  # next 20%


def test_scoring_without_a_saved_model_stops_before_writing(tmp_path) -> None:
    csv = write_csv(make_upload().drop(columns="Churn"), tmp_path / "new.csv")
    output = tmp_path / "scores.csv"

    with pytest.raises(FileNotFoundError, match="No saved model"):
        score_file(csv, tmp_path / "missing.joblib", output)

    assert not output.exists()


def test_scoring_a_file_that_fails_the_checks_writes_nothing(tmp_path) -> None:
    model_path = tmp_path / "baseline.joblib"
    train_from_file(write_csv(make_upload(), tmp_path / "train.csv"), model_path)
    broken = make_upload().drop(columns=["Contract", "Churn"])
    output = tmp_path / "scores.csv"

    with pytest.raises(InvalidFileError, match="Contract"):
        score_file(write_csv(broken, tmp_path / "bad.csv"), model_path, output)

    assert not output.exists()


def test_an_unreadable_file_is_reported_as_an_upload_error(tmp_path) -> None:
    with pytest.raises(UploadError, match="could not find"):
        train_from_file(tmp_path / "missing.csv", tmp_path / "baseline.joblib")


def test_training_saves_a_reference_next_to_the_model(tmp_path) -> None:
    model_path = tmp_path / "baseline.joblib"

    train_from_file(write_csv(make_upload(), tmp_path / "train.csv"), model_path)

    assert reference_path(model_path).exists()


def test_a_file_like_the_training_data_shows_no_drift(tmp_path) -> None:
    model_path = tmp_path / "baseline.joblib"
    csv = write_csv(make_upload(), tmp_path / "train.csv")
    train_from_file(csv, model_path)

    report = drift_report(csv, model_path)

    assert report is not None
    assert report.reliable
    assert report.flagged().empty


def test_a_file_with_a_different_tenure_spread_is_flagged(tmp_path) -> None:
    model_path = tmp_path / "baseline.joblib"
    train_from_file(write_csv(make_upload(), tmp_path / "train.csv"), model_path)
    shifted = make_upload().drop(columns="Churn").assign(tenure=1)
    csv = write_csv(shifted, tmp_path / "shifted.csv")

    report = drift_report(csv, model_path)

    assert report is not None
    assert "tenure" in report.flagged().index


def test_a_model_without_a_reference_file_is_not_checked(tmp_path) -> None:
    model_path = tmp_path / "baseline.joblib"
    csv = write_csv(make_upload(), tmp_path / "train.csv")
    train_from_file(csv, model_path)
    reference_path(model_path).unlink()

    assert drift_report(csv, model_path) is None
