"""Tests for churn_platform.ingest."""

import numpy as np
import pandas as pd
import pytest

from churn_platform.ingest import (
    UploadError,
    load_csv,
    validate_upload,
)
from churn_platform.preprocessing import CATEGORICAL_COLUMNS


def make_upload(n: int = 600, churners: int = 120) -> pd.DataFrame:
    """A table that passes every check: all columns present, 600 rows, 120 churners."""
    rng = np.random.RandomState(0)
    table = {"customerID": [f"C{i:04d}" for i in range(n)]}
    for column in CATEGORICAL_COLUMNS:
        table[column] = rng.choice(["A", "B", "C"], size=n)
    table["tenure"] = rng.randint(1, 72, size=n)
    table["MonthlyCharges"] = rng.uniform(20, 120, size=n).round(2)
    table["TotalCharges"] = (table["tenure"] * 50.5).astype(str)
    for column in ("Partner", "Dependents", "PhoneService", "PaperlessBilling"):
        table[column] = rng.choice(["Yes", "No"], size=n)
    table["SeniorCitizen"] = rng.randint(0, 2, size=n)
    table["Churn"] = ["Yes"] * churners + ["No"] * (n - churners)
    return pd.DataFrame(table)


def test_a_good_training_file_has_no_problems_and_is_in_train_mode() -> None:
    report = validate_upload(make_upload())

    assert report.is_valid
    assert report.mode == "train"
    assert report.errors == () and report.warnings == ()
    assert "No problems found" in report.describe()


def test_a_file_without_a_churn_column_is_in_score_mode() -> None:
    report = validate_upload(make_upload().drop(columns=["Churn"]))

    assert report.is_valid
    assert report.mode == "score"


def test_missing_id_column_is_reported() -> None:
    report = validate_upload(make_upload().drop(columns=["customerID"]))

    assert not report.is_valid
    assert "customerID" in report.errors[0].message


def test_repeated_ids_are_reported_with_a_count() -> None:
    table = make_upload()
    table.loc[1, "customerID"] = table.loc[0, "customerID"]

    report = validate_upload(table)

    assert any("1 rows repeat" in e.message for e in report.errors)


def test_missing_feature_columns_are_all_listed() -> None:
    report = validate_upload(make_upload().drop(columns=["Contract", "tenure"]))

    message = report.errors[0].message
    assert "Contract" in message and "tenure" in message


def test_a_value_other_than_yes_or_no_in_a_flag_column_is_reported() -> None:
    table = make_upload()
    table.loc[0, "Partner"] = "Maybe"

    report = validate_upload(table)

    assert any("Partner" in e.message and "Maybe" in e.message for e in report.errors)


def test_text_in_a_number_column_is_reported() -> None:
    table = make_upload()
    table["tenure"] = table["tenure"].astype(str)
    table.loc[0, "tenure"] = "12 months"

    report = validate_upload(table)

    assert any("tenure" in e.message for e in report.errors)


def test_non_numeric_total_charges_is_reported_but_blanks_are_fine() -> None:
    table = make_upload()
    table.loc[0, "TotalCharges"] = " "
    assert validate_upload(table).is_valid

    table.loc[1, "TotalCharges"] = "abc"
    report = validate_upload(table)

    messages = [e.message for e in report.errors]
    assert any("TotalCharges" in m and "abc" in m for m in messages)


def test_churn_values_other_than_yes_or_no_are_reported() -> None:
    table = make_upload()
    table.loc[0, "Churn"] = "left"

    report = validate_upload(table)

    assert any("Churn" in e.message and "left" in e.message for e in report.errors)


def test_a_churn_column_with_one_outcome_is_reported() -> None:
    report = validate_upload(make_upload(churners=0))

    assert any("only one outcome" in e.message for e in report.errors)


def test_a_small_file_only_warns_it_does_not_block() -> None:
    report = validate_upload(make_upload(n=100, churners=30))

    assert report.is_valid
    assert any("only 100 customers" in w.message for w in report.warnings)


def test_few_churners_only_warns() -> None:
    report = validate_upload(make_upload(n=600, churners=10))

    assert report.is_valid
    assert any("Only 10 customers left" in w.message for w in report.warnings)


def test_blank_total_charges_warns_only_for_customers_past_month_zero() -> None:
    table = make_upload()
    table.loc[0, ["tenure", "TotalCharges"]] = [0, " "]
    assert validate_upload(table).warnings == ()

    table.loc[1, "TotalCharges"] = " "  # tenure above 0
    report = validate_upload(table)

    assert report.is_valid
    assert any("1 customers" in w.message for w in report.warnings)


def test_every_problem_is_reported_together_each_with_a_fix() -> None:
    table = make_upload().drop(columns=["customerID", "Contract"])
    table.loc[0, "Partner"] = "Maybe"

    report = validate_upload(table)

    assert len(report.errors) == 3
    assert all(issue.fix for issue in report.errors)
    text = report.describe()
    assert "PROBLEM:" in text and "FIX:" in text and "Traceback" not in text


def test_an_empty_table_is_reported() -> None:
    report = validate_upload(make_upload().iloc[0:0])

    assert not report.is_valid
    assert "no customer rows" in report.errors[0].message


def test_load_csv_reads_a_valid_file(tmp_path) -> None:
    path = tmp_path / "customers.csv"
    make_upload(n=10, churners=3).to_csv(path, index=False)

    table = load_csv(path)

    assert len(table) == 10 and "customerID" in table.columns


@pytest.mark.parametrize(
    ("make_file", "expected"),
    [
        (lambda p: p / "missing.csv", "could not find"),
        (lambda p: _write(p / "empty.csv", ""), "empty"),
        (lambda p: _write(p / "book.xlsx", "x"), "not supported"),
    ],
)
def test_load_csv_explains_unreadable_files(tmp_path, make_file, expected) -> None:
    with pytest.raises(UploadError, match=expected) as caught:
        load_csv(make_file(tmp_path))

    assert caught.value.issue.fix


def test_load_csv_refuses_a_file_over_the_size_limit(tmp_path) -> None:
    path = tmp_path / "customers.csv"
    make_upload(n=50, churners=10).to_csv(path, index=False)

    with pytest.raises(UploadError, match="limit"):
        load_csv(path, max_mb=0.0001)


def _write(path, text):
    path.write_text(text)
    return path
