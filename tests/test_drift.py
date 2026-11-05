"""Tests for churn_platform.drift."""

import json

import numpy as np
import pandas as pd
import pytest

from churn_platform.baselines import build_logistic_baseline
from churn_platform.drift import (
    MIN_ROWS,
    build_reference,
    check_drift,
    column_kinds,
    compute_psi,
    level_of,
    load_reference,
    reference_path,
    save_reference,
)
from churn_platform.preprocessing import build_preprocessor


def make_customers(n: int = 1000, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Raw-style features; short tenure predicts churn."""
    rng = np.random.RandomState(seed)
    tenure = rng.randint(0, 72, size=n)
    churn = (tenure < 20).astype(int)
    X = pd.DataFrame(
        {
            "customerID": [f"C{i:05d}" for i in range(n)],
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
            "tenure": tenure,
            "TotalCharges": rng.uniform(0, 3000, size=n).round(2).astype(str),
            "Partner": rng.choice(["Yes", "No"], size=n),
            "Dependents": rng.choice(["Yes", "No"], size=n),
            "PhoneService": rng.choice(["Yes", "No"], size=n),
            "PaperlessBilling": rng.choice(["Yes", "No"], size=n),
        }
    )
    return X, pd.Series(churn, name="Churn")


def make_fitted_pipeline(X: pd.DataFrame, y: pd.Series):
    preprocessor = build_preprocessor(
        categorical_columns=["Contract"],
        numeric_columns=["tenure", "TotalCharges"],
        flag_columns=["Partner", "Dependents", "PhoneService", "PaperlessBilling"],
    )
    return build_logistic_baseline(preprocessor).fit(X, y)


@pytest.fixture
def trained():
    X, y = make_customers()
    pipeline = make_fitted_pipeline(X, y)
    return X, build_reference(pipeline, X), pipeline


def test_numbers_and_categories_are_told_apart_from_the_preprocessor(trained) -> None:
    _, _, pipeline = trained

    kinds = column_kinds(pipeline)

    assert kinds["tenure"] == kinds["TotalCharges"] == "numeric"
    assert kinds["Contract"] == kinds["Partner"] == "categorical"
    assert "customerID" not in kinds


def test_the_reference_describes_each_input_column_and_not_the_id(trained) -> None:
    _, reference, _ = trained

    assert set(reference) == {
        "Contract",
        "tenure",
        "TotalCharges",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    }
    for entry in reference.values():
        assert sum(entry["shares"]) == pytest.approx(1.0)


def test_the_same_data_shows_no_drift_in_any_column(trained) -> None:
    X, reference, _ = trained

    psi = compute_psi(reference, X)

    assert (psi < 0.01).all()


def test_fresh_customers_from_the_same_population_show_no_drift(trained) -> None:
    _, reference, _ = trained
    fresh, _ = make_customers(n=1000, seed=99)

    report = check_drift(reference, fresh)

    assert report.reliable
    assert report.flagged().empty


def test_a_big_shift_in_a_number_is_flagged_as_major_and_only_that_column(
    trained,
) -> None:
    X, reference, _ = trained
    shifted = X.assign(tenure=np.random.RandomState(1).randint(0, 8, size=len(X)))

    report = check_drift(reference, shifted)

    assert report.table.loc["tenure", "level"] == "major"
    assert report.flagged().index.tolist() == ["tenure"]


def test_a_category_the_model_never_saw_is_flagged(trained) -> None:
    X, reference, _ = trained
    shifted = X.assign(Contract="Weekly")

    report = check_drift(reference, shifted)

    assert report.table.loc["Contract", "level"] == "major"


def test_the_table_is_sorted_with_the_biggest_shift_first(trained) -> None:
    X, reference, _ = trained
    shifted = X.assign(tenure=np.random.RandomState(1).randint(0, 8, size=len(X)))

    report = check_drift(reference, shifted)

    assert report.table.index[0] == "tenure"
    assert report.table["psi"].is_monotonic_decreasing


def test_a_column_missing_from_the_new_file_is_skipped_not_a_crash(trained) -> None:
    X, reference, _ = trained

    psi = compute_psi(reference, X.drop(columns=["Contract"]))

    assert "Contract" not in psi.index
    assert "tenure" in psi.index


def test_blank_cells_do_not_crash_and_are_not_treated_as_drift(trained) -> None:
    X, reference, _ = trained
    with_blanks = X.copy()
    with_blanks.loc[:4, "TotalCharges"] = " "  # unreadable number
    with_blanks.loc[:4, "Partner"] = None  # missing category

    report = check_drift(reference, with_blanks)

    assert report.table["level"].eq("none").all()


@pytest.mark.parametrize(
    ("psi", "level"),
    [(0.0, "none"), (0.0999, "none"), (0.1, "moderate"), (0.2499, "moderate")]
    + [(0.25, "major"), (1.5, "major")],
)
def test_the_levels_change_at_the_agreed_cutoffs(psi, level) -> None:
    assert level_of(psi) == level


def test_a_small_file_is_marked_as_unreliable(trained) -> None:
    X, reference, _ = trained

    small = check_drift(reference, X.head(MIN_ROWS - 1))
    big = check_drift(reference, X.head(MIN_ROWS))

    assert not small.reliable
    assert big.reliable
    assert small.rows == MIN_ROWS - 1


def test_the_reference_is_saved_next_to_the_model_as_json(trained, tmp_path) -> None:
    _, reference, _ = trained
    model_path = tmp_path / "baseline.joblib"

    written = save_reference(reference, model_path)

    assert written == tmp_path / "baseline.reference.json"
    assert reference_path(model_path) == written
    assert json.loads(written.read_text()) == reference
    assert load_reference(model_path) == reference


def test_no_saved_reference_gives_none_not_an_error(tmp_path) -> None:
    assert load_reference(tmp_path / "baseline.joblib") is None


def test_the_reference_holds_no_customer_rows(trained, tmp_path) -> None:
    X, reference, _ = trained
    path = save_reference(reference, tmp_path / "m.joblib")

    text = path.read_text()

    assert X["customerID"].iloc[0] not in text
