"""Tests for churn_platform.persistence."""

import joblib
import numpy as np
import pandas as pd
import pytest

from churn_platform.baselines import build_logistic_baseline
from churn_platform.persistence import (
    DEMO_MODE_ENV,
    demo_mode_enabled,
    load_model,
    save_model,
)
from churn_platform.preprocessing import build_preprocessor
from churn_platform.scoring import score_customers


def make_raw_customers(n: int = 60) -> pd.DataFrame:
    """A small raw table whose IDs (C0000, C0001, ...) are easy to search for."""
    rng = np.random.RandomState(0)
    tenure = rng.randint(0, 72, size=n)
    return pd.DataFrame(
        {
            "customerID": [f"C{i:04d}" for i in range(n)],
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
            "tenure": tenure,
            "TotalCharges": (tenure * 30.5).astype(str),
            "Partner": rng.choice(["Yes", "No"], size=n),
            "Dependents": rng.choice(["Yes", "No"], size=n),
            "PhoneService": rng.choice(["Yes", "No"], size=n),
            "PaperlessBilling": rng.choice(["Yes", "No"], size=n),
        }
    )


def make_fitted_pipeline(raw: pd.DataFrame):
    preprocessor = build_preprocessor(
        categorical_columns=["Contract"],
        numeric_columns=["tenure", "TotalCharges"],
        flag_columns=["Partner", "Dependents", "PhoneService", "PaperlessBilling"],
    )
    labels = (raw["tenure"] < 20).astype(int)
    return build_logistic_baseline(preprocessor).fit(raw, labels)


def test_a_loaded_model_gives_the_same_scores_as_the_original(tmp_path) -> None:
    raw = make_raw_customers()
    pipeline = make_fitted_pipeline(raw)

    path = save_model(pipeline, tmp_path / "model.joblib")
    loaded = load_model(path)

    pd.testing.assert_frame_equal(
        score_customers(pipeline, raw), score_customers(loaded, raw)
    )


def test_save_creates_missing_folders_and_returns_the_path(tmp_path) -> None:
    pipeline = make_fitted_pipeline(make_raw_customers())
    target = tmp_path / "models" / "nested" / "model.joblib"

    returned = save_model(pipeline, target)

    assert returned == target
    assert target.exists()


def test_the_saved_file_contains_no_customer_rows(tmp_path) -> None:
    raw = make_raw_customers()
    path = save_model(make_fitted_pipeline(raw), tmp_path / "model.joblib")

    contents = path.read_bytes()

    # What the model learned is saved (the category names), the customers are not.
    assert b"Month-to-month" in contents
    assert b"C0007" not in contents


def test_loading_a_missing_file_gives_a_clear_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="No saved model"):
        load_model(tmp_path / "missing.joblib")


def test_loading_something_that_is_not_a_pipeline_is_refused(tmp_path) -> None:
    path = tmp_path / "other.joblib"
    joblib.dump({"not": "a pipeline"}, path)

    with pytest.raises(TypeError, match="dict"):
        load_model(path)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes "])
def test_demo_mode_is_on_for_the_accepted_values(monkeypatch, value) -> None:
    monkeypatch.setenv(DEMO_MODE_ENV, value)

    assert demo_mode_enabled()


@pytest.mark.parametrize("value", ["", "0", "false", "no", "banana"])
def test_demo_mode_is_off_for_everything_else(monkeypatch, value) -> None:
    monkeypatch.setenv(DEMO_MODE_ENV, value)

    assert not demo_mode_enabled()


def test_demo_mode_is_off_when_the_setting_is_missing(monkeypatch) -> None:
    monkeypatch.delenv(DEMO_MODE_ENV, raising=False)

    assert not demo_mode_enabled()
