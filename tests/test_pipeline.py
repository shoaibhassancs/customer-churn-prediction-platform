"""Tests for churn_platform.pipeline."""

import pandas as pd
import pytest

from churn_platform.pipeline import build_pipeline, clean_features
from churn_platform.preprocessing import build_preprocessor


def make_raw_features() -> pd.DataFrame:
    """Four customers as they arrive: text flags, text TotalCharges, one blank."""
    return pd.DataFrame(
        {
            "customerID": ["C000", "C001", "C002", "C003"],
            "Contract": ["Month-to-month", "Two year", "Month-to-month", "Two year"],
            "tenure": [0, 10, 20, 30],
            "TotalCharges": [" ", "100.5", "250.0", "900.25"],
            "Partner": ["Yes", "No", "No", "Yes"],
            "Dependents": ["No", "No", "Yes", "No"],
            "PhoneService": ["Yes", "Yes", "No", "Yes"],
            "PaperlessBilling": ["Yes", "No", "Yes", "No"],
        }
    )


def make_pipeline():
    """The real builder, pointed at the small table's columns."""
    preprocessor = build_preprocessor(
        categorical_columns=["Contract"],
        numeric_columns=["tenure", "TotalCharges"],
        flag_columns=["Partner", "Dependents", "PhoneService", "PaperlessBilling"],
    )
    return build_pipeline(preprocessor)


def test_pipeline_has_a_clean_step_then_a_preprocess_step() -> None:
    assert [name for name, _ in make_pipeline().steps] == ["clean", "preprocess"]


def test_clean_features_fixes_total_charges_and_yes_no_flags() -> None:
    cleaned = clean_features(make_raw_features())

    assert cleaned["TotalCharges"].tolist() == [0.0, 100.5, 250.0, 900.25]
    assert cleaned["Partner"].tolist() == [1, 0, 0, 1]


def test_raw_text_table_goes_in_and_numbers_come_out_in_one_call() -> None:
    out = make_pipeline().fit_transform(make_raw_features())

    assert out.isna().sum().sum() == 0
    assert all(pd.api.types.is_numeric_dtype(t) for t in out.dtypes)
    assert out["Partner"].tolist() == [1, 0, 0, 1]
    assert "customerID" not in out.columns


def test_scaling_still_uses_training_statistics_only() -> None:
    train = make_raw_features().iloc[:2]  # tenure 0 and 10: mean 5, std 5
    new = make_raw_features().iloc[[3]]  # tenure 30
    pipeline = make_pipeline().fit(train)

    out = pipeline.transform(new)

    assert out["tenure"].tolist() == pytest.approx([5.0])


def test_one_customer_gets_the_same_features_alone_as_inside_a_batch() -> None:
    raw = make_raw_features()
    pipeline = make_pipeline().fit(raw)

    in_batch = pipeline.transform(raw).iloc[[1]]
    alone = pipeline.transform(raw.iloc[[1]])

    pd.testing.assert_frame_equal(in_batch, alone)


def test_input_dataframe_is_not_modified() -> None:
    raw = make_raw_features()

    make_pipeline().fit_transform(raw)

    assert raw["TotalCharges"].tolist() == [" ", "100.5", "250.0", "900.25"]
    assert raw["Partner"].tolist() == ["Yes", "No", "No", "Yes"]


def test_a_model_is_added_as_the_last_step_when_given() -> None:
    from sklearn.dummy import DummyClassifier

    pipeline = build_pipeline(model=DummyClassifier())

    assert [name for name, _ in pipeline.steps] == ["clean", "preprocess", "model"]
