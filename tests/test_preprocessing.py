"""Tests for churn_platform.preprocessing."""

import pandas as pd
import pytest

from churn_platform.preprocessing import build_preprocessor


def make_customers(
    contracts: list[str], tenures: list[int], partners: list[int]
) -> pd.DataFrame:
    """A small table with one text, one numeric, one flag and two unused columns."""
    return pd.DataFrame(
        {
            "customerID": [f"C{i:03d}" for i in range(len(contracts))],
            "Contract": contracts,
            "tenure": tenures,
            "Partner": partners,
            "Churn": [0] * len(contracts),
        }
    )


def make_preprocessor():
    """The real builder, pointed at the small table's columns."""
    return build_preprocessor(
        categorical_columns=["Contract"],
        numeric_columns=["tenure"],
        flag_columns=["Partner"],
    )


def test_output_has_one_hot_scaled_and_flag_columns_only() -> None:
    train = make_customers(["Month-to-month", "Two year"], [0, 10], [1, 0])

    out = make_preprocessor().fit_transform(train)

    assert list(out.columns) == [
        "Contract_Month-to-month",
        "Contract_Two year",
        "tenure",
        "Partner",
    ]
    # customerID and the Churn target are not model inputs.
    assert "customerID" not in out.columns
    assert "Churn" not in out.columns


def test_flag_columns_pass_through_unchanged() -> None:
    train = make_customers(["Month-to-month", "Two year"], [0, 10], [1, 0])

    out = make_preprocessor().fit_transform(train)

    assert out["Partner"].tolist() == [1, 0]


def test_scaling_uses_training_statistics_not_the_new_rows() -> None:
    train = make_customers(["Month-to-month", "Two year"], [0, 10], [1, 0])
    new = make_customers(["Two year"], [15], [0])
    preprocessor = make_preprocessor().fit(train)

    out = preprocessor.transform(new)

    # Training tenure has mean 5 and standard deviation 5, so 15 scales to 2.0.
    # If the new row were used to learn the scale, the answer would be 0.0.
    assert out["tenure"].tolist() == pytest.approx([2.0])


def test_unseen_category_becomes_all_zeros_instead_of_crashing() -> None:
    train = make_customers(["Month-to-month", "Two year"], [0, 10], [1, 0])
    new = make_customers(["One year"], [5], [1])
    preprocessor = make_preprocessor().fit(train)

    out = preprocessor.transform(new)

    assert out[["Contract_Month-to-month", "Contract_Two year"]].values.tolist() == [
        [0.0, 0.0]
    ]


def test_training_and_prediction_give_the_same_columns() -> None:
    train = make_customers(["Month-to-month", "Two year"], [0, 10], [1, 0])
    new = make_customers(["Two year"], [3], [1])
    preprocessor = make_preprocessor().fit(train)

    assert list(preprocessor.transform(new).columns) == list(
        preprocessor.transform(train).columns
    )
