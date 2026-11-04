"""Tests for churn_platform.cleaning."""

import pandas as pd
import pytest

from churn_platform.cleaning import (
    encode_churn,
    encode_yes_no_columns,
    fix_total_charges,
)


def make_raw_customers() -> pd.DataFrame:
    """Three customers shaped like the raw CSV: TotalCharges is text, one blank."""
    return pd.DataFrame(
        {
            "customerID": ["7590-VHVEG", "5575-GNVDE", "4472-LVYGI"],
            "tenure": [1, 34, 0],
            "TotalCharges": ["29.85", "1889.5", " "],
        }
    )


def test_blank_total_charges_becomes_zero() -> None:
    cleaned = fix_total_charges(make_raw_customers())

    assert cleaned.loc[2, "TotalCharges"] == 0.0


def test_total_charges_is_numeric_and_real_values_survive() -> None:
    cleaned = fix_total_charges(make_raw_customers())

    assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])
    assert cleaned["TotalCharges"].tolist() == [29.85, 1889.5, 0.0]


def test_input_dataframe_is_not_modified() -> None:
    raw = make_raw_customers()

    fix_total_charges(raw)

    assert raw["TotalCharges"].tolist() == ["29.85", "1889.5", " "]


def test_non_numeric_text_raises_instead_of_being_hidden() -> None:
    raw = make_raw_customers()
    raw.loc[0, "TotalCharges"] = "abc"

    with pytest.raises(ValueError):
        fix_total_charges(raw)


def test_total_charges_already_read_as_numbers_is_accepted() -> None:
    raw = make_raw_customers()
    raw["TotalCharges"] = [29.85, 1889.5, 12.0]

    cleaned = fix_total_charges(raw)

    assert cleaned["TotalCharges"].tolist() == [29.85, 1889.5, 12.0]


def test_empty_cells_become_zero_in_text_and_in_numeric_columns() -> None:
    as_text = make_raw_customers()
    as_text["TotalCharges"] = pd.Series(["29.85", None, " "], dtype="str")
    as_numbers = make_raw_customers()
    as_numbers["TotalCharges"] = [29.85, float("nan"), 12.0]

    assert fix_total_charges(as_text)["TotalCharges"].tolist() == [29.85, 0.0, 0.0]
    assert fix_total_charges(as_numbers)["TotalCharges"].tolist() == [29.85, 0.0, 12.0]


def make_churn_customers(labels: list[str]) -> pd.DataFrame:
    """A small table with a text Churn column holding the given labels."""
    return pd.DataFrame(
        {
            "customerID": [f"C{i:03d}" for i in range(len(labels))],
            "Churn": labels,
        }
    )


def test_encode_churn_maps_yes_to_one_and_no_to_zero() -> None:
    encoded = encode_churn(make_churn_customers(["No", "Yes", "No"]))

    assert encoded["Churn"].tolist() == [0, 1, 0]
    assert pd.api.types.is_integer_dtype(encoded["Churn"])


def test_encode_churn_does_not_modify_input() -> None:
    raw = make_churn_customers(["No", "Yes"])

    encode_churn(raw)

    assert raw["Churn"].tolist() == ["No", "Yes"]


def test_encode_churn_rejects_unexpected_values() -> None:
    raw = make_churn_customers(["No", "Maybe", "Yes"])

    with pytest.raises(ValueError, match="Maybe"):
        encode_churn(raw)


def test_encode_churn_accepts_custom_labels() -> None:
    raw = make_churn_customers(["stayed", "left", "left"])

    encoded = encode_churn(raw, positive_label="left", negative_label="stayed")

    assert encoded["Churn"].tolist() == [0, 1, 1]


def make_flag_customers() -> pd.DataFrame:
    """Two Yes/No flags, one three-value column, and a number, as in the raw CSV."""
    return pd.DataFrame(
        {
            "Partner": ["Yes", "No", "No"],
            "Dependents": ["No", "No", "Yes"],
            "MultipleLines": ["No phone service", "No", "Yes"],
            "tenure": [1, 34, 2],
        }
    )


def test_yes_no_columns_become_one_and_zero() -> None:
    encoded = encode_yes_no_columns(
        make_flag_customers(), columns=("Partner", "Dependents")
    )

    assert encoded["Partner"].tolist() == [1, 0, 0]
    assert encoded["Dependents"].tolist() == [0, 0, 1]


def test_columns_not_listed_are_left_alone() -> None:
    encoded = encode_yes_no_columns(
        make_flag_customers(), columns=("Partner", "Dependents")
    )

    assert encoded["MultipleLines"].tolist() == ["No phone service", "No", "Yes"]
    assert encoded["tenure"].tolist() == [1, 34, 2]


def test_yes_no_encoding_does_not_modify_input() -> None:
    raw = make_flag_customers()

    encode_yes_no_columns(raw, columns=("Partner", "Dependents"))

    assert raw["Partner"].tolist() == ["Yes", "No", "No"]


def test_yes_no_encoding_rejects_a_third_value_in_a_listed_column() -> None:
    raw = make_flag_customers()

    with pytest.raises(ValueError, match="MultipleLines"):
        encode_yes_no_columns(raw, columns=("MultipleLines",))
