"""Tests for churn_platform.synthetic."""

import pandas as pd
import pytest

from churn_platform.ingest import load_csv, validate_upload
from churn_platform.synthetic import make_customers, write_customers_csv
from churn_platform.training import train_baseline

TELCO_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]  # fmt: skip


def churn_rate(table: pd.DataFrame) -> float:
    return float((table["Churn"] == "Yes").mean())


def test_columns_and_their_order_match_the_telecom_file() -> None:
    table = make_customers(200)

    assert table.columns.tolist() == TELCO_COLUMNS
    assert len(table) == 200


def test_customer_ids_are_unique() -> None:
    table = make_customers(1000)

    assert table["customerID"].is_unique
    assert table["customerID"].iloc[0] == "SYN-000001"


def test_the_same_seed_gives_the_same_table_and_another_seed_does_not() -> None:
    first = make_customers(500, seed=1)

    pd.testing.assert_frame_equal(first, make_customers(500, seed=1))
    assert not first.equals(make_customers(500, seed=2))


def test_the_checker_accepts_it_with_no_problems_and_no_warnings() -> None:
    report = validate_upload(make_customers(2000))

    assert report.is_valid
    assert report.mode == "train"
    assert report.warnings == ()


def test_planted_drivers_show_up_in_the_churn_rates() -> None:
    table = make_customers(30000)

    by_contract = table.groupby("Contract").apply(churn_rate, include_groups=False)
    assert by_contract["Month-to-month"] > by_contract["One year"] + 0.2
    assert by_contract["One year"] > by_contract["Two year"]
    assert (
        churn_rate(table[table["tenure"] < 12])
        > churn_rate(table[table["tenure"] > 48]) + 0.3
    )
    by_payment = table.groupby("PaymentMethod").apply(churn_rate, include_groups=False)
    assert by_payment["Electronic check"] > by_payment["Mailed check"]


def test_overall_churn_rate_is_close_to_a_quarter() -> None:
    assert 0.2 < churn_rate(make_customers(20000)) < 0.35


def test_total_charges_is_text_and_blank_exactly_when_tenure_is_zero() -> None:
    table = make_customers(5000)

    blank = table["TotalCharges"].str.strip() == ""
    assert (blank == (table["tenure"] == 0)).all()
    assert blank.any()
    assert pd.to_numeric(table.loc[~blank, "TotalCharges"]).gt(0).all()


def test_add_on_and_phone_columns_agree_with_the_services_taken() -> None:
    table = make_customers(5000)

    no_internet = table["InternetService"] == "No"
    for column in ("OnlineSecurity", "TechSupport", "StreamingTV"):
        assert (table.loc[no_internet, column] == "No internet service").all()
        assert (table.loc[~no_internet, column] != "No internet service").all()
    no_phone = table["PhoneService"] == "No"
    assert (table.loc[no_phone, "MultipleLines"] == "No phone service").all()
    assert (table.loc[~no_phone, "MultipleLines"] != "No phone service").all()


@pytest.mark.filterwarnings("ignore:Random split")
def test_a_model_can_learn_the_planted_pattern() -> None:
    result = train_baseline(make_customers(4000))

    assert result.cv_scores.loc["logistic", "roc_auc_mean"] > 0.75


def test_a_csv_can_be_written_and_read_back_by_the_checker(tmp_path) -> None:
    path = write_customers_csv(tmp_path / "out" / "customers.csv", 300)

    table = load_csv(path)

    assert len(table) == 300
    assert validate_upload(table).is_valid


def test_asking_for_no_rows_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        make_customers(0)
