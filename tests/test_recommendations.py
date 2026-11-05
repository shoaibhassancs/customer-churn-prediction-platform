"""Tests for churn_platform.recommendations."""

import pandas as pd

from churn_platform.recommendations import (
    DEFAULT_ACTION,
    RULES,
    recommend_actions,
)


def make_table(**columns) -> pd.DataFrame:
    """A small raw-style table; one list per column, one value per customer."""
    return pd.DataFrame(columns)


def everyone_high(table: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=table.index)


def test_each_rule_gives_its_own_action_and_reason() -> None:
    table = make_table(
        tenure=[2, 30, 30, 30],
        InternetService=["DSL", "Fiber optic", "DSL", "DSL"],
        TechSupport=["Yes", "No", "Yes", "Yes"],
        PaymentMethod=[
            "Mailed check",
            "Mailed check",
            "Electronic check",
            "Mailed check",
        ],
        Contract=["One year", "One year", "One year", "Month-to-month"],
    )

    result = recommend_actions(table, everyone_high(table))

    assert result["recommended_action"].tolist() == [
        "Onboarding check-in call",
        "Offer a free tech-support trial",
        "Move to automatic payment with reminders",
        "Offer a discounted 12-month plan",
    ]
    assert result["reason"].tolist() == [
        "New customer: under 6 months with the company",
        "Fibre internet without tech support",
        "Pays by electronic check",
        "Month-to-month contract",
    ]


def test_the_most_specific_matching_rule_wins() -> None:
    # A new customer on a month-to-month contract matches two rules; the first wins.
    table = make_table(tenure=[2], Contract=["Month-to-month"])

    result = recommend_actions(table, everyone_high(table))

    assert result["recommended_action"].tolist() == ["Onboarding check-in call"]


def test_a_high_customer_who_matches_nothing_gets_the_default_action() -> None:
    table = make_table(tenure=[40], Contract=["Two year"])

    result = recommend_actions(table, everyone_high(table))

    assert result["recommended_action"].tolist() == [DEFAULT_ACTION]
    assert result["reason"].iloc[0] != ""


def test_customers_who_are_not_high_risk_get_no_action() -> None:
    table = make_table(tenure=[2, 2], Contract=["Month-to-month", "Month-to-month"])
    is_high = pd.Series([True, False])

    result = recommend_actions(table, is_high)

    assert result.loc[0, "recommended_action"] != ""
    assert result.loc[1].tolist() == ["", ""]


def test_rules_whose_columns_are_missing_are_skipped_not_a_crash() -> None:
    # A file with none of the telecom columns still gives every High customer an action.
    table = make_table(Balance=[100.0, 200.0])

    result = recommend_actions(table, everyone_high(table))

    assert result["recommended_action"].tolist() == [DEFAULT_ACTION, DEFAULT_ACTION]


def test_a_missing_value_does_not_match_a_rule() -> None:
    table = make_table(tenure=[None], Contract=[None])

    result = recommend_actions(table, everyone_high(table))

    assert result["recommended_action"].tolist() == [DEFAULT_ACTION]


def test_the_result_keeps_the_index_of_the_input_table() -> None:
    table = make_table(tenure=[2, 50]).set_axis(["a", "b"])

    result = recommend_actions(table, pd.Series([True, True], index=table.index))

    assert result.index.tolist() == ["a", "b"]


def test_the_input_table_is_not_modified() -> None:
    table = make_table(tenure=[2, 50], Contract=["Month-to-month", "Two year"])
    before = table.copy()

    recommend_actions(table, everyone_high(table))

    pd.testing.assert_frame_equal(table, before)


def test_every_rule_names_its_columns_an_action_and_a_reason() -> None:
    for rule in RULES:
        assert rule.columns and rule.action and rule.reason
