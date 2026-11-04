"""Tests for churn_platform.drivers."""

import numpy as np
import pandas as pd
import pytest

from churn_platform.baselines import build_logistic_baseline
from churn_platform.drivers import model_input_columns, rank_drivers, stable_top_drivers
from churn_platform.preprocessing import build_preprocessor


def make_customers(n: int = 600) -> tuple[pd.DataFrame, pd.Series]:
    """Raw-style features where only short tenure predicts churn; the rest is noise."""
    rng = np.random.RandomState(0)
    tenure = rng.randint(0, 72, size=n)
    churn = (tenure < 20).astype(int)
    X = pd.DataFrame(
        {
            "customerID": [f"C{i:04d}" for i in range(n)],
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


def test_the_input_columns_are_the_ones_the_preprocessor_uses_not_the_id() -> None:
    X, y = make_customers()

    columns = model_input_columns(make_fitted_pipeline(X, y))

    assert columns == [
        "Contract",
        "tenure",
        "TotalCharges",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ]
    assert "customerID" not in columns


def test_the_column_that_really_drives_churn_is_ranked_first() -> None:
    X, y = make_customers()
    pipeline = make_fitted_pipeline(X, y)

    ranking = rank_drivers(pipeline, X, y)

    assert ranking.index[0] == "tenure"
    assert ranking.columns.tolist() == ["importance_mean", "importance_std"]
    assert ranking.loc["tenure", "importance_mean"] > 0.1
    # A column with no link to churn costs almost nothing when shuffled.
    assert abs(ranking.loc["Partner", "importance_mean"]) < 0.02


def test_the_ranking_is_sorted_from_most_to_least_important() -> None:
    X, y = make_customers()

    ranking = rank_drivers(make_fitted_pipeline(X, y), X, y)

    assert ranking["importance_mean"].is_monotonic_decreasing


def test_the_same_seed_gives_the_same_ranking() -> None:
    X, y = make_customers()
    pipeline = make_fitted_pipeline(X, y)

    first = rank_drivers(pipeline, X, y, seed=7)
    again = rank_drivers(pipeline, X, y, seed=7)

    pd.testing.assert_frame_equal(first, again)


def test_the_input_table_is_not_changed_by_the_shuffling() -> None:
    X, y = make_customers()
    pipeline = make_fitted_pipeline(X, y)
    before = X.copy()

    rank_drivers(pipeline, X, y)

    pd.testing.assert_frame_equal(X, before)


def test_a_clear_top_driver_stays_in_the_top_group_for_every_seed() -> None:
    X, y = make_customers()
    pipeline = make_fitted_pipeline(X, y)

    ranking, shared = stable_top_drivers(pipeline, X, y, top_n=1)

    assert ranking.index[0] == "tenure"
    assert shared == 1


def test_the_shared_count_is_between_zero_and_top_n() -> None:
    X, y = make_customers()
    pipeline = make_fitted_pipeline(X, y)

    _, shared = stable_top_drivers(pipeline, X, y, top_n=4)

    assert 0 <= shared <= 4
    # tenure is always first, so at least that one column is shared.
    assert shared >= 1


def test_drivers_need_a_fitted_pipeline() -> None:
    X, y = make_customers()
    unfitted = build_logistic_baseline(build_preprocessor())

    with pytest.raises(AttributeError):
        rank_drivers(unfitted, X, y)
