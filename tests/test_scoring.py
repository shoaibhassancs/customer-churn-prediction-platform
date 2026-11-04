"""Tests for churn_platform.scoring."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from churn_platform.baselines import build_logistic_baseline
from churn_platform.preprocessing import build_preprocessor
from churn_platform.scoring import score_customers


def make_customers(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    """Raw-style features where short tenure predicts churn, with some noise."""
    rng = np.random.RandomState(0)
    tenure = rng.randint(0, 72, size=n)
    churn = np.where(rng.rand(n) < 0.1, (tenure >= 20), (tenure < 20)).astype(int)
    X = pd.DataFrame(
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
    return X, pd.Series(churn, name="Churn")


def make_fitted_pipeline(X: pd.DataFrame, y: pd.Series):
    preprocessor = build_preprocessor(
        categorical_columns=["Contract"],
        numeric_columns=["tenure", "TotalCharges"],
        flag_columns=["Partner", "Dependents", "PhoneService", "PaperlessBilling"],
    )
    return build_logistic_baseline(preprocessor).fit(X, y)


def test_one_row_per_customer_in_the_same_order() -> None:
    X, y = make_customers()
    pipeline = make_fitted_pipeline(X, y)

    scored = score_customers(pipeline, X)

    assert scored.columns.tolist() == ["customerID", "churn_probability"]
    assert scored["customerID"].tolist() == X["customerID"].tolist()


def test_probabilities_are_between_zero_and_one() -> None:
    X, y = make_customers()
    pipeline = make_fitted_pipeline(X, y)

    scored = score_customers(pipeline, X)

    assert scored["churn_probability"].between(0, 1).all()


def test_short_tenure_customers_score_higher_than_long_tenure_ones() -> None:
    X, y = make_customers()
    pipeline = make_fitted_pipeline(X, y)

    scored = score_customers(pipeline, X)

    short = scored.loc[X["tenure"] < 10, "churn_probability"].mean()
    long = scored.loc[X["tenure"] > 60, "churn_probability"].mean()
    assert short > long + 0.3


def test_an_extra_churn_column_is_ignored() -> None:
    X, y = make_customers()
    pipeline = make_fitted_pipeline(X, y)

    without = score_customers(pipeline, X)
    with_labels = score_customers(pipeline, X.assign(Churn=y))

    pd.testing.assert_frame_equal(without, with_labels)


def test_input_dataframe_is_not_modified() -> None:
    X, y = make_customers()
    pipeline = make_fitted_pipeline(X, y)
    before = X.copy()

    score_customers(pipeline, X)

    pd.testing.assert_frame_equal(X, before)


def test_missing_id_column_gives_a_clear_error() -> None:
    X, y = make_customers()
    pipeline = make_fitted_pipeline(X, y)

    with pytest.raises(ValueError, match="customerID"):
        score_customers(pipeline, X.drop(columns=["customerID"]))


def test_an_unfitted_pipeline_raises_instead_of_guessing() -> None:
    X, _ = make_customers()
    unfitted = build_logistic_baseline(
        build_preprocessor(
            categorical_columns=["Contract"],
            numeric_columns=["tenure", "TotalCharges"],
            flag_columns=["Partner", "Dependents", "PhoneService", "PaperlessBilling"],
        )
    )

    with pytest.raises(NotFittedError):
        score_customers(unfitted, X)
