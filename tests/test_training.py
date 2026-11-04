"""Tests for churn_platform.training."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from churn_platform.preprocessing import build_preprocessor
from churn_platform.scoring import score_customers
from churn_platform.training import TrainingResult, train_baseline

# The random-split warning is tested in test_splitting; here it would only add noise.
pytestmark = pytest.mark.filterwarnings("ignore:Random split")


def make_raw_customers(n: int = 300) -> pd.DataFrame:
    """A raw labelled table: text flags, text Churn, short tenure predicts churn."""
    rng = np.random.RandomState(0)
    tenure = rng.randint(0, 72, size=n)
    churned = np.where(rng.rand(n) < 0.1, (tenure >= 20), (tenure < 20))
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
            "Churn": np.where(churned, "Yes", "No"),
        }
    )


def make_small_preprocessor():
    return build_preprocessor(
        categorical_columns=["Contract"],
        numeric_columns=["tenure", "TotalCharges"],
        flag_columns=["Partner", "Dependents", "PhoneService", "PaperlessBilling"],
    )


def test_result_holds_a_fitted_pipeline_that_can_score_raw_customers() -> None:
    raw = make_raw_customers()

    result = train_baseline(raw, make_small_preprocessor())

    assert isinstance(result, TrainingResult)
    scored = score_customers(result.pipeline, result.X_test)
    assert len(scored) == len(result.X_test)


def test_cv_scores_compare_both_baselines_and_logistic_wins() -> None:
    result = train_baseline(make_raw_customers(), make_small_preprocessor())

    scores = result.cv_scores

    assert scores.index.tolist() == ["majority", "logistic"]
    assert scores.loc["majority", "roc_auc_mean"] == pytest.approx(0.5)
    assert scores.loc["logistic", "roc_auc_mean"] > 0.8


def test_test_rows_are_held_back_with_matching_zero_one_labels() -> None:
    raw = make_raw_customers()

    result = train_baseline(raw, make_small_preprocessor())

    assert len(result.X_test) == 60  # 20% of 300
    assert "Churn" not in result.X_test.columns
    assert set(result.y_test.unique()) <= {0, 1}
    assert result.X_test.index.tolist() == result.y_test.index.tolist()


def test_input_table_is_not_modified() -> None:
    raw = make_raw_customers()
    before = raw.copy()

    train_baseline(raw, make_small_preprocessor())

    pd.testing.assert_frame_equal(raw, before)


def test_same_input_gives_the_same_held_back_rows() -> None:
    first = train_baseline(make_raw_customers(), make_small_preprocessor())
    again = train_baseline(make_raw_customers(), make_small_preprocessor())

    assert first.X_test.index.tolist() == again.X_test.index.tolist()


def test_an_unexpected_churn_value_stops_training() -> None:
    raw = make_raw_customers()
    raw.loc[0, "Churn"] = "Maybe"

    with pytest.raises(ValueError, match="Maybe"):
        train_baseline(raw, make_small_preprocessor())


def test_result_cannot_be_changed_after_it_is_created() -> None:
    result = train_baseline(make_raw_customers(), make_small_preprocessor())

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.pipeline = None  # type: ignore[misc]
