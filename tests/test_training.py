"""Tests for churn_platform.training."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from churn_platform import training
from churn_platform.preprocessing import build_preprocessor
from churn_platform.scoring import score_customers
from churn_platform.training import TrainingResult, train_baseline

# The random-split warning is tested in test_splitting; here it would only add noise.
pytestmark = pytest.mark.filterwarnings("ignore:Random split")

CLASS_NAMES = {
    "logistic": "LogisticRegression",
    "random_forest": "RandomForestClassifier",
    "gradient_boosting": "HistGradientBoostingClassifier",
}


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


def test_cv_scores_compare_all_four_models_and_logistic_beats_the_floor() -> None:
    result = train_baseline(make_raw_customers(), make_small_preprocessor())

    scores = result.cv_scores

    assert scores.index.tolist() == [
        "majority",
        "logistic",
        "random_forest",
        "gradient_boosting",
    ]
    assert scores.loc["majority", "roc_auc_mean"] == pytest.approx(0.5)
    assert scores.loc["logistic", "roc_auc_mean"] > 0.8


def test_the_verdict_covers_each_challenger_with_a_true_or_false() -> None:
    result = train_baseline(make_raw_customers(), make_small_preprocessor())

    verdict = result.beats_baselines

    assert verdict.index.tolist() == ["random_forest", "gradient_boosting"]
    assert verdict.dtype == bool


def test_the_chosen_model_gets_a_one_time_check_on_the_held_back_rows() -> None:
    result = train_baseline(make_raw_customers(), make_small_preprocessor())

    assert result.test_scores.index.tolist() == [
        "pr_auc",
        "roc_auc",
        "recall_top_10",
        "brier",
    ]
    assert result.test_scores["roc_auc"] > 0.8


def test_drivers_rank_the_model_input_columns_and_count_the_stable_ones() -> None:
    result = train_baseline(make_raw_customers(), make_small_preprocessor())

    # In this table TotalCharges is exactly tenure * 30.5, so the two carry the same
    # information and swap first place between machines; either may come first.
    assert set(result.drivers.index[:2]) == {"tenure", "TotalCharges"}
    assert "customerID" not in result.drivers.index
    assert 0 <= result.shared_drivers <= 10


def test_the_result_carries_a_drift_reference_for_the_model_input_columns() -> None:
    result = train_baseline(make_raw_customers(), make_small_preprocessor())

    assert set(result.drift_reference) == {
        "Contract",
        "tenure",
        "TotalCharges",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    }


def test_the_fitted_model_is_the_one_the_result_names() -> None:
    result = train_baseline(make_raw_customers(), make_small_preprocessor())

    fitted_class = type(result.pipeline.steps[-1][1]).__name__
    assert result.chosen_model in {"logistic", "random_forest", "gradient_boosting"}
    assert fitted_class == CLASS_NAMES[result.chosen_model]


def fake_scores(forest_pr_auc: float) -> pd.DataFrame:
    """A score table where the forest has the given PR-AUC and the rest are fixed."""
    pr_auc = {
        "majority": 0.27,
        "logistic": 0.60,
        "random_forest": forest_pr_auc,
        "gradient_boosting": 0.60,
    }
    return pd.DataFrame({"pr_auc_mean": pr_auc})


def test_a_challenger_that_clears_the_margin_is_fitted_and_chosen(monkeypatch) -> None:
    monkeypatch.setattr(training, "cross_validate_models", lambda *a: fake_scores(0.65))

    result = train_baseline(make_raw_customers(), make_small_preprocessor())

    assert result.chosen_model == "random_forest"
    assert type(result.pipeline.steps[-1][1]).__name__ == "RandomForestClassifier"


def test_logistic_is_fitted_when_no_challenger_clears_the_margin(monkeypatch) -> None:
    monkeypatch.setattr(training, "cross_validate_models", lambda *a: fake_scores(0.61))

    result = train_baseline(make_raw_customers(), make_small_preprocessor())

    assert result.chosen_model == "logistic"
    assert type(result.pipeline.steps[-1][1]).__name__ == "LogisticRegression"


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
