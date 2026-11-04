"""Tests for churn_platform.baselines and churn_platform.evaluation."""

import numpy as np
import pandas as pd
import pytest

from churn_platform.baselines import build_logistic_baseline, build_majority_baseline
from churn_platform.evaluation import cross_validate_models, evaluate_on_test
from churn_platform.preprocessing import build_preprocessor


def make_customers(n: int = 300) -> tuple[pd.DataFrame, pd.Series]:
    """Raw-style features where short tenure really does predict churn, plus noise."""
    rng = np.random.RandomState(0)
    tenure = rng.randint(0, 72, size=n)
    churn = (tenure < 20).astype(int)
    flip = rng.rand(n) < 0.1  # 10% of answers are wrong, so the signal is not perfect
    churn = np.where(flip, 1 - churn, churn)
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


def make_small_preprocessor():
    """The real builder, pointed at the small table's columns."""
    return build_preprocessor(
        categorical_columns=["Contract"],
        numeric_columns=["tenure", "TotalCharges"],
        flag_columns=["Partner", "Dependents", "PhoneService", "PaperlessBilling"],
    )


def make_models() -> dict:
    return {
        "majority": build_majority_baseline(make_small_preprocessor()),
        "logistic": build_logistic_baseline(make_small_preprocessor()),
    }


def test_result_has_one_row_per_model_and_the_four_score_columns() -> None:
    X, y = make_customers()

    result = cross_validate_models(make_models(), X, y)

    assert result.index.tolist() == ["majority", "logistic"]
    assert result.columns.tolist() == [
        "pr_auc_mean",
        "pr_auc_std",
        "roc_auc_mean",
        "roc_auc_std",
    ]


def test_majority_baseline_scores_like_a_coin_flip() -> None:
    X, y = make_customers()

    result = cross_validate_models(make_models(), X, y)

    # A constant prediction cannot rank anyone: ROC-AUC 0.5, PR-AUC = churn share.
    assert result.loc["majority", "roc_auc_mean"] == pytest.approx(0.5)
    assert result.loc["majority", "pr_auc_mean"] == pytest.approx(y.mean(), abs=0.02)


def test_logistic_baseline_beats_the_majority_baseline() -> None:
    X, y = make_customers()

    result = cross_validate_models(make_models(), X, y)

    # 10% of the labels are flipped, so even a perfect rule scores only about 0.85.
    assert result.loc["logistic", "roc_auc_mean"] > 0.8
    assert result.loc["logistic", "pr_auc_mean"] > result.loc["majority", "pr_auc_mean"]


def test_same_seed_gives_the_same_scores() -> None:
    X, y = make_customers()

    first = cross_validate_models(make_models(), X, y, seed=1)
    again = cross_validate_models(make_models(), X, y, seed=1)

    pd.testing.assert_frame_equal(first, again)


def test_models_passed_in_are_left_unfitted() -> None:
    X, y = make_customers()
    models = make_models()

    cross_validate_models(models, X, y)

    assert not hasattr(models["logistic"].named_steps["model"], "coef_")


class FixedProbabilities:
    """A stand-in fitted model that returns chosen churn probabilities."""

    classes_ = np.array([0, 1])

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = np.array(probabilities)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([1 - self.probabilities, self.probabilities])


def make_ten_customers(labels: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    return pd.DataFrame({"x": range(len(labels))}), pd.Series(labels)


def test_a_perfect_ranking_scores_one_on_every_ranking_metric() -> None:
    X, y = make_ten_customers([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    model = FixedProbabilities([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    scores = evaluate_on_test(model, X, y)

    assert scores["pr_auc"] == pytest.approx(1.0)
    assert scores["roc_auc"] == pytest.approx(1.0)
    # Top 10% of 10 customers is one customer; it holds 1 of the 2 churners.
    assert scores["recall_top_10"] == pytest.approx(0.5)


def test_a_reversed_ranking_finds_no_churners_in_the_top_group() -> None:
    X, y = make_ten_customers([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    # The two churners get the two lowest probabilities, with no ties.
    model = FixedProbabilities([0.01, 0.02, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9])

    scores = evaluate_on_test(model, X, y)

    assert scores["recall_top_10"] == 0.0
    assert scores["roc_auc"] == pytest.approx(0.0)


def test_brier_is_zero_for_certain_correct_probabilities_and_worse_otherwise() -> None:
    X, y = make_ten_customers([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    sure = evaluate_on_test(FixedProbabilities([1.0] + [0.0] * 9), X, y)
    unsure = evaluate_on_test(FixedProbabilities([0.5] * 10), X, y)

    assert sure["brier"] == pytest.approx(0.0)
    assert unsure["brier"] == pytest.approx(0.25)


def test_a_test_set_without_churners_is_refused() -> None:
    X, y = make_ten_customers([0] * 10)

    with pytest.raises(ValueError, match="no churned customers"):
        evaluate_on_test(FixedProbabilities([0.5] * 10), X, y)
