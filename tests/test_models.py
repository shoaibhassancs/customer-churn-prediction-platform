"""Tests for churn_platform.models and the margin rule in evaluation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from churn_platform.evaluation import (
    PR_AUC_MARGIN,
    choose_scoring_model,
    cross_validate_models,
    models_beating_baselines,
)
from churn_platform.models import (
    build_candidates,
    build_gradient_boosting,
    build_random_forest,
)
from churn_platform.preprocessing import build_preprocessor
from churn_platform.splitting import RANDOM_SEED


def make_customers(n: int = 300) -> tuple[pd.DataFrame, pd.Series]:
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


def make_small_preprocessor():
    return build_preprocessor(
        categorical_columns=["Contract"],
        numeric_columns=["tenure", "TotalCharges"],
        flag_columns=["Partner", "Dependents", "PhoneService", "PaperlessBilling"],
    )


def test_each_builder_puts_the_right_model_last_with_the_project_seed() -> None:
    forest = build_random_forest().steps[-1][1]
    boosting = build_gradient_boosting().steps[-1][1]

    assert isinstance(forest, RandomForestClassifier)
    assert isinstance(boosting, HistGradientBoostingClassifier)
    assert forest.random_state == RANDOM_SEED
    assert boosting.random_state == RANDOM_SEED
    assert forest.class_weight == boosting.class_weight == "balanced"


def test_candidates_are_two_baselines_and_two_challengers_in_a_fixed_order() -> None:
    assert list(build_candidates()) == [
        "majority",
        "logistic",
        "random_forest",
        "gradient_boosting",
    ]


@pytest.mark.parametrize("name", ["random_forest", "gradient_boosting"])
def test_a_challenger_fits_and_predicts_probabilities_for_raw_customers(name) -> None:
    X, y = make_customers()
    pipeline = build_candidates(make_small_preprocessor())[name].fit(X, y)

    probabilities = pipeline.predict_proba(X)

    assert probabilities.shape == (len(X), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_all_four_models_train_in_one_run_and_the_pattern_is_learned() -> None:
    X, y = make_customers()

    scores = cross_validate_models(build_candidates(make_small_preprocessor()), X, y)

    assert scores.index.tolist() == [
        "majority",
        "logistic",
        "random_forest",
        "gradient_boosting",
    ]
    assert scores.loc["majority", "roc_auc_mean"] == pytest.approx(0.5)
    assert (scores.drop(index="majority")["roc_auc_mean"] > 0.75).all()


def test_the_same_seed_reproduces_every_model_score() -> None:
    X, y = make_customers()

    first = cross_validate_models(build_candidates(make_small_preprocessor()), X, y)
    again = cross_validate_models(build_candidates(make_small_preprocessor()), X, y)

    pd.testing.assert_frame_equal(first, again)


def make_scores(**pr_auc: float) -> pd.DataFrame:
    return pd.DataFrame({"pr_auc_mean": pr_auc})


def test_a_challenger_must_clear_the_best_baseline_by_the_margin() -> None:
    scores = make_scores(
        majority=0.27, logistic=0.66, forest=0.67, boosting=0.68, linear_plus=0.70
    )

    verdict = models_beating_baselines(scores)

    assert PR_AUC_MARGIN == 0.02
    # 0.66 + 0.02 = 0.68: the boundary counts as beating, below it does not.
    assert verdict.to_dict() == {"forest": False, "boosting": True, "linear_plus": True}


def test_the_margin_can_be_changed() -> None:
    scores = make_scores(majority=0.27, logistic=0.66, forest=0.67)

    assert not models_beating_baselines(scores)["forest"]
    assert models_beating_baselines(scores, margin=0.005)["forest"]


def test_logistic_is_chosen_when_no_challenger_clears_the_margin() -> None:
    scores = make_scores(majority=0.27, logistic=0.66, forest=0.67, boosting=0.64)

    assert choose_scoring_model(scores) == "logistic"


def test_the_best_challenger_is_chosen_when_several_clear_the_margin() -> None:
    scores = make_scores(majority=0.27, logistic=0.66, forest=0.69, boosting=0.72)

    assert choose_scoring_model(scores) == "boosting"


def test_a_strong_majority_row_cannot_be_chosen() -> None:
    scores = make_scores(majority=0.90, logistic=0.66, forest=0.67)

    assert choose_scoring_model(scores) == "logistic"
