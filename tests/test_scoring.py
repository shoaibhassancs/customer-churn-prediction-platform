"""Tests for churn_platform.scoring."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from churn_platform.baselines import build_logistic_baseline
from churn_platform.preprocessing import build_preprocessor
from churn_platform.scoring import assign_risk_bands, score_customers


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

    assert scored.columns.tolist() == [
        "customerID",
        "churn_probability",
        "risk_band",
        "recommended_action",
        "reason",
    ]
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


def test_the_riskiest_tenth_is_high_and_the_next_fifth_is_medium() -> None:
    probability = pd.Series(np.linspace(0.01, 1.0, 100))

    bands = assign_risk_bands(probability)

    assert bands.value_counts().to_dict() == {"Low": 70, "Medium": 20, "High": 10}
    assert (bands[probability >= 0.91] == "High").all()
    assert (bands[probability <= 0.70] == "Low").all()


def test_bands_follow_the_ranking_not_the_size_of_the_probabilities() -> None:
    low_numbers = pd.Series(np.linspace(0.001, 0.01, 50))
    high_numbers = pd.Series(np.linspace(0.90, 0.99, 50))

    first = assign_risk_bands(low_numbers)
    second = assign_risk_bands(high_numbers)

    assert first.tolist() == second.tolist()


def test_a_bigger_high_share_gives_a_bigger_high_group() -> None:
    probability = pd.Series(np.linspace(0.01, 1.0, 100))

    bands = assign_risk_bands(probability, high_share=0.30)

    assert (bands == "High").sum() == 30
    assert (bands == "Medium").sum() == 20


def test_at_least_one_customer_is_high_even_in_a_tiny_file() -> None:
    bands = assign_risk_bands(pd.Series([0.2, 0.9, 0.5]))

    assert bands.tolist() == ["Low", "High", "Medium"]


def test_equal_probabilities_still_split_into_groups_by_position() -> None:
    bands = assign_risk_bands(pd.Series([0.5] * 10))

    assert bands.value_counts().to_dict() == {"Low": 7, "Medium": 2, "High": 1}


def test_bands_keep_the_index_of_the_probabilities() -> None:
    probability = pd.Series([0.1, 0.8], index=["a", "b"])

    bands = assign_risk_bands(probability)

    assert bands.index.tolist() == ["a", "b"]
    assert bands.name == "risk_band"


@pytest.mark.parametrize("share", [0, -0.1, 1.5])
def test_an_impossible_high_share_is_refused(share) -> None:
    with pytest.raises(ValueError, match="high_share"):
        assign_risk_bands(pd.Series([0.1, 0.9]), high_share=share)


def test_every_high_risk_customer_has_an_action_and_others_do_not() -> None:
    X, y = make_customers()
    pipeline = make_fitted_pipeline(X, y)

    scored = score_customers(pipeline, X)

    high = scored["risk_band"] == "High"
    assert high.sum() > 0
    assert (scored.loc[high, "recommended_action"] != "").all()
    assert (scored.loc[high, "reason"] != "").all()
    assert (scored.loc[~high, "recommended_action"] == "").all()


def test_actions_line_up_with_customers_when_the_index_is_scrambled() -> None:
    X, y = make_customers()
    pipeline = make_fitted_pipeline(X, y)
    shuffled = X.sample(frac=1, random_state=3)  # same rows, scrambled index labels

    scored = score_customers(pipeline, shuffled)

    new_customers = shuffled.reset_index(drop=True)
    new_in_high = (scored["risk_band"] == "High") & (new_customers["tenure"] < 6)
    assert new_in_high.any()
    assert (
        scored.loc[new_in_high, "recommended_action"] == "Onboarding check-in call"
    ).all()
