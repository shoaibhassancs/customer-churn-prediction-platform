"""Tests for churn_platform.splitting."""

import pandas as pd
import pytest

from churn_platform.splitting import split_data

# The warning is tested once below; elsewhere it would only add noise.
pytestmark = pytest.mark.filterwarnings("ignore:Random split")


def make_customers() -> pd.DataFrame:
    """100 customers, 20 of whom churned (20%), with one feature column."""
    return pd.DataFrame(
        {
            "tenure": list(range(100)),
            "Churn": [1] * 20 + [0] * 80,
        }
    )


def test_split_sizes_are_80_and_20_percent() -> None:
    X_train, X_test, y_train, y_test = split_data(make_customers())

    assert (len(X_train), len(X_test)) == (80, 20)
    assert (len(y_train), len(y_test)) == (80, 20)


def test_target_is_removed_from_the_features() -> None:
    X_train, X_test, _, _ = split_data(make_customers())

    assert "Churn" not in X_train.columns
    assert "Churn" not in X_test.columns


def test_churn_rate_is_the_same_in_both_parts() -> None:
    _, _, y_train, y_test = split_data(make_customers())

    assert y_train.mean() == pytest.approx(0.2)
    assert y_test.mean() == pytest.approx(0.2)


def test_no_customer_is_in_both_parts() -> None:
    X_train, X_test, _, _ = split_data(make_customers())

    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(X_train) + len(X_test) == 100


def test_labels_stay_matched_to_their_rows() -> None:
    df = make_customers()

    X_train, _, y_train, _ = split_data(df)

    # The first 20 customers are the churners, so tenure < 20 must mean Churn == 1.
    assert (y_train == (X_train["tenure"] < 20).astype(int)).all()


def test_same_seed_gives_the_same_split_and_a_different_seed_does_not() -> None:
    first, _, _, _ = split_data(make_customers(), seed=42)
    again, _, _, _ = split_data(make_customers(), seed=42)
    other, _, _, _ = split_data(make_customers(), seed=7)

    assert first.index.tolist() == again.index.tolist()
    assert first.index.tolist() != other.index.tolist()


def test_input_dataframe_is_not_modified() -> None:
    df = make_customers()

    split_data(df)

    assert "Churn" in df.columns
    assert len(df) == 100


def test_warns_that_the_split_is_random() -> None:
    with pytest.warns(UserWarning, match="Random split"):
        split_data(make_customers())
