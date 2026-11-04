"""Split the data into rows the model learns from and rows held back for testing."""

import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
RANDOM_SEED = 42


def split_data(
    df: pd.DataFrame,
    target: str = "Churn",
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return ``X_train, X_test, y_train, y_test`` from a table with a 0/1 target.

    ``X`` holds the feature columns (everything except ``target``) and ``y`` the
    target. The split is random but stratified, so the churn rate is the same in both
    parts, and the fixed ``seed`` makes it repeatable. There is no date column to
    split by time, so a ``UserWarning`` says the split is random.
    """
    warnings.warn(
        "Random split: the data has no date column, so rows are split at random, "
        "not by time. Scores may look better than future performance.",
        UserWarning,
        stacklevel=2,
    )
    features = df.drop(columns=[target])
    labels = df[target]
    # stratify=labels keeps the share of churners the same in train and test.
    return train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
