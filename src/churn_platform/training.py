"""The training path: from a raw labelled table to a fitted baseline model."""

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from churn_platform.baselines import build_logistic_baseline, build_majority_baseline
from churn_platform.cleaning import encode_churn
from churn_platform.evaluation import cross_validate_models
from churn_platform.splitting import split_data


@dataclass(frozen=True)
class TrainingResult:
    """Everything one training run produces.

    ``pipeline`` is fitted on the training rows. ``cv_scores`` compares the baselines
    by cross-validation. ``X_test`` and ``y_test`` are the held-back rows, kept here so
    the final check can use them later without having to split the data again.
    """

    pipeline: Pipeline
    cv_scores: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.Series


def train_baseline(
    raw: pd.DataFrame, preprocessor: ColumnTransformer | None = None
) -> TrainingResult:
    """Train the logistic baseline on a raw table that has a ``Churn`` column.

    Steps: encode the target, split into train and test, score both baselines with
    cross-validation on the training rows only, then fit the logistic pipeline on
    those same rows. The test rows are not used for anything here.
    """
    X_train, X_test, y_train, y_test = split_data(encode_churn(raw))
    cv_scores = cross_validate_models(
        {
            "majority": build_majority_baseline(preprocessor),
            "logistic": build_logistic_baseline(preprocessor),
        },
        X_train,
        y_train,
    )
    pipeline = build_logistic_baseline(preprocessor).fit(X_train, y_train)
    return TrainingResult(
        pipeline=pipeline, cv_scores=cv_scores, X_test=X_test, y_test=y_test
    )
