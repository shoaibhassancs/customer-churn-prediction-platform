"""The training path: from a raw labelled table to the fitted scoring model."""

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from churn_platform.cleaning import encode_churn
from churn_platform.drivers import TOP_N, stable_top_drivers
from churn_platform.evaluation import (
    choose_scoring_model,
    cross_validate_models,
    evaluate_on_test,
    models_beating_baselines,
)
from churn_platform.models import build_candidates
from churn_platform.splitting import split_data


@dataclass(frozen=True)
class TrainingResult:
    """Everything one training run produces.

    ``pipeline`` is the model picked by the margin rule, fitted on the training rows,
    and ``chosen_model`` is its name. ``cv_scores`` compares all four models by
    cross-validation, and ``beats_baselines`` says which challengers beat the
    baselines by the agreed margin. ``test_scores`` is the one-time check of the
    chosen model on the held-back rows. ``drivers`` ranks the columns the model relies
    on, and ``shared_drivers`` counts how many of the top ten hold for three different
    shuffles. ``X_test`` and ``y_test`` are
    the held-back rows, kept here so the final check can use them later without having
    to split the data again.
    """

    pipeline: Pipeline
    chosen_model: str
    cv_scores: pd.DataFrame
    beats_baselines: pd.Series
    test_scores: pd.Series
    drivers: pd.DataFrame
    shared_drivers: int
    X_test: pd.DataFrame
    y_test: pd.Series


def train_baseline(
    raw: pd.DataFrame, preprocessor: ColumnTransformer | None = None
) -> TrainingResult:
    """Compare four models on a raw table with a ``Churn`` column; fit the winner.

    Steps: encode the target, split into train and test, score all four models with
    cross-validation on the training rows only, pick the scoring model by the margin
    rule (logistic unless a challenger clearly beats the baselines), then fit that
    model on those same rows. Only then are the test rows used, to score it and to
    rank its drivers. Nothing is changed because of what they show.
    """
    X_train, X_test, y_train, y_test = split_data(encode_churn(raw))
    candidates = build_candidates(preprocessor)
    cv_scores = cross_validate_models(candidates, X_train, y_train)
    chosen = choose_scoring_model(cv_scores)
    # cross_validate worked on clones, so this candidate is still unfitted.
    pipeline = candidates[chosen].fit(X_train, y_train)
    drivers, shared_drivers = stable_top_drivers(pipeline, X_test, y_test, TOP_N)
    return TrainingResult(
        pipeline=pipeline,
        chosen_model=chosen,
        cv_scores=cv_scores,
        beats_baselines=models_beating_baselines(cv_scores),
        test_scores=evaluate_on_test(pipeline, X_test, y_test),
        drivers=drivers,
        shared_drivers=shared_drivers,
        X_test=X_test,
        y_test=y_test,
    )
