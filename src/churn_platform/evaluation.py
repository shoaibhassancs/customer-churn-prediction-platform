"""Score models: cross-validation on training rows, one final check on test rows."""

import math

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from churn_platform.scoring import CHURN_CLASS
from churn_platform.splitting import RANDOM_SEED

FOLDS = 5
# scikit-learn's name for PR-AUC is "average_precision".
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}


def cross_validate_models(
    models: dict[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    folds: int = FOLDS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Return one row per model with the mean and spread of PR-AUC and ROC-AUC.

    Each model is trained on ``folds - 1`` slices of ``X`` and scored on the slice
    left out, once per slice. Every model gets the same slices, so the comparison is
    fair, and every slice keeps the churn rate (stratified). Pass training rows only.
    """
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = {}
    for name, model in models.items():
        # cross_validate clones the model per slice, so ``model`` itself stays unfitted.
        scores = cross_validate(model, X, y, cv=splitter, scoring=SCORING)
        rows[name] = {
            "pr_auc_mean": scores["test_pr_auc"].mean(),
            "pr_auc_std": scores["test_pr_auc"].std(),
            "roc_auc_mean": scores["test_roc_auc"].mean(),
            "roc_auc_std": scores["test_roc_auc"].std(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


# A challenger must beat the best baseline's PR-AUC by this much to count as better.
# 0.02 is about one fold-to-fold standard deviation of the logistic baseline, so a
# smaller gap could just be noise.
PR_AUC_MARGIN = 0.02
BASELINE_NAMES = ("majority", "logistic")


def models_beating_baselines(
    scores: pd.DataFrame, margin: float = PR_AUC_MARGIN
) -> pd.Series:
    """Say which challengers beat the baselines by at least ``margin`` PR-AUC.

    ``scores`` is the table from ``cross_validate_models``. The bar is the better
    baseline's mean PR-AUC plus ``margin``. Returns True or False per challenger.
    """
    bar = scores.loc[list(BASELINE_NAMES), "pr_auc_mean"].max() + margin
    challengers = scores.drop(index=list(BASELINE_NAMES))
    return challengers["pr_auc_mean"] >= bar


def choose_scoring_model(scores: pd.DataFrame, margin: float = PR_AUC_MARGIN) -> str:
    """Return the name of the model to score with, by the margin rule.

    It is logistic regression, unless at least one challenger beats the baselines by
    ``margin`` PR-AUC. If several do, the one with the highest mean PR-AUC wins.
    """
    winners = models_beating_baselines(scores, margin)
    if not winners.any():
        return "logistic"
    return scores.loc[winners[winners].index, "pr_auc_mean"].idxmax()


# The riskiest share of customers a retention team might contact.
TOP_FRACTION = 0.10


def evaluate_on_test(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.Series:
    """Score a fitted pipeline once on the held-back rows; return four metrics.

    ``pr_auc`` and ``roc_auc`` judge the ranking of customers. ``recall_top_10`` is
    the share of customers who really churned that sit in the 10% ranked riskiest.
    ``brier`` is the mean squared gap between probability and outcome (0 is perfect);
    it also judges the probabilities, which class weights push too high.
    """
    if y_test.sum() == 0:
        raise ValueError("The test rows contain no churned customers to score.")
    position = list(pipeline.classes_).index(CHURN_CLASS)
    scores = pipeline.predict_proba(X_test)[:, position]
    probability = pd.Series(scores, index=y_test.index)
    top_count = max(1, math.ceil(TOP_FRACTION * len(y_test)))
    ranked = probability.sort_values(ascending=False, kind="stable")
    riskiest = ranked.index[:top_count]
    return pd.Series(
        {
            "pr_auc": average_precision_score(y_test, probability),
            "roc_auc": roc_auc_score(y_test, probability),
            "recall_top_10": y_test.loc[riskiest].sum() / y_test.sum(),
            "brier": brier_score_loss(y_test, probability),
        }
    )
