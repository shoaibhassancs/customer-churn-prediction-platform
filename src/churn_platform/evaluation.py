"""Score models with cross-validation, without touching the held-back test set."""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_validate

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
