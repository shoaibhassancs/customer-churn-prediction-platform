"""Find which customer columns the fitted model relies on most."""

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from churn_platform.splitting import RANDOM_SEED

N_REPEATS = 5
TOP_N = 10
# Three different shuffles; the top drivers should not depend on which one is used.
STABILITY_SEEDS = (RANDOM_SEED, RANDOM_SEED + 1, RANDOM_SEED + 2)


def model_input_columns(pipeline: Pipeline) -> list[str]:
    """Return the raw columns the fitted preprocessor reads, in order.

    ``customerID`` and ``Churn`` are not among them because the preprocessor drops
    columns it was not told to use.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    return [
        column
        for name, _, columns in preprocessor.transformers_
        if name != "remainder"
        for column in columns
    ]


def rank_drivers(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = RANDOM_SEED,
    n_repeats: int = N_REPEATS,
) -> pd.DataFrame:
    """Rank the model's input columns by how much shuffling each one hurts PR-AUC.

    For each column, its values are shuffled among the customers ``n_repeats`` times
    and the model is scored again. The drop in PR-AUC is that column's importance:
    large means the model leaned on it, near zero means it barely mattered. The
    columns are shuffled before the pipeline runs, so ``Contract`` counts as one
    driver, not as several one-hot columns. Returns a table sorted from most to least
    important with ``importance_mean`` and ``importance_std``. Pass held-back rows.
    """
    columns = model_input_columns(pipeline)
    result = permutation_importance(
        pipeline,
        X[columns],
        y,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=seed,
    )
    table = pd.DataFrame(
        {
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        },
        index=columns,
    )
    return table.sort_values("importance_mean", ascending=False)


def stable_top_drivers(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = TOP_N,
    seeds: tuple[int, ...] = STABILITY_SEEDS,
) -> tuple[pd.DataFrame, int]:
    """Rank the drivers and say how many of the top ``top_n`` hold for every seed.

    Returns the ranking from the first seed and a count from 0 to ``top_n``: the
    number of columns that appear in the top ``top_n`` of every seed's ranking.
    """
    rankings = [rank_drivers(pipeline, X, y, seed=seed) for seed in seeds]
    top_sets = [set(ranking.index[:top_n]) for ranking in rankings]
    return rankings[0], len(set.intersection(*top_sets))
