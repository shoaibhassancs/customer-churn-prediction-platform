"""Turn a fitted pipeline and a raw customer table into churn probabilities."""

import pandas as pd
from sklearn.pipeline import Pipeline

ID_COLUMN = "customerID"
PROBABILITY_COLUMN = "churn_probability"
# encode_churn turned "churned" into 1, so class 1 is the one we report.
CHURN_CLASS = 1


def score_customers(
    pipeline: Pipeline, raw: pd.DataFrame, id_column: str = ID_COLUMN
) -> pd.DataFrame:
    """Return one row per customer with their ID and churn probability.

    ``pipeline`` must already be fitted and end in a classifier. ``raw`` is a table in
    the original file format (text values and all); the pipeline cleans it itself.
    Rows keep the order of ``raw``. A ``Churn`` column, if present, is ignored.

    The probability comes from a model trained with class weights, so it ranks
    customers correctly but overstates the true chance of churn until it is calibrated.
    """
    if id_column not in raw.columns:
        raise ValueError(f"The table has no {id_column!r} column to name customers.")
    # Predict first, so an unfitted pipeline raises its own clear NotFittedError.
    probabilities = pipeline.predict_proba(raw)
    churn_position = list(pipeline.classes_).index(CHURN_CLASS)
    return pd.DataFrame(
        {
            id_column: raw[id_column].to_numpy(),
            PROBABILITY_COLUMN: probabilities[:, churn_position],
        }
    )
