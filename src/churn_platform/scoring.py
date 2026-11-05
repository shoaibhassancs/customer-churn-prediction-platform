"""Turn a fitted pipeline and a raw customer table into churn probabilities."""

import math

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churn_platform.recommendations import recommend_actions

ID_COLUMN = "customerID"
PROBABILITY_COLUMN = "churn_probability"
RISK_BAND_COLUMN = "risk_band"
HIGH, MEDIUM, LOW = "High", "Medium", "Low"
# Share of customers a retention team can contact; the next MEDIUM_SHARE are Medium.
HIGH_SHARE = 0.10
MEDIUM_SHARE = 0.20
# encode_churn turned "churned" into 1, so class 1 is the one we report.
CHURN_CLASS = 1


def assign_risk_bands(
    probability: pd.Series,
    high_share: float = HIGH_SHARE,
    medium_share: float = MEDIUM_SHARE,
) -> pd.Series:
    """Label customers High, Medium or Low by their rank, not by the probability.

    The riskiest ``high_share`` of customers are High and the next ``medium_share``
    are Medium; everyone else is Low. Only the order matters, so it still works when
    the probabilities run too high. At least one customer is High, and customers with
    equal probabilities are separated by their position in the table. Every file
    therefore gets a High group, even if nobody in it is likely to leave.
    """
    if not 0 < high_share <= 1:
        raise ValueError("high_share must be above 0 and at most 1.")
    if not 0 <= medium_share <= 1:
        raise ValueError("medium_share must be between 0 and 1.")
    count = len(probability)
    high_count = max(1, math.ceil(high_share * count)) if count else 0
    medium_count = min(math.ceil(medium_share * count), count - high_count)
    # rank 1 is the riskiest customer; "first" breaks ties by table position.
    rank = probability.rank(method="first", ascending=False)
    bands = np.select(
        [rank <= high_count, rank <= high_count + medium_count],
        [HIGH, MEDIUM],
        default=LOW,
    )
    return pd.Series(bands, index=probability.index, name=RISK_BAND_COLUMN)


def score_customers(
    pipeline: Pipeline,
    raw: pd.DataFrame,
    id_column: str = ID_COLUMN,
    high_share: float = HIGH_SHARE,
) -> pd.DataFrame:
    """Return one row per customer: ID, probability, risk band, action and reason.

    ``pipeline`` must already be fitted and end in a classifier. ``raw`` is a table in
    the original file format (text values and all); the pipeline cleans it itself.
    Rows keep the order of ``raw``. A ``Churn`` column, if present, is ignored.

    The probability comes from a model trained with class weights, so it ranks
    customers correctly but overstates the true chance of churn until it is calibrated.
    The risk band uses the ranking only; see ``assign_risk_bands``. High-risk
    customers also get a recommended action and a reason; see ``recommendations``.
    """
    if id_column not in raw.columns:
        raise ValueError(f"The table has no {id_column!r} column to name customers.")
    # Predict first, so an unfitted pipeline raises its own clear NotFittedError.
    probabilities = pipeline.predict_proba(raw)
    churn_position = list(pipeline.classes_).index(CHURN_CLASS)
    probability = pd.Series(probabilities[:, churn_position])
    bands = assign_risk_bands(probability, high_share)
    # The scored table has a fresh 0..n-1 index, so line the raw rows up the same way.
    actions = recommend_actions(raw.reset_index(drop=True), bands == HIGH)
    scored = pd.DataFrame(
        {
            id_column: raw[id_column].to_numpy(),
            PROBABILITY_COLUMN: probability,
            RISK_BAND_COLUMN: bands,
        }
    )
    return pd.concat([scored, actions], axis=1)
