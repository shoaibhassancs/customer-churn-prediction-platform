"""One object that cleans and prepares raw customer features for any use."""

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from churn_platform.cleaning import encode_yes_no_columns, fix_total_charges
from churn_platform.preprocessing import build_preprocessor


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the fixed cleaning rules to the feature columns of a raw table."""
    return encode_yes_no_columns(fix_total_charges(df))


def build_pipeline(
    preprocessor: ColumnTransformer | None = None,
    model: BaseEstimator | None = None,
) -> Pipeline:
    """Return an unfitted pipeline: clean the raw features, preprocess, then predict.

    Call ``fit`` on the training rows only, then ``transform`` any raw table, whether
    it is the test rows or a new customer file. The same object does both jobs, so
    training and scoring cannot drift apart. ``preprocessor`` defaults to the full
    telecom one and can be replaced, which the tests use to keep their tables small.
    ``model`` is an optional classifier added as the last step; without it the
    pipeline only prepares features.
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()
    steps = [
        # Stateless: wraps a plain function so it can be a pipeline step.
        ("clean", FunctionTransformer(clean_features)),
        # Stateful: learns categories and scaling from the training rows.
        ("preprocess", preprocessor),
    ]
    if model is not None:
        steps.append(("model", model))
    return Pipeline(steps=steps)
