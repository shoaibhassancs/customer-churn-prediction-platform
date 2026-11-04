"""The preprocessing step that learns from training data and is reused to predict."""

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_platform.cleaning import YES_NO_COLUMNS

# Text columns, including the ones with a third value such as "No phone service".
CATEGORICAL_COLUMNS = (
    "gender",
    "InternetService",
    "Contract",
    "PaymentMethod",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)
NUMERIC_COLUMNS = ("tenure", "MonthlyCharges", "TotalCharges")
# Already 0/1 after cleaning, so they pass through unchanged.
FLAG_COLUMNS = (*YES_NO_COLUMNS, "SeniorCitizen")


def build_preprocessor(
    categorical_columns: Sequence[str] = CATEGORICAL_COLUMNS,
    numeric_columns: Sequence[str] = NUMERIC_COLUMNS,
    flag_columns: Sequence[str] = FLAG_COLUMNS,
) -> ColumnTransformer:
    """Return an unfitted transformer that one-hot encodes, scales and passes through.

    Call ``fit`` on the training rows only, then ``transform`` on any other rows. That
    way categories and scaling statistics come from training data alone and are reused
    unchanged at prediction time. Columns not listed (such as ``customerID`` and the
    ``Churn`` target) are dropped.
    """
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                # Unknown category at prediction time becomes all zeros, not a crash.
                # sparse_output=False because labelled-table output needs dense data.
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_columns),
            ),
            ("numeric", StandardScaler(), list(numeric_columns)),
            ("flags", "passthrough", list(flag_columns)),
        ],
        remainder="drop",
        # Keep plain column names such as "tenure" instead of "numeric__tenure".
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
