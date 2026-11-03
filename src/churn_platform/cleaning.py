"""Cleaning steps for the raw telecom churn data."""

import pandas as pd

TOTAL_CHARGES = "TotalCharges"
CHURN = "Churn"
# Telecom columns whose only values are "Yes" and "No" (checked against the data).
YES_NO_COLUMNS = ("Partner", "Dependents", "PhoneService", "PaperlessBilling")


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``TotalCharges`` converted to numbers.

    In the raw file the column is text, and brand-new customers (tenure 0) have a
    blank instead of a number. They have not been billed yet, so a blank becomes 0.
    Any other non-numeric text raises ``ValueError`` instead of being hidden.
    """
    # Copy first so the caller's DataFrame is never changed behind their back.
    cleaned = df.copy()
    text = cleaned[TOTAL_CHARGES].str.strip()
    # Only blanks are replaced. Converting without errors="coerce" means a value like
    # "abc" fails loudly, which is what we want from a data-quality step.
    cleaned[TOTAL_CHARGES] = pd.to_numeric(text.replace("", "0"))
    return cleaned


def encode_churn(
    df: pd.DataFrame,
    positive_label: str = "Yes",
    negative_label: str = "No",
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``Churn`` as 1 (churned) and 0 (stayed).

    ``positive_label`` is the text that means the customer left. It is a parameter
    because other files may say "left" or "True" instead of "Yes". Any value that is
    neither label (a typo, a blank) raises ``ValueError`` so bad targets are never
    guessed at.
    """
    cleaned = df.copy()
    cleaned[CHURN] = _to_binary(cleaned[CHURN], positive_label, negative_label)
    return cleaned


def encode_yes_no_columns(
    df: pd.DataFrame, columns: tuple[str, ...] = YES_NO_COLUMNS
) -> pd.DataFrame:
    """Return a copy of ``df`` with each of ``columns`` as 1 ("Yes") or 0 ("No").

    The default columns are the telecom flags that hold only Yes or No. Columns with
    a third value, such as "No phone service", are not in the list and are left
    alone. A value other than Yes or No in a listed column raises ``ValueError``.
    """
    cleaned = df.copy()
    for column in columns:
        cleaned[column] = _to_binary(cleaned[column], "Yes", "No")
    return cleaned


def _to_binary(
    series: pd.Series, positive_label: str, negative_label: str
) -> pd.Series:
    """Map two labels to 1 and 0, raising ``ValueError`` on any other value."""
    encoded = series.map({positive_label: 1, negative_label: 0})
    # map() gives NaN for anything not in the dictionary, which marks the bad values.
    unexpected = series[encoded.isna()].unique().tolist()
    if unexpected:
        raise ValueError(
            f"{series.name} has values other than {positive_label!r} or "
            f"{negative_label!r}: {unexpected}"
        )
    return encoded.astype("int64")
