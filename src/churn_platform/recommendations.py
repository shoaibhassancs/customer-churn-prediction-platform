"""Suggest a retention action, with a reason, for each high-risk customer.

The rules are plain business judgements written for the telecom columns, not
something the data proved. Whether an action really keeps customers needs campaign
results that do not exist yet.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

ACTION_COLUMN = "recommended_action"
REASON_COLUMN = "reason"
DEFAULT_ACTION = "Account review call"
DEFAULT_REASON = "High risk, with no more specific warning sign in the file"
NEW_CUSTOMER_MONTHS = 6


@dataclass(frozen=True)
class Rule:
    """One trigger: the columns it reads, a test on them, and what it recommends."""

    columns: tuple[str, ...]
    applies: Callable[[pd.DataFrame], pd.Series]
    action: str
    reason: str


# Most specific first: a customer gets the action of the first rule that matches, so
# the broad "month-to-month" rule does not swallow customers with a sharper signal.
RULES = (
    Rule(
        columns=("tenure",),
        applies=lambda df: (
            pd.to_numeric(df["tenure"], errors="coerce") < NEW_CUSTOMER_MONTHS
        ),
        action="Onboarding check-in call",
        reason=f"New customer: under {NEW_CUSTOMER_MONTHS} months with the company",
    ),
    Rule(
        columns=("InternetService", "TechSupport"),
        applies=lambda df: (
            (df["InternetService"] == "Fiber optic") & (df["TechSupport"] == "No")
        ),
        action="Offer a free tech-support trial",
        reason="Fibre internet without tech support",
    ),
    Rule(
        columns=("PaymentMethod",),
        applies=lambda df: df["PaymentMethod"] == "Electronic check",
        action="Move to automatic payment with reminders",
        reason="Pays by electronic check",
    ),
    Rule(
        columns=("Contract",),
        applies=lambda df: df["Contract"] == "Month-to-month",
        action="Offer a discounted 12-month plan",
        reason="Month-to-month contract",
    ),
)


def recommend_actions(raw: pd.DataFrame, is_high: pd.Series) -> pd.DataFrame:
    """Return an action and a reason for every high-risk customer in ``raw``.

    ``is_high`` is True for the customers to act on, in the same order as ``raw``.
    Each of them gets the action of the first rule that matches, or the default action
    if none does, so none is left without one. A rule whose columns are missing from
    the file is skipped. Customers who are not high risk get empty text.
    """
    count = len(raw)
    action = np.full(count, "", dtype=object)
    reason = np.full(count, "", dtype=object)
    waiting = is_high.to_numpy(dtype=bool).copy()
    for rule in RULES:
        if not set(rule.columns) <= set(raw.columns):
            continue
        match = rule.applies(raw).to_numpy(dtype=bool) & waiting
        action[match] = rule.action
        reason[match] = rule.reason
        waiting &= ~match
    action[waiting] = DEFAULT_ACTION
    reason[waiting] = DEFAULT_REASON
    return pd.DataFrame({ACTION_COLUMN: action, REASON_COLUMN: reason}, index=raw.index)
