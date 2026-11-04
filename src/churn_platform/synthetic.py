"""Made-up customers in the same shape as the telecom file, with planted churn drivers.

Use it for demos and for testing the plumbing (large files, the screen). The drivers
are planted on purpose, so models score unrealistically well on it: it says nothing
about how good a model is on real customers.

Planted drivers: month-to-month contracts, short tenure, electronic-check payment,
fiber internet and senior customers churn more; two-year contracts churn less.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from churn_platform.splitting import RANDOM_SEED

ADD_ON_COLUMNS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)
PAYMENT_METHODS = (
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
)
ELECTRONIC_CHECK = PAYMENT_METHODS[0]
# Intercept chosen so roughly a quarter of the customers churn, like the real file.
_BASE_LOG_ODDS = -1.55


def make_customers(n_rows: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Return ``n_rows`` made-up customers as a raw table, ``Churn`` included.

    The columns, their order and their text values match the telecom CSV, so the
    checker, the pipeline and the screen accept the result unchanged. Customers get
    unique IDs (``SYN-000001`` and so on). The same ``seed`` always gives the same
    table. As in the real file, ``TotalCharges`` is text and is a blank for customers
    with tenure 0.
    """
    if n_rows < 1:
        raise ValueError("n_rows must be at least 1.")
    rng = np.random.default_rng(seed)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n_rows, p=[0.55, 0.21, 0.24]
    )
    tenure = _draw_tenure(rng, contract)
    phone = rng.random(n_rows) < 0.9
    internet = rng.choice(["DSL", "Fiber optic", "No"], n_rows, p=[0.34, 0.44, 0.22])
    has_internet = internet != "No"
    payment = rng.choice(PAYMENT_METHODS, n_rows, p=[0.34, 0.23, 0.22, 0.21])
    senior = (rng.random(n_rows) < 0.16).astype(int)

    table = {
        "customerID": [f"SYN-{i:06d}" for i in range(1, n_rows + 1)],
        "gender": rng.choice(["Female", "Male"], n_rows),
        "SeniorCitizen": senior,
        "Partner": _yes_no(rng.random(n_rows) < 0.48),
        "Dependents": _yes_no(rng.random(n_rows) < 0.30),
        "tenure": tenure,
        "PhoneService": _yes_no(phone),
        "MultipleLines": np.where(
            phone, _yes_no(rng.random(n_rows) < 0.42), "No phone service"
        ),
        "InternetService": internet,
    }
    add_on_count = np.zeros(n_rows)
    for column in ADD_ON_COLUMNS:
        taken = has_internet & (rng.random(n_rows) < 0.4)
        add_on_count += taken
        table[column] = np.where(has_internet, _yes_no(taken), "No internet service")
    table["Contract"] = contract
    table["PaperlessBilling"] = _yes_no(rng.random(n_rows) < 0.6)
    table["PaymentMethod"] = payment

    monthly = _monthly_charges(rng, internet, phone, add_on_count)
    table["MonthlyCharges"] = monthly
    table["TotalCharges"] = _total_charges(rng, tenure, monthly)
    chance = _churn_probability(contract, tenure, payment, internet, senior)
    table["Churn"] = _yes_no(rng.random(n_rows) < chance)
    return pd.DataFrame(table)


def write_customers_csv(path: Path, n_rows: int, seed: int = RANDOM_SEED) -> Path:
    """Write ``n_rows`` made-up customers to a CSV file and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_customers(n_rows, seed).to_csv(path, index=False)
    return path


def _yes_no(flags: np.ndarray) -> np.ndarray:
    return np.where(flags, "Yes", "No")


def _draw_tenure(rng: np.random.Generator, contract: np.ndarray) -> np.ndarray:
    """Month-to-month customers are newer; longer contracts have longer tenure."""
    n_rows = len(contract)
    short = np.clip(rng.exponential(18, n_rows), 0, 72)
    one_year = rng.uniform(6, 72, n_rows)
    two_year = rng.uniform(12, 72, n_rows)
    is_monthly = contract == "Month-to-month"
    is_one_year = contract == "One year"
    months = np.select([is_monthly, is_one_year], [short, one_year], two_year)
    return months.astype(int)


def _monthly_charges(
    rng: np.random.Generator,
    internet: np.ndarray,
    phone: np.ndarray,
    add_on_count: np.ndarray,
) -> np.ndarray:
    base = np.select(
        [internet == "No", internet == "DSL"],
        [rng.uniform(18, 30, len(internet)), rng.uniform(35, 65, len(internet))],
        rng.uniform(65, 95, len(internet)),
    )
    return (base + 5 * add_on_count + 2 * phone).round(2)


def _total_charges(
    rng: np.random.Generator, tenure: np.ndarray, monthly: np.ndarray
) -> np.ndarray:
    """Text totals: months times monthly price, blank before the first bill."""
    total = (tenure * monthly * rng.uniform(0.95, 1.05, len(tenure))).round(2)
    return np.where(tenure == 0, " ", pd.Series(total).astype(str).to_numpy())


def _churn_probability(
    contract: np.ndarray,
    tenure: np.ndarray,
    payment: np.ndarray,
    internet: np.ndarray,
    senior: np.ndarray,
) -> np.ndarray:
    log_odds = (
        _BASE_LOG_ODDS
        + 1.5 * (contract == "Month-to-month")
        - 1.2 * (contract == "Two year")
        - 0.035 * tenure
        + 0.6 * (payment == ELECTRONIC_CHECK)
        + 0.5 * (internet == "Fiber optic")
        + 0.3 * senior
    )
    return 1 / (1 + np.exp(-log_odds))
