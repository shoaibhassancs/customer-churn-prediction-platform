"""Tests for the Streamlit screen, driven with Streamlit's own AppTest."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from churn_platform.preprocessing import CATEGORICAL_COLUMNS

# The app reads its model location from this variable; tests point it at a temp folder.
MODEL_PATH_ENV = "CHURN_MODEL_PATH"
APP_FILE = Path(__file__).resolve().parents[1] / "src" / "churn_platform" / "app.py"


def make_upload(n: int = 600, churners: int = 120) -> pd.DataFrame:
    """A table that passes validation: every required column, 120 of 600 churned."""
    rng = np.random.RandomState(0)
    table = {"customerID": [f"C{i:05d}" for i in range(n)]}
    for column in CATEGORICAL_COLUMNS:
        table[column] = rng.choice(["A", "B", "C"], size=n)
    table["tenure"] = rng.randint(1, 72, size=n)
    table["MonthlyCharges"] = rng.uniform(20, 120, size=n).round(2)
    table["TotalCharges"] = (table["tenure"] * 50.5).astype(str)
    for column in ("Partner", "Dependents", "PhoneService", "PaperlessBilling"):
        table[column] = rng.choice(["Yes", "No"], size=n)
    table["SeniorCitizen"] = rng.randint(0, 2, size=n)
    table["Churn"] = rng.permutation(["Yes"] * churners + ["No"] * (n - churners))
    return pd.DataFrame(table)


@pytest.fixture
def app(tmp_path, monkeypatch) -> AppTest:
    """The app, started with its model file pointed at a temporary folder."""
    monkeypatch.setenv(MODEL_PATH_ENV, str(tmp_path / "model.joblib"))
    return AppTest.from_file(str(APP_FILE), default_timeout=60).run()


def upload(app: AppTest, table: pd.DataFrame | bytes) -> AppTest:
    data = table if isinstance(table, bytes) else table.to_csv(index=False).encode()
    app.file_uploader[0].set_value(("customers.csv", data, "text/csv"))
    return app.run()


def button(app: AppTest, label_start: str):
    return next(b for b in app.button if b.label.startswith(label_start))


def test_first_screen_asks_for_a_file_and_warns_about_personal_data(app) -> None:
    assert not app.exception
    assert app.title[0].value == "Customer churn predictor"
    assert "personal data" in app.info[0].value
    assert len(app.file_uploader) == 1
    assert len(app.button) == 0


def test_a_good_training_file_shows_success_and_both_buttons(app) -> None:
    upload(app, make_upload())

    assert not app.exception
    assert app.success[0].value == "No problems found in this file."
    assert button(app, "Train a model")
    assert button(app, "Score customers")


def test_training_shows_scores_a_saved_model_and_the_random_split_warning(
    app, tmp_path
) -> None:
    upload(app, make_upload())

    button(app, "Train a model").click()
    app.run()

    assert not app.exception
    assert (tmp_path / "model.joblib").exists()
    assert any("Four models compared" in s.value for s in app.success)
    scores = app.dataframe[0].value
    assert scores.index.tolist() == [
        "majority",
        "logistic",
        "random_forest",
        "gradient_boosting",
    ]
    assert any("PR-AUC" in i.value for i in app.info[1:])
    assert "Held-back check" in [h.value for h in app.subheader]
    held_back = app.dataframe[1].value
    assert held_back.shape == (4, 1)
    assert "Top drivers" in [h.value for h in app.subheader]
    assert app.dataframe[2].value.columns.tolist() == ["PR-AUC drop when shuffled"]
    assert any("Random split" in w.value for w in app.warning)


def test_scoring_after_training_shows_a_preview_and_a_download(app) -> None:
    upload(app, make_upload())
    button(app, "Train a model").click()
    app.run()

    button(app, "Score customers").click()
    app.run()

    assert not app.exception
    assert any("Scored 600 customers" in s.value for s in app.success)
    assert any("No drift found" in s.value for s in app.success)
    preview = next(d.value for d in app.dataframe if len(d.value) == 20)
    assert preview.columns.tolist() == [
        "customerID",
        "churn_probability",
        "risk_band",
        "recommended_action",
        "reason",
    ]
    assert any("High risk:" in m.value for m in app.markdown)
    assert any("have a recommended action" in m.value for m in app.markdown)
    assert len(preview) == 20


def test_a_file_without_churn_can_only_be_scored_not_trained(app) -> None:
    upload(app, make_upload().drop(columns="Churn"))

    assert [b.label for b in app.button] == ["Score customers in this file"]


def test_scoring_before_any_model_exists_explains_what_to_do(app) -> None:
    upload(app, make_upload().drop(columns="Churn"))

    button(app, "Score customers").click()
    app.run()

    assert not app.exception
    assert any("no trained model yet" in w.value for w in app.warning)


def test_a_file_with_problems_lists_them_with_fixes_and_offers_no_buttons(
    app,
) -> None:
    broken = make_upload().drop(columns=["customerID", "Contract"])

    upload(app, broken)

    assert not app.exception
    text = " ".join(e.value for e in app.error)
    assert "customerID" in text and "Contract" in text
    assert "How to fix it" in text
    assert len(app.button) == 0


def test_an_unreadable_file_gets_a_plain_message_not_a_crash(app) -> None:
    upload(app, b"")

    assert not app.exception
    assert "empty" in app.error[0].value
