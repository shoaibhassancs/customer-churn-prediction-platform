"""The Streamlit screen: upload a CSV, read the checks, train a model, score customers.

Run it from the project root with ``uv run streamlit run src/churn_platform/app.py``.
The screen only calls the tested functions in ``ingest`` and ``workflow``.
"""

import os
import tempfile
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

from churn_platform.ingest import (
    TRAIN_MODE,
    UploadError,
    ValidationReport,
    load_csv,
    validate_upload,
)
from churn_platform.workflow import InvalidFileError, score_file, train_from_file

MODEL_PATH_ENV = "CHURN_MODEL_PATH"
DEFAULT_MODEL_PATH = "models/baseline.joblib"
PREVIEW_ROWS = 20
METRIC_HELP = (
    "**PR-AUC**: how well the model ranks customers who really leave above the "
    "rest. Guessing scores about the share of customers who leave. "
    "**ROC-AUC**: the chance a customer who left is ranked above one who stayed "
    "(0.5 is a coin flip, 1.0 is perfect). Higher is better for both."
)


def main() -> None:
    """Draw the whole screen. Streamlit calls this again after every click."""
    st.set_page_config(page_title="Customer churn predictor")
    st.title("Customer churn predictor")
    st.info("Demo only: please do not upload real personal data.")
    st.write(
        "Upload a CSV with one row per customer. With a `Churn` column (Yes or No) "
        "you can train a model; with or without it you can score the customers."
    )
    uploaded = st.file_uploader("Customer file (CSV)", type="csv")
    if uploaded is None:
        return
    file_key = (uploaded.name, uploaded.size)
    # The functions read from a path, so the upload is written to a temporary folder
    # that is deleted when this block ends.
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "upload.csv"
        path.write_bytes(uploaded.getvalue())
        _show_file(path, file_key)


def _model_path() -> Path:
    return Path(os.environ.get(MODEL_PATH_ENV, DEFAULT_MODEL_PATH))


def _show_file(path: Path, file_key: tuple) -> None:
    try:
        table = load_csv(path)
    except UploadError as error:
        st.error(f"{error.issue.message} {error.issue.fix}")
        return
    st.caption(f"{len(table):,} customers, {len(table.columns)} columns")
    report = validate_upload(table)
    _show_report(report)
    if not report.is_valid:
        return
    if report.mode == TRAIN_MODE:
        _train_section(path, file_key)
    _score_section(path, file_key)


def _show_report(report: ValidationReport) -> None:
    for issue in report.errors:
        st.error(f"{issue.message}  \n**How to fix it:** {issue.fix}")
    for issue in report.warnings:
        st.warning(f"{issue.message}  \n**Suggestion:** {issue.fix}")
    if not report.errors and not report.warnings:
        st.success("No problems found in this file.")


def _train_section(path: Path, file_key: tuple) -> None:
    st.subheader("Train a model")
    if st.button("Train a model on this file"):
        try:
            # Record warnings so the "random split" notice is visible on the screen.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = train_from_file(path, _model_path())
        except InvalidFileError as error:
            st.error(str(error))
            return
        notes = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        st.session_state["trained"] = {
            "key": file_key,
            "scores": result.cv_scores.round(3),
            "notes": notes,
        }
    trained = st.session_state.get("trained")
    if trained and trained["key"] == file_key:
        st.success(f"Model trained and saved to {_model_path()}")
        st.dataframe(trained["scores"])
        st.caption(METRIC_HELP)
        for note in trained["notes"]:
            st.warning(note)


def _score_section(path: Path, file_key: tuple) -> None:
    st.subheader("Score customers")
    if st.button("Score customers in this file"):
        try:
            scored = score_file(path, _model_path(), path.with_name("scores.csv"))
        except FileNotFoundError:
            st.warning(
                "There is no trained model yet. Upload a file that has a Churn "
                "column and train a model first."
            )
            return
        except InvalidFileError as error:
            st.error(str(error))
            return
        st.session_state["scored"] = {"key": file_key, "table": scored}
    scored_state = st.session_state.get("scored")
    if scored_state and scored_state["key"] == file_key:
        _show_scores(scored_state["table"])


def _show_scores(scored: pd.DataFrame) -> None:
    st.success(f"Scored {len(scored):,} customers.")
    st.caption(
        "Probabilities rank customers correctly but run higher than the real chance "
        "of leaving, until they are calibrated."
    )
    st.dataframe(scored.head(PREVIEW_ROWS))
    st.download_button(
        "Download scored customers (CSV)",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="scored_customers.csv",
        mime="text/csv",
    )


main()
