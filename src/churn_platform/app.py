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

from churn_platform.drivers import STABILITY_SEEDS, TOP_N
from churn_platform.evaluation import PR_AUC_MARGIN
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
    "PR-AUC: Measures how well the model ranks customers who leave above those "
    "who stay. A guessing model scores about the same as the churn rate.\n\n"
    "ROC-AUC: Measures how often the model ranks a customer who leaves above one "
    "who stays. A score of 0.5 is like a coin flip; 1.0 is perfect.\n\n"
    "Higher scores are better for both."
)
TEST_HELP = (
    "**Test results:** The model was checked on 20% of the customers it had not "
    "seen during training.\n\n"
    "**Riskiest 10%:** If you contact the 10% of customers the model ranks most "
    "likely to leave, this shows what share of all customers who left would be in "
    "that group.\n\n"
    "**Brier score:** Measures how close the model’s predicted chances are to what "
    "actually happened. Lower is better; 0 is perfect. This score is high partly "
    "because class weighting pushes the predicted chances upward."
)
DRIVERS_HELP = (
    "**Top drivers:** The columns the model relies on most. Each column's values are "
    "shuffled among the held-back customers, and the drop in PR-AUC shows how much "
    "the model needed that column. A bigger drop means a stronger driver.\n\n"
    "This shows which columns matter, not whether they raise or lower the risk."
)


def main() -> None:
    """Draw the whole screen. Streamlit calls this again after every click."""
    st.set_page_config(page_title="Customer churn predictor")
    st.title("Customer churn predictor")
    st.info("Demo only: please do not upload real personal data.")
    st.write(
        "Upload a CSV with one row per customer. Include a Churn column with “Yes” "
        "or “No” to train a model. You can also upload a file without that column "
        "to get churn risk scores for each customer."
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
    st.subheader("Train and compare models")
    if st.button("Train a model on this file"):
        try:
            # Record warnings so the "random split" notice is visible on the screen.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with st.spinner("Training and comparing four models. Please wait..."):
                    result = train_from_file(path, _model_path())
        except InvalidFileError as error:
            st.error(str(error))
            return
        notes = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        st.session_state["trained"] = {
            "key": file_key,
            "chosen": result.chosen_model,
            "scores": result.cv_scores.round(3),
            "verdict": result.beats_baselines.to_dict(),
            "test": result.test_scores.round(3),
            "drivers": result.drivers.head(TOP_N).round(3),
            "shared": result.shared_drivers,
            "notes": notes,
        }
    trained = st.session_state.get("trained")
    if trained and trained["key"] == file_key:
        st.success(
            f"Four models compared. The {trained['chosen'].replace('_', ' ')} model "
            f"was chosen by the margin rule, saved to {_model_path()} and is used "
            "for scoring."
        )
        st.dataframe(trained["scores"])
        st.caption(METRIC_HELP)
        _show_verdict(trained["verdict"])
        _show_test_check(trained["test"])
        _show_drivers(trained["drivers"], trained["shared"])
        for note in trained["notes"]:
            st.warning(note)


def _show_test_check(test: pd.Series) -> None:
    st.subheader("Held-back check")
    st.caption(TEST_HELP)
    labels = {
        "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC",
        "recall_top_10": "Churners found in the riskiest 10%",
        "brier": "Brier score (lower is better)",
    }
    st.dataframe(test.rename(index=labels).to_frame("chosen model"))


def _show_drivers(drivers: pd.DataFrame, shared: int) -> None:
    st.subheader("Top drivers")
    st.caption(DRIVERS_HELP)
    table = drivers[["importance_mean"]].rename(
        columns={"importance_mean": "PR-AUC drop when shuffled"}
    )
    st.dataframe(table)
    st.caption(
        f"{shared} of the top {TOP_N} columns stayed in the top {TOP_N} for all "
        f"{len(STABILITY_SEEDS)} different shuffles. Columns near the bottom of the "
        "list have scores too close together for their order to be reliable."
    )


def _show_verdict(verdict: dict) -> None:
    winners = [name.replace("_", " ") for name, beats in verdict.items() if beats]
    if winners:
        st.info(
            f"Better than the baselines by at least {PR_AUC_MARGIN} PR-AUC: "
            f"{', '.join(winners)}. The best of them is used for scoring."
        )
    else:
        st.info(
            f"No model beat the baselines by at least {PR_AUC_MARGIN} PR-AUC, so the "
            "simple logistic regression is still the best choice. Smaller differences "
            "can be noise."
        )


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
