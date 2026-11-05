"""Warn when a file being scored looks different from the data the model learned from.

Each input column is compared with a reference saved at training time, using the
population stability index (PSI). The check only warns; scoring still happens.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

BINS = 10
# Rules of thumb from credit-risk practice, not tuned to this project's data.
MODERATE_PSI = 0.1
MAJOR_PSI = 0.25
# PSI is noisy on small files: a few hundred rows can move it by chance.
MIN_ROWS = 200
# Stops log(0) when a group is empty in one of the two files.
SHARE_FLOOR = 1e-4
BLANK = "(blank)"

NONE, MODERATE, MAJOR = "none", "moderate", "major"


@dataclass(frozen=True)
class DriftReport:
    """PSI per column (``psi`` and ``level``), the file's size and whether it is big
    enough for the numbers to be reliable."""

    table: pd.DataFrame
    rows: int
    reliable: bool

    def flagged(self) -> pd.DataFrame:
        """The columns with a moderate or major shift, biggest first."""
        return self.table[self.table["level"] != NONE]


def column_kinds(pipeline: Pipeline) -> dict[str, str]:
    """Say which model input columns are numbers and which are categories.

    Read from the fitted preprocessor: its ``numeric`` group is numbers, and
    everything else (text categories and yes/no flags) is treated as categories.
    """
    kinds = {}
    for name, _, columns in pipeline.named_steps["preprocess"].transformers_:
        if name != "remainder":
            for column in columns:
                kinds[column] = "numeric" if name == "numeric" else "categorical"
    return kinds


def _numbers(column: pd.Series) -> np.ndarray:
    # Blank or unreadable cells become missing and are left out of the comparison.
    return pd.to_numeric(column, errors="coerce").dropna().to_numpy(dtype=float)


def _labels(column: pd.Series) -> pd.Series:
    return column.fillna(BLANK).astype(str)


def _bin_shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    groups = np.searchsorted(edges, values, side="right")
    return np.bincount(groups, minlength=len(edges) + 1) / len(values)


def build_reference(pipeline: Pipeline, X: pd.DataFrame) -> dict[str, Any]:
    """Describe how each input column was spread across the training rows.

    Numbers are split into up to ten groups at the training quantiles; categories
    keep their share of rows. The result is plain lists and numbers, so it can be
    saved as JSON. It holds no customer rows.
    """
    reference: dict[str, Any] = {}
    for column, kind in column_kinds(pipeline).items():
        if kind == "numeric":
            values = _numbers(X[column])
            if len(values) == 0:
                continue
            cut_points = np.linspace(0, 1, BINS + 1)[1:-1]
            edges = np.unique(np.quantile(values, cut_points))
            shares = _bin_shares(values, edges)
            reference[column] = {
                "kind": kind,
                "edges": edges.tolist(),
                "shares": shares.tolist(),
            }
        else:
            counts = _labels(X[column]).value_counts(normalize=True)
            reference[column] = {
                "kind": kind,
                "categories": counts.index.tolist(),
                "shares": counts.tolist(),
            }
    return reference


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.maximum(expected, SHARE_FLOOR)
    a = np.maximum(actual, SHARE_FLOOR)
    return float(np.sum((a - e) * np.log(a / e)))


def compute_psi(reference: dict[str, Any], table: pd.DataFrame) -> pd.Series:
    """Return the PSI of every referenced column that is in ``table``, biggest first.

    0 means the column is spread the way it was in training. A category the model
    never saw counts as its own group. Columns missing from ``table`` are skipped.
    """
    scores = {}
    for column, ref in reference.items():
        if column not in table.columns:
            continue
        expected = np.array(ref["shares"])
        if ref["kind"] == "numeric":
            values = _numbers(table[column])
            if len(values) == 0:
                continue
            actual = _bin_shares(values, np.array(ref["edges"]))
        else:
            labels = _labels(table[column])
            known = ref["categories"]
            share = labels.value_counts(normalize=True)
            actual = np.array([share.get(c, 0.0) for c in known])
            actual = np.append(actual, max(0.0, 1.0 - actual.sum()))  # unseen
            expected = np.append(expected, 0.0)
        scores[column] = _psi(expected, actual)
    return pd.Series(scores, dtype=float).sort_values(ascending=False)


def level_of(psi: float) -> str:
    """Name the size of a shift: none, moderate (from 0.1) or major (from 0.25)."""
    if psi >= MAJOR_PSI:
        return MAJOR
    return MODERATE if psi >= MODERATE_PSI else NONE


def check_drift(reference: dict[str, Any], table: pd.DataFrame) -> DriftReport:
    """Compare ``table`` with the training reference, column by column."""
    psi = compute_psi(reference, table)
    result = pd.DataFrame({"psi": psi, "level": [level_of(v) for v in psi]})
    return DriftReport(table=result, rows=len(table), reliable=len(table) >= MIN_ROWS)


def reference_path(model_path: Path) -> Path:
    """Where the reference lives: next to the model (``baseline.reference.json``)."""
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}.reference.json")


def save_reference(reference: dict[str, Any], model_path: Path) -> Path:
    """Write the reference next to the model file and return its path."""
    path = reference_path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reference, indent=2))
    return path


def load_reference(model_path: Path) -> dict[str, Any] | None:
    """Read the reference saved next to the model, or None if there is none.

    A model saved before this check existed has no reference, so it is not checked.
    """
    path = reference_path(model_path)
    if not path.exists():
        return None
    return json.loads(path.read_text())
