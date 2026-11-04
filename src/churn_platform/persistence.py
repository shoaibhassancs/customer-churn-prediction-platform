"""Save a fitted pipeline to disk and load it back in a later session.

Only load files this project saved itself. Loading a joblib file runs code from it, so
a file from an untrusted source (such as a user upload) must never be loaded.
"""

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


def save_model(pipeline: Pipeline, path: Path) -> Path:
    """Write ``pipeline`` to ``path`` (creating its folder) and return the path.

    The file holds what the pipeline learned (categories, scaling numbers, model
    weights). It does not hold any customer rows.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(path: Path) -> Pipeline:
    """Read a pipeline saved by ``save_model``.

    Raises ``FileNotFoundError`` if there is no file, and ``TypeError`` if the file
    holds something other than a pipeline.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}. Train and save one first.")
    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise TypeError(f"{path} holds a {type(model).__name__}, not a Pipeline.")
    return model
