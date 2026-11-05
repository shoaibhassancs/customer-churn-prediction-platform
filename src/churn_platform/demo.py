"""A demo model that is trained when the Docker image is built and restored at start-up.

On a host with no persistent disk (such as a free Render service) the saved model is
wiped on every restart. The image therefore carries a model trained on made-up data,
and the container copies it into place at start-up when no model is there yet. Run it
as ``python -m churn_platform.demo build`` or ``... restore``.
"""

import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

from churn_platform.drift import reference_path
from churn_platform.persistence import configured_model_path
from churn_platform.splitting import RANDOM_SEED
from churn_platform.synthetic import write_customers_csv
from churn_platform.workflow import train_from_file

DEMO_PATH_ENV = "CHURN_DEMO_MODEL_PATH"
DEFAULT_DEMO_PATH = "demo-model/baseline.joblib"
# About the size of the Telco file, so training time and memory are similar.
DEMO_ROWS = 7000


def configured_demo_path() -> Path:
    """Where the baked-in demo model lives: ``CHURN_DEMO_MODEL_PATH`` or the default."""
    return Path(os.environ.get(DEMO_PATH_ENV, DEFAULT_DEMO_PATH))


def build_demo_model(
    demo_path: Path, n_rows: int = DEMO_ROWS, seed: int = RANDOM_SEED
) -> Path:
    """Train a model on ``n_rows`` made-up customers and save it at ``demo_path``.

    The saved model and its drift reference hold what was learned, not the customers,
    and the made-up file is deleted afterwards. No real data is involved.
    """
    with tempfile.TemporaryDirectory() as folder:
        csv = write_customers_csv(Path(folder) / "demo.csv", n_rows, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the random-split notice is for the screen
            train_from_file(csv, demo_path)
    return Path(demo_path)


def restore_demo_model(demo_path: Path, model_path: Path) -> bool:
    """Copy the demo model (and its drift reference) to ``model_path`` if it is missing.

    Returns True if it copied, and False if a model is already there or there is no
    demo model. It never overwrites a model somebody trained.
    """
    demo_path, model_path = Path(demo_path), Path(model_path)
    if model_path.exists() or not demo_path.exists():
        return False
    model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(demo_path, model_path)
    demo_reference = reference_path(demo_path)
    if demo_reference.exists():
        shutil.copy2(demo_reference, reference_path(model_path))
    return True


def main(argv: list[str]) -> int:
    """Run ``build`` or ``restore``; return the exit code."""
    if len(argv) != 1 or argv[0] not in ("build", "restore"):
        print("Usage: python -m churn_platform.demo build|restore", file=sys.stderr)
        return 2
    if argv[0] == "build":
        path = build_demo_model(configured_demo_path())
        print(f"Demo model saved at {path}")
    else:
        copied = restore_demo_model(configured_demo_path(), configured_model_path())
        print("Demo model restored." if copied else "Kept the existing model.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
