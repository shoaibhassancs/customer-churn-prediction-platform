"""The models that compete with the baselines, built the same way."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from churn_platform.baselines import build_logistic_baseline, build_majority_baseline
from churn_platform.pipeline import build_pipeline
from churn_platform.splitting import RANDOM_SEED


def build_random_forest(preprocessor: ColumnTransformer | None = None) -> Pipeline:
    """Return a pipeline whose model is a random forest.

    A random forest averages the votes of many decision trees, each trained on a random
    sample of the customers and columns. ``min_samples_leaf`` keeps single trees from
    memorising individual customers (overfitting). Class weights make the rare class
    (churners) count more. Settings are sensible defaults, not tuned.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    return build_pipeline(preprocessor, model)


def build_gradient_boosting(preprocessor: ColumnTransformer | None = None) -> Pipeline:
    """Return a pipeline whose model is histogram gradient boosting.

    Gradient boosting builds trees one after another, each correcting the mistakes of
    the trees before it. ``early_stopping`` is off so the model behaves the same on
    small and large files. Settings are sensible defaults, not tuned.
    """
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        early_stopping=False,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    return build_pipeline(preprocessor, model)


def build_candidates(
    preprocessor: ColumnTransformer | None = None,
) -> dict[str, Pipeline]:
    """Return every model to compare, by name: two baselines and two challengers."""
    return {
        "majority": build_majority_baseline(preprocessor),
        "logistic": build_logistic_baseline(preprocessor),
        "random_forest": build_random_forest(preprocessor),
        "gradient_boosting": build_gradient_boosting(preprocessor),
    }
