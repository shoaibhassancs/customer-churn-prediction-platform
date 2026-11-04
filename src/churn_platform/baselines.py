"""The two baseline models every later model has to beat."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn_platform.pipeline import build_pipeline
from churn_platform.splitting import RANDOM_SEED


def build_majority_baseline(preprocessor: ColumnTransformer | None = None) -> Pipeline:
    """Return a pipeline whose model ignores every column.

    It predicts the overall churn rate for everyone, so it is the floor: a model that
    scores no better than this has learned nothing.
    """
    # strategy="prior" returns the class shares from training as the probabilities.
    return build_pipeline(preprocessor, DummyClassifier(strategy="prior"))


def build_logistic_baseline(preprocessor: ColumnTransformer | None = None) -> Pipeline:
    """Return a pipeline with a logistic regression as the model.

    ``class_weight="balanced"`` makes mistakes on the rare class (churners) count
    more, so the model does not simply favour "stays". ``max_iter`` is raised so the
    optimiser has room to converge on the 41 prepared columns.
    """
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
    )
    return build_pipeline(preprocessor, model)
