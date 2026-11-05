"""Tests for churn_platform.demo."""

import pandas as pd
import pytest

from churn_platform.demo import (
    DEMO_PATH_ENV,
    build_demo_model,
    configured_demo_path,
    main,
    restore_demo_model,
)
from churn_platform.drift import reference_path
from churn_platform.persistence import MODEL_PATH_ENV, load_model
from churn_platform.scoring import score_customers
from churn_platform.synthetic import make_customers

# Enough rows to pass the upload checks (500 rows, 50 churners) but quick to train.
SMALL = 1500


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One small demo model, trained once for the whole module."""
    path = tmp_path_factory.mktemp("demo") / "baseline.joblib"
    return build_demo_model(path, n_rows=SMALL)


def test_building_saves_a_model_and_its_drift_reference(built) -> None:
    assert built.exists()
    assert reference_path(built).exists()


def test_the_built_model_scores_new_customers(built) -> None:
    fresh = make_customers(50, seed=7).drop(columns="Churn")

    scored = score_customers(load_model(built), fresh)

    assert len(scored) == 50
    assert scored["churn_probability"].between(0, 1).all()


def test_the_same_seed_builds_the_same_demo_model(built, tmp_path) -> None:
    again = build_demo_model(tmp_path / "again.joblib", n_rows=SMALL)
    fresh = make_customers(50, seed=7).drop(columns="Churn")

    first = score_customers(load_model(built), fresh)
    second = score_customers(load_model(again), fresh)

    pd.testing.assert_frame_equal(first, second)


def test_restore_copies_the_model_and_reference_when_none_is_there(
    built, tmp_path
) -> None:
    target = tmp_path / "data" / "models" / "baseline.joblib"

    copied = restore_demo_model(built, target)

    assert copied
    assert target.exists()
    assert reference_path(target).exists()


def test_restore_never_overwrites_a_model_somebody_trained(built, tmp_path) -> None:
    target = tmp_path / "baseline.joblib"
    target.write_text("my own model")

    copied = restore_demo_model(built, target)

    assert not copied
    assert target.read_text() == "my own model"


def test_restore_does_nothing_when_there_is_no_demo_model(tmp_path) -> None:
    copied = restore_demo_model(tmp_path / "missing.joblib", tmp_path / "m.joblib")

    assert not copied
    assert not (tmp_path / "m.joblib").exists()


def test_the_demo_path_comes_from_the_setting(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(DEMO_PATH_ENV, str(tmp_path / "x.joblib"))

    assert configured_demo_path() == tmp_path / "x.joblib"


def test_the_restore_command_uses_the_two_settings(
    built, monkeypatch, tmp_path, capsys
) -> None:
    target = tmp_path / "live" / "baseline.joblib"
    monkeypatch.setenv(DEMO_PATH_ENV, str(built))
    monkeypatch.setenv(MODEL_PATH_ENV, str(target))

    assert main(["restore"]) == 0
    assert "restored" in capsys.readouterr().out
    assert main(["restore"]) == 0  # second start-up: a model is already there
    assert "Kept the existing model" in capsys.readouterr().out
    assert target.exists()


@pytest.mark.parametrize("argv", [[], ["fly"], ["build", "extra"]])
def test_an_unknown_command_prints_usage_and_fails(argv, capsys) -> None:
    assert main(argv) == 2
    assert "Usage" in capsys.readouterr().err
