import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cancer import get_features_and_target, load_config, load_data


@pytest.fixture(scope="module")
def cancer_config():
    return load_config()


@pytest.fixture(scope="module")
def trained_model(cancer_config, tmp_path_factory):
    from cancer import train_and_save_model

    model_path = tmp_path_factory.mktemp("model") / "cancer_model.joblib"
    model, _ = train_and_save_model(load_data(), cancer_config, model_path=str(model_path))
    return model


@pytest.fixture(scope="module")
def one_sample(cancer_config):
    X, _ = get_features_and_target(load_data(), cancer_config)
    return X.head(1)


def test_same_sample_gives_same_answer(trained_model, one_sample):
    first = trained_model.predict(one_sample)[0]
    second = trained_model.predict(one_sample)[0]
    assert first == second


def test_prediction_is_a_known_class(trained_model, one_sample):
    assert trained_model.predict(one_sample)[0] in (0, 1)
