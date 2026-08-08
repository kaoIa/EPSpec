import pytest
from pydantic import ValidationError

from epspec_agents.schemas import ComparisonConfig, ExperimentIntent, PreprocessConfig


def valid_intent(**updates):
    value = {
        "dataset_name": "corn",
        "task_type": "regression",
        "preprocess": {"enabled": False, "method": None},
        "model": "EPSpec_plsr",
        "compare": {"enabled": False, "models": []},
    }
    value.update(updates)
    return value


def test_valid_intent():
    intent = ExperimentIntent.model_validate(valid_intent())
    assert intent.dataset_name == "corn"
    assert intent.model == "EPSpec_plsr"


@pytest.mark.parametrize("dataset", ["diesel", "cassava", "milk"])
def test_unsupported_dataset_rejected(dataset):
    with pytest.raises(ValidationError):
        ExperimentIntent.model_validate(valid_intent(dataset_name=dataset))


def test_unsupported_preprocessing_rejected():
    with pytest.raises(ValidationError):
        PreprocessConfig(enabled=True, method="msc")


def test_unsupported_model_rejected():
    with pytest.raises(ValidationError):
        ExperimentIntent.model_validate(valid_intent(model="random_forest"))


def test_duplicate_comparison_rejected():
    with pytest.raises(ValidationError):
        ComparisonConfig(enabled=True, models=["plsr", "plsr"])


def test_primary_comparison_duplication_rejected():
    with pytest.raises(ValidationError):
        ExperimentIntent.model_validate(valid_intent(compare={"enabled": True, "models": ["EPSpec_plsr"]}))
