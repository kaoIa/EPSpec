import json

import pytest
from pydantic import ValidationError

from epspec_agents.schemas import ComparisonConfig, ExperimentIntent, PlanningOutput, PreprocessConfig
from epspec_agents.services.model_factory import HeuristicModelAdapter


def test_paper_datasets_and_strict_contract() -> None:
    for dataset in ("shootout", "corn", "soil", "tecator"):
        intent = ExperimentIntent(dataset_name=dataset, model="EPSpec_plsr")
        assert intent.dataset_name == dataset
    with pytest.raises(ValidationError):
        ExperimentIntent.model_validate({"dataset_name": "milk", "model": "EPSpec_plsr"})
    with pytest.raises(ValidationError):
        ExperimentIntent.model_validate({"dataset_name": "corn", "model": "EPSpec_plsr", "unknown": True})


def test_schema_invariants() -> None:
    with pytest.raises(ValidationError):
        PreprocessConfig(enabled=True, method=None)
    with pytest.raises(ValidationError):
        ComparisonConfig(enabled=True, models=[])
    with pytest.raises(ValidationError):
        ExperimentIntent.model_validate(
            {
                "dataset_name": "corn",
                "model": "EPSpec_plsr",
                "compare": {"enabled": True, "models": ["EPSpec_plsr"]},
            }
        )


def test_offline_planner_extracts_multilingual_intent() -> None:
    request = "在玉米数据集上使用 SNV 运行 EPSpec，并和 PLSR、iPLS 与 CARS 比较。"
    payload = json.dumps({"conversation": [{"role": "user", "content": request}]}, ensure_ascii=False)
    output = HeuristicModelAdapter().generate("planner", "", payload, PlanningOutput).value
    assert output.status == "ready"
    assert output.intent is not None
    assert output.intent.dataset_name == "corn"
    assert output.intent.model == "EPSpec_plsr"
    assert output.intent.preprocess.method == "snv"
    assert set(output.intent.compare.models) == {"plsr", "ipls_plsr", "cars_plsr"}


def test_offline_planner_requests_missing_dataset() -> None:
    payload = json.dumps({"conversation": [{"role": "user", "content": "Run EPSpec"}]})
    output = HeuristicModelAdapter().generate("planner", "", payload, PlanningOutput).value
    assert output.status == "needs_clarification"
    assert output.intent is None


def test_offline_planner_uses_first_mentioned_model_as_primary() -> None:
    payload = json.dumps({"conversation": [{"role": "user", "content": "Run PLSR on corn and compare it with EPSpec."}]})
    output = HeuristicModelAdapter().generate("planner", "", payload, PlanningOutput).value
    assert output.intent is not None
    assert output.intent.model == "plsr"
    assert output.intent.compare.models == ["EPSpec_plsr"]
