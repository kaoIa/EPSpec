import pytest

from epspec_agents.exceptions import PlanValidationError, ToolResolutionError
from epspec_agents.guardrails.execution import validate_execution_plan, validate_interpretation_plan
from epspec_agents.schemas import ExperimentIntent
from epspec_agents.services.plan_compiler import PlanCompiler
from epspec_agents.tools.registry import ScientificToolRegistry


def test_plan_compiler_contains_every_output(runtime_config) -> None:
    intent = ExperimentIntent.model_validate(
        {
            "dataset_name": "shootout",
            "preprocess": {"enabled": True, "method": "savitzky_golay"},
            "model": "EPSpec_plsr",
            "compare": {"enabled": True, "models": ["plsr", "ipls_plsr", "cars_plsr"]},
        }
    )
    plan = PlanCompiler(runtime_config.project_root, runtime_config.agents_dir).compile(intent, "run_a")
    run_root = runtime_config.agents_dir / "runs" / "run_a"
    outputs = [
        plan.step_preprocess.output_path,
        plan.step_model_main.out_dir,
        *[step.out_dir for step in plan.step_model_compare.models],
        plan.step_report.output_path,
    ]
    assert all(path.resolve().is_relative_to(run_root.resolve()) for path in outputs)
    validate_execution_plan(plan, ScientificToolRegistry(runtime_config.project_root), runtime_config.project_root, runtime_config.agents_dir)


def test_plan_guardrail_rejects_output_escape(runtime_config) -> None:
    intent = ExperimentIntent(dataset_name="corn", model="plsr")
    plan = PlanCompiler(runtime_config.project_root, runtime_config.agents_dir).compile(intent, "run_b")
    escaped_step = plan.step_model_main.model_copy(update={"out_dir": runtime_config.project_root / "escaped"})
    escaped = plan.model_copy(update={"step_model_main": escaped_step})
    with pytest.raises(PlanValidationError):
        validate_execution_plan(escaped, ScientificToolRegistry(runtime_config.project_root), runtime_config.project_root, runtime_config.agents_dir)


def test_interpretation_guardrail_allows_only_run_scoped_results(runtime_config) -> None:
    intent = ExperimentIntent(dataset_name="corn", model="plsr")
    plan = PlanCompiler(runtime_config.project_root, runtime_config.agents_dir).compile(intent, "source_run")
    plan.step_model_main.out_dir.mkdir(parents=True)
    registry = ScientificToolRegistry(runtime_config.project_root)
    validate_interpretation_plan(plan, registry, runtime_config.project_root, runtime_config.agents_dir)
    escaped_dir = runtime_config.project_root / "escaped"
    escaped_dir.mkdir()
    escaped_main = plan.step_model_main.model_copy(update={"out_dir": escaped_dir})
    escaped_report = plan.step_report.model_copy(update={"input_dir_main": escaped_dir})
    escaped = plan.model_copy(update={"step_model_main": escaped_main, "step_report": escaped_report})
    with pytest.raises(PlanValidationError):
        validate_interpretation_plan(escaped, registry, runtime_config.project_root, runtime_config.agents_dir)


def test_registry_is_metadata_only(runtime_config) -> None:
    registry = ScientificToolRegistry(runtime_config.project_root)
    assert registry.resolve("EPSpec_plsr", "modeling").runtime_patch == "epspec"
    assert registry.resolve("snv", "preprocessing").function_name == "preprocess_file"
    assert all(item["available"] for item in registry.capabilities())
    with pytest.raises(ToolResolutionError):
        registry.resolve("unknown")
    with pytest.raises(ToolResolutionError):
        registry.resolve("plsr", "preprocessing")
