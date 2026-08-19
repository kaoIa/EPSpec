from pathlib import Path

import pytest

from epspec_agents.exceptions import RunNotFoundError, RunStateError
from epspec_agents.execution.simulator import ScientificSimulator
from epspec_agents.runtime.repository import RunRepository
from epspec_agents.runtime.runner import RuntimeRunner
from epspec_agents.schemas import ExperimentIntent, RunStatus
from epspec_agents.services.plan_compiler import PlanCompiler
from epspec_agents.services.result_parser import ResultParser


def test_stage_specific_planning_execution_and_interpretation(runtime_config) -> None:
    runner = RuntimeRunner(runtime_config)
    planning_created = runner.create_run("planning", "Run EPSpec on soil and compare PLSR")
    planning = runner.execute(planning_created.run_id)
    assert planning.status == RunStatus.completed
    assert planning.result is not None
    assert planning.result["plan"]["dataset_name"] == "soil"
    assert "experiment_result" not in planning.result

    source_plan = PlanCompiler(runtime_config.project_root, runtime_config.agents_dir).compile(
        ExperimentIntent(dataset_name="corn", model="plsr"),
        "source_results",
    )
    execution_created = runner.create_run("execution", "")
    execution = runner.execute(execution_created.run_id, source_plan)
    assert execution.status == RunStatus.completed
    assert execution.result is not None
    assert execution.result["main_result"]["status"] == "completed"
    assert "report" not in execution.result

    ScientificSimulator().model(
        source_plan.step_model_main.method,
        source_plan.dataset_name,
        source_plan.step_model_main.input_path,
        source_plan.step_model_main.out_dir,
    )
    interpretation_created = runner.create_run("interpretation", "")
    interpretation = runner.execute(interpretation_created.run_id, source_plan)
    assert interpretation.status == RunStatus.completed
    assert interpretation.result is not None
    assert interpretation.result["report"]["markdown"]
    assert interpretation.result["experiment_result"]["dataset_name"] == "corn"
    assert runner.load_plan().dataset_name in {"soil", "corn"}


def test_guarded_graph_failure_routes_to_terminal_manifest(runtime_config, monkeypatch) -> None:
    def fail_parse(self, plan, simulated):
        raise ValueError("forced parser failure")

    monkeypatch.setattr(ResultParser, "parse", fail_parse)
    runner = RuntimeRunner(runtime_config)
    created = runner.create_run("full", "Run EPSpec on corn")
    failed = runner.execute(created.run_id)
    assert failed.status == RunStatus.failed
    assert failed.current_stage == "failure"
    assert failed.result is not None
    assert failed.result["errors"][0]["stage"] == "aggregate_results"
    assert (runtime_config.runs_dir / created.run_id / "manifest.json").is_file()


def test_runtime_invocation_failure_is_persisted(runtime_config, monkeypatch) -> None:
    from epspec_agents.runtime import runner as runner_module

    def fail_workflow(*args, **kwargs):
        raise RuntimeError("forced workflow failure")

    monkeypatch.setattr(runner_module, "build_workflow", fail_workflow)
    runner = RuntimeRunner(runtime_config)
    created = runner.create_run("full", "Run EPSpec on corn")
    with pytest.raises(RuntimeError, match="forced workflow failure"):
        runner.execute(created.run_id)
    failed = runner.get(created.run_id)
    assert failed.status == RunStatus.failed
    assert failed.current_stage == "runtime"
    assert (runtime_config.runs_dir / created.run_id / "runtime_error.json").is_file()


def test_runtime_state_guards_and_prestart_cancellation(runtime_config) -> None:
    runner = RuntimeRunner(runtime_config)
    with pytest.raises(RunStateError):
        runner.create_run("unknown")
    created = runner.create_run("full", "Run EPSpec on corn")
    runner.repository.request_cancel(created.run_id)
    cancelled = runner.execute(created.run_id)
    assert cancelled.status == RunStatus.cancelled
    with pytest.raises(RunStateError):
        runner.execute(created.run_id)
    with pytest.raises(RunStateError):
        runner.resume(created.run_id, "approve")
    with pytest.raises(FileNotFoundError):
        runner.report(created.run_id)
    with pytest.raises(RunNotFoundError):
        runner.get("missing")


def test_repository_filters_metadata_and_terminal_guards(tmp_path: Path) -> None:
    repository = RunRepository(tmp_path / "runs.sqlite")
    repository.create("first", "first", "full", "request", {"one": 1})
    repository.create("second", "second", "planning", "request")
    updated = repository.update(
        "first",
        status="completed",
        current_stage="finalization",
        interruption={"type": "approval"},
        result={"plan": {"dataset_name": "corn"}},
        metadata={"two": 2},
    )
    assert updated.metadata == {"one": 1, "two": 2}
    cleared = repository.update("first", clear_interruption=True)
    assert cleared.interruption is None
    assert repository.list(0)[0]
    assert repository.list(5000, "completed")[0].run_id == "first"
    assert repository.latest_plan_run().run_id == "first"
    assert not repository.is_cancel_requested("missing")
    with pytest.raises(RunStateError):
        repository.request_cancel("first")
