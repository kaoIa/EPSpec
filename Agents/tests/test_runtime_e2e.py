from epspec_agents.runtime.runner import RuntimeRunner
from epspec_agents.schemas import RunStatus


def test_offline_simulated_full_run(runtime_config) -> None:
    runner = RuntimeRunner(runtime_config)
    created = runner.create_run("full", "Run EPSpec_plsr on corn with SNV and compare plsr, ipls, and cars.")
    completed = runner.execute(created.run_id)
    assert completed.status == RunStatus.completed
    assert completed.result is not None
    assert completed.result["experiment_result"]["simulated"] is True
    assert completed.result["report"]["markdown"]
    run_dir = runtime_config.runs_dir / created.run_id
    assert (run_dir / "plan.json").is_file()
    assert (run_dir / "experiment_result.json").is_file()
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "report" / "summary.md").is_file()
    assert not (runtime_config.project_root / "Experiments" / "Tests").exists()


def test_approval_and_clarification_resume(runtime_config) -> None:
    config = runtime_config.with_overrides(auto_approve=False)
    runner = RuntimeRunner(config)
    created = runner.create_run("full", "Run EPSpec")
    clarification = runner.execute(created.run_id)
    assert clarification.status == RunStatus.awaiting_clarification
    approval = runner.resume(created.run_id, "Use soil and compare PLSR")
    assert approval.status == RunStatus.awaiting_approval
    completed = runner.resume(created.run_id, "approve")
    assert completed.status == RunStatus.completed


def test_waiting_run_can_be_cancelled(runtime_config) -> None:
    config = runtime_config.with_overrides(auto_approve=False)
    runner = RuntimeRunner(config)
    created = runner.create_run("full", "Run EPSpec on corn")
    waiting = runner.execute(created.run_id)
    assert waiting.status == RunStatus.awaiting_approval
    cancelled = runner.cancel(created.run_id)
    assert cancelled.status == RunStatus.cancelled
    assert cancelled.cancel_requested
