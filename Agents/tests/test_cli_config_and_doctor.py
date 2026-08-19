import json
from dataclasses import replace
from pathlib import Path

import pytest

from epspec_agents import cli
from epspec_agents.config import ModelProfile, RuntimeConfig
from epspec_agents.exceptions import ConfigurationError
from epspec_agents.runtime.doctor import run_doctor
from epspec_agents.runtime.runner import RuntimeRunner
from epspec_agents.schemas import RunStatus


def bind_runtime_config(monkeypatch, runtime_config) -> None:
    monkeypatch.setattr(cli.RuntimeConfig, "from_env", classmethod(lambda cls: runtime_config))


def test_runtime_config_environment_and_validation(runtime_config, monkeypatch) -> None:
    monkeypatch.setenv("EPSPEC_EXECUTION_MODE", "simulate")
    monkeypatch.setenv("EPSPEC_OFFLINE", "yes")
    monkeypatch.setenv("EPSPEC_AUTO_APPROVE", "on")
    monkeypatch.setenv("EPSPEC_AGENT_MAX_CONCURRENCY", "0")
    monkeypatch.setenv("EPSPEC_WORKER_TIMEOUT", "0")
    monkeypatch.setenv("EPSPEC_PLANNER_API_KEY", "planner-key")
    monkeypatch.setenv("EPSPEC_INTERPRETER_API_KEY", "interpreter-key")
    monkeypatch.setenv("EPSPEC_SCIENTIFIC_API_KEY", "scientific-key")
    config = RuntimeConfig.from_env(runtime_config.agents_dir)
    assert config.execution_mode == "simulate"
    assert config.offline
    assert config.auto_approve
    assert config.max_concurrency == 1
    assert config.worker_timeout_seconds == 1.0
    assert config.planner.api_key == "planner-key"
    assert config.interpreter.api_key == "interpreter-key"
    assert config.scientific.api_key == "scientific-key"
    assert config.public()["planner"] == config.planner.public()
    monkeypatch.setenv("EPSPEC_EXECUTION_MODE", "invalid")
    with pytest.raises(ConfigurationError):
        RuntimeConfig.from_env(runtime_config.agents_dir)
    monkeypatch.setenv("EPSPEC_EXECUTION_MODE", "simulate")
    monkeypatch.setenv("EPSPEC_AGENT_MAX_CONCURRENCY", "invalid")
    with pytest.raises(ConfigurationError):
        RuntimeConfig.from_env(runtime_config.agents_dir)
    monkeypatch.setenv("EPSPEC_AGENT_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("EPSPEC_WORKER_TIMEOUT", "invalid")
    with pytest.raises(ConfigurationError):
        RuntimeConfig.from_env(runtime_config.agents_dir)


def test_model_profile_contract() -> None:
    missing = ModelProfile("planning", "model", "your key", None, 0.0, 30.0, 0)
    assert not missing.configured
    with pytest.raises(ConfigurationError):
        missing.require()
    configured = replace(missing, api_key="secret")
    configured.require()
    assert configured.public()["configured"] is True
    assert "api_key" not in configured.public()


def test_doctor_success_and_failure(runtime_config) -> None:
    healthy = run_doctor(runtime_config)
    assert healthy.ready
    assert all(check.status == "pass" for check in healthy.checks)
    missing_dataset = runtime_config.project_root / "Data" / "Raw Data" / "soil.csv"
    missing_dataset.unlink()
    unhealthy = run_doctor(runtime_config)
    assert not unhealthy.ready
    assert any(check.name == "dataset:soil" and check.status == "fail" for check in unhealthy.checks)


def test_cli_read_operations(runtime_config, monkeypatch, capsys) -> None:
    bind_runtime_config(monkeypatch, runtime_config)
    runner = RuntimeRunner(runtime_config)
    created = runner.create_run("full", "Run EPSpec_plsr on corn with SNV and compare PLSR")
    completed = runner.execute(created.run_id)
    assert completed.status == RunStatus.completed
    commands = [
        ["doctor", "--json"],
        ["capabilities"],
        ["status", completed.run_id],
        ["list", "--limit", "10", "--status", "completed"],
        ["report", completed.run_id],
        ["artifacts", completed.run_id],
        ["events", completed.run_id],
        ["eval", "--cases", str(Path(__file__).resolve().parents[1] / "evals" / "cases.json")],
    ]
    for command in commands:
        assert cli.main(command) == 0
        assert capsys.readouterr().out.strip()
    waiting = runner.create_run("full", "Run EPSpec on corn")
    assert cli.main(["cancel", waiting.run_id]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "cancelled"
    assert cli.main(["status", "missing-run"]) == 1
    assert "运行失败" in capsys.readouterr().err


def test_cli_execution_resume_and_service_dispatch(runtime_config, monkeypatch, capsys) -> None:
    bind_runtime_config(monkeypatch, runtime_config)
    runner = RuntimeRunner(runtime_config)
    created = runner.create_run("full", "Run EPSpec on corn")
    completed = runner.execute(created.run_id)
    plan_path = runtime_config.runs_dir / completed.run_id / "plan.json"

    def return_completed(config, target_stage, request, plan, interactive):
        return completed

    monkeypatch.setattr(cli, "_run_snapshot", return_completed)
    assert cli.main(["demo", "Run EPSpec on corn", "--json"]) == 0
    capsys.readouterr()
    assert cli.main(["plan", "Run EPSpec on corn", "--offline", "--simulate", "--approve", "--json"]) == 0
    capsys.readouterr()
    assert cli.main(["run", "--plan", str(plan_path), "--stage", "execution", "--offline", "--simulate", "--non-interactive", "--json"]) == 0
    capsys.readouterr()
    assert cli.main(["--stage", "full", "--request", "Run EPSpec on corn"]) == 0
    capsys.readouterr()

    import uvicorn

    uvicorn_calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: uvicorn_calls.append((args, kwargs)))
    assert cli.main(["serve", "--reload", "--port", "8011"]) == 0
    assert uvicorn_calls[0][1]["reload"] is True

    from epspec_agents import mcp_server

    mcp_calls = []
    monkeypatch.setattr(mcp_server, "serve", lambda *args: mcp_calls.append(args))
    assert cli.main(["mcp", "--transport", "streamable-http", "--port", "8012"]) == 0
    assert mcp_calls[0][0] == "streamable-http"

    interactive_config = runtime_config.with_overrides(auto_approve=False)
    waiting_runner = RuntimeRunner(interactive_config)
    waiting_created = waiting_runner.create_run("full", "Run EPSpec on corn")
    waiting = waiting_runner.execute(waiting_created.run_id)
    assert waiting.status == RunStatus.awaiting_approval
    monkeypatch.undo()
    bind_runtime_config(monkeypatch, runtime_config)
    assert cli.main(["resume", waiting.run_id, "approve", "--non-interactive", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_cli_input_and_interrupt_paths(runtime_config, monkeypatch, capsys) -> None:
    bind_runtime_config(monkeypatch, runtime_config)
    monkeypatch.setattr("builtins.input", lambda prompt: "Run PLSR on soil")
    assert cli._request(None) == "Run PLSR on soil"
    capsys.readouterr()
    monkeypatch.setattr(cli, "_legacy", lambda args, config: (_ for _ in ()).throw(EOFError()))
    assert cli.main([]) == 130
    assert "已中断" in capsys.readouterr().err
