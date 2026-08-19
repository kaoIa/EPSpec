import json
import sys
from dataclasses import replace
from types import ModuleType

import pytest

from epspec_agents.exceptions import RunCancelledError, ToolExecutionError, WorkerTimeoutError
from epspec_agents.execution import worker
from epspec_agents.execution.subprocess_runner import SubprocessToolRunner
from epspec_agents.runtime.repository import RunRepository
from epspec_agents.services.artifact_store import ArtifactStore
from epspec_agents.services.tracing import LocalTracer
from epspec_agents.tools.registry import ScientificToolRegistry


def build_subprocess_runner(config, run_id: str) -> tuple[SubprocessToolRunner, RunRepository, ArtifactStore]:
    repository = RunRepository(config.repository_path)
    repository.create(run_id, run_id, "execution", "")
    store = ArtifactStore(config.agents_dir, run_id)
    runner = SubprocessToolRunner(
        config,
        ScientificToolRegistry(config.project_root),
        repository,
        run_id,
        LocalTracer(store.run_dir / "events.jsonl", run_id),
    )
    return runner, repository, store


def worker_payload(runtime_config, tool_id: str, kind: str, input_path, output_path) -> dict:
    spec = ScientificToolRegistry(runtime_config.project_root).resolve(tool_id, kind)
    return {
        "run_id": "direct-worker",
        "stage": kind,
        "kind": kind,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "tool": spec.payload(),
    }


def test_direct_worker_model_preprocess_and_main(runtime_config, tmp_path) -> None:
    input_path = runtime_config.project_root / "Data" / "Raw Data" / "corn.csv"
    model_output = tmp_path / "model"
    model = worker.execute(worker_payload(runtime_config, "plsr", "modeling", input_path, model_output))
    assert model["status"] == "completed"
    assert (model_output / "summary.json").is_file()
    preprocess_output = tmp_path / "preprocessed.csv"
    preprocessing = worker.execute(worker_payload(runtime_config, "snv", "preprocessing", input_path, preprocess_output))
    assert preprocessing["status"] == "completed"
    assert preprocess_output.is_file()
    payload_path = tmp_path / "payload.json"
    result_path = tmp_path / "result.json"
    payload_path.write_text(json.dumps(worker_payload(runtime_config, "plsr", "modeling", input_path, tmp_path / "main-output")), encoding="utf-8")
    assert worker.main(["--payload", str(payload_path), "--result", str(result_path)]) == 0
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "completed"
    failed_payload = worker_payload(runtime_config, "plsr", "modeling", tmp_path / "missing.csv", tmp_path / "failed-output")
    payload_path.write_text(json.dumps(failed_payload), encoding="utf-8")
    assert worker.main(["--payload", str(payload_path), "--result", str(result_path)]) == 1
    assert json.loads(result_path.read_text(encoding="utf-8"))["error_type"] == "FileNotFoundError"


def test_worker_runtime_patches_and_resolution(runtime_config, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("EPSPEC_WORKER_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        worker._scientific_call([{"role": "user", "content": "rank"}])
    module = ModuleType("sliding_fixture")
    module.PRIOR_KB_PATH = "original"
    monkeypatch.setenv("EPSPEC_WORKER_API_KEY", "key")
    monkeypatch.setenv("EPSPEC_WORKER_MODEL", "model")
    worker._patch_sliding(module)
    assert module.API_KEY == "key"
    assert module.CLIENT_MODEL == "model"
    assert module._client is None
    helper_path = tmp_path / "ep_global_prior_and_ranking_regression.py"
    helper_path.write_text("PRIOR_KB_PATH = 'original'\n", encoding="utf-8")
    sys.modules.pop("ep_global_prior_and_ranking_regression", None)
    worker._patch_epspec(tmp_path)
    helper = sys.modules["ep_global_prior_and_ranking_regression"]
    assert helper.call_llm is worker._scientific_call
    assert helper.call_llm_agent35 is worker._scientific_call
    with pytest.raises(RuntimeError):
        worker._load(
            {
                "search_dir": str(tmp_path),
                "module_names": ["missing_worker_module"],
                "function_name": "run",
            }
        )


def test_subprocess_runner_path_boundaries(runtime_config) -> None:
    config = runtime_config.with_overrides(execution_mode="native")
    runner, repository, store = build_subprocess_runner(config, "boundary-run")
    input_path = config.project_root / "Data" / "Raw Data" / "corn.csv"
    outside_input = config.project_root.parent / "outside.csv"
    outside_input.write_text("value\n1\n", encoding="utf-8")
    with pytest.raises(ToolExecutionError, match="输入路径越界"):
        runner.run("plsr", "modeling", outside_input, store.run_dir / "results", "outside-input")
    with pytest.raises(ToolExecutionError, match="输出路径越界"):
        runner.run("plsr", "modeling", input_path, config.project_root / "escaped", "outside-output")
    with pytest.raises(ToolExecutionError, match="输入不存在"):
        runner.run("plsr", "modeling", config.project_root / "missing.csv", store.run_dir / "results", "missing-input")
    assert not repository.is_cancel_requested("boundary-run")


def test_subprocess_runner_failure_timeout_and_cancellation(runtime_config) -> None:
    model_path = runtime_config.project_root / "Baseline Algorithm" / "Regression" / "plsr.py"
    input_path = runtime_config.project_root / "Data" / "Raw Data" / "corn.csv"

    failed_config = runtime_config.with_overrides(execution_mode="native")
    failed_runner, _, failed_store = build_subprocess_runner(failed_config, "failed-worker")
    model_path.write_text("def run_regression(input_path, out_dir, **kwargs):\n    raise RuntimeError('worker failure')\n", encoding="utf-8")
    with pytest.raises(ToolExecutionError, match="执行失败"):
        failed_runner.run("plsr", "modeling", input_path, failed_store.run_dir / "results", "failed")

    model_path.write_text("import time\ndef run_regression(input_path, out_dir, **kwargs):\n    time.sleep(10)\n", encoding="utf-8")
    timeout_config = replace(runtime_config, execution_mode="native", worker_timeout_seconds=0.05)
    timeout_runner, _, timeout_store = build_subprocess_runner(timeout_config, "timeout-worker")
    with pytest.raises(WorkerTimeoutError):
        timeout_runner.run("plsr", "modeling", input_path, timeout_store.run_dir / "results", "timeout")

    cancelled_config = runtime_config.with_overrides(execution_mode="native")
    cancelled_runner, cancelled_repository, cancelled_store = build_subprocess_runner(cancelled_config, "cancelled-worker")
    cancelled_repository.request_cancel("cancelled-worker")
    with pytest.raises(RunCancelledError):
        cancelled_runner.run("plsr", "modeling", input_path, cancelled_store.run_dir / "results", "cancelled")
