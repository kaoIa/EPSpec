import asyncio
import time

from fastapi.testclient import TestClient

from epspec_agents.api import create_app
from epspec_agents.execution.subprocess_runner import SubprocessToolRunner
from epspec_agents.mcp_server import build_server
from epspec_agents.runtime.repository import RunRepository
from epspec_agents.services.artifact_store import ArtifactStore
from epspec_agents.services.tracing import LocalTracer
from epspec_agents.tools.registry import ScientificToolRegistry


def test_native_worker_isolated_contract(runtime_config) -> None:
    config = runtime_config.with_overrides(execution_mode="native")
    repository = RunRepository(config.repository_path)
    repository.create("worker_run", "worker_run", "execution", "")
    store = ArtifactStore(config.agents_dir, "worker_run")
    tracer = LocalTracer(store.run_dir / "events.jsonl", "worker_run")
    runner = SubprocessToolRunner(config, ScientificToolRegistry(config.project_root), repository, "worker_run", tracer)
    output = store.run_dir / "results" / "plsr"
    result = runner.run("plsr", "modeling", config.project_root / "Data" / "Raw Data" / "corn.csv", output, "worker_test")
    assert result["status"] == "completed"
    assert (output / "summary.json").is_file()
    assert (store.run_dir / "logs" / "worker_test.log").is_file()


def test_worker_environment_contains_only_scientific_credentials(runtime_config, monkeypatch) -> None:
    monkeypatch.setenv("EPSPEC_PLANNER_API_KEY", "planner-secret")
    repository = RunRepository(runtime_config.repository_path)
    repository.create("worker_environment", "worker_environment", "execution", "")
    store = ArtifactStore(runtime_config.agents_dir, "worker_environment")
    runner = SubprocessToolRunner(
        runtime_config,
        ScientificToolRegistry(runtime_config.project_root),
        repository,
        "worker_environment",
        LocalTracer(store.run_dir / "events.jsonl", "worker_environment"),
    )
    environment = runner._environment()
    assert "EPSPEC_PLANNER_API_KEY" not in environment
    assert environment["EPSPEC_WORKER_API_KEY"] == runtime_config.scientific.api_key
    assert environment["OPENAI_API_KEY"] == runtime_config.scientific.api_key
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_rest_api_background_lifecycle(runtime_config) -> None:
    with TestClient(create_app(runtime_config)) as client:
        response = client.post(
            "/v1/runs",
            json={
                "request": "Run EPSpec on corn and compare PLSR",
                "offline": True,
                "simulate": True,
                "auto_approve": True,
            },
        )
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        status = None
        for _ in range(100):
            status = client.get(f"/v1/runs/{run_id}").json()["status"]
            if status in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.05)
        assert status == "completed"
        assert client.get(f"/v1/runs/{run_id}/report").status_code == 200
        assert client.get(f"/v1/runs/{run_id}/artifacts").json()


def test_mcp_v2_in_memory_contract(runtime_config) -> None:
    async def verify() -> None:
        from mcp import Client

        async with Client(build_server(runtime_config), raise_exceptions=True) as client:
            tools = await client.list_tools()
            names = {tool.name for tool in tools.tools}
            assert {"epspec_capabilities", "epspec_run", "epspec_status", "epspec_resume"}.issubset(names)
            result = await client.call_tool("epspec_capabilities", {})
            assert result.is_error is False
            assert result.structured_content is not None

    asyncio.run(verify())
