import asyncio
import time
from dataclasses import replace

from fastapi.testclient import TestClient

from epspec_agents.api import create_app
from epspec_agents.mcp_server import build_server
from epspec_agents.runtime.runner import RuntimeRunner
from epspec_agents.schemas import RunStatus


def wait_for_status(client: TestClient, run_id: str, headers: dict[str, str], expected: set[str]) -> str:
    observed = ""
    for _ in range(200):
        response = client.get(f"/v1/runs/{run_id}", headers=headers)
        assert response.status_code == 200
        observed = response.json()["status"]
        if observed in expected:
            return observed
        time.sleep(0.05)
    return observed


def test_rest_auth_resume_events_errors_and_cancellation(runtime_config) -> None:
    config = replace(runtime_config, auto_approve=False, server_token="server-secret")
    bearer = {"Authorization": "Bearer server-secret"}
    alternate = {"X-EPSpec-Token": "server-secret"}
    with TestClient(create_app(config)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/capabilities").status_code == 401
        assert client.get("/v1/capabilities", headers=bearer).status_code == 200
        assert client.get("/v1/capabilities", headers=alternate).status_code == 200
        assert client.get("/v1/runs/missing", headers=bearer).status_code == 404
        assert client.post("/v1/runs", headers=bearer, json={"request": "x", "unknown": True}).status_code == 422

        created_response = client.post(
            "/v1/runs",
            headers=bearer,
            json={
                "request": "Run EPSpec on corn",
                "offline": True,
                "simulate": True,
                "auto_approve": False,
            },
        )
        assert created_response.status_code == 202
        run_id = created_response.json()["run_id"]
        assert wait_for_status(client, run_id, bearer, {"awaiting_approval", "failed"}) == "awaiting_approval"
        assert client.post(f"/v1/runs/{run_id}/resume", headers=bearer, json={"response": ""}).status_code == 422
        assert client.post(f"/v1/runs/{run_id}/resume", headers=bearer, json={"response": "approve"}).status_code == 202
        assert wait_for_status(client, run_id, bearer, {"completed", "failed"}) == "completed"
        assert client.get("/v1/runs", headers=bearer, params={"status": "completed"}).json()
        assert client.get(f"/v1/runs/{run_id}/report", headers=bearer).status_code == 200
        assert client.get(f"/v1/runs/{run_id}/artifacts", headers=bearer).json()
        event_response = client.get(f"/v1/runs/{run_id}/events", headers=bearer)
        assert event_response.status_code == 200
        assert "event: terminal" in event_response.text
        assert client.post(f"/v1/runs/{run_id}/cancel", headers=bearer).status_code == 409

        cancellable = RuntimeRunner(config).create_run("full", "Run EPSpec on soil")
        cancelled_response = client.post(f"/v1/runs/{cancellable.run_id}/cancel", headers=bearer)
        assert cancelled_response.status_code == 200
        assert cancelled_response.json()["status"] == "cancelled"
        assert client.get(f"/v1/runs/{cancellable.run_id}/report", headers=bearer).status_code == 404


def test_mcp_all_runtime_tools(runtime_config) -> None:
    async def verify() -> None:
        from mcp import Client

        server = build_server(runtime_config)
        async with Client(server, raise_exceptions=True) as client:
            run_result = await client.call_tool(
                "epspec_run",
                {
                    "request": "Run EPSpec on corn and compare PLSR",
                    "offline": True,
                    "simulate": True,
                    "auto_approve": True,
                },
            )
            assert run_result.is_error is False
            assert run_result.structured_content is not None
            run_id = run_result.structured_content["run_id"]
            status_result = await client.call_tool("epspec_status", {"run_id": run_id})
            assert status_result.structured_content["status"] == "completed"
            assert (await client.call_tool("epspec_report", {"run_id": run_id})).is_error is False
            assert (await client.call_tool("epspec_artifacts", {"run_id": run_id})).is_error is False
            events_result = await client.call_tool("epspec_events", {"run_id": run_id, "offset": 0})
            assert events_result.structured_content["events"]
            plan_result = await client.call_tool("epspec_plan", {"request": "Run PLSR on soil", "offline": True})
            assert plan_result.structured_content["status"] == "completed"

            waiting_config = runtime_config.with_overrides(auto_approve=False)
            waiting_runner = RuntimeRunner(waiting_config)
            waiting_created = waiting_runner.create_run("full", "Run EPSpec on tecator")
            waiting = waiting_runner.execute(waiting_created.run_id)
            assert waiting.status == RunStatus.awaiting_approval
            resumed = await client.call_tool("epspec_resume", {"run_id": waiting.run_id, "response": "approve"})
            assert resumed.structured_content["status"] == "completed"

            cancellable = RuntimeRunner(runtime_config).create_run("full", "Run EPSpec on shootout")
            cancelled = await client.call_tool("epspec_cancel", {"run_id": cancellable.run_id})
            assert cancelled.structured_content["status"] == "cancelled"

    asyncio.run(verify())
