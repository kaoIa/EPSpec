from typing import Any

from .config import ExecutionMode, RuntimeConfig
from .runtime.runner import RuntimeRunner
from .tools.registry import ScientificToolRegistry


def build_server(config: RuntimeConfig | None = None):
    try:
        from mcp.server import MCPServer
    except ImportError as exc:
        raise RuntimeError("缺少 MCP Python SDK v2") from exc
    runtime = config or RuntimeConfig.from_env()
    mcp = MCPServer(
        "epspec-agents",
        title="EPSpec Agents",
        description="Stateful evidence-guided near-infrared spectral experiment runtime.",
        instructions="Inspect capabilities, create a plan or run, then use status and resume until the run reaches a terminal state.",
        version="1.0.0",
    )

    @mcp.tool(
        title="EPSpec capabilities",
        description="List supported datasets, preprocessing methods, models, and scientific tool availability.",
        structured_output=True,
    )
    def epspec_capabilities() -> dict[str, Any]:
        return {
            "datasets": ["shootout", "corn", "soil", "tecator"],
            "preprocessors": ["savitzky_golay", "snv"],
            "models": ["plsr", "ipls_plsr", "cars_plsr", "EPSpec_plsr", "EPSpec_plsr_sliding"],
            "tools": ScientificToolRegistry(runtime.project_root).capabilities(),
        }

    @mcp.tool(
        title="Run an EPSpec experiment",
        description="Create and execute a stateful experiment; the result may request clarification or approval.",
        structured_output=True,
    )
    def epspec_run(
        request: str,
        offline: bool | None = None,
        simulate: bool | None = None,
        auto_approve: bool | None = None,
    ) -> dict[str, Any]:
        execution_mode: ExecutionMode | None = None if simulate is None else "simulate" if simulate else "native"
        effective = runtime.with_overrides(offline=offline, execution_mode=execution_mode, auto_approve=auto_approve)
        runner = RuntimeRunner(effective)
        snapshot = runner.create_run("full", request)
        return runner.execute(snapshot.run_id).model_dump(mode="json")

    @mcp.tool(
        title="Plan an EPSpec experiment",
        description="Compile and validate a run-scoped experiment plan without executing scientific algorithms.",
        structured_output=True,
    )
    def epspec_plan(request: str, offline: bool | None = None) -> dict[str, Any]:
        effective = runtime.with_overrides(offline=offline)
        runner = RuntimeRunner(effective)
        snapshot = runner.create_run("planning", request)
        return runner.execute(snapshot.run_id).model_dump(mode="json")

    @mcp.tool(
        title="Inspect an EPSpec run",
        description="Return status, current stage, pending interruption, metadata, and the latest structured state.",
        structured_output=True,
    )
    def epspec_status(run_id: str) -> dict[str, Any]:
        return RuntimeRunner(runtime).get(run_id).model_dump(mode="json")

    @mcp.tool(
        title="Resume an EPSpec run",
        description="Answer a clarification or human approval interruption for a durable run.",
        structured_output=True,
    )
    def epspec_resume(run_id: str, response: str) -> dict[str, Any]:
        return RuntimeRunner(runtime).resume(run_id, response).model_dump(mode="json")

    @mcp.tool(
        title="Cancel an EPSpec run",
        description="Request cooperative cancellation and return the updated lifecycle snapshot.",
        structured_output=True,
    )
    def epspec_cancel(run_id: str) -> dict[str, Any]:
        return RuntimeRunner(runtime).cancel(run_id).model_dump(mode="json")

    @mcp.tool(
        title="Read an EPSpec report",
        description="Return the grounded Markdown scientific report for a completed run.",
    )
    def epspec_report(run_id: str) -> str:
        return RuntimeRunner(runtime).report(run_id)

    @mcp.tool(
        title="List EPSpec artifacts",
        description="Return the run-scoped artifact inventory with paths, roles, sizes, media types, and SHA-256 hashes.",
        structured_output=True,
    )
    def epspec_artifacts(run_id: str) -> list[dict[str, Any]]:
        return RuntimeRunner(runtime).artifacts(run_id)

    @mcp.tool(
        title="Read EPSpec events",
        description="Read redacted structured events from a byte offset and return the next offset.",
        structured_output=True,
    )
    def epspec_events(run_id: str, offset: int = 0) -> dict[str, Any]:
        events, next_offset = RuntimeRunner(runtime).events(run_id, offset)
        return {"events": events, "next_offset": next_offset}

    return mcp


def serve(
    transport: str = "stdio",
    config: RuntimeConfig | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    server = build_server(config)
    if transport == "streamable-http":
        server.run(transport="streamable-http", host=host, port=port, stateless_http=True, json_response=True)
    else:
        server.run(transport="stdio")
