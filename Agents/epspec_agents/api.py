import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import ExecutionMode, RuntimeConfig
from .exceptions import AgentRuntimeError, RunNotFoundError, RunStateError
from .runtime.runner import RuntimeRunner
from .schemas import ExperimentPlan, RunStatus
from .tools.registry import ScientificToolRegistry


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunCreateRequest(ApiModel):
    request: str = ""
    target_stage: Literal["full", "planning", "execution", "interpretation"] = "full"
    plan: dict[str, Any] | None = None
    offline: bool | None = None
    simulate: bool | None = None
    auto_approve: bool | None = None


class ResumeRequest(ApiModel):
    response: str = Field(min_length=1)


class RunManager:
    def __init__(self, max_workers: int):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="epspec-api")
        self.futures: dict[str, Future[Any]] = {}
        self.lock = threading.Lock()

    def submit(self, run_id: str, function: Any, *arguments: Any) -> None:
        future = self.executor.submit(function, *arguments)
        with self.lock:
            self.futures[run_id] = future
        future.add_done_callback(lambda _: self._discard(run_id))

    def _discard(self, run_id: str) -> None:
        with self.lock:
            self.futures.pop(run_id, None)

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)


def create_app(config: RuntimeConfig | None = None):
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
        from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError("缺少 fastapi") from exc
    runtime = config or RuntimeConfig.from_env()
    runner = RuntimeRunner(runtime)
    manager = RunManager(runtime.max_concurrency)

    @asynccontextmanager
    async def lifespan(app: Any):
        yield
        manager.close()

    app = FastAPI(
        title="EPSpec Multi-Agent Runtime API",
        version="1.0.0",
        description="Evidence-guided planning, deterministic spectral execution, and grounded scientific interpretation.",
        lifespan=lifespan,
    )

    @app.exception_handler(RunNotFoundError)
    async def handle_not_found(request: Any, exc: RunNotFoundError):
        return JSONResponse(status_code=404, content={"error": "run_not_found", "message": str(exc)})

    @app.exception_handler(FileNotFoundError)
    async def handle_file_not_found(request: Any, exc: FileNotFoundError):
        return JSONResponse(status_code=404, content={"error": "artifact_not_found", "message": str(exc)})

    @app.exception_handler(RunStateError)
    async def handle_run_state(request: Any, exc: RunStateError):
        return JSONResponse(status_code=409, content={"error": "invalid_run_state", "message": str(exc)})

    @app.exception_handler(AgentRuntimeError)
    async def handle_agent_error(request: Any, exc: AgentRuntimeError):
        return JSONResponse(status_code=422, content={"error": type(exc).__name__, "message": str(exc)})

    def authorize(authorization: str | None = Header(default=None), x_epspec_token: str | None = Header(default=None)) -> None:
        if not runtime.server_token:
            return
        bearer = authorization.removeprefix("Bearer ").strip() if authorization else ""
        if runtime.server_token not in {bearer, x_epspec_token or ""}:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": "epspec-agents", "version": "1.0.0"}

    @app.get("/v1/capabilities", dependencies=[Depends(authorize)])
    def capabilities() -> dict[str, Any]:
        return {
            "datasets": ["shootout", "corn", "soil", "tecator"],
            "preprocessors": ["savitzky_golay", "snv"],
            "models": ["plsr", "ipls_plsr", "cars_plsr", "EPSpec_plsr", "EPSpec_plsr_sliding"],
            "tools": ScientificToolRegistry(runtime.project_root).capabilities(),
            "stages": ["full", "planning", "execution", "interpretation"],
        }

    @app.post("/v1/runs", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(authorize)])
    def create_run(payload: RunCreateRequest) -> dict[str, Any]:
        execution_mode: ExecutionMode | None = None if payload.simulate is None else "simulate" if payload.simulate else "native"
        effective = runtime.with_overrides(
            offline=payload.offline,
            execution_mode=execution_mode,
            auto_approve=payload.auto_approve,
        )
        local_runner = RuntimeRunner(effective)
        plan = ExperimentPlan.from_legacy_dict(payload.plan) if payload.plan else None
        snapshot = local_runner.create_run(payload.target_stage, payload.request)
        manager.submit(snapshot.run_id, local_runner.execute, snapshot.run_id, plan)
        return snapshot.model_dump(mode="json")

    @app.get("/v1/runs", dependencies=[Depends(authorize)])
    def list_runs(limit: int = Query(default=50, ge=1, le=500), run_status: str | None = Query(default=None, alias="status")) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in runner.list(limit, run_status)]

    @app.get("/v1/runs/{run_id}", dependencies=[Depends(authorize)])
    def get_run(run_id: str) -> dict[str, Any]:
        return runner.get(run_id).model_dump(mode="json")

    @app.post("/v1/runs/{run_id}/resume", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(authorize)])
    def resume_run(run_id: str, payload: ResumeRequest) -> dict[str, Any]:
        snapshot = runner.get(run_id)
        manager.submit(run_id, runner.resume, run_id, payload.response)
        return snapshot.model_dump(mode="json")

    @app.post("/v1/runs/{run_id}/cancel", dependencies=[Depends(authorize)])
    def cancel_run(run_id: str) -> dict[str, Any]:
        return runner.cancel(run_id).model_dump(mode="json")

    @app.get("/v1/runs/{run_id}/artifacts", dependencies=[Depends(authorize)])
    def list_artifacts(run_id: str) -> list[dict[str, Any]]:
        return runner.artifacts(run_id)

    @app.get("/v1/runs/{run_id}/report", response_class=PlainTextResponse, dependencies=[Depends(authorize)])
    def get_report(run_id: str) -> str:
        return runner.report(run_id)

    @app.get("/v1/runs/{run_id}/events", dependencies=[Depends(authorize)])
    def stream_events(run_id: str, offset: int = Query(default=0, ge=0)):
        runner.get(run_id)

        def generate():
            position = offset
            while True:
                events, position = runner.events(run_id, position)
                for event in events:
                    yield "event: epspec\ndata: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                snapshot = runner.get(run_id)
                if snapshot.status in {RunStatus.completed, RunStatus.failed, RunStatus.cancelled} and not events:
                    yield "event: terminal\ndata: " + json.dumps({"status": snapshot.status.value}, ensure_ascii=False) + "\n\n"
                    break
                time.sleep(0.5)

        return StreamingResponse(generate(), media_type="text/event-stream")

    return app
