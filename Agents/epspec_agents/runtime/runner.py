from __future__ import annotations

import builtins
import json
import traceback
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..config import RuntimeConfig
from ..exceptions import RunStateError
from ..graph.workflow import build_workflow
from ..guardrails.execution import validate_interpretation_plan
from ..schemas import ComparisonConfig, ExperimentIntent, ExperimentPlan, PreprocessConfig, RunSnapshot, RunStatus
from ..services.artifact_store import ArtifactStore
from ..services.model_factory import ModelFactory, StructuredModelAdapter
from ..services.plan_compiler import PlanCompiler
from ..services.tracing import LocalTracer
from ..tools.registry import ScientificToolRegistry
from .checkpoint import CheckpointManager
from .repository import RunRepository


class RuntimeRunner:
    def __init__(self, config: RuntimeConfig | None = None, adapter: StructuredModelAdapter | None = None):
        self.config = config or RuntimeConfig.from_env()
        self.adapter = adapter
        self.repository = RunRepository(self.config.repository_path)

    def create_run(
        self,
        target_stage: str = "full",
        user_request: str = "",
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunSnapshot:
        if target_stage not in {"full", "planning", "execution", "interpretation"}:
            raise RunStateError(f"未知目标阶段: {target_stage}")
        identifier = run_id or uuid4().hex
        runtime_metadata = {
            "offline": self.config.offline,
            "execution_mode": self.config.execution_mode,
            "auto_approve": self.config.auto_approve,
            **(metadata or {}),
        }
        store = ArtifactStore(self.config.agents_dir, identifier)
        snapshot = self.repository.create(identifier, identifier, target_stage, user_request, runtime_metadata)
        artifact = store.write_request(user_request)
        LocalTracer(store.run_dir / "events.jsonl", identifier).emit("runtime", "run_created", target_stage=target_stage, request_artifact=str(artifact.path))
        return snapshot

    def execute(self, run_id: str, plan: ExperimentPlan | None = None) -> RunSnapshot:
        snapshot = self.repository.get(run_id)
        if snapshot.status not in {RunStatus.created, RunStatus.queued}:
            raise RunStateError(f"运行状态不允许启动: {snapshot.status.value}")
        config = self._config_for(snapshot)
        if plan is None:
            normalized_plan = None
        elif snapshot.target_stage == "interpretation":
            normalized_plan = self._prepare_interpretation_plan(plan, run_id, config)
        else:
            normalized_plan = self._rebase_plan(plan, run_id, config)
        initial: dict[str, Any] = {
            "run_id": run_id,
            "thread_id": snapshot.thread_id,
            "target_stage": snapshot.target_stage,
            "user_request": snapshot.user_request,
            "messages": [{"role": "user", "content": snapshot.user_request}] if snapshot.user_request else [],
            "plan": normalized_plan.to_legacy_dict() if normalized_plan else None,
            "artifacts": [],
            "errors": [],
            "comparison_results": [],
            "created_at": snapshot.created_at,
            "auto_approve": config.auto_approve,
            "offline": config.offline,
            "execution_mode": config.execution_mode,
        }
        self.repository.update(run_id, status=RunStatus.queued, current_stage="queued", clear_interruption=True)
        return self._invoke(config, snapshot.thread_id, initial)

    def resume(self, run_id: str, response: str) -> RunSnapshot:
        snapshot = self.repository.get(run_id)
        if snapshot.status not in {RunStatus.awaiting_clarification, RunStatus.awaiting_approval}:
            raise RunStateError(f"运行状态不允许恢复: {snapshot.status.value}")
        if snapshot.cancel_requested:
            return self.repository.update(run_id, status=RunStatus.cancelled, current_stage="cancellation", clear_interruption=True)
        try:
            from langgraph.types import Command
        except ImportError as exc:
            raise RunStateError("缺少 langgraph") from exc
        config = self._config_for(snapshot)
        self.repository.update(run_id, status=RunStatus.queued, current_stage="resume", clear_interruption=True)
        return self._invoke(config, snapshot.thread_id, Command(resume=response))

    def run(
        self,
        target_stage: str = "full",
        user_request: str = "",
        plan: ExperimentPlan | None = None,
    ) -> dict[str, Any]:
        snapshot = self.create_run(target_stage, user_request)
        completed = self.execute(snapshot.run_id, plan)
        return completed.model_dump(mode="json")

    def run_interactive(
        self,
        target_stage: str = "full",
        user_request: str = "",
        plan: ExperimentPlan | None = None,
    ) -> RunSnapshot:
        snapshot = self.create_run(target_stage, user_request)
        snapshot = self.execute(snapshot.run_id, plan)
        while snapshot.interruption is not None:
            print(json.dumps(snapshot.interruption, ensure_ascii=False, indent=2))
            response = input("输入：").strip()
            snapshot = self.resume(snapshot.run_id, response)
        return snapshot

    def get(self, run_id: str) -> RunSnapshot:
        return self.repository.get(run_id)

    def list(self, limit: int = 50, status: str | None = None) -> builtins.list[RunSnapshot]:
        return self.repository.list(limit, status)

    def cancel(self, run_id: str) -> RunSnapshot:
        snapshot = self.repository.request_cancel(run_id)
        if snapshot.status in {RunStatus.created, RunStatus.queued, RunStatus.awaiting_clarification, RunStatus.awaiting_approval}:
            return self.repository.update(run_id, status=RunStatus.cancelled, current_stage="cancellation", clear_interruption=True)
        return snapshot

    def events(self, run_id: str, offset: int = 0) -> tuple[builtins.list[dict[str, Any]], int]:
        self.repository.get(run_id)
        store = ArtifactStore(self.config.agents_dir, run_id)
        return LocalTracer(store.run_dir / "events.jsonl", run_id).read(offset)

    def artifacts(self, run_id: str) -> builtins.list[dict[str, Any]]:
        self.repository.get(run_id)
        return [item.model_dump(mode="json") for item in ArtifactStore(self.config.agents_dir, run_id).collect()]

    def report(self, run_id: str) -> str:
        self.repository.get(run_id)
        path = ArtifactStore(self.config.agents_dir, run_id).run_dir / "report" / "summary.md"
        if not path.is_file():
            raise FileNotFoundError(f"报告不存在: {run_id}")
        return path.read_text(encoding="utf-8")

    def load_plan(self, path: Path | None = None) -> ExperimentPlan:
        if path is not None:
            return ExperimentPlan.from_legacy_dict(json.loads(path.read_text(encoding="utf-8")))
        snapshot = self.repository.latest_plan_run()
        if snapshot and snapshot.result and snapshot.result.get("plan"):
            return ExperimentPlan.from_legacy_dict(snapshot.result["plan"])
        candidates = [
            self.config.agents_dir / "plan.json",
            self.config.agents_dir / "examples" / "legacy_corn_run" / "plan.json",
        ]
        selected = next((candidate for candidate in candidates if candidate.is_file()), None)
        if selected is None:
            raise FileNotFoundError("未找到可执行计划")
        return ExperimentPlan.from_legacy_dict(json.loads(selected.read_text(encoding="utf-8")))

    def _invoke(self, config: RuntimeConfig, thread_id: str, command: Any) -> RunSnapshot:
        manager = CheckpointManager(config.checkpoint_path)
        checkpointer = manager.open()
        try:
            factory = ModelFactory(config)
            planner = self.adapter or factory.create("planning")
            interpreter = self.adapter or factory.create("interpretation")
            graph = build_workflow(config, planner, interpreter, checkpointer, self.repository)
            invocation_config = {"configurable": {"thread_id": thread_id}}
            result = graph.invoke(command, invocation_config)
            return self._record_invocation(thread_id, result)
        except Exception as exc:
            trace = traceback.format_exc()
            payload = {
                "status": "failed",
                "current_stage": "runtime",
                "errors": [{"error_type": type(exc).__name__, "message": str(exc)}],
            }
            store = ArtifactStore(config.agents_dir, thread_id)
            store.write_json(
                "runtime_error.json",
                {"error_type": type(exc).__name__, "message": str(exc), "traceback": trace},
                "error",
            )
            LocalTracer(store.run_dir / "events.jsonl", thread_id).emit("runtime", "runtime_failed", error_type=type(exc).__name__, error=str(exc))
            self.repository.update(
                thread_id,
                status=RunStatus.failed,
                current_stage="runtime",
                result=payload,
                clear_interruption=True,
            )
            raise
        finally:
            manager.close()

    def _record_invocation(self, run_id: str, result: dict[str, Any]) -> RunSnapshot:
        interruption = self._interrupt_payload(result.get("__interrupt__"))
        payload = self._jsonable({key: value for key, value in result.items() if key != "__interrupt__"})
        if interruption is not None:
            kind = interruption.get("type") if isinstance(interruption, dict) else None
            status = RunStatus.awaiting_clarification if kind == "clarification" else RunStatus.awaiting_approval
            return self.repository.update(
                run_id,
                status=status,
                current_stage=kind or "interrupt",
                interruption=self._jsonable(interruption),
                result=payload,
            )
        status = RunStatus(payload.get("status", "failed"))
        return self.repository.update(
            run_id,
            status=status,
            current_stage=payload.get("current_stage", "unknown"),
            result=payload,
            clear_interruption=True,
        )

    def _interrupt_payload(self, value: Any) -> Any:
        if not value:
            return None
        item = value[0] if isinstance(value, (builtins.list, tuple)) else value
        return getattr(item, "value", item)

    def _jsonable(self, value: Any) -> Any:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))

    def _config_for(self, snapshot: RunSnapshot) -> RuntimeConfig:
        mode = snapshot.metadata.get("execution_mode")
        return self.config.with_overrides(
            offline=bool(snapshot.metadata.get("offline", self.config.offline)),
            execution_mode=mode if mode in {"native", "simulate"} else self.config.execution_mode,
            auto_approve=bool(snapshot.metadata.get("auto_approve", self.config.auto_approve)),
        )

    def _rebase_plan(self, plan: ExperimentPlan, run_id: str, config: RuntimeConfig) -> ExperimentPlan:
        intent = ExperimentIntent(
            dataset_name=plan.dataset_name,
            task_type=plan.task_type,
            preprocess=PreprocessConfig(enabled=plan.step_preprocess.enabled, method=plan.step_preprocess.method),
            model=plan.step_model_main.method,
            compare=ComparisonConfig(
                enabled=plan.step_model_compare.enabled,
                models=[step.method for step in plan.step_model_compare.models],
            ),
        )
        return PlanCompiler(config.project_root, config.agents_dir).compile(intent, run_id)

    def _prepare_interpretation_plan(self, plan: ExperimentPlan, run_id: str, config: RuntimeConfig) -> ExperimentPlan:
        validate_interpretation_plan(plan, ScientificToolRegistry(config.project_root), config.project_root, config.agents_dir)
        payload = plan.model_dump(mode="python")
        payload["run_id"] = run_id
        payload["step_report"]["output_path"] = ArtifactStore(config.agents_dir, run_id).run_dir / "report"
        return ExperimentPlan.model_validate(payload)
