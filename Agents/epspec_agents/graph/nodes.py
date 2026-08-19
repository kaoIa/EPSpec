import json
import traceback
from collections.abc import Callable, Mapping
from typing import Any
from uuid import uuid4

from ..agents.execution import ScientificExecutionAgent
from ..agents.interpretation import ScientificInterpretationAgent
from ..agents.planning import ExperimentPlanningAgent
from ..config import RuntimeConfig
from ..exceptions import RunCancelledError
from ..guardrails.execution import validate_execution_plan
from ..guardrails.interpretation import validate_experiment_result, validate_report
from ..runtime.repository import RunRepository
from ..schemas import ExecutionError, ExperimentIntent, ExperimentPlan, ExperimentResult, ScientificReport
from ..services.artifact_store import ArtifactStore
from ..services.model_factory import StructuredModelAdapter
from ..services.plan_compiler import PlanCompiler
from ..services.provenance import build_manifest, utc_now
from ..services.result_parser import ResultParser
from ..services.tracing import LocalTracer
from ..state import RuntimeState
from ..tools.registry import ScientificToolRegistry


class GuardedNode:
    def __init__(self, function: Callable[[RuntimeState], dict[str, Any]]):
        self.function = function

    def __call__(self, state: RuntimeState) -> dict[str, Any]:
        return self.function(state)


class WorkflowNodes:
    def __init__(
        self,
        config: RuntimeConfig,
        planner_adapter: StructuredModelAdapter,
        interpreter_adapter: StructuredModelAdapter,
        repository: RunRepository,
    ):
        self.config = config
        self.planner_adapter = planner_adapter
        self.interpreter_adapter = interpreter_adapter
        self.repository = repository
        self.compiler = PlanCompiler(config.project_root, config.agents_dir)
        self.registry = ScientificToolRegistry(config.project_root)
        self.parser = ResultParser()

    def initialize_run(self, state: RuntimeState) -> dict[str, Any]:
        now = utc_now()
        run_id = state.get("run_id") or uuid4().hex
        thread_id = state.get("thread_id") or run_id
        update = {
            "run_id": run_id,
            "thread_id": thread_id,
            "target_stage": state.get("target_stage", "full"),
            "messages": state.get("messages", []),
            "comparison_results": state.get("comparison_results", []),
            "artifacts": state.get("artifacts", []),
            "errors": state.get("errors", []),
            "status": "initialized",
            "current_stage": "initialize_run",
            "created_at": state.get("created_at", now),
            "updated_at": now,
            "max_concurrency": self.config.max_concurrency,
            "offline": self.config.offline,
            "execution_mode": self.config.execution_mode,
            "auto_approve": state.get("auto_approve", self.config.auto_approve),
        }
        self._tracer(update).emit("initialize_run", "run_initialized", thread_id=thread_id)
        return update

    def planning(self, state: RuntimeState) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        if state.get("user_request") and not messages:
            messages.append({"role": "user", "content": state["user_request"]})
        agent = ExperimentPlanningAgent(self.planner_adapter, self.config.prompts_dir / "planning.txt")
        output, raw = agent.plan(messages)
        self._tracer(state).emit("planning", "planning_completed", planning_status=output.status)
        update = {
            "messages": messages,
            "planning_output": output.model_dump(mode="json"),
            "planning_raw": raw,
            "status": "awaiting_clarification" if output.status == "needs_clarification" else "planning",
            "current_stage": "planning",
            "updated_at": utc_now(),
        }
        if output.intent is not None:
            update["intent"] = output.intent.model_dump(mode="json")
        return update

    def clarification(self, state: RuntimeState) -> dict[str, Any]:
        from langgraph.types import interrupt

        question = (state.get("planning_output") or {}).get("message", "请补充实验需求。")
        pending = {**state, "status": "awaiting_clarification", "current_stage": "clarification", "updated_at": utc_now()}
        self._sync(pending)
        self._tracer(state).emit("clarification", "clarification_requested", question=question)
        answer = str(interrupt({"type": "clarification", "message": question})).strip()
        messages = [*state.get("messages", []), {"role": "assistant", "content": question}, {"role": "user", "content": answer}]
        self._tracer(state).emit("clarification", "clarification_received")
        return {"messages": messages, "user_request": answer, "status": "planning", "current_stage": "clarification", "updated_at": utc_now()}

    def compile_plan(self, state: RuntimeState) -> dict[str, Any]:
        intent = ExperimentIntent.model_validate(state["intent"])
        plan = self.compiler.compile(intent, state["run_id"])
        artifact = self._store(state).write_plan(plan)
        self._tracer(state).emit("compile_plan", "plan_compiled", artifact=str(artifact.path))
        return {
            "plan": plan.to_legacy_dict(),
            "artifacts": [*state.get("artifacts", []), artifact.model_dump(mode="json")],
            "current_stage": "compile_plan",
            "updated_at": utc_now(),
        }

    def validate_plan(self, state: RuntimeState) -> dict[str, Any]:
        plan = ExperimentPlan.from_legacy_dict(state["plan"])
        if plan.run_id != state["run_id"]:
            raise ValueError("plan.run_id 与当前运行不一致")
        validate_execution_plan(plan, self.registry, self.config.project_root, self.config.agents_dir, require_input=True)
        self._tracer(state).emit("validate_plan", "plan_validated")
        return {"current_stage": "validate_plan", "updated_at": utc_now()}

    def plan_approval(self, state: RuntimeState) -> dict[str, Any]:
        from langgraph.types import interrupt

        if state.get("auto_approve") or self.config.auto_approve:
            text = "approve"
            self._tracer(state).emit("plan_approval", "approval_automated")
        else:
            payload = {
                "type": "approval",
                "message": "请审阅实验计划。回复 approve 继续，或输入修改意见。",
                "plan": state["plan"],
            }
            pending = {**state, "status": "awaiting_approval", "current_stage": "plan_approval", "updated_at": utc_now()}
            self._sync(pending)
            self._tracer(state).emit("plan_approval", "approval_requested")
            text = str(interrupt(payload)).strip()
        approved = text.lower() in {"确认", "confirm", "approve", "approved", "yes", "y"}
        self._tracer(state).emit("plan_approval", "approval_received", approved=approved)
        if approved:
            return {
                "approval_status": "approved",
                "status": "queued",
                "current_stage": "plan_approval",
                "updated_at": utc_now(),
            }
        messages = [*state.get("messages", []), {"role": "user", "content": text}]
        return {
            "approval_status": "rejected",
            "messages": messages,
            "user_request": text,
            "status": "planning",
            "current_stage": "plan_approval",
            "updated_at": utc_now(),
        }

    def preprocessing(self, state: RuntimeState) -> dict[str, Any]:
        plan = ExperimentPlan.from_legacy_dict(state["plan"])
        result = self._executor(state).preprocess(plan)
        return {"preprocess_result": result, "status": "executing", "current_stage": "preprocessing", "updated_at": utc_now()}

    def primary_model_execution(self, state: RuntimeState) -> dict[str, Any]:
        plan = ExperimentPlan.from_legacy_dict(state["plan"])
        result = self._executor(state).run_model(plan.step_model_main, "primary_model_execution", plan.dataset_name)
        return {"main_result": result, "status": "executing", "current_stage": "primary_model_execution", "updated_at": utc_now()}

    def comparison_model_execution(self, state: RuntimeState) -> dict[str, Any]:
        plan = ExperimentPlan.from_legacy_dict(state["plan"])
        results = self._executor(state).run_comparisons(plan)
        return {"comparison_results": results, "status": "executing", "current_stage": "comparison_model_execution", "updated_at": utc_now()}

    def aggregate_results(self, state: RuntimeState) -> dict[str, Any]:
        plan = ExperimentPlan.from_legacy_dict(state["plan"])
        result = self.parser.parse(plan, simulated=self.config.execution_mode == "simulate")
        artifact = self._store(state).write_json("experiment_result.json", result, "result")
        self._tracer(state).emit("aggregate_results", "aggregation_completed", artifact=str(artifact.path))
        return {
            "experiment_result": result.model_dump(mode="json"),
            "artifacts": [*state.get("artifacts", []), artifact.model_dump(mode="json")],
            "current_stage": "aggregate_results",
            "updated_at": utc_now(),
        }

    def validate_results(self, state: RuntimeState) -> dict[str, Any]:
        result = validate_experiment_result(self._result(state))
        self._tracer(state).emit("validate_results", "results_validated")
        return {"experiment_result": result.model_dump(mode="json"), "current_stage": "validate_results", "updated_at": utc_now()}

    def interpretation(self, state: RuntimeState) -> dict[str, Any]:
        result = self._result(state)
        self._tracer(state).emit("interpretation", "interpretation_started")
        agent = ScientificInterpretationAgent(self.interpreter_adapter, self.config.prompts_dir / "interpretation.txt")
        report, raw, prompt = agent.interpret(result)
        try:
            raw_parsed = json.loads(raw)
        except Exception:
            raw_parsed = None
        response = {
            "run_id": state["run_id"],
            "model": self.config.interpreter.model if not self.config.offline else "offline-grounded-interpreter",
            "report": report.markdown,
            "structured_model_output": raw_parsed,
        }
        artifacts = self._store(state).write_report_artifacts(response, report.markdown, prompt if self.config.capture_prompts else None)
        self._tracer(state).emit("interpretation", "interpretation_completed", artifact=str(self._store(state).run_dir / "report" / "summary.md"))
        return {
            "report": report.model_dump(mode="json"),
            "artifacts": [*state.get("artifacts", []), *[item.model_dump(mode="json") for item in artifacts]],
            "status": "interpreting",
            "current_stage": "interpretation",
            "updated_at": utc_now(),
        }

    def validate_report(self, state: RuntimeState) -> dict[str, Any]:
        report = validate_report(ScientificReport.model_validate(state["report"]), self._result(state))
        self._tracer(state).emit("validate_report", "report_validated")
        return {"report": report.model_dump(mode="json"), "current_stage": "validate_report", "updated_at": utc_now()}

    def finalization(self, state: RuntimeState) -> dict[str, Any]:
        completed = {**state, "status": "completed", "current_stage": "finalization", "updated_at": utc_now()}
        store = self._store(state)
        self._tracer(state).emit("finalization", "run_completed", manifest=str(store.run_dir / "manifest.json"))
        manifest = build_manifest(completed, self.config, self.registry, store)
        store.write_json("manifest.json", manifest, "manifest")
        artifacts = [item.model_dump(mode="json") for item in store.collect()]
        return {"status": "completed", "current_stage": "finalization", "updated_at": utc_now(), "artifacts": artifacts}

    def failure(self, state: RuntimeState) -> dict[str, Any]:
        failed = {**state, "status": "failed", "current_stage": "failure", "updated_at": utc_now()}
        store = self._store(state)
        self._tracer(state).emit("failure", "run_failed", errors=state.get("errors", []))
        manifest = build_manifest(failed, self.config, self.registry, store)
        store.write_json("manifest.json", manifest, "manifest")
        return {"status": "failed", "current_stage": "failure", "updated_at": utc_now(), "artifacts": [item.model_dump(mode="json") for item in store.collect()]}

    def cancellation(self, state: RuntimeState) -> dict[str, Any]:
        cancelled = {**state, "status": "cancelled", "current_stage": "cancellation", "updated_at": utc_now()}
        store = self._store(state)
        self._tracer(state).emit("cancellation", "run_cancelled")
        manifest = build_manifest(cancelled, self.config, self.registry, store)
        store.write_json("manifest.json", manifest, "manifest")
        return {"status": "cancelled", "current_stage": "cancellation", "updated_at": utc_now(), "artifacts": [item.model_dump(mode="json") for item in store.collect()]}

    def guarded(self, stage: str, function: Callable[[RuntimeState], dict[str, Any]]) -> GuardedNode:
        def invoke(state: RuntimeState) -> dict[str, Any]:
            try:
                if self.repository.is_cancel_requested(state["run_id"]):
                    raise RunCancelledError(f"运行已取消: {state['run_id']}")
                self._tracer(state).emit(stage, "stage_started")
                update = function(state)
                merged = {**state, **update}
                self._sync(merged)
                if stage != "finalization":
                    self._tracer(state).emit(stage, "stage_completed", status=merged.get("status"))
                return update
            except RunCancelledError:
                update = {"status": "cancelled", "current_stage": stage, "updated_at": utc_now()}
                self._sync({**state, **update})
                return update
            except Exception as exc:
                error = ExecutionError(stage=stage, error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
                update = {
                    "status": "failed",
                    "current_stage": stage,
                    "updated_at": utc_now(),
                    "errors": [*state.get("errors", []), error.model_dump(mode="json")],
                }
                self._tracer(state).emit(stage, "stage_failed", error_type=type(exc).__name__, error=str(exc))
                self._sync({**state, **update})
                return update

        return GuardedNode(invoke)

    def _store(self, state: Mapping[str, Any]) -> ArtifactStore:
        return ArtifactStore(self.config.agents_dir, state["run_id"])

    def _tracer(self, state: Mapping[str, Any]) -> LocalTracer:
        store = self._store(state)
        return LocalTracer(store.run_dir / "events.jsonl", state["run_id"])

    def _executor(self, state: Mapping[str, Any]) -> ScientificExecutionAgent:
        return ScientificExecutionAgent(self.config, self.registry, self.repository, self._tracer(state), state["run_id"])

    def _result(self, state: Mapping[str, Any]) -> ExperimentResult:
        return ExperimentResult.model_validate(state["experiment_result"])

    def _sync(self, state: Mapping[str, Any]) -> None:
        payload = json.loads(json.dumps({key: value for key, value in state.items() if key != "__interrupt__"}, ensure_ascii=False, default=str))
        self.repository.update(
            state["run_id"],
            status=state.get("status", "initialized"),
            current_stage=state.get("current_stage", "unknown"),
            result=payload,
        )
