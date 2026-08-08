from datetime import datetime, timezone
import json
import traceback
from typing import Any, Callable
from uuid import uuid4

from ..agents.execution import ScientificExecutionAgent
from ..agents.interpretation import ScientificInterpretationAgent
from ..agents.planning import ExperimentPlanningAgent
from ..config import RuntimeConfig
from ..guardrails.execution import validate_execution_plan
from ..guardrails.interpretation import validate_experiment_result, validate_report
from ..schemas import ExecutionError, ExperimentIntent, ExperimentPlan, ScientificReport
from ..services.artifact_store import ArtifactStore
from ..services.model_factory import StructuredModelAdapter
from ..services.plan_compiler import PlanCompiler
from ..services.provenance import build_manifest, utc_now
from ..services.result_parser import ResultParser
from ..services.tracing import LocalTracer
from ..tools.registry import ScientificToolRegistry


class WorkflowNodes:
    def __init__(self, config: RuntimeConfig, adapter: StructuredModelAdapter):
        self.config = config
        self.adapter = adapter
        self.compiler = PlanCompiler(config.project_root, config.agents_dir)
        self.registry = ScientificToolRegistry(config.project_root)
        self.parser = ResultParser()

    def initialize_run(self, state: dict[str, Any]) -> dict[str, Any]:
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
        }
        self._tracer(update).emit("initialize_run", "run_initialized", thread_id=thread_id)
        return update

    def planning(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        if state.get("user_request") and not messages:
            messages.append({"role": "user", "content": state["user_request"]})
        agent = ExperimentPlanningAgent(self.adapter, self.config.prompts_dir / "planning.txt")
        output, raw = agent.plan(messages)
        self._tracer(state).emit("planning", "planning_completed", planning_status=output.status)
        return {
            "messages": messages,
            "planning_output": output.model_dump(mode="json"),
            "intent": output.intent.model_dump(mode="json") if output.intent else None,
            "status": "awaiting_clarification" if output.status == "needs_clarification" else "planning",
            "current_stage": "planning",
            "updated_at": utc_now(),
            "planning_raw": raw,
        }

    def clarification(self, state: dict[str, Any]) -> dict[str, Any]:
        from langgraph.types import interrupt
        question = (state.get("planning_output") or {}).get("message", "请补充实验需求。")
        self._tracer(state).emit("clarification", "clarification_requested", question=question)
        answer = str(interrupt({"type": "clarification", "message": question})).strip()
        messages = [*state.get("messages", []), {"role": "assistant", "content": question}, {"role": "user", "content": answer}]
        return {"messages": messages, "user_request": answer, "status": "planning", "current_stage": "clarification", "updated_at": utc_now()}

    def compile_plan(self, state: dict[str, Any]) -> dict[str, Any]:
        intent = ExperimentIntent.model_validate(state["intent"])
        plan = self.compiler.compile(intent)
        store = self._store(state)
        artifact = store.write_run_plan(plan)
        artifacts = [*state.get("artifacts", []), artifact.model_dump(mode="json")]
        self._tracer(state).emit("compile_plan", "plan_compiled", artifact=str(artifact.path))
        return {"plan": plan.to_legacy_dict(), "artifacts": artifacts, "current_stage": "compile_plan", "updated_at": utc_now()}

    def validate_plan(self, state: dict[str, Any]) -> dict[str, Any]:
        plan = ExperimentPlan.from_legacy_dict(state["plan"])
        validate_execution_plan(plan, self.registry, self.config.project_root, require_input=True)
        self._tracer(state).emit("validate_plan", "plan_validated")
        return {"current_stage": "validate_plan", "updated_at": utc_now()}

    def plan_approval(self, state: dict[str, Any]) -> dict[str, Any]:
        from langgraph.types import interrupt
        self._tracer(state).emit("plan_approval", "approval_requested")
        response = interrupt({"type": "approval", "message": "请确认实验计划，输入 确认/confirm/approve 继续，或输入修改意见。", "plan": state["plan"]})
        text = str(response).strip()
        approved = text.lower() in {"确认", "confirm", "approve", "approved", "yes", "y"}
        self._tracer(state).emit("plan_approval", "approval_received", approved=approved)
        if approved:
            plan = ExperimentPlan.from_legacy_dict(state["plan"])
            artifact = self._store(state).write_legacy_plan(plan)
            return {
                "approval_status": "approved",
                "status": "executing",
                "current_stage": "plan_approval",
                "updated_at": utc_now(),
                "artifacts": [*state.get("artifacts", []), artifact.model_dump(mode="json")],
            }
        messages = [*state.get("messages", []), {"role": "user", "content": text}]
        return {"approval_status": "rejected", "messages": messages, "user_request": text, "status": "planning", "current_stage": "plan_approval", "updated_at": utc_now()}

    def preprocessing(self, state: dict[str, Any]) -> dict[str, Any]:
        plan = ExperimentPlan.from_legacy_dict(state["plan"])
        result = self._executor(state).preprocess(plan)
        return {"preprocess_result": result, "status": "executing", "current_stage": "preprocessing", "updated_at": utc_now()}

    def primary_model_execution(self, state: dict[str, Any]) -> dict[str, Any]:
        plan = ExperimentPlan.from_legacy_dict(state["plan"])
        result = self._executor(state).run_model(plan.step_model_main, "primary_model_execution")
        return {"main_result": result, "current_stage": "primary_model_execution", "updated_at": utc_now()}

    def comparison_model_execution(self, state: dict[str, Any]) -> dict[str, Any]:
        plan = ExperimentPlan.from_legacy_dict(state["plan"])
        results = self._executor(state).run_comparisons(plan)
        return {"comparison_results": results, "current_stage": "comparison_model_execution", "updated_at": utc_now()}

    def aggregate_results(self, state: dict[str, Any]) -> dict[str, Any]:
        plan = ExperimentPlan.from_legacy_dict(state["plan"])
        result = self.parser.parse(plan)
        artifact = self._store(state).write_json(self._store(state).run_dir / "experiment_result.json", result)
        self._tracer(state).emit("aggregate_results", "aggregation_completed", artifact=str(artifact.path))
        return {
            "experiment_result": result.model_dump(mode="json"),
            "artifacts": [*state.get("artifacts", []), artifact.model_dump(mode="json")],
            "current_stage": "aggregate_results",
            "updated_at": utc_now(),
        }

    def validate_results(self, state: dict[str, Any]) -> dict[str, Any]:
        result = validate_experiment_result(state_to_result(state))
        return {"experiment_result": result.model_dump(mode="json"), "current_stage": "validate_results", "updated_at": utc_now()}

    def interpretation(self, state: dict[str, Any]) -> dict[str, Any]:
        result = state_to_result(state)
        self._tracer(state).emit("interpretation", "interpretation_started")
        agent = ScientificInterpretationAgent(self.adapter, self.config.prompts_dir / "interpretation.txt")
        report, raw, send = agent.interpret(result)
        try:
            raw_parsed = json.loads(raw)
        except Exception:
            raw_parsed = None
        response = {
            "report_raw": report.markdown,
            "report_md": report.markdown,
            "raw_json_str": raw,
            "raw_json_parsed": raw_parsed,
            "response_lines": {
                "role": "assistant",
                "model": self.config.model_name,
                "content": report.markdown,
                "content_lines": report.markdown.splitlines(),
            },
        }
        artifacts = self._store(state).write_report_artifacts(send, response, report.markdown)
        self._tracer(state).emit("interpretation", "interpretation_completed", artifact=str(self.config.agents_dir / "summary_report.md"))
        return {
            "report": report.model_dump(mode="json"),
            "artifacts": [*state.get("artifacts", []), *[item.model_dump(mode="json") for item in artifacts]],
            "status": "interpreting",
            "current_stage": "interpretation",
            "updated_at": utc_now(),
        }

    def validate_report(self, state: dict[str, Any]) -> dict[str, Any]:
        report = validate_report(ScientificReport.model_validate(state["report"]), state_to_result(state))
        self._tracer(state).emit("validate_report", "report_validated")
        return {"report": report.model_dump(mode="json"), "current_stage": "validate_report", "updated_at": utc_now()}

    def finalization(self, state: dict[str, Any]) -> dict[str, Any]:
        completed = {**state, "status": "completed", "current_stage": "finalization", "updated_at": utc_now()}
        manifest = build_manifest(completed)
        artifact = self._store(state).write_json(self._store(state).run_dir / "manifest.json", manifest)
        self._tracer(state).emit("finalization", "run_completed", artifact=str(artifact.path))
        return {
            "status": "completed",
            "current_stage": "finalization",
            "updated_at": utc_now(),
            "artifacts": [*state.get("artifacts", []), artifact.model_dump(mode="json")],
        }

    def failure(self, state: dict[str, Any]) -> dict[str, Any]:
        failed = {**state, "status": "failed", "current_stage": "failure", "updated_at": utc_now()}
        manifest = build_manifest(failed)
        artifact = self._store(state).write_json(self._store(state).run_dir / "manifest.json", manifest)
        self._tracer(state).emit("failure", "run_failed", errors=state.get("errors", []))
        return {"status": "failed", "current_stage": "failure", "updated_at": utc_now(), "artifacts": [*state.get("artifacts", []), artifact.model_dump(mode="json")]}

    def guarded(self, stage: str, function: Callable[[dict[str, Any]], dict[str, Any]]) -> Callable[[dict[str, Any]], dict[str, Any]]:
        def invoke(state: dict[str, Any]) -> dict[str, Any]:
            try:
                return function(state)
            except Exception as exc:
                error = ExecutionError(stage=stage, error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
                if state.get("run_id"):
                    self._tracer(state).emit(stage, "tool_failed", error_type=type(exc).__name__, error=str(exc))
                return {
                    "status": "failed",
                    "current_stage": stage,
                    "updated_at": utc_now(),
                    "errors": [*state.get("errors", []), error.model_dump(mode="json")],
                }
        return invoke

    def _store(self, state: dict[str, Any]) -> ArtifactStore:
        return ArtifactStore(self.config.agents_dir, state["run_id"])

    def _tracer(self, state: dict[str, Any]) -> LocalTracer:
        run_id = state["run_id"]
        return LocalTracer(self.config.runs_dir / run_id / "trace.jsonl", run_id)

    def _executor(self, state: dict[str, Any]) -> ScientificExecutionAgent:
        return ScientificExecutionAgent(self.registry, self._tracer(state))


def state_to_result(state: dict[str, Any]):
    from ..schemas import ExperimentResult
    return ExperimentResult.model_validate(state["experiment_result"])
