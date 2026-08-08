from pathlib import Path
from typing import Any

from ..schemas import ExperimentPlan, ModelExecutionStep
from ..services.tracing import LocalTracer
from ..tools.modeling import execute_model
from ..tools.preprocessing import execute_preprocessing
from ..tools.registry import ScientificToolRegistry


class ScientificExecutionAgent:
    name = "Scientific Execution Agent"

    def __init__(self, registry: ScientificToolRegistry, tracer: LocalTracer | None = None):
        self.registry = registry
        self.tracer = tracer

    def preprocess(self, plan: ExperimentPlan) -> dict[str, Any]:
        step = plan.step_preprocess
        if not step.enabled:
            return {"status": "skipped", "method": None, "output_path": str(step.output_path)}
        self._emit("preprocessing", "tool_started", tool=step.method)
        execute_preprocessing(self.registry, str(step.method), step.input_path, step.output_path)
        self._emit("preprocessing", "tool_completed", tool=step.method, artifact=str(step.output_path))
        return {"status": "completed", "method": step.method, "output_path": str(step.output_path)}

    def run_model(self, step: ModelExecutionStep, role: str) -> dict[str, Any]:
        self._emit(role, "tool_started", tool=step.method)
        execute_model(self.registry, step.method, step.input_path, step.out_dir)
        self._emit(role, "tool_completed", tool=step.method, artifact=str(step.out_dir))
        return {"status": "completed", "role": role, "method": step.method, "out_dir": str(step.out_dir)}

    def run_comparisons(self, plan: ExperimentPlan) -> list[dict[str, Any]]:
        results = []
        for index, step in enumerate(plan.step_model_compare.models, start=1):
            results.append(self.run_model(step, f"comparison_model_{index}"))
        return results

    def _emit(self, stage: str, event_type: str, **payload: Any) -> None:
        if self.tracer:
            self.tracer.emit(stage, event_type, **payload)
