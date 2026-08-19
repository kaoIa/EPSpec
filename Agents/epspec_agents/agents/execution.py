from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ..config import RuntimeConfig
from ..execution.simulator import ScientificSimulator
from ..execution.subprocess_runner import SubprocessToolRunner
from ..runtime.repository import RunRepository
from ..schemas import ExperimentPlan, ModelExecutionStep
from ..services.tracing import LocalTracer
from ..tools.registry import ScientificToolRegistry


class ScientificExecutionAgent:
    name = "Scientific Execution Agent"

    def __init__(
        self,
        config: RuntimeConfig,
        registry: ScientificToolRegistry,
        repository: RunRepository,
        tracer: LocalTracer,
        run_id: str,
    ):
        self.config = config
        self.registry = registry
        self.repository = repository
        self.tracer = tracer
        self.run_id = run_id
        self.runner = SubprocessToolRunner(config, registry, repository, run_id, tracer)
        self.simulator = ScientificSimulator()

    def preprocess(self, plan: ExperimentPlan) -> dict[str, Any]:
        step = plan.step_preprocess
        if not step.enabled:
            return {"status": "skipped", "method": None, "output_path": str(step.output_path), "simulated": self.config.execution_mode == "simulate"}
        self.tracer.emit("preprocessing", "tool_started", tool=step.method)
        if self.config.execution_mode == "simulate":
            result = self.simulator.preprocess(str(step.method), step.input_path, step.output_path)
        else:
            result = self.runner.run(str(step.method), "preprocessing", step.input_path, step.output_path, "preprocessing")
        self.tracer.emit("preprocessing", "tool_completed", tool=step.method, artifact=str(step.output_path))
        return result

    def run_model(self, step: ModelExecutionStep, role: str, dataset: str) -> dict[str, Any]:
        self.tracer.emit(role, "tool_started", tool=step.method)
        if self.config.execution_mode == "simulate":
            result = self.simulator.model(step.method, dataset, step.input_path, step.out_dir)
        else:
            result = self.runner.run(step.method, "modeling", step.input_path, step.out_dir, role)
        self.tracer.emit(role, "tool_completed", tool=step.method, artifact=str(step.out_dir))
        return {**result, "role": role, "method": step.method, "out_dir": str(step.out_dir)}

    def run_comparisons(self, plan: ExperimentPlan) -> list[dict[str, Any]]:
        steps = list(plan.step_model_compare.models)
        if not steps:
            return []
        workers = min(self.config.max_concurrency, len(steps))
        if workers == 1:
            return [self.run_model(step, f"comparison_model_{index}", plan.dataset_name) for index, step in enumerate(steps, start=1)]
        results: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="epspec-comparison") as executor:
            futures = {executor.submit(self.run_model, step, f"comparison_model_{index}", plan.dataset_name): index for index, step in enumerate(steps, start=1)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [results[index] for index in sorted(results)]
