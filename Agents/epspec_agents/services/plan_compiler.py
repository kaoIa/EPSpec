from pathlib import Path

from ..exceptions import PlanCompilationError
from ..schemas import ComparisonExecutionStep, ExperimentIntent, ExperimentPlan, ModelExecutionStep, ModelFamily, PreprocessStep, ReportStep
from .artifact_store import ArtifactStore


class PlanCompiler:
    def __init__(self, project_root: Path, agents_dir: Path):
        self.project_root = project_root.resolve()
        self.agents_dir = agents_dir.resolve()

    def model_family(self, model_id: str) -> ModelFamily:
        if model_id == "plsr":
            return "baseline_regression"
        if model_id in {"ipls_plsr", "cars_plsr"}:
            return "ipls_cars_regression"
        if model_id in {"EPSpec_plsr", "EPSpec_plsr_sliding"}:
            return "wavelength_selection_regression"
        raise PlanCompilationError(f"无法解析模型: {model_id}")

    def compile(self, intent: ExperimentIntent, run_id: str) -> ExperimentPlan:
        store = ArtifactStore(self.agents_dir, run_id)
        raw_input = (self.project_root / "Data" / "Raw Data" / f"{intent.dataset_name}.csv").resolve()
        if intent.preprocess.enabled:
            processed = store.run_dir / "work" / "input" / f"{intent.dataset_name}_{intent.preprocess.method}.csv"
            preprocess = PreprocessStep(enabled=True, method=intent.preprocess.method, input_path=raw_input, output_path=processed)
            model_input = processed
        else:
            preprocess = PreprocessStep(enabled=False, method=None, input_path=raw_input, output_path=raw_input)
            model_input = raw_input
        main = ModelExecutionStep(
            method=intent.model,
            family=self.model_family(intent.model),
            input_path=model_input,
            out_dir=store.run_dir / "results" / "primary" / intent.model,
        )
        comparisons = [
            ModelExecutionStep(
                method=model_id,
                family=self.model_family(model_id),
                input_path=model_input,
                out_dir=store.run_dir / "results" / "comparisons" / f"{index:02d}_{model_id}",
            )
            for index, model_id in enumerate(intent.compare.models, start=1)
        ]
        return ExperimentPlan(
            run_id=run_id,
            dataset_name=intent.dataset_name,
            step_preprocess=preprocess,
            step_model_main=main,
            step_model_compare=ComparisonExecutionStep(enabled=bool(comparisons), models=comparisons),
            step_report=ReportStep(
                input_dir_main=main.out_dir,
                input_dirs_compare=[step.out_dir for step in comparisons],
                output_path=store.run_dir / "report",
            ),
        )
