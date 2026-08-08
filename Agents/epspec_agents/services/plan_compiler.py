from pathlib import Path

from ..exceptions import PlanCompilationError
from ..schemas import (
    ComparisonExecutionStep,
    ExperimentIntent,
    ExperimentPlan,
    ModelExecutionStep,
    PreprocessStep,
    ReportStep,
)


class PlanCompiler:
    def __init__(self, project_root: Path, agents_dir: Path):
        self.project_root = project_root.resolve()
        self.agents_dir = agents_dir.resolve()
        self.tests_root = self.project_root / "Experiments" / "Tests"

    def model_location(self, dataset: str, model_id: str) -> tuple[str, Path]:
        if model_id == "plsr":
            return "baseline_regression", self.tests_root / "Baseline" / dataset / f"{model_id}_cv_results"
        if model_id in {"ipls_plsr", "cars_plsr"}:
            return "ipls_cars_regression", self.tests_root / "ipls and cars" / dataset / f"{model_id}_cv_results"
        if model_id == "EPSpec_plsr":
            return "wavelength_selection_regression", self.tests_root / "wavelength selection" / dataset / "Epspec"
        if model_id == "EPSpec_plsr_sliding":
            return "wavelength_selection_regression", self.tests_root / "wavelength selection" / dataset / "Epspec_sliding"
        raise PlanCompilationError(f"无法解析模型路径: {model_id}")

    def compile(self, intent: ExperimentIntent) -> ExperimentPlan:
        raw_input = self.project_root / "Data" / "Raw Data" / f"{intent.dataset_name}.csv"
        if intent.preprocess.enabled:
            processed = self.tests_root / "Preprocessed Data" / f"{intent.dataset_name}_{intent.preprocess.method}_preprocessed.csv"
            preprocess = PreprocessStep(enabled=True, method=intent.preprocess.method, input_path=raw_input, output_path=processed)
            model_input = processed
        else:
            preprocess = PreprocessStep(enabled=False, method=None, input_path=raw_input, output_path=raw_input)
            model_input = raw_input
        main_family, main_dir = self.model_location(intent.dataset_name, intent.model)
        main = ModelExecutionStep(
            method=intent.model,
            family=main_family,
            input_path=model_input,
            out_dir=main_dir,
        )
        comparisons = []
        for model_id in intent.compare.models:
            family, out_dir = self.model_location(intent.dataset_name, model_id)
            comparisons.append(ModelExecutionStep(method=model_id, family=family, input_path=model_input, out_dir=out_dir))
        compare = ComparisonExecutionStep(enabled=bool(comparisons), models=comparisons)
        report = ReportStep(
            input_dir_main=main_dir,
            input_dirs_compare=[step.out_dir for step in comparisons],
            output_path=self.agents_dir,
        )
        return ExperimentPlan(
            dataset_name=intent.dataset_name,
            step_preprocess=preprocess,
            step_model_main=main,
            step_model_compare=compare,
            step_report=report,
        )
