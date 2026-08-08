from pathlib import Path

from epspec_agents.schemas import ExperimentIntent, ExperimentPlan
from epspec_agents.services.plan_compiler import PlanCompiler


def test_plan_compiler_preserves_legacy_layout(tmp_path):
    agents_dir = tmp_path / "Agents"
    compiler = PlanCompiler(tmp_path, agents_dir)
    intent = ExperimentIntent.model_validate({
        "dataset_name": "corn",
        "task_type": "regression",
        "preprocess": {"enabled": True, "method": "savitzky_golay"},
        "model": "EPSpec_plsr",
        "compare": {"enabled": True, "models": ["plsr", "ipls_plsr", "cars_plsr", "EPSpec_plsr_sliding"]},
    })
    plan = compiler.compile(intent)
    legacy = plan.to_legacy_dict()
    assert set(legacy) == {"dataset_name", "task_type", "step_preprocess", "step_model_main", "step_model_compare", "step_report"}
    assert Path(legacy["step_preprocess"]["output_path"]).name == "corn_savitzky_golay_preprocessed.csv"
    assert legacy["step_model_main"]["family"] == "wavelength_selection_regression"
    assert Path(legacy["step_model_main"]["out_dir"]).name == "Epspec"
    assert [item["method"] for item in legacy["step_model_compare"]["models"]] == ["plsr", "ipls_plsr", "cars_plsr", "EPSpec_plsr_sliding"]
    assert [Path(item).name for item in legacy["step_report"]["input_dirs_compare"]] == ["plsr_cv_results", "ipls_plsr_cv_results", "cars_plsr_cv_results", "Epspec_sliding"]
    assert ExperimentPlan.from_legacy_dict(legacy) == plan


def test_no_preprocessing_uses_raw_input(tmp_path):
    plan = PlanCompiler(tmp_path, tmp_path / "Agents").compile(ExperimentIntent.model_validate({
        "dataset_name": "soil",
        "model": "plsr",
    }))
    assert not plan.step_preprocess.enabled
    assert plan.step_preprocess.input_path == plan.step_preprocess.output_path
    assert plan.step_model_main.input_path == plan.step_preprocess.input_path
