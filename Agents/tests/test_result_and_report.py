import pytest

from epspec_agents.exceptions import ReportValidationError
from epspec_agents.execution.simulator import ScientificSimulator
from epspec_agents.guardrails.interpretation import validate_report
from epspec_agents.schemas import ExperimentIntent, ScientificReport
from epspec_agents.services.plan_compiler import PlanCompiler
from epspec_agents.services.result_parser import ResultParser


def test_simulator_result_is_parseable(runtime_config) -> None:
    intent = ExperimentIntent(dataset_name="shootout", model="EPSpec_plsr")
    plan = PlanCompiler(runtime_config.project_root, runtime_config.agents_dir).compile(intent, "parse_run")
    ScientificSimulator().model(plan.step_model_main.method, plan.dataset_name, plan.step_model_main.input_path, plan.step_model_main.out_dir)
    result = ResultParser().parse(plan, simulated=True)
    assert result.run_id == "parse_run"
    assert result.simulated
    assert result.main_result.metrics_summary["R2"]["mean"]
    assert result.task_level_prior["predicted_substance"] == "active pharmaceutical ingredient"
    assert result.main_result.selection_details["selection_type"] == "epspec_topk_intervals"
    assert len(result.main_result.selection_details["per_fold_selected_topk_intervals"]) == 5
    assert result.main_result.selection_details["consensus_selected_intervals"][0]["selected_frequency"] == 5


def test_report_metric_grounding(runtime_config) -> None:
    intent = ExperimentIntent(dataset_name="corn", model="plsr")
    plan = PlanCompiler(runtime_config.project_root, runtime_config.agents_dir).compile(intent, "report_run")
    ScientificSimulator().model("plsr", "corn", plan.step_model_main.input_path, plan.step_model_main.out_dir)
    result = ResultParser().parse(plan, simulated=True)
    known = result.main_result.metrics_summary["R2"]["mean"]
    accepted = ScientificReport(markdown=f"# Simulated Result\nR2 {known:.6f}")
    assert validate_report(accepted, result) == accepted
    with pytest.raises(ReportValidationError):
        validate_report(ScientificReport(markdown="# Simulated Result\nR2 0.123456"), result)
