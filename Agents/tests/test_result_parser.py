import json

from epspec_agents.schemas import ExperimentIntent
from epspec_agents.services.plan_compiler import PlanCompiler
from epspec_agents.services.result_parser import ResultParser


def test_result_parser_basic_contract(tmp_path):
    agents_dir = tmp_path / "Agents"
    plan = PlanCompiler(tmp_path, agents_dir).compile(ExperimentIntent.model_validate({"dataset_name": "corn", "model": "plsr"}))
    result_dir = plan.step_model_main.out_dir
    result_dir.mkdir(parents=True)
    (result_dir / "summary.json").write_text(json.dumps({"R2": {"mean": 0.91, "std": 0.02}, "RMSE": {"mean": 0.12, "std": 0.01}}), encoding="utf-8")
    (result_dir / "metrics_per_fold.csv").write_text("fold,R2,RMSE\n1,0.90,0.13\n2,0.92,0.11\n", encoding="utf-8")
    result = ResultParser().parse(plan)
    assert result.main_result.method == "plsr"
    assert result.main_result.metrics_summary["R2"]["mean"] == 0.91
    assert result.main_result.selection_details["selection_type"] == "full_spectrum"
    assert result.task_level_prior["predicted_substance"] == "starch"
