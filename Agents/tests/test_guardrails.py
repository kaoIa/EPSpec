import pytest

from epspec_agents.exceptions import ReportValidationError
from epspec_agents.guardrails.interpretation import validate_report
from epspec_agents.schemas import ExperimentResult, ScientificReport


def result_payload():
    return ExperimentResult.model_validate({
        "dataset_name": "corn",
        "preprocess": {"enabled": False, "method": None},
        "main_result": {
            "role": "main_model",
            "method": "plsr",
            "family": "baseline_regression",
            "result_dir": ".",
            "metrics_summary": {"R2": {"mean": 0.91, "std": 0.02}, "RMSE": {"mean": 0.12, "std": 0.01}},
            "metrics_per_fold": [],
            "selection_details": {"selection_type": "full_spectrum"},
        },
    })


def test_report_metric_grounding_accepts_known_values():
    report = ScientificReport(markdown="# 结果\nPLSR 的 R2 为 0.91，RMSE 为 0.12。")
    assert validate_report(report, result_payload()) == report


def test_report_metric_grounding_rejects_unknown_values():
    report = ScientificReport(markdown="# 结果\nPLSR 的 R2 为 0.99。")
    with pytest.raises(ReportValidationError):
        validate_report(report, result_payload())
