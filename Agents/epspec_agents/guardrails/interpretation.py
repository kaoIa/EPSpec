import re

from ..exceptions import ReportValidationError, ResultParsingError
from ..schemas import ExperimentResult, ScientificReport


def validate_experiment_result(result: ExperimentResult) -> ExperimentResult:
    if result.main_result.method in {item.method for item in result.comparison_results}:
        raise ResultParsingError("主模型与对比模型 identity 冲突")
    if not result.main_result.metrics_summary and not result.main_result.metrics_per_fold:
        raise ResultParsingError("主模型结果不完整，未读取到 metrics")
    return result


def _known_metric_values(result: ExperimentResult) -> list[float]:
    values = []
    for model in [result.main_result, *result.comparison_results]:
        for payload in model.metrics_summary.values():
            for value in payload.values():
                if isinstance(value, (int, float)):
                    values.append(float(value))
        for row in model.metrics_per_fold:
            for key in ("R2", "RMSE", "MAE", "Bias", "RPD", "RPIQ"):
                value = row.get(key)
                if isinstance(value, (int, float)):
                    values.append(float(value))
    return values


def validate_report(report: ScientificReport, result: ExperimentResult) -> ScientificReport:
    text = report.markdown.strip()
    if not text:
        raise ReportValidationError("报告为空")
    known = _known_metric_values(result)
    claims = re.findall(r"(?:R\^?2|R²|RMSE|MAE|Bias|RPD|RPIQ)[^\d-]{0,16}(-?\d+(?:\.\d+)?)", text, flags=re.I)
    unmatched = []
    for raw in claims:
        value = float(raw)
        tolerance = max(1e-4, abs(value) * 5e-4)
        if not any(abs(value - candidate) <= tolerance for candidate in known):
            unmatched.append(raw)
    if unmatched:
        raise ReportValidationError("报告包含无法由结果直接支撑的关键指标: " + ", ".join(unmatched[:8]))
    return report
