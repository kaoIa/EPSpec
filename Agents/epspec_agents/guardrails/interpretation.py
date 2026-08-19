import math
import re
from typing import Any

from ..exceptions import ReportValidationError, ResultParsingError
from ..schemas import ExperimentResult, ScientificReport

METRICS = ("R2", "RMSE", "MAE", "Bias", "RPD", "RPIQ")
METRIC_PATTERN = re.compile(r"R\^?2|R²|RMSE|MAE|Bias|RPD|RPIQ", flags=re.I)
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", flags=re.I)


def validate_experiment_result(result: ExperimentResult) -> ExperimentResult:
    methods = [item.method for item in result.comparison_results]
    if result.main_result.method in set(methods):
        raise ResultParsingError("主模型与对比模型 identity 冲突")
    if len(methods) != len(set(methods)):
        raise ResultParsingError("对比模型 identity 重复")
    for model in [result.main_result, *result.comparison_results]:
        if not model.metrics_summary and not model.metrics_per_fold:
            raise ResultParsingError(f"模型 {model.method} 未读取到 metrics")
        for value in _numbers(model.metrics_summary):
            if not math.isfinite(value):
                raise ResultParsingError(f"模型 {model.method} 包含非有限指标")
    return result


def _numbers(value: Any) -> list[float]:
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, dict):
        output: list[float] = []
        for item in value.values():
            output.extend(_numbers(item))
        return output
    if isinstance(value, list):
        output = []
        for item in value:
            output.extend(_numbers(item))
        return output
    return []


def _canonical_metric(value: str) -> str:
    normalized = value.upper().replace("^", "").replace("²", "2")
    return "Bias" if normalized == "BIAS" else normalized


def _known_metrics(result: ExperimentResult) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for model in [result.main_result, *result.comparison_results]:
        for metric in METRICS:
            output[metric].extend(_numbers(model.metrics_summary.get(metric, {})))
        for row in model.metrics_per_fold:
            for metric in METRICS:
                value = row.get(metric)
                if isinstance(value, int | float):
                    output[metric].append(float(value))
    return output


def _known_wavelengths(result: ExperimentResult) -> list[float]:
    output: list[float] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"start_nm", "end_nm"} and isinstance(item, int | float):
                    output.append(float(item))
                else:
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    for model in [result.main_result, *result.comparison_results]:
        collect(model.selection_details)
    return output


def _supported(value: float, candidates: list[float]) -> bool:
    tolerance = max(1e-4, abs(value) * 5e-4)
    return any(abs(value - candidate) <= tolerance for candidate in candidates)


def validate_report(report: ScientificReport, result: ExperimentResult) -> ScientificReport:
    text = report.markdown.strip()
    if not text:
        raise ReportValidationError("报告为空")
    if result.simulated and not re.search(r"simulat|模拟", text, flags=re.I):
        raise ReportValidationError("模拟运行报告必须明确披露 simulated 状态")
    known_metrics = _known_metrics(result)
    metric_matches = list(METRIC_PATTERN.finditer(text))
    unmatched: list[str] = []
    delimiters = {";", "；", "。", "\n"}
    for index, match in enumerate(metric_matches):
        end = metric_matches[index + 1].start() if index + 1 < len(metric_matches) else len(text)
        end = min(end, match.end() + 100)
        body = text[match.end() : end]
        boundary = next((position for position, character in enumerate(body) if character in delimiters), len(body))
        metric = _canonical_metric(match.group())
        for raw in NUMBER_PATTERN.findall(body[:boundary]):
            if not _supported(float(raw), known_metrics.get(metric, [])):
                unmatched.append(f"{metric}={raw}")
    known_wavelengths = _known_wavelengths(result)
    wavelength_pattern = re.compile(r"(?:(-?\d+(?:\.\d+)?)\s*(?:-|–|—|to|至)\s*)?(-?\d+(?:\.\d+)?)\s*nm", flags=re.I)
    for start, end in wavelength_pattern.findall(text):
        for raw in (start, end):
            if raw and not _supported(float(raw), known_wavelengths):
                unmatched.append(f"wavelength={raw}nm")
    if unmatched:
        raise ReportValidationError("报告包含无法由结果直接支撑的数值: " + ", ".join(unmatched[:12]))
    return report
