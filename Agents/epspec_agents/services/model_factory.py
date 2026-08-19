import json
import re
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel

from ..config import ModelProfile, RuntimeConfig
from ..exceptions import ConfigurationError, DependencyError
from ..schemas import ComparisonConfig, DatasetId, ExperimentIntent, ModelId, PlanningOutput, PreprocessConfig, PreprocessId, ScientificReport

T = TypeVar("T", bound=BaseModel)
ModelRole = Literal["planning", "interpretation"]


@dataclass(frozen=True)
class GenerationResult(Generic[T]):
    value: T
    raw: str


class StructuredModelAdapter(Protocol):
    def generate(self, name: str, instructions: str, input_text: str, output_type: type[T]) -> GenerationResult[T]: ...


class AgentsSDKAdapter:
    def __init__(self, profile: ModelProfile, allow_text_fallback: bool, tracing_enabled: bool):
        self.profile = profile
        self.allow_text_fallback = allow_text_fallback
        self.tracing_enabled = tracing_enabled
        self._agents: dict[tuple[str, str, str], Any] = {}

    def generate(self, name: str, instructions: str, input_text: str, output_type: type[T]) -> GenerationResult[T]:
        self.profile.require()
        try:
            from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, RunConfig, Runner
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise DependencyError("缺少 openai-agents 或 openai") from exc
        key = (name, instructions, output_type.__name__)
        agent = self._agents.get(key)
        if agent is None:
            client_kwargs: dict[str, Any] = {
                "api_key": self.profile.api_key,
                "timeout": self.profile.timeout_seconds,
                "max_retries": self.profile.max_retries,
            }
            if self.profile.base_url:
                client_kwargs["base_url"] = self.profile.base_url
            client = AsyncOpenAI(**client_kwargs)
            model = OpenAIChatCompletionsModel(model=self.profile.model, openai_client=client)
            agent = Agent(
                name=name,
                instructions=instructions,
                model=model,
                output_type=output_type,
                model_settings=ModelSettings(temperature=self.profile.temperature),
            )
            self._agents[key] = agent
        try:
            result = Runner.run_sync(
                agent,
                input=input_text,
                run_config=RunConfig(
                    tracing_disabled=not self.tracing_enabled,
                    trace_include_sensitive_data=False,
                    workflow_name=f"EPSpec {self.profile.role}",
                ),
            )
            value = result.final_output
            if not isinstance(value, output_type):
                value = output_type.model_validate(value)
            return GenerationResult(value=value, raw=value.model_dump_json(indent=2))
        except Exception as exc:
            if not self.allow_text_fallback:
                raise ConfigurationError(f"OpenAI Agents SDK structured output 调用失败: {exc}") from exc
            return self._fallback(instructions, input_text, output_type, exc)

    def _fallback(self, instructions: str, input_text: str, output_type: type[T], original_error: Exception) -> GenerationResult[T]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise DependencyError("缺少 openai Python SDK") from exc
        client_kwargs: dict[str, Any] = {
            "api_key": self.profile.api_key,
            "timeout": self.profile.timeout_seconds,
            "max_retries": self.profile.max_retries,
        }
        if self.profile.base_url:
            client_kwargs["base_url"] = self.profile.base_url
        client = OpenAI(**client_kwargs)
        schema = json.dumps(output_type.model_json_schema(), ensure_ascii=False)
        system = instructions + "\n\nOnly return one JSON object conforming to this schema:\n" + schema
        try:
            response = client.chat.completions.create(
                model=self.profile.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": input_text}],
                temperature=self.profile.temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            return GenerationResult(value=output_type.model_validate_json(content), raw=response.model_dump_json(indent=2))
        except Exception as exc:
            raise ConfigurationError(f"structured output 与兼容 fallback 均失败: {original_error}; {exc}") from exc


class HeuristicModelAdapter:
    def generate(self, name: str, instructions: str, input_text: str, output_type: type[T]) -> GenerationResult[T]:
        value: BaseModel
        if output_type is PlanningOutput:
            value = self._planning(input_text)
        elif output_type is ScientificReport:
            value = self._report(input_text)
        else:
            raise ConfigurationError(f"离线适配器不支持输出类型: {output_type.__name__}")
        parsed = output_type.model_validate(value)
        return GenerationResult(value=parsed, raw=parsed.model_dump_json(indent=2))

    def _planning(self, input_text: str) -> PlanningOutput:
        try:
            conversation = json.loads(input_text).get("conversation", [])
        except json.JSONDecodeError:
            conversation = []
        text = " ".join(str(item.get("content", "")) for item in conversation if item.get("role") == "user").lower()
        aliases: dict[DatasetId, tuple[str, ...]] = {
            "shootout": ("shootout", "药片", "制药", "pharmaceutical"),
            "corn": ("corn", "玉米"),
            "soil": ("soil", "土壤"),
            "tecator": ("tecator", "肉样", "脂肪"),
        }
        dataset: DatasetId | None = next((key for key, values in aliases.items() if any(value in text for value in values)), None)
        if dataset is None:
            return PlanningOutput(status="needs_clarification", message="请选择数据集：shootout、corn、soil 或 tecator。")
        detected: list[ModelId] = []
        positions: dict[ModelId, int] = {}
        model_patterns: list[tuple[ModelId, tuple[str, ...]]] = [
            ("EPSpec_plsr_sliding", ("epspec_plsr_sliding", "epspec sliding", "滑动窗口")),
            ("EPSpec_plsr", ("epspec_plsr", "epspec", "证据引导", "波段选择")),
            ("ipls_plsr", ("ipls_plsr", "ipls")),
            ("cars_plsr", ("cars_plsr", "cars")),
            ("plsr", ("plsr", "pls")),
        ]
        for model, patterns in model_patterns:
            found = []
            for pattern in patterns:
                if pattern.isascii():
                    match = re.search(rf"(?<![a-z0-9_]){re.escape(pattern)}(?![a-z0-9_])", text)
                    if match:
                        found.append(match.start())
                else:
                    position = text.find(pattern)
                    if position >= 0:
                        found.append(position)
            if found:
                detected.append(model)
                positions[model] = min(found)
        if "EPSpec_plsr_sliding" in detected and "EPSpec_plsr" in detected:
            remaining = text.replace("epspec sliding", "").replace("滑动窗口", "").replace("滑窗", "")
            if not any(token in remaining for token in ("epspec_plsr", "证据引导", "波段选择")):
                detected.remove("EPSpec_plsr")
                positions.pop("EPSpec_plsr", None)
        if not detected:
            return PlanningOutput(status="needs_clarification", message="请选择主模型：plsr、ipls_plsr、cars_plsr、EPSpec_plsr 或 EPSpec_plsr_sliding。")
        primary = min(detected, key=lambda model: positions[model])
        comparison_requested = any(token in text for token in ("compare", "comparison", "versus", " vs ", "对比", "比较", "baseline", "基线"))
        comparisons = [model for model in sorted(detected, key=lambda item: positions[item]) if model != primary] if comparison_requested else []
        if comparison_requested and not comparisons:
            if any(token in text for token in ("baseline", "基线")) and primary != "plsr":
                comparisons = ["plsr"]
            else:
                return PlanningOutput(status="needs_clarification", message="请明确要比较的模型。")
        preprocess_method: PreprocessId | None = None
        if "snv" in text:
            preprocess_method = "snv"
        elif any(token in text for token in ("savitzky", "golay", "sg预处理", "sg preprocessing")):
            preprocess_method = "savitzky_golay"
        intent = ExperimentIntent(
            dataset_name=dataset,
            preprocess=PreprocessConfig(enabled=preprocess_method is not None, method=preprocess_method),
            model=primary,
            compare=ComparisonConfig(enabled=bool(comparisons), models=comparisons),
        )
        return PlanningOutput(status="ready", message="实验意图已结构化。", intent=intent)

    def _report(self, input_text: str) -> ScientificReport:
        start = input_text.find("{")
        payload = json.loads(input_text[start:])
        main = payload["main_result"]
        comparisons = payload.get("comparison_results", [])
        selection = main.get("selection_details", {})
        prior = payload.get("task_level_prior", {})
        mode = "simulated" if payload.get("simulated") else "native"
        configuration = f"Primary method: {main['method']}. Preprocessing: {payload['preprocess']['method'] or 'none'}. Execution mode: {mode}."
        consensus = [
            item for item in selection.get("consensus_selected_intervals", []) if isinstance(item.get("start_nm"), int | float) and isinstance(item.get("end_nm"), int | float)
        ]
        interval_text = ", ".join(
            f"{float(item['start_nm']):.6g}–{float(item['end_nm']):.6g} nm (frequency {item.get('selected_frequency', 'n/a')}, average rank {item.get('avg_rank', 'n/a')})"
            for item in consensus[:5]
        )
        selection_summary = f"Selection type: {selection.get('selection_type', 'not available')}."
        if interval_text:
            selection_summary += " Leading consensus intervals: " + interval_text + "."
        else:
            selection_summary += " No consensus wavelength interval was available in the parsed artifacts."
        if prior.get("available"):
            prior_summary = (
                f"Prior evidence for {prior.get('predicted_substance') or 'the target'} was recovered from "
                f"{prior.get('source_model') or 'EPSpec'} artifacts and linked to the band analysis."
            )
        else:
            prior_summary = "No task-level prior was recovered from the execution artifacts."
        ranked = []
        for model in [main, *comparisons]:
            r2 = model.get("metrics_summary", {}).get("R2", {}).get("mean")
            if isinstance(r2, int | float):
                ranked.append((float(r2), model["method"]))
        if comparisons and ranked:
            best_r2, best_method = max(ranked)
            comparison_summary = (
                f"Within the executed model set, {best_method} had the highest mean R2: {best_r2:.6f}. This is a descriptive result, not a statistical significance claim."
            )
        else:
            comparison_summary = "No comparison model was requested."
        boundary = (
            "All quantitative statements above are copied from the validated execution result. "
            "Simulated runs demonstrate orchestration only and must not be interpreted as scientific findings."
            if payload.get("simulated")
            else "All quantitative statements above are copied from the validated execution result; scientific conclusions remain bounded by the dataset and validation protocol."
        )
        lines = [
            f"# EPSpec Scientific Report: {payload['dataset_name']}",
            "",
            "## Experimental configuration",
            "",
            configuration,
            "",
            "## Quantitative results",
            "",
        ]
        for model in [main, *comparisons]:
            metrics = model.get("metrics_summary", {})
            values = []
            for metric in ("R2", "RMSE", "MAE", "Bias", "RPD", "RPIQ"):
                if metric in metrics:
                    values.append(f"{metric} {metrics[metric]['mean']:.6f} ± {metrics[metric]['std']:.6f}")
            lines.append(f"- {model['method']}: " + "; ".join(values))
        lines.extend(
            [
                "",
                "## Model comparison",
                "",
                comparison_summary,
                "",
                "## Evidence-guided band selection",
                "",
                selection_summary,
                "",
                "## Prior retrieval",
                "",
                prior_summary,
                "",
                "## Interpretation boundary",
                "",
                boundary,
            ]
        )
        return ScientificReport(markdown="\n".join(lines))


class ModelFactory:
    def __init__(self, config: RuntimeConfig):
        self.config = config

    def create(self, role: ModelRole = "planning") -> StructuredModelAdapter:
        if self.config.offline:
            return HeuristicModelAdapter()
        profile = self.config.planner if role == "planning" else self.config.interpreter
        return AgentsSDKAdapter(profile, self.config.allow_text_fallback, self.config.sdk_tracing)
