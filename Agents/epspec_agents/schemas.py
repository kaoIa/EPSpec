from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DatasetId = Literal["corn", "soil", "tecator"]
TaskType = Literal["regression"]
PreprocessId = Literal["savitzky_golay", "snv"]
ModelId = Literal["plsr", "ipls_plsr", "cars_plsr", "EPSpec_plsr", "EPSpec_plsr_sliding"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PreprocessConfig(StrictModel):
    enabled: bool = False
    method: PreprocessId | None = None

    @model_validator(mode="after")
    def validate_method(self) -> "PreprocessConfig":
        if self.enabled != (self.method is not None):
            raise ValueError("preprocess.enabled 与 preprocess.method 不一致")
        return self


class ModelConfig(StrictModel):
    method: ModelId


class ComparisonConfig(StrictModel):
    enabled: bool = False
    models: list[ModelId] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_enabled(self) -> "ComparisonConfig":
        if not self.enabled and self.models:
            raise ValueError("compare.enabled=false 时 models 必须为空")
        if self.enabled and not self.models:
            raise ValueError("compare.enabled=true 时 models 不能为空")
        if len(self.models) != len(set(self.models)):
            raise ValueError("comparison models 不允许重复")
        return self


class ExperimentIntent(StrictModel):
    dataset_name: DatasetId
    task_type: TaskType = "regression"
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    model: ModelId
    compare: ComparisonConfig = Field(default_factory=ComparisonConfig)

    @model_validator(mode="after")
    def validate_comparisons(self) -> "ExperimentIntent":
        if self.model in self.compare.models:
            raise ValueError("comparison model 不得与 primary model 相同")
        return self


class PlanningOutput(StrictModel):
    status: Literal["ready", "needs_clarification"]
    message: str = ""
    intent: ExperimentIntent | None = None

    @model_validator(mode="after")
    def validate_status(self) -> "PlanningOutput":
        if self.status == "ready" and self.intent is None:
            raise ValueError("ready 状态必须包含 intent")
        if self.status == "needs_clarification" and (self.intent is not None or not self.message.strip()):
            raise ValueError("needs_clarification 状态必须仅包含澄清问题")
        return self


class PreprocessStep(StrictModel):
    enabled: bool
    method: PreprocessId | None
    input_path: Path
    output_path: Path


class ModelExecutionStep(StrictModel):
    method: ModelId
    family: Literal["baseline_regression", "ipls_cars_regression", "wavelength_selection_regression"]
    task_type: TaskType = "regression"
    input_path: Path
    out_dir: Path


class ComparisonExecutionStep(StrictModel):
    enabled: bool
    models: list[ModelExecutionStep] = Field(default_factory=list)


class ReportStep(StrictModel):
    enabled: bool = True
    input_dir_main: Path
    input_dirs_compare: list[Path] = Field(default_factory=list)
    output_path: Path


class ExperimentPlan(StrictModel):
    dataset_name: DatasetId
    task_type: TaskType = "regression"
    step_preprocess: PreprocessStep
    step_model_main: ModelExecutionStep
    step_model_compare: ComparisonExecutionStep
    step_report: ReportStep

    def to_legacy_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_legacy_dict(cls, value: dict[str, Any]) -> "ExperimentPlan":
        compare = value.get("step_model_compare", {})
        if isinstance(compare, list):
            value = dict(value)
            value["step_model_compare"] = {"enabled": bool(compare), "models": compare}
        return cls.model_validate(value)


class ArtifactRef(StrictModel):
    name: str
    path: Path
    media_type: str
    sha256: str | None = None
    size_bytes: int | None = None


class ModelMetrics(StrictModel):
    R2: float | None = None
    RMSE: float | None = None
    MAE: float | None = None
    Bias: float | None = None
    RPD: float | None = None
    RPIQ: float | None = None


class ModelRunResult(StrictModel):
    role: str
    method: ModelId
    family: str
    result_dir: Path
    metrics_summary: dict[str, dict[str, float]] = Field(default_factory=dict)
    metrics_per_fold: list[dict[str, Any]] = Field(default_factory=list)
    selection_details: dict[str, Any] = Field(default_factory=dict)
    task_context: dict[str, Any] | None = None


class ExperimentResult(StrictModel):
    dataset_name: DatasetId
    task_type: TaskType = "regression"
    preprocess: PreprocessConfig
    main_result: ModelRunResult
    comparison_results: list[ModelRunResult] = Field(default_factory=list)
    task_level_prior: dict[str, Any] = Field(default_factory=dict)


class ScientificReport(StrictModel):
    markdown: str = Field(min_length=1)


class ExecutionError(StrictModel):
    stage: str
    error_type: str
    message: str
    traceback: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RunStatus(str, Enum):
    initialized = "initialized"
    planning = "planning"
    awaiting_clarification = "awaiting_clarification"
    awaiting_approval = "awaiting_approval"
    executing = "executing"
    interpreting = "interpreting"
    completed = "completed"
    failed = "failed"


class RunManifest(StrictModel):
    run_id: str
    thread_id: str
    status: RunStatus
    current_stage: str
    created_at: str
    updated_at: str
    plan: ExperimentPlan | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    errors: list[ExecutionError] = Field(default_factory=list)
    environment: dict[str, Any] = Field(default_factory=dict)
