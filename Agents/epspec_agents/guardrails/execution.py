import re
from pathlib import Path

from ..exceptions import PlanValidationError
from ..schemas import ExperimentPlan
from ..tools.registry import ScientificToolRegistry


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_execution_plan(
    plan: ExperimentPlan,
    registry: ScientificToolRegistry,
    project_root: Path,
    agents_dir: Path,
    require_input: bool = True,
) -> ExperimentPlan:
    project_root = project_root.resolve()
    raw_root = (project_root / "Data" / "Raw Data").resolve()
    run_root = (agents_dir.resolve() / "runs" / plan.run_id).resolve()
    raw_input = plan.step_preprocess.input_path.resolve()
    expected_raw = (raw_root / f"{plan.dataset_name}.csv").resolve()
    if raw_input != expected_raw:
        raise PlanValidationError("原始输入路径与数据集标识不一致")
    if require_input and not raw_input.is_file():
        raise PlanValidationError(f"输入数据不存在: {raw_input}")
    if not _within(raw_input, raw_root):
        raise PlanValidationError("原始输入路径必须位于 Data/Raw Data")
    if plan.step_preprocess.enabled:
        registry.resolve(str(plan.step_preprocess.method), "preprocessing")
        if not _within(plan.step_preprocess.output_path, run_root):
            raise PlanValidationError("预处理输出必须位于当前运行目录")
    elif plan.step_preprocess.output_path.resolve() != raw_input:
        raise PlanValidationError("未启用预处理时输出路径必须等于原始输入")
    expected_model_input = plan.step_preprocess.output_path.resolve() if plan.step_preprocess.enabled else raw_input
    steps = [plan.step_model_main, *plan.step_model_compare.models]
    families = {
        "plsr": "baseline_regression",
        "ipls_plsr": "ipls_cars_regression",
        "cars_plsr": "ipls_cars_regression",
        "EPSpec_plsr": "wavelength_selection_regression",
        "EPSpec_plsr_sliding": "wavelength_selection_regression",
    }
    for step in steps:
        registry.resolve(step.method, "modeling")
        if step.task_type != "regression":
            raise PlanValidationError(f"工具 {step.method} 不支持任务 {step.task_type}")
        if not _within(step.input_path, project_root):
            raise PlanValidationError(f"模型 {step.method} 的输入路径越界")
        if step.input_path.resolve() != expected_model_input:
            raise PlanValidationError(f"模型 {step.method} 的输入未连接到计划数据流")
        if not _within(step.out_dir, run_root):
            raise PlanValidationError(f"模型 {step.method} 的输出路径越界")
        if step.family != families[step.method]:
            raise PlanValidationError(f"模型 {step.method} 的 family 不匹配")
    if [step.out_dir.resolve() for step in plan.step_model_compare.models] != [path.resolve() for path in plan.step_report.input_dirs_compare]:
        raise PlanValidationError("报告对比目录与模型执行目录不一致")
    if plan.step_model_main.out_dir.resolve() != plan.step_report.input_dir_main.resolve():
        raise PlanValidationError("报告主模型目录与模型执行目录不一致")
    if not _within(plan.step_report.output_path, run_root):
        raise PlanValidationError("报告输出路径必须位于当前运行目录")
    if not plan.step_report.enabled:
        raise PlanValidationError("报告阶段必须启用")
    output_paths = [step.out_dir.resolve() for step in steps]
    if len(output_paths) != len(set(output_paths)):
        raise PlanValidationError("模型输出目录不允许重复")
    return plan


def validate_interpretation_plan(
    plan: ExperimentPlan,
    registry: ScientificToolRegistry,
    project_root: Path,
    agents_dir: Path,
) -> ExperimentPlan:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", plan.run_id):
        raise PlanValidationError("解释源 run_id 格式不合法")
    project_root = project_root.resolve()
    runs_root = (agents_dir.resolve() / "runs").resolve()
    source_root = (runs_root / plan.run_id).resolve()
    if not _within(source_root, runs_root) or not source_root.is_dir():
        raise PlanValidationError("解释源运行目录不存在或越界")
    raw_input = plan.step_preprocess.input_path.resolve()
    expected_raw = (project_root / "Data" / "Raw Data" / f"{plan.dataset_name}.csv").resolve()
    if raw_input != expected_raw or not raw_input.is_file():
        raise PlanValidationError("解释计划的原始输入与数据集不一致")
    if plan.step_preprocess.enabled:
        registry.resolve(str(plan.step_preprocess.method), "preprocessing")
        expected_model_input = plan.step_preprocess.output_path.resolve()
        if not _within(expected_model_input, source_root) or not expected_model_input.is_file():
            raise PlanValidationError("解释计划的预处理产物不存在或越界")
    else:
        expected_model_input = raw_input
        if plan.step_preprocess.output_path.resolve() != raw_input:
            raise PlanValidationError("解释计划未启用预处理时路径不一致")
    families = {
        "plsr": "baseline_regression",
        "ipls_plsr": "ipls_cars_regression",
        "cars_plsr": "ipls_cars_regression",
        "EPSpec_plsr": "wavelength_selection_regression",
        "EPSpec_plsr_sliding": "wavelength_selection_regression",
    }
    steps = [plan.step_model_main, *plan.step_model_compare.models]
    for step in steps:
        registry.resolve(step.method, "modeling")
        if step.family != families[step.method]:
            raise PlanValidationError(f"模型 {step.method} 的 family 不匹配")
        if step.input_path.resolve() != expected_model_input:
            raise PlanValidationError(f"模型 {step.method} 的解释输入路径不一致")
        if not _within(step.out_dir, source_root) or not step.out_dir.resolve().is_dir():
            raise PlanValidationError(f"模型 {step.method} 的解释产物目录不存在或越界")
    if plan.step_model_main.out_dir.resolve() != plan.step_report.input_dir_main.resolve():
        raise PlanValidationError("报告主模型目录与解释产物目录不一致")
    if [step.out_dir.resolve() for step in plan.step_model_compare.models] != [path.resolve() for path in plan.step_report.input_dirs_compare]:
        raise PlanValidationError("报告对比目录与解释产物目录不一致")
    if not plan.step_report.enabled:
        raise PlanValidationError("解释报告阶段必须启用")
    return plan
