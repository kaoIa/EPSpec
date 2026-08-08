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


def validate_execution_plan(plan: ExperimentPlan, registry: ScientificToolRegistry, project_root: Path, require_input: bool = True) -> ExperimentPlan:
    project_root = project_root.resolve()
    if require_input and not plan.step_preprocess.input_path.is_file():
        raise PlanValidationError(f"输入数据不存在: {plan.step_preprocess.input_path}")
    if not _within(plan.step_preprocess.input_path, project_root):
        raise PlanValidationError("输入路径超出项目目录")
    if not _within(plan.step_preprocess.output_path, project_root):
        raise PlanValidationError("预处理输出路径超出项目目录")
    if plan.step_preprocess.enabled:
        registry.resolve(str(plan.step_preprocess.method), "preprocessing")
    steps = [plan.step_model_main, *plan.step_model_compare.models]
    for step in steps:
        registry.resolve(step.method, "modeling")
        if step.task_type != "regression":
            raise PlanValidationError(f"工具 {step.method} 不支持任务 {step.task_type}")
        if not _within(step.input_path, project_root) or not _within(step.out_dir, project_root):
            raise PlanValidationError(f"模型 {step.method} 的路径超出项目目录")
    if [step.out_dir for step in plan.step_model_compare.models] != plan.step_report.input_dirs_compare:
        raise PlanValidationError("报告对比目录与模型执行目录不一致")
    if plan.step_model_main.out_dir != plan.step_report.input_dir_main:
        raise PlanValidationError("报告主模型目录与模型执行目录不一致")
    return plan
