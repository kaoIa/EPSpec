import importlib.util
import sys

from ..config import RuntimeConfig
from ..schemas import DoctorCheck, DoctorReport
from ..tools.registry import ScientificToolRegistry


def run_doctor(config: RuntimeConfig) -> DoctorReport:
    checks: list[DoctorCheck] = []

    def available(module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError):
            return False

    def add(name: str, condition: bool, success: str, failure: str, warning: bool = False) -> None:
        checks.append(DoctorCheck(name=name, status="pass" if condition else "warn" if warning else "fail", message=success if condition else failure))

    add("python", sys.version_info >= (3, 10), sys.version.split()[0], "需要 Python 3.10 或更高版本")
    add("project_root", config.project_root.is_dir(), str(config.project_root), "项目根目录不存在")
    add("agents_dir", config.agents_dir.is_dir(), str(config.agents_dir), "Agents 目录不存在")
    for dataset in ("shootout", "corn", "soil", "tecator"):
        path = config.project_root / "Data" / "Raw Data" / f"{dataset}.csv"
        add(f"dataset:{dataset}", path.is_file(), str(path), f"缺少数据文件: {path}")
    dependencies = {
        "openai-agents": "agents",
        "openai": "openai",
        "langgraph": "langgraph",
        "langgraph-sqlite": "langgraph.checkpoint.sqlite",
        "pydantic": "pydantic",
        "pandas": "pandas",
        "numpy": "numpy",
        "scipy": "scipy",
        "scikit-learn": "sklearn",
        "fastapi": "fastapi",
        "mcp": "mcp",
    }
    for name, module in dependencies.items():
        installed = available(module)
        add(f"dependency:{name}", installed, "available", "not installed")
    registry = ScientificToolRegistry(config.project_root)
    for capability in registry.capabilities():
        add(f"tool:{capability['tool_id']}", bool(capability["available"]), str(capability["source"]), "算法入口不存在")
    add("prior_knowledge", config.prior_path.is_file(), str(config.prior_path), "先验知识库不存在")
    add("planner_credentials", config.offline or config.planner.configured, "configured", "在线规划模型凭据未配置", warning=config.execution_mode == "simulate")
    add("interpreter_credentials", config.offline or config.interpreter.configured, "configured", "在线解释模型凭据未配置", warning=config.execution_mode == "simulate")
    add("scientific_credentials", config.execution_mode == "simulate" or config.scientific.configured, "configured", "原生 EPSpec 排序模型凭据未配置", warning=True)
    config.runtime_dir.mkdir(parents=True, exist_ok=True)
    config.runs_dir.mkdir(parents=True, exist_ok=True)
    add("runtime_storage", config.runtime_dir.is_dir() and config.runs_dir.is_dir(), "writable", "无法创建运行存储")
    return DoctorReport(ready=not any(check.status == "fail" for check in checks), checks=checks, runtime=config.public())
