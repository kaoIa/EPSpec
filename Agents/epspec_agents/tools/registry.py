from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import sys
from typing import Callable, Literal

from ..exceptions import ToolExecutionError, ToolResolutionError


ToolKind = Literal["preprocessing", "modeling"]


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    kind: ToolKind
    task_type: str
    parallel_safe: bool
    loader: Callable[[], Callable[..., object]]


class ScientificToolRegistry:
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self._tools: dict[str, ToolSpec] = {}
        self._register_defaults()

    def register(self, spec: ToolSpec) -> None:
        if spec.tool_id in self._tools:
            raise ToolResolutionError(f"工具已注册: {spec.tool_id}")
        self._tools[spec.tool_id] = spec

    def resolve(self, tool_id: str, kind: ToolKind | None = None) -> ToolSpec:
        spec = self._tools.get(tool_id)
        if spec is None:
            raise ToolResolutionError(f"未知 scientific tool: {tool_id}")
        if kind is not None and spec.kind != kind:
            raise ToolResolutionError(f"工具 {tool_id} 不具备 {kind} capability")
        return spec

    def invoke(self, tool_id: str, input_path: Path, output_path: Path, kind: ToolKind) -> object:
        spec = self.resolve(tool_id, kind)
        try:
            runner = spec.loader()
            output_path.parent.mkdir(parents=True, exist_ok=True) if kind == "preprocessing" else output_path.mkdir(parents=True, exist_ok=True)
            return runner(str(input_path), str(output_path))
        except ToolResolutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"工具 {tool_id} 执行失败: {exc}") from exc

    def ids(self, kind: ToolKind | None = None) -> list[str]:
        return sorted(key for key, value in self._tools.items() if kind is None or value.kind == kind)

    def _module_loader(self, search_dir: Path, module_names: list[str], function_name: str, kwargs: dict[str, object] | None = None) -> Callable[[], Callable[..., object]]:
        def load() -> Callable[..., object]:
            if str(search_dir) not in sys.path:
                sys.path.insert(0, str(search_dir))
            module = None
            failures = []
            for module_name in module_names:
                try:
                    module = importlib.import_module(module_name)
                    break
                except Exception as exc:
                    failures.append(f"{module_name}: {exc}")
            if module is None:
                raise ToolResolutionError("; ".join(failures))
            function = getattr(module, function_name, None)
            if not callable(function):
                raise ToolResolutionError(f"{module.__name__}.{function_name} 不可调用")
            if not kwargs:
                return function
            return lambda input_path, output_path: function(input_path, output_path, **kwargs)
        return load

    def _file_loader(self, path: Path, alias: str, function_name: str) -> Callable[[], Callable[..., object]]:
        def load() -> Callable[..., object]:
            if not path.is_file():
                raise ToolResolutionError(f"算法文件不存在: {path}")
            spec = importlib.util.spec_from_file_location(alias, path)
            if spec is None or spec.loader is None:
                raise ToolResolutionError(f"无法创建算法加载器: {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[alias] = module
            spec.loader.exec_module(module)
            function = getattr(module, function_name, None)
            if not callable(function):
                raise ToolResolutionError(f"{path.name}.{function_name} 不可调用")
            return function
        return load

    def _register_defaults(self) -> None:
        preprocessing = self.project_root / "Baseline Algorithm" / "Preprocessing"
        regression = self.project_root / "Baseline Algorithm" / "Regression"
        wavelength = self.project_root / "Wavelength selection" / "Regression"
        sliding = self.project_root / "Experiments" / "Ablation" / "Code" / "滑动窗口和分段数" / "Sliding Window Segmentation Version.py"
        self.register(ToolSpec("savitzky_golay", "preprocessing", "regression", False, self._module_loader(preprocessing, ["savitzky_golay"], "preprocess_file")))
        self.register(ToolSpec("snv", "preprocessing", "regression", False, self._module_loader(preprocessing, ["snv"], "preprocess_file")))
        self.register(ToolSpec("plsr", "modeling", "regression", False, self._module_loader(regression, ["plsr"], "run_regression")))
        self.register(ToolSpec("ipls_plsr", "modeling", "regression", False, self._module_loader(regression, ["ipls_plsr", "ipls_plsr_no_full_lv_cap"], "run_regression", {"use_ipls": True})))
        self.register(ToolSpec("cars_plsr", "modeling", "regression", False, self._module_loader(regression, ["cars_plsr", "cars_plsr_no_full_lv_cap"], "run_regression", {"use_cars": True})))
        self.register(ToolSpec("EPSpec_plsr", "modeling", "regression", False, self._module_loader(wavelength, ["EPSpec_plsr_joink"], "run_regression")))
        self.register(ToolSpec("EPSpec_plsr_sliding", "modeling", "regression", False, self._file_loader(sliding, "epspec_plsr_sliding_runtime", "run_regression")))
