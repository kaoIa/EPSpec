from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..exceptions import ToolResolutionError

ToolKind = Literal["preprocessing", "modeling"]


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    kind: ToolKind
    task_type: str
    parallel_safe: bool
    search_dir: Path
    module_names: tuple[str, ...] = ()
    file_path: Path | None = None
    function_name: str = "run_regression"
    call_kwargs: dict[str, Any] = field(default_factory=dict)
    runtime_patch: str | None = None

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["search_dir"] = str(self.search_dir)
        value["file_path"] = str(self.file_path) if self.file_path else None
        value["module_names"] = list(self.module_names)
        return value

    def source_candidates(self) -> list[Path]:
        if self.file_path:
            return [self.file_path]
        return [self.search_dir / f"{name}.py" for name in self.module_names]

    def available_source(self) -> Path | None:
        return next((path for path in self.source_candidates() if path.is_file()), None)

    def source_files(self) -> list[Path]:
        primary = self.available_source()
        if primary is None:
            return []
        if self.runtime_patch == "epspec":
            return sorted({primary, *self.search_dir.glob("*.py")})
        return [primary]


class ScientificToolRegistry:
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self._tools: dict[str, ToolSpec] = {}
        self._register_defaults()

    def register(self, spec: ToolSpec) -> None:
        if spec.tool_id in self._tools:
            raise ToolResolutionError(f"工具已注册: {spec.tool_id}")
        self._tools[spec.tool_id] = spec

    def resolve(self, tool_id: str, kind: ToolKind | None = None, require_source: bool = True) -> ToolSpec:
        spec = self._tools.get(tool_id)
        if spec is None:
            raise ToolResolutionError(f"未知 scientific tool: {tool_id}")
        if kind is not None and spec.kind != kind:
            raise ToolResolutionError(f"工具 {tool_id} 不具备 {kind} capability")
        if require_source and spec.available_source() is None:
            raise ToolResolutionError(f"工具源文件不存在: {tool_id}")
        return spec

    def ids(self, kind: ToolKind | None = None) -> list[str]:
        return sorted(key for key, value in self._tools.items() if kind is None or value.kind == kind)

    def capabilities(self) -> list[dict[str, Any]]:
        return [
            {
                "tool_id": spec.tool_id,
                "kind": spec.kind,
                "task_type": spec.task_type,
                "parallel_safe": spec.parallel_safe,
                "available": spec.available_source() is not None,
                "source": str(spec.available_source()) if spec.available_source() else None,
            }
            for spec in sorted(self._tools.values(), key=lambda item: item.tool_id)
        ]

    def _register_defaults(self) -> None:
        preprocessing = self.project_root / "Baseline Algorithm" / "Preprocessing"
        regression = self.project_root / "Baseline Algorithm" / "Regression"
        wavelength = self.project_root / "Wavelength selection" / "Regression"
        sliding = self.project_root / "Experiments" / "Ablation" / "Code" / "滑动窗口和分段数" / "Sliding Window Segmentation Version.py"
        self.register(ToolSpec("savitzky_golay", "preprocessing", "regression", False, preprocessing, ("savitzky_golay",), function_name="preprocess_file"))
        self.register(ToolSpec("snv", "preprocessing", "regression", False, preprocessing, ("snv",), function_name="preprocess_file"))
        self.register(ToolSpec("plsr", "modeling", "regression", True, regression, ("plsr",)))
        self.register(ToolSpec("ipls_plsr", "modeling", "regression", True, regression, ("ipls_plsr", "ipls_plsr_no_full_lv_cap"), call_kwargs={"use_ipls": True}))
        self.register(ToolSpec("cars_plsr", "modeling", "regression", True, regression, ("cars_plsr_no_full_lv_cap",), call_kwargs={"use_cars": True}))
        self.register(ToolSpec("EPSpec_plsr", "modeling", "regression", True, wavelength, ("EPSpec_plsr_joink",), runtime_patch="epspec"))
        self.register(ToolSpec("EPSpec_plsr_sliding", "modeling", "regression", True, sliding.parent, file_path=sliding, runtime_patch="sliding"))
