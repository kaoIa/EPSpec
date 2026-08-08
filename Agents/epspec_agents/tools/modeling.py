from pathlib import Path

from .registry import ScientificToolRegistry


def execute_model(registry: ScientificToolRegistry, method: str, input_path: Path, output_dir: Path) -> object:
    if method in {"EPSpec_plsr", "EPSpec_plsr_sliding"}:
        (output_dir / "EP").mkdir(parents=True, exist_ok=True)
    return registry.invoke(method, input_path, output_dir, "modeling")
