from pathlib import Path

from .registry import ScientificToolRegistry


def execute_preprocessing(registry: ScientificToolRegistry, method: str, input_path: Path, output_path: Path) -> object:
    return registry.invoke(method, input_path, output_path, "preprocessing")
