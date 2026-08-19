import importlib.util
from pathlib import Path


def test_legacy_entrypoints_are_importable() -> None:
    agents = Path(__file__).resolve().parents[1]
    for name in ("Agent 1_Process Planning.py", "Agent 2_Process execution.py", "Agent 3_Summary.py", "run.py"):
        path = agents / name
        spec = importlib.util.spec_from_file_location(path.stem.replace(" ", "_"), path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert callable(module.main)
