import importlib.util
from pathlib import Path


def import_file(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem.replace(" ", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_entrypoints_are_importable():
    agents = Path(__file__).resolve().parents[1]
    for name in ["Agent 1_Process Planning.py", "Agent 2_Process execution.py", "Agent 3_Summary.py", "run.py"]:
        module = import_file(agents / name)
        assert callable(module.main)
