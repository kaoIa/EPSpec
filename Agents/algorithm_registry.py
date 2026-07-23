import sys
import os
import importlib
import importlib.util

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

PREPROCESS_DIR = os.path.join(PROJECT_ROOT, "Baseline Algorithm", "Preprocessing")
BASELINE_REGRESSION_DIR = os.path.join(PROJECT_ROOT, "Baseline Algorithm", "Regression")
WAVELENGTH_REGRESSION_DIR = os.path.join(PROJECT_ROOT, "Wavelength selection", "Regression")

SLIDING_ABLATION_DIR = os.path.join(
    PROJECT_ROOT,
    "Experiments",
    "Ablation",
    "Code",
    "滑动窗口和分段数"
)

for p in [PREPROCESS_DIR, BASELINE_REGRESSION_DIR, WAVELENGTH_REGRESSION_DIR]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

def _warn(msg: str):
    print(f"[Warning] {msg}")

def _import_module_safely(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        _warn(f"导入模块失败：{module_name} -> {e}")
        return None

def _import_module_from_file(alias_name: str, file_path: str):
    if not os.path.isfile(file_path):
        _warn(f"文件不存在，无法导入：{file_path}")
        return None

    try:
        spec = importlib.util.spec_from_file_location(alias_name, file_path)
        if spec is None or spec.loader is None:
            _warn(f"无法为文件创建 import spec：{file_path}")
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[alias_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        _warn(f"从文件导入模块失败：{file_path} -> {e}")
        return None

savitzky_golay = _import_module_safely("savitzky_golay")
snv = _import_module_safely("snv")

plsr = _import_module_safely("plsr")
ipls_plsr = _import_module_safely("ipls_plsr")
cars_plsr = _import_module_safely("cars_plsr")

EPSpec_plsr = _import_module_safely("EPSpec_plsr_joink")

EPSPEC_SLIDING_FILE = os.path.join(
    SLIDING_ABLATION_DIR,
    "Sliding Window Segmentation Version.py"
)

EPSpec_plsr_sliding = _import_module_from_file(
    alias_name="EPSpec_plsr_sliding",
    file_path=EPSPEC_SLIDING_FILE
)

__all__ = [

    "savitzky_golay",
    "snv",

    "plsr",
    "ipls_plsr",
    "cars_plsr",

    "EPSpec_plsr",
    "EPSpec_plsr_sliding",
]
