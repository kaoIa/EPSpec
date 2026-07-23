import os
import sys
import json
import traceback

AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(AGENTS_DIR)

if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)

try:
    import algorithm_registry as algs
except ImportError as e:
    print("[FATAL] 无法导入 algorithm_registry.py，请确认该文件位于 Agents 目录下。")
    print(e)
    sys.exit(1)

def _build_preprocess_mapping():
    mapping = {}

    if hasattr(algs, "savitzky_golay") and algs.savitzky_golay is not None:
        if hasattr(algs.savitzky_golay, "preprocess_file"):
            mapping["savitzky_golay"] = algs.savitzky_golay.preprocess_file

    if hasattr(algs, "snv") and algs.snv is not None:
        if hasattr(algs.snv, "preprocess_file"):
            mapping["snv"] = algs.snv.preprocess_file

    return mapping

PREPROCESS_FUNCS = _build_preprocess_mapping()

def _build_model_mapping():
    mapping = {}

    if hasattr(algs, "plsr") and algs.plsr is not None:
        if hasattr(algs.plsr, "run_regression"):
            mapping["plsr"] = lambda ip, od: algs.plsr.run_regression(ip, od)

    if hasattr(algs, "ipls_plsr") and algs.ipls_plsr is not None:
        if hasattr(algs.ipls_plsr, "run_regression"):
            mapping["ipls_plsr"] = lambda ip, od: algs.ipls_plsr.run_regression(ip, od, use_ipls=True)

    if hasattr(algs, "cars_plsr") and algs.cars_plsr is not None:
        if hasattr(algs.cars_plsr, "run_regression"):
            mapping["cars_plsr"] = lambda ip, od: algs.cars_plsr.run_regression(ip, od, use_cars=True)

    if hasattr(algs, "EPSpec_plsr") and algs.EPSpec_plsr is not None:
        if hasattr(algs.EPSpec_plsr, "run_regression"):
            mapping["EPSpec_plsr"] = lambda ip, od: algs.EPSpec_plsr.run_regression(ip, od)

    if hasattr(algs, "EPSpec_plsr_sliding") and algs.EPSpec_plsr_sliding is not None:
        if hasattr(algs.EPSpec_plsr_sliding, "run_regression"):
            mapping["EPSpec_plsr_sliding"] = lambda ip, od: algs.EPSpec_plsr_sliding.run_regression(ip, od)

    return mapping

MODEL_RUNNERS = _build_model_mapping()

def _print_divider(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def _load_plan(plan_path: str) -> dict:
    if not os.path.exists(plan_path):
        raise FileNotFoundError(f"未找到 plan.json：{plan_path}")

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    if not isinstance(plan, dict):
        raise ValueError("plan.json 的根节点必须是一个 JSON 对象。")

    return plan

def _get_compare_model_list(compare_cfg) -> list:
    if isinstance(compare_cfg, dict):
        enabled = bool(compare_cfg.get("enabled", False))
        models = compare_cfg.get("models") or []
        if enabled and isinstance(models, list):
            return models
        return []

    if isinstance(compare_cfg, list):
        return compare_cfg

    return []

def run_preprocess_step(step_cfg: dict) -> bool:
    if not step_cfg:
        print("[Step 1] 未找到 step_preprocess 配置，视为跳过。")
        return True

    enabled = bool(step_cfg.get("enabled", False))
    method = step_cfg.get("method")
    input_path = step_cfg.get("input_path")
    output_path = step_cfg.get("output_path")

    if not enabled:
        print("[Step 1] 未启用预处理，跳过。")
        return True

    if not method:
        print("[Step 1] 已启用预处理，但 method 为空。")
        return False

    func = PREPROCESS_FUNCS.get(method)
    if func is None:
        print(f"[Step 1] 未找到预处理函数：{method}")
        print(f"         当前可用预处理：{sorted(PREPROCESS_FUNCS.keys())}")
        return False

    if not input_path or not os.path.exists(input_path):
        print(f"[Step 1] 输入文件不存在：{input_path}")
        return False

    if not output_path:
        print("[Step 1] 预处理输出路径 output_path 为空。")
        return False

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Step 1] 开始预处理：{method}")
    print(f"         输入：{input_path}")
    print(f"         输出：{output_path}")

    try:
        func(input_path, output_path)
    except Exception:
        print("[Step 1] 预处理执行失败。")
        traceback.print_exc()
        return False

    print("[Step 1] 预处理完成。")
    return True

def run_model_step(step_cfg: dict, step_name: str = "Model") -> bool:
    if not step_cfg:
        print(f"[{step_name}] 模型配置为空，无法执行。")
        return False

    method = step_cfg.get("method")
    input_path = step_cfg.get("input_path")
    out_dir = step_cfg.get("out_dir")
    task_type = step_cfg.get("task_type")

    if not method:
        print(f"[{step_name}] 未指定 method。")
        return False

    func = MODEL_RUNNERS.get(method)
    if func is None:
        print(f"[{step_name}] 未找到模型执行函数：{method}")
        print(f"            当前可用模型：{sorted(MODEL_RUNNERS.keys())}")
        return False

    if not input_path or not os.path.exists(input_path):
        print(f"[{step_name}] 输入文件不存在：{input_path}")
        return False

    if not out_dir:
        print(f"[{step_name}] 输出目录 out_dir 为空。")
        return False

    os.makedirs(out_dir, exist_ok=True)

    print(f"[{step_name}] 开始执行模型：{method}")

    if method in {"EPSpec_plsr", "EPSpec_plsr_sliding"}:
        os.makedirs(os.path.join(out_dir, "EP"), exist_ok=True)
    print(f"            task_type：{task_type}")
    print(f"            输入：{input_path}")
    print(f"            输出目录：{out_dir}")

    try:
        func(input_path, out_dir)
    except Exception:
        print(f"[{step_name}] 模型执行失败：{method}")
        traceback.print_exc()
        return False

    print(f"[{step_name}] 模型执行完成：{method}")
    return True

def main():
    _print_divider("Agent 2 - Process Execution：基于 plan.json 执行实验流程")

    plan_path = os.path.join(AGENTS_DIR, "plan.json")

    try:
        plan = _load_plan(plan_path)
    except Exception as e:
        print(f"[FATAL] 读取 plan.json 失败：{e}")
        print("请先运行 Agent 1，确保 plan.json 已正确生成。")
        sys.exit(1)

    dataset_name = plan.get("dataset_name")
    task_type = plan.get("task_type")

    print(f"当前实验配置：dataset={dataset_name}, task_type={task_type}")
    print(f"plan.json 路径：{plan_path}")

    _print_divider("Step 1 - 预处理")
    if not run_preprocess_step(plan.get("step_preprocess") or {}):
        print("流程在 Step 1（预处理）失败，后续步骤停止。")
        sys.exit(1)

    _print_divider("Step 2 - 主模型实验")
    main_cfg = plan.get("step_model_main") or {}
    if not run_model_step(main_cfg, step_name="Main Model"):
        print("流程在 Step 2（主模型实验）失败。")
        sys.exit(1)

    _print_divider("Step 3 - 对比模型实验")
    compare_models = _get_compare_model_list(plan.get("step_model_compare"))

    if not compare_models:
        print("未配置对比模型，或对比未启用，跳过。")
    else:
        for idx, cfg in enumerate(compare_models, start=1):
            method_name = cfg.get("method", f"compare_{idx}")
            step_name = f"Compare Model {idx}"
            print(f"\n[{step_name}] 准备执行：{method_name}")

            if not run_model_step(cfg, step_name=step_name):
                print(f"流程在 Step 3-{idx}（对比模型 {method_name}）失败。")
                sys.exit(1)

    _print_divider("全部实验流程执行完成（不含总结步骤）")

if __name__ == "__main__":
    main()
