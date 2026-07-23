import os
import json
from openai import OpenAI

API_KEY = "your key"
API_URL = "your URL"
MODEL_NAME = "your model name"

client = OpenAI(
    api_key=API_KEY,
    base_url=API_URL,
    timeout=600,
    max_retries=0,
)

def call_llm(messages) -> str:
    if not API_KEY:
        return "调用大模型失败：未检测到 API_KEY。"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            max_tokens=64000,
            seed=0,
            response_format={"type": "text"},
            stream=False,
        )
    except Exception as e:
        return f"调用大模型失败：{e}"

    try:
        content = response.choices[0].message.content
        if content is None:
            return "API 响应格式错误：message.content 为空"
        return content
    except Exception as e:
        try:
            raw = response.model_dump_json(indent=2)
        except Exception:
            raw = str(response)
        return f"解析响应失败：{e}，原始响应：{raw}"

SYSTEM_PROMPT = r"""
你是“近红外光谱实验规划助手”，负责在多轮对话中将用户需求整理为一次实验的语义配置 JSON。

你的唯一任务：和用户多轮对话后，给出一次实验的**语义配置 JSON**，字段固定为：
{
  "dataset_name": "...",
  "task_type": "regression",
  "preprocess": {
    "enabled": true/false,
    "method": 预处理英文标识或 null
  },
  "model": "一个主模型的英文名",
  "compare": {
    "enabled": true/false,
    "models": [对比模型1, 对比模型2, ...]
  }
}

请特别注意：
- task_type 固定为 "regression"。
- "model" 表示本次实验的**主模型**，只能有一个。
- "compare.models" 表示用户**显式要求比较**的其他模型。
- 你只负责生成语义配置，不负责生成最终实验结果。

-------------------------
【语言理解要求】
-------------------------
你必须支持以下表达方式，并能正确映射到统一 JSON：
- 中文
- English
- 中文/English 混合
- 简称、别名、口语化表达

例如：
- “土壤，用 EPSpec，SNV，和基线比一下”
- “soil + EPSpec+PLSR + snv + compare baseline”
- “tecator, no preprocessing, PLSR”
- “玉米，SG-d1，主模型用滑窗版 EPSpec，再加一个 ipls_plsr 对比”
这些都应能正确理解。

当用户用中文、英文、简称或中英混合描述时：
- 你要正常理解；
- 但最终 JSON 中必须统一输出规范英文标识；
- JSON 中不能出现中文别名。

默认回复语言：
- 优先使用中文与用户交流；
- 如需举例，可适当同时给出中英文对应；
- 最终 JSON 必须是英文标识。

-------------------------
【默认规则】
-------------------------
如果用户没有提到“对比 / 比较 / baseline / 基线 / compare”等内容，
则必须输出：
"compare": {
  "enabled": false,
  "models": []
}

如果用户没有提到预处理，
则必须输出：
"preprocess": {
  "enabled": false,
  "method": null
}

-------------------------
【1. 数据集 dataset_name】
-------------------------
JSON 中允许的 dataset_name 只有以下 3 个英文标识：
["corn", "soil", "tecator"]

常见映射示例：
- 玉米 / corn -> "corn"
- 土壤 / soil -> "soil"
- 肉糜 / 肉类 / tecator -> "tecator"

如果用户提到其他数据集（例如 cassav, diesel, grape, milk 等），
请礼貌说明可用的数据集名称，并请用户改成其中之一。
在这种情况下，不要直接给最终 JSON，先澄清。

-------------------------
【2. 任务类型 task_type】
-------------------------
task_type 固定只能是：
- "regression"

不能输出 classification。
如果用户提到分类任务，请礼貌说明这里使用的是 regression 设置，并请用户改写需求。
在这种情况下，不要直接给最终 JSON，先澄清。

-------------------------
【3. 预处理 preprocess】
-------------------------
preprocess 是一个对象：

- enabled: true / false
- method:
    - enabled = true 时，method 必须是以下之一：
      ["savitzky_golay", "snv"]
    - enabled = false 时，method 必须为 null

映射规则：
- SNV / snv / 标准正态变量变换 -> "snv"
- SG-d1 / sg-d1 / SG一阶导 / SG导数 / Savitzky-Golay一阶导 -> "savitzky_golay"
- 不预处理 / 原始光谱 / none / no preprocessing / raw spectrum -> enabled=false, method=null

说明：
- 若用户只说“一阶导”，只有在语境明确指向 SG 一阶导时，才映射为 "savitzky_golay"。
- 若用户提到不在本流程中的预处理（例如 msc / detrend / asls 等），
  请礼貌说明可识别的预处理写法，并请用户修改。
  在这种情况下，不要直接给最终 JSON，先澄清。

-------------------------
【4. 主模型 model】
-------------------------
你只负责确定一个主模型，写入 JSON 的 "model" 字段。
JSON 中允许的主模型只有以下 5 个英文标识：

["plsr", "ipls_plsr", "cars_plsr", "EPSpec_plsr", "EPSpec_plsr_sliding"]

常见映射示例：
- PLSR / plsr / 偏最小二乘回归 / 全谱PLSR -> "plsr"
- iPLS+PLSR / iPLS / ipls_plsr -> "ipls_plsr"
- CARS+PLSR / CARS / cars_plsr -> "cars_plsr"
- EPSpec+PLSR / EPSpec / epspec / 默认EPSpec -> "EPSpec_plsr"
- EPSpec+PLSR-sliding / EPSpec-sliding / EPSpec滑动窗口版 / EPSpec滑窗版 / sliding EPSpec -> "EPSpec_plsr_sliding"

注意：
- 如果用户只说“EPSpec”，默认理解为 "EPSpec_plsr"。
- 只有在用户明确提到 sliding / 滑动窗口 / 滑窗 时，才理解为 "EPSpec_plsr_sliding"。

如果用户的描述不足以唯一确定主模型，请简要追问，不要直接猜。

-------------------------
【5. 对比模型 compare】
-------------------------
compare 字段结构为：

"compare": {
  "enabled": true/false,
  "models": [ ... ]
}

语义如下：
- enabled = false：
  - 不做额外模型对比；
  - models 必须为 []。
- enabled = true：
  - 用户明确表达了要比较、对比、加模型、compare 等意思；
  - models 按用户提及顺序填写。

compare.models 中每个元素必须来自：
["plsr", "ipls_plsr", "cars_plsr", "EPSpec_plsr", "EPSpec_plsr_sliding"]

当用户说“对比基线 / baseline / compare baseline”时：
- 你要根据主模型自动映射出具体模型名；
- JSON 中不要写 "baseline"，要写真实模型名。

映射规则如下：
- 主模型是 "EPSpec_plsr" -> baseline 为 "plsr"
- 主模型是 "EPSpec_plsr_sliding" -> baseline 为 "plsr"
- 主模型是 "ipls_plsr" -> baseline 为 "plsr"
- 主模型是 "cars_plsr" -> baseline 为 "plsr"

特殊情况：
- 如果主模型本身已经是 "plsr"，而用户又说“对比基线 / baseline”，
  需要先礼貌说明 plsr 本身就是基线，请用户明确写出要比较的其他模型。
  在这种情况下，不要直接给最终 JSON，先澄清。

去重规则：
- compare.models 不能包含主模型本身；
- compare.models 不能重复；
- 若用户重复添加同一个对比模型，可简要提示“该模型已在对比列表中”，并保持 JSON 不重复。

执行顺序规则：
- 主模型始终先执行；
- compare.models 按数组顺序依次执行。

-------------------------
【6. 何时直接输出 JSON，何时继续追问】
-------------------------
当以下信息已经能够唯一确定时：
- dataset_name
- task_type = regression
- 一个合法的主模型 model
则可以直接输出 JSON。

若用户没有提到预处理，则使用默认：
"preprocess": {"enabled": false, "method": null}

若用户没有提到对比模型，则使用默认：
"compare": {"enabled": false, "models": []}

只有以下情况才需要追问：
- 数据集不明确或不在可识别范围内；
- 用户提到 classification 或其他非 regression 任务；
- 预处理描述无法唯一映射；
- 主模型描述无法唯一映射；
- 用户明确要求对比，但没有说明对比谁；
- 主模型已是 plsr，但用户又说要对比 baseline。

-------------------------
【7. 输出格式要求】
-------------------------
1）最终最好输出一个纯 JSON 对象。
2）如果你需要先用中文简要说明，也可以写在前面；
   但最后必须给出且只给出一份完整 JSON。
3）禁止一次输出多个 JSON。
4）JSON 中禁止出现中文、别名或解释性文字，必须全部使用规范英文标识。

你的目标是：
- 准确理解用户需求；
- 在必要时做最少量澄清；
- 尽快给出一份结构正确、字段合法、可直接解析的语义 JSON。
"""

def extract_last_json_block(text: str):
    blocks = []
    brace_level = 0
    start_idx = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if brace_level == 0:
                start_idx = i
            brace_level += 1
        elif ch == "}":
            if brace_level > 0:
                brace_level -= 1
                if brace_level == 0 and start_idx != -1:
                    blocks.append(text[start_idx:i + 1])
                    start_idx = -1
    if blocks:
        return blocks[-1].strip()
    return None

AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(AGENTS_DIR)
TESTS_ROOT = os.path.join(ROOT_DIR, "Experiments", "Tests")

SUPPORTED_DATASETS = {"corn", "soil", "tecator"}
SUPPORTED_TASK_TYPE = "regression"
SUPPORTED_PREPROCESS = {"savitzky_golay", "snv"}
SUPPORTED_MODELS = {"plsr", "ipls_plsr", "cars_plsr", "EPSpec_plsr", "EPSpec_plsr_sliding"}

BASELINE_REG = {"plsr"}
IPLS_CARS_REG = {"cars_plsr", "ipls_plsr"}
EPSPEC_REG = {"EPSpec_plsr", "EPSpec_plsr_sliding"}

def model_to_family_and_outdir(dataset: str, model_id: str):

    if model_id in BASELINE_REG:
        family = "baseline_regression"
        base_dir = os.path.join(TESTS_ROOT, "Baseline", dataset)
        out_dir = os.path.join(base_dir, f"{model_id}_cv_results")
        return family, out_dir

    if model_id in IPLS_CARS_REG:
        family = "ipls_cars_regression"
        base_dir = os.path.join(TESTS_ROOT, "ipls and cars", dataset)
        out_dir = os.path.join(base_dir, f"{model_id}_cv_results")
        return family, out_dir

    if model_id in EPSPEC_REG:
        family = "wavelength_selection_regression"
        base_dir = os.path.join(TESTS_ROOT, "wavelength selection", dataset)

        if model_id == "EPSpec_plsr":
            short_name = "Epspec"
        elif model_id == "EPSpec_plsr_sliding":
            short_name = "Epspec_sliding"
        else:
            short_name = model_id.lower()

        out_dir = os.path.join(base_dir, short_name)
        return family, out_dir

    family = "unknown"
    base_dir = os.path.join(TESTS_ROOT, "unknown", dataset)
    out_dir = os.path.join(base_dir, model_id)
    return family, out_dir

def validate_semantic_config(config: dict):
    if not isinstance(config, dict):
        raise ValueError("语义配置不是 JSON 对象。")

    dataset = config.get("dataset_name")
    task_type = config.get("task_type")
    preprocess_cfg = config.get("preprocess")
    main_model_id = config.get("model")
    compare_cfg = config.get("compare")

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"不支持的数据集: {dataset}。可选值为 {sorted(SUPPORTED_DATASETS)}")

    if task_type != SUPPORTED_TASK_TYPE:
        raise ValueError(f"task_type 必须为 '{SUPPORTED_TASK_TYPE}'，收到: {task_type}")

    if not isinstance(preprocess_cfg, dict):
        raise ValueError("preprocess 字段必须是对象。")

    pre_enabled = bool(preprocess_cfg.get("enabled", False))
    pre_method = preprocess_cfg.get("method")

    if pre_enabled:
        if pre_method not in SUPPORTED_PREPROCESS:
            raise ValueError(f"不支持的预处理方法: {pre_method}。可选值为 {sorted(SUPPORTED_PREPROCESS)}")
    else:
        if pre_method is not None:
            raise ValueError("当 preprocess.enabled = false 时，preprocess.method 必须为 null。")

    if main_model_id not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的主模型: {main_model_id}。可选值为 {sorted(SUPPORTED_MODELS)}")

    if not isinstance(compare_cfg, dict):
        raise ValueError("compare 字段必须是对象。")

    compare_enabled = bool(compare_cfg.get("enabled", False))
    compare_models = compare_cfg.get("models", [])

    if not isinstance(compare_models, list):
        raise ValueError("compare.models 必须是列表。")

    if not compare_enabled and compare_models:
        raise ValueError("当 compare.enabled = false 时，compare.models 必须为空列表。")

    seen = {main_model_id}
    for cm in compare_models:
        if cm not in SUPPORTED_MODELS:
            raise ValueError(f"不支持的对比模型: {cm}。可选值为 {sorted(SUPPORTED_MODELS)}")
        if cm in seen:
            raise ValueError(f"对比模型重复，或与主模型相同: {cm}")
        seen.add(cm)

def build_plan_from_semantic(config: dict) -> dict:
    dataset = config.get("dataset_name")
    task_type = config.get("task_type")
    preprocess_cfg = config.get("preprocess") or {}
    main_model_id = config.get("model")

    compare_cfg = config.get("compare") or {}
    compare_enabled = bool(compare_cfg.get("enabled", False))
    compare_models = compare_cfg.get("models") or []
    if not compare_enabled:
        compare_models = []

    unique_compare_models = []
    seen = {main_model_id}
    for cm in compare_models:
        if cm in seen:
            print(f"[提示] 对比模型 {cm} 已存在，将不会重复添加到 plan 中。")
            continue
        unique_compare_models.append(cm)
        seen.add(cm)
    compare_models = unique_compare_models

    raw_input_path = os.path.join(
        ROOT_DIR,
        "Data", "Raw Data",
        f"{dataset}.csv"
    )

    pre_enabled = bool(preprocess_cfg.get("enabled", False))
    method = preprocess_cfg.get("method") if pre_enabled else None

    if pre_enabled:
        preprocessed_output_path = os.path.join(
            TESTS_ROOT,
            "Preprocessed Data",
            f"{dataset}_{method}_preprocessed.csv"
        )
        step_preprocess = {
            "enabled": True,
            "method": method,
            "input_path": raw_input_path,
            "output_path": preprocessed_output_path,
        }
        main_input_path = preprocessed_output_path
    else:
        step_preprocess = {
            "enabled": False,
            "method": None,
            "input_path": raw_input_path,
            "output_path": raw_input_path,
        }
        main_input_path = raw_input_path

    main_family, main_out_dir = model_to_family_and_outdir(dataset, main_model_id)
    step_model_main = {
        "method": main_model_id,
        "family": main_family,
        "task_type": task_type,
        "input_path": main_input_path,
        "out_dir": main_out_dir
    }

    compare_steps = []
    compare_out_dirs = []

    for cm in compare_models:
        family, out_dir = model_to_family_and_outdir(dataset, cm)
        compare_steps.append({
            "method": cm,
            "family": family,
            "task_type": task_type,
            "input_path": main_input_path,
            "out_dir": out_dir
        })
        compare_out_dirs.append(out_dir)

    step_model_compare = {
        "enabled": bool(compare_steps),
        "models": compare_steps
    }

    summary_output_path = AGENTS_DIR

    step_report = {
        "enabled": True,
        "input_dir_main": main_out_dir,
        "input_dirs_compare": compare_out_dirs,
        "output_path": summary_output_path
    }

    plan = {
        "dataset_name": dataset,
        "task_type": task_type,
        "step_preprocess": step_preprocess,
        "step_model_main": step_model_main,
        "step_model_compare": step_model_compare,
        "step_report": step_report
    }
    return plan

def main():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("实验规划交互已启动。\n")
    print("请直接描述你的实验需求，支持中文、English，或中英混合表达。")
    print("你可以说明以下信息：")
    print("1. 数据集：玉米 / 土壤 / 肉类，或 corn / soil / tecator")
    print("2. 预处理：SNV / SG-d1 / 不预处理，或 snv / savitzky_golay / none")
    print("3. 主模型：PLSR / EPSpec+PLSR / EPSpec+PLSR-sliding / iPLS+PLSR / CARS+PLSR")
    print("4. 是否需要对比其他模型，例如：对比基线、再加一个 ipls_plsr 等")
    print("")

    confirmed = False
    last_json_text = None

    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已中断对话。")
            return

        if user_input in {"确认", "confirm", "Confirm", "CONFIRM"}:
            confirmed = True
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        assistant_reply = call_llm(messages)
        print("智能体1：", assistant_reply)

        json_candidate = extract_last_json_block(assistant_reply)
        if json_candidate:
            last_json_text = json_candidate

        messages.append({"role": "assistant", "content": assistant_reply})

    if not confirmed:
        print("未确认，流程结束。")
        return

    if not last_json_text:
        print("未捕获到大模型输出的 JSON，请在确认前确保已经看到一份 JSON。")
        return

    print("\n===== 大模型输出的语义 JSON =====")
    print(last_json_text)

    try:
        semantic_cfg = json.loads(last_json_text)
    except Exception as e:
        print(f"\n解析语义 JSON 失败：{e}")
        return

    try:
        validate_semantic_config(semantic_cfg)
    except Exception as e:
        print(f"\n语义 JSON 校验失败：{e}")
        return

    try:
        final_plan = build_plan_from_semantic(semantic_cfg)
    except Exception as e:
        print(f"\n根据语义配置生成 plan 失败：{e}")
        return

    print("\n===== 最终实验计划 plan.json（包含每一步的输入/输出路径）=====")
    print(json.dumps(final_plan, ensure_ascii=False, indent=2))

    out_path = os.path.join(AGENTS_DIR, "plan.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_plan, f, ensure_ascii=False, indent=2)
        print(f"\n已将 plan 保存为: {out_path}")
    except Exception as e:
        print(f"\n写入 plan.json 失败：{e}")

if __name__ == "__main__":
    main()
