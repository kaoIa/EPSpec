import os
import sys
import re
import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
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

def call_llm(messages: List[Dict[str, str]]) -> Tuple[str, str]:
    if not API_KEY:
        return "调用大模型失败：未检测到 API_KEY。", ""

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
        return f"调用大模型失败：{e}", ""

    try:
        raw_json_str = response.model_dump_json(indent=2)
    except Exception:
        raw_json_str = str(response)

    try:
        content = response.choices[0].message.content
        if content is None:
            return "API 响应格式错误：message.content 为空", raw_json_str
        return str(content), raw_json_str
    except Exception as e:
        return f"解析响应失败：{e}", raw_json_str

SUMMARY_SYSTEM_PROMPT = r"""
你是“智能体3：近红外光谱结果总结与解释专家”。

你将收到一份结构化实验摘要。该摘要通常包含三部分信息：

第一部分是任务说明，包括：
- 当前数据集名称；
- 当前预测目标物质/指标；
- 是否使用预处理、使用了什么预处理；
- 当前任务级别的全局先验知识。

第二部分是主模型结果，包括：
- 五折综合数值结果；
- 五折逐折指标；
- 该模型的选段/选变量结果。
其中：
- 对于 EPSpec 类模型，会给出每一折最优 best_k 对应的 Top-k 区间；
- 对于 PLSR，会明确说明其为全谱建模；
- 对于 iPLS 和 CARS，会给出每一折最终选中的波长范围或点段。

第三部分是对比模型结果，结构与主模型类似。

你的任务是：
基于这些结构化信息，输出一篇中文 Markdown 报告，可直接用于论文或实验报告中的“结果与讨论”。

写作要求：
1. 语言风格为科研论文式中文，避免口语化。
2. 只基于输入信息写作，不要编造不存在的实验、数值或结论。
3. 如果某些信息缺失，请明确说明“当前结果中未提供……”。
4. 开头要先准确说明当前任务：数据集、预测目标、预处理方式。
5. 若提供了任务级先验，要讨论其与最终选段之间的一致、邻近或偏离关系。
6. 当存在主模型与对比模型时，要明确比较它们的性能差异与方法差异。
7. 要结合逐折选段结果讨论跨折一致性、稳定性与可解释性。
8. 对于 PLSR，要明确其为全谱基线，不做显式选段。
9. 不要出现“你”“我”“大模型”这类元表述。

输出结构建议：
1. 实验任务与配置
2. 主模型结果概述
3. 主模型与对比模型的性能比较
4. 波段选择结果与可解释性分析
5. 方法讨论与局限性
6. 小结

输出要求：
- 只输出 Markdown 正文
- 不要输出 JSON
- 尽量写得完整、自然、可直接粘贴使用
"""

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def strip_md_fences(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()

def shorten_text(text: str, max_chars: int = 2200) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[已截断]"

def parse_fold_id_from_name(name: str) -> Optional[int]:
    m = re.search(r"outerfold0*([0-9]+)", name, flags=re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"fold0*([0-9]+)", name, flags=re.I)
    if m:
        return int(m.group(1))
    return None

def safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def find_file(root_dir: str, target_filename: str) -> Optional[str]:
    if not root_dir or not os.path.isdir(root_dir):
        return None

    direct_path = os.path.join(root_dir, target_filename)
    if os.path.isfile(direct_path):
        return direct_path

    for dirpath, _, filenames in os.walk(root_dir):
        if target_filename in filenames:
            return os.path.join(dirpath, target_filename)
    return None

def find_files_by_regex(root_dir: str, pattern: str) -> List[str]:
    if not root_dir or not os.path.isdir(root_dir):
        return []

    rx = re.compile(pattern, flags=re.I)
    hits = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if rx.search(fname):
                hits.append(os.path.join(dirpath, fname))
    return sorted(hits)

def dataset_to_target_name(dataset_name: str) -> str:
    mapping = {
        "corn": "starch",
        "soil": "soil organic matter",
        "tecator": "fat",
        "shootout": "active pharmaceutical ingredient",
        "diesel": "cetane value",
        "cassava": "β-carotene",
        "cassav": "β-carotene",
        "gasoline": "octane number",
    }
    return mapping.get((dataset_name or "").lower(), "")

def parse_feature_value(v: Any) -> Optional[float]:
    s = str(v).strip()
    if s == "" or s.lower() == "intercept":
        return None
    try:
        return float(s)
    except Exception:
        m = re.search(r"(\d+(?:\.\d+)?)", s)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None

def infer_step(values: List[float]) -> float:
    if len(values) < 2:
        return 1.0
    diffs = []
    for i in range(len(values) - 1):
        d = values[i + 1] - values[i]
        if d > 0:
            diffs.append(round(d, 10))
    if not diffs:
        return 1.0
    return float(sorted(set(diffs))[0])

def merge_values_to_ranges(values: List[float]) -> List[Dict[str, Any]]:
    if not values:
        return []

    vals = sorted(set(float(v) for v in values))
    step = infer_step(vals)
    tol = step * 1.5 + 1e-9

    ranges = []
    start = vals[0]
    prev = vals[0]
    current = [vals[0]]

    for v in vals[1:]:
        if (v - prev) <= tol:
            current.append(v)
            prev = v
        else:
            ranges.append({
                "start_nm": float(start),
                "end_nm": float(prev),
                "n_points": int(len(current)),
            })
            start = v
            prev = v
            current = [v]

    ranges.append({
        "start_nm": float(start),
        "end_nm": float(prev),
        "n_points": int(len(current)),
    })
    return ranges

def summarize_numeric_columns(df: pd.DataFrame, metric_names: List[str]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for col in metric_names:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if not s.empty:
                summary[col] = {
                    "mean": float(s.mean()),
                    "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                }
    return summary

def build_metrics_records(metrics_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if not isinstance(metrics_df, pd.DataFrame) or metrics_df.empty:
        return []

    preferred = [
        "fold", "outer_fold", "pc", "best_k", "n_features", "n_selected_features",
        "R2", "RMSE", "MAE", "Bias", "RPD", "RPIQ"
    ]
    cols = [c for c in preferred if c in metrics_df.columns]
    if not cols:
        cols = list(metrics_df.columns)

    records: List[Dict[str, Any]] = []
    for row in metrics_df[cols].to_dict(orient="records"):
        clean_row = {}
        for k, v in row.items():
            try:
                if pd.isna(v):
                    clean_row[k] = None
                elif hasattr(v, "item"):
                    clean_row[k] = v.item()
                else:
                    clean_row[k] = v
            except Exception:
                clean_row[k] = v
        records.append(clean_row)
    return records

def normalize_summary_object(summary_obj: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "metrics_summary": {},
        "pc_per_fold": None,
        "best_k_per_fold": None,
        "n_features_per_fold": None,
        "n_selected_features_per_fold": None,
        "ipls_interval_width": None,
        "cars_n_mc": None,
        "cars_ratio_samples": None,
    }

    if not isinstance(summary_obj, dict):
        return result

    metric_names = ["R2", "RMSE", "MAE", "Bias", "RPD", "RPIQ"]
    for m in metric_names:
        v = summary_obj.get(m)
        if isinstance(v, dict) and "mean" in v and "std" in v:
            try:
                result["metrics_summary"][m] = {
                    "mean": float(v["mean"]),
                    "std": float(v["std"]),
                }
            except Exception:
                pass

    if isinstance(summary_obj.get("pc"), dict) and isinstance(summary_obj["pc"].get("per_fold"), list):
        result["pc_per_fold"] = [int(x) for x in summary_obj["pc"]["per_fold"]]
    elif isinstance(summary_obj.get("pc_per_fold"), list):
        result["pc_per_fold"] = [int(x) for x in summary_obj["pc_per_fold"]]

    if isinstance(summary_obj.get("best_k"), dict) and isinstance(summary_obj["best_k"].get("per_fold"), list):
        result["best_k_per_fold"] = [int(x) for x in summary_obj["best_k"]["per_fold"]]
    elif isinstance(summary_obj.get("best_k_per_fold"), list):
        result["best_k_per_fold"] = [int(x) for x in summary_obj["best_k_per_fold"]]

    if isinstance(summary_obj.get("n_features"), dict) and isinstance(summary_obj["n_features"].get("per_fold"), list):
        result["n_features_per_fold"] = [int(x) for x in summary_obj["n_features"]["per_fold"]]
    elif isinstance(summary_obj.get("n_features_per_fold"), list):
        result["n_features_per_fold"] = [int(x) for x in summary_obj["n_features_per_fold"]]

    if isinstance(summary_obj.get("n_selected_features_per_fold"), list):
        result["n_selected_features_per_fold"] = [int(x) for x in summary_obj["n_selected_features_per_fold"]]

    if summary_obj.get("ipls_interval_width") is not None:
        try:
            result["ipls_interval_width"] = int(summary_obj["ipls_interval_width"])
        except Exception:
            pass

    if summary_obj.get("cars_n_mc") is not None:
        try:
            result["cars_n_mc"] = int(summary_obj["cars_n_mc"])
        except Exception:
            pass

    if summary_obj.get("cars_ratio_samples") is not None:
        try:
            result["cars_ratio_samples"] = float(summary_obj["cars_ratio_samples"])
        except Exception:
            pass

    return result

def build_summary_from_metrics_csv(metrics_df: pd.DataFrame) -> Dict[str, Any]:
    out = {
        "metrics_summary": summarize_numeric_columns(
            metrics_df,
            ["R2", "RMSE", "MAE", "Bias", "RPD", "RPIQ"]
        ),
        "pc_per_fold": None,
        "best_k_per_fold": None,
        "n_features_per_fold": None,
        "n_selected_features_per_fold": None,
        "ipls_interval_width": None,
        "cars_n_mc": None,
        "cars_ratio_samples": None,
    }

    if "pc" in metrics_df.columns:
        s = pd.to_numeric(metrics_df["pc"], errors="coerce").dropna()
        if not s.empty:
            out["pc_per_fold"] = [int(x) for x in s.tolist()]

    if "best_k" in metrics_df.columns:
        s = pd.to_numeric(metrics_df["best_k"], errors="coerce").dropna()
        if not s.empty:
            out["best_k_per_fold"] = [int(x) for x in s.tolist()]

    if "n_features" in metrics_df.columns:
        s = pd.to_numeric(metrics_df["n_features"], errors="coerce").dropna()
        if not s.empty:
            out["n_features_per_fold"] = [int(x) for x in s.tolist()]

    if "n_selected_features" in metrics_df.columns:
        s = pd.to_numeric(metrics_df["n_selected_features"], errors="coerce").dropna()
        if not s.empty:
            out["n_selected_features_per_fold"] = [int(x) for x in s.tolist()]

    return out

def extract_basic_result_summary(result_dir: str) -> Dict[str, Any]:
    summary_path = find_file(result_dir, "summary.json")
    metrics_path = find_file(result_dir, "metrics_per_fold.csv")

    summary_obj = safe_read_json(summary_path) if summary_path else None
    metrics_df = safe_read_csv(metrics_path) if metrics_path else None

    if isinstance(summary_obj, dict):
        normalized = normalize_summary_object(summary_obj)
    elif isinstance(metrics_df, pd.DataFrame):
        normalized = build_summary_from_metrics_csv(metrics_df)
    else:
        normalized = {
            "metrics_summary": {},
            "pc_per_fold": None,
            "best_k_per_fold": None,
            "n_features_per_fold": None,
            "n_selected_features_per_fold": None,
            "ipls_interval_width": None,
            "cars_n_mc": None,
            "cars_ratio_samples": None,
        }

    normalized["metrics_per_fold_records"] = build_metrics_records(metrics_df) if isinstance(metrics_df, pd.DataFrame) else []
    return normalized

def extract_task_context_from_epspec_dir(result_dir: str) -> Dict[str, Any]:
    interval_metric_files = find_files_by_regex(result_dir, r"interval_metrics_outerfold.*\.json$")
    if not interval_metric_files:
        return {
            "dataset_key": None,
            "predicted_substance": None,
            "n_intervals": None,
            "global_prior_knowledge": None,
        }

    interval_metric_files = sorted(interval_metric_files)
    for path in interval_metric_files:
        obj = safe_read_json(path)
        if not isinstance(obj, dict):
            continue

        info = obj.get("info", {}) if isinstance(obj.get("info"), dict) else {}
        gp = (obj.get("global_prior_knowledge") or "").strip()

        return {
            "dataset_key": info.get("dataset_key"),
            "predicted_substance": info.get("predicted_substance"),
            "n_intervals": info.get("n_intervals"),
            "global_prior_knowledge": shorten_text(gp, max_chars=2200) if gp else None,
        }

    return {
        "dataset_key": None,
        "predicted_substance": None,
        "n_intervals": None,
        "global_prior_knowledge": None,
    }

def extract_epspec_selected_topk_per_fold(result_dir: str, summary_info: Dict[str, Any]) -> Dict[str, Any]:
    interval_metric_files = find_files_by_regex(result_dir, r"interval_metrics_outerfold.*\.json$")
    ranking_files = find_files_by_regex(result_dir, r"interval_ranking_outerfold(?!.*_nms).*\.json$")

    metrics_by_fold: Dict[int, Dict[str, Any]] = {}
    rankings_by_fold: Dict[int, Dict[str, Any]] = {}

    for path in interval_metric_files:
        fold_id = parse_fold_id_from_name(os.path.basename(path))
        obj = safe_read_json(path)
        if fold_id is not None and isinstance(obj, dict):
            metrics_by_fold[fold_id] = obj

    for path in ranking_files:
        fold_id = parse_fold_id_from_name(os.path.basename(path))
        obj = safe_read_json(path)
        if fold_id is not None and isinstance(obj, dict):
            rankings_by_fold[fold_id] = obj

    best_k_list = summary_info.get("best_k_per_fold")
    best_k_map = {idx + 1: int(v) for idx, v in enumerate(best_k_list)} if isinstance(best_k_list, list) else {}

    per_fold_selected = []
    consensus_map: Dict[str, Dict[str, Any]] = {}

    all_folds = sorted(set(metrics_by_fold.keys()) | set(rankings_by_fold.keys()))
    for fold_id in all_folds:
        metric_obj = metrics_by_fold.get(fold_id, {})
        ranking_obj = rankings_by_fold.get(fold_id, {})

        interval_recs = metric_obj.get("intervals", []) if isinstance(metric_obj, dict) else []
        metric_index = {}
        for rec in interval_recs:
            iid = rec.get("interval_id") or rec.get("id")
            if iid is not None:
                metric_index[str(iid)] = rec

        ranking_list = ranking_obj.get("interval_ranking", []) if isinstance(ranking_obj, dict) else []
        ranking_list = sorted(ranking_list, key=lambda x: float(x.get("rank", 1e9)))

        if not ranking_list and interval_recs:
            ranking_list = [
                {
                    "id": rec.get("interval_id"),
                    "start": rec.get("start_nm"),
                    "end": rec.get("end_nm"),
                    "rank": i + 1,
                    "importance_level": "fallback",
                    "reason": "未读取到显式排序文件，使用 local_r2 作为兜底排序。"
                }
                for i, rec in enumerate(
                    sorted(interval_recs, key=lambda x: float(x.get("local_r2", 0.0)), reverse=True)
                )
            ]

        best_k = int(best_k_map.get(fold_id, 0)) if best_k_map else 0
        if best_k <= 0:
            best_k = min(5, len(ranking_list))

        selected_items = ranking_list[:best_k]

        fold_payload = {
            "fold": int(fold_id),
            "best_k": int(best_k),
            "selected_topk_intervals": []
        }

        for item in selected_items:
            iid = str(item.get("id"))
            metric_rec = metric_index.get(iid, {})
            rec = {
                "interval_id": item.get("id"),
                "start_nm": item.get("start"),
                "end_nm": item.get("end"),
                "rank": item.get("rank"),
                "importance_level": item.get("importance_level"),
                "reason": item.get("reason"),
            }
            for k in ["local_r2", "stability", "snr_band", "artifact_risk", "corr_max", "corr_mean"]:
                if isinstance(metric_rec, dict) and k in metric_rec:
                    rec[k] = metric_rec.get(k)

            fold_payload["selected_topk_intervals"].append(rec)

            try:
                key = f"{float(item.get('start')):.6f}-{float(item.get('end')):.6f}"
            except Exception:
                continue

            if key not in consensus_map:
                consensus_map[key] = {
                    "start_nm": float(item.get("start")),
                    "end_nm": float(item.get("end")),
                    "selected_in_folds": [],
                    "ranks": [],
                }

            consensus_map[key]["selected_in_folds"].append(int(fold_id))
            if item.get("rank") is not None:
                try:
                    consensus_map[key]["ranks"].append(float(item.get("rank")))
                except Exception:
                    pass

        per_fold_selected.append(fold_payload)

    consensus_selected = []
    for _, agg in consensus_map.items():
        consensus_selected.append({
            "start_nm": agg["start_nm"],
            "end_nm": agg["end_nm"],
            "selected_frequency": int(len(agg["selected_in_folds"])),
            "avg_rank": round(sum(agg["ranks"]) / len(agg["ranks"]), 4) if agg["ranks"] else None,
        })

    consensus_selected = sorted(
        consensus_selected,
        key=lambda x: (-x["selected_frequency"], x["avg_rank"] if x["avg_rank"] is not None else 1e9, x["start_nm"])
    )

    task_ctx = extract_task_context_from_epspec_dir(result_dir)

    return {
        "selection_type": "epspec_topk_intervals",
        "dataset_key": task_ctx.get("dataset_key"),
        "predicted_substance": task_ctx.get("predicted_substance"),
        "n_intervals": task_ctx.get("n_intervals"),
        "best_k_per_fold": best_k_list,
        "per_fold_selected_topk_intervals": per_fold_selected,
        "consensus_selected_intervals": consensus_selected[:12],
    }

def extract_selected_ranges_from_coefficients(result_dir: str, method: str, summary_info: Dict[str, Any]) -> Dict[str, Any]:
    coefs_path = find_file(result_dir, "coefficients.csv")
    coefs_df = safe_read_csv(coefs_path) if coefs_path else None

    if method == "plsr":
        return {
            "selection_type": "full_spectrum",
            "note": "该模型为全谱 PLSR，不进行显式选段。",
        }

    if not isinstance(coefs_df, pd.DataFrame) or coefs_df.empty:
        return {
            "selection_type": "unknown",
            "note": "当前结果中未读取到 coefficients.csv，无法恢复逐折选段。",
        }

    if "fold" not in coefs_df.columns or "feature" not in coefs_df.columns:
        return {
            "selection_type": "unknown",
            "note": "coefficients.csv 缺少 fold/feature 字段，无法恢复逐折选段。",
        }

    df = coefs_df.copy()
    df["feature_parsed"] = df["feature"].map(parse_feature_value)
    df = df[df["feature_parsed"].notna()].copy()

    if df.empty:
        return {
            "selection_type": "unknown",
            "note": "coefficients.csv 中未解析出有效波长变量。",
        }

    per_fold_selected = []
    for fold_id, sub in df.groupby("fold"):
        try:
            fold_int = int(fold_id)
        except Exception:
            continue

        selected_values = sorted(set(float(v) for v in sub["feature_parsed"].tolist()))
        per_fold_selected.append({
            "fold": int(fold_int),
            "n_selected_features": int(len(selected_values)),
            "selected_ranges": merge_values_to_ranges(selected_values),
        })

    per_fold_selected = sorted(per_fold_selected, key=lambda x: x["fold"])

    all_values = sorted(set(float(v) for v in df["feature_parsed"].tolist()))
    consensus_ranges = merge_values_to_ranges(all_values)

    payload = {
        "selection_type": "selected_ranges_from_coefficients",
        "per_fold_selected_ranges": per_fold_selected,
        "consensus_selected_ranges": consensus_ranges,
    }

    if method == "ipls_plsr":
        payload["ipls_interval_width"] = summary_info.get("ipls_interval_width")
    if method == "cars_plsr":
        payload["cars_n_mc"] = summary_info.get("cars_n_mc")
        payload["cars_ratio_samples"] = summary_info.get("cars_ratio_samples")

    return payload

DATASET_DESC = {
    "corn": "corn（玉米，当前任务通常预测淀粉等理化指标）",
    "soil": "soil（土壤样品，当前任务通常预测土壤有机质等指标）",
    "tecator": "tecator（肉糜样品，当前任务通常预测脂肪等理化指标）",
}

def describe_model(method: str, family: str, task_type: str) -> str:
    task_cn = "回归模型" if task_type == "regression" else "模型"

    if family.startswith("wavelength_selection_"):
        if method == "EPSpec_plsr":
            return f"EPSpec 等分区间版 {task_cn}（基于区间证据、先验检索与排序进行选段，再用 PLSR 建模）"
        if method == "EPSpec_plsr_sliding":
            return f"EPSpec 滑动窗口版 {task_cn}（采用滑动窗口候选区间并结合区间证据、先验检索与排序进行选段，再用 PLSR 建模）"
        return f"波长选择类 {task_cn}"

    if family.startswith("ipls_cars_"):
        if method == "ipls_plsr":
            return f"iPLS + PLSR {task_cn}（基于区间搜索进行变量筛选后建模）"
        if method == "cars_plsr":
            return f"CARS + PLSR {task_cn}（基于竞争性自适应重加权采样进行变量筛选后建模）"
        return f"变量筛选类 {task_cn}"

    if family.startswith("baseline_regression"):
        return "全谱 PLSR 基线模型（不做显式波段选择，直接在完整光谱上建模）"

    return f"其它类型的{task_cn}（family={family}）"

def build_model_result_block(
    role_name: str,
    method: str,
    family: str,
    task_type: str,
    result_dir: str,
) -> Dict[str, Any]:
    core_summary = extract_basic_result_summary(result_dir)

    if method in {"EPSpec_plsr", "EPSpec_plsr_sliding"}:
        selection_details = extract_epspec_selected_topk_per_fold(result_dir, core_summary)
        epspec_task_context = extract_task_context_from_epspec_dir(result_dir)
    else:
        selection_details = extract_selected_ranges_from_coefficients(result_dir, method, core_summary)
        epspec_task_context = None

    return {
        "role": role_name,
        "method": method,
        "family": family,
        "task_type": task_type,
        "description": describe_model(method, family, task_type),
        "five_fold_result_summary": {
            "metrics_summary": core_summary.get("metrics_summary"),
            "pc_per_fold": core_summary.get("pc_per_fold"),
            "best_k_per_fold": core_summary.get("best_k_per_fold"),
            "n_features_per_fold": core_summary.get("n_features_per_fold"),
            "n_selected_features_per_fold": core_summary.get("n_selected_features_per_fold"),
        },
        "five_fold_metrics_records": core_summary.get("metrics_per_fold_records"),
        "selection_details": selection_details,
        "task_context_from_result": epspec_task_context,
    }

def choose_task_level_prior(
    dataset_name: str,
    main_block: Dict[str, Any],
    compare_blocks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    candidate_blocks = [main_block] + compare_blocks

    for block in candidate_blocks:
        ctx = block.get("task_context_from_result")
        if isinstance(ctx, dict) and ctx.get("global_prior_knowledge"):
            predicted_substance = ctx.get("predicted_substance") or dataset_to_target_name(dataset_name)
            return {
                "available": True,
                "source_model": block.get("method"),
                "dataset_key": ctx.get("dataset_key") or dataset_name,
                "predicted_substance": predicted_substance,
                "n_intervals": ctx.get("n_intervals"),
                "global_prior_knowledge": ctx.get("global_prior_knowledge"),
            }

    return {
        "available": False,
        "source_model": None,
        "dataset_key": dataset_name,
        "predicted_substance": dataset_to_target_name(dataset_name),
        "n_intervals": None,
        "global_prior_knowledge": None,
    }

def build_prompt_payload(plan: Dict[str, Any]) -> Dict[str, Any]:
    step_report = plan.get("step_report") or {}
    main_dir = step_report.get("input_dir_main") or (plan.get("step_model_main") or {}).get("out_dir")
    compare_dirs = step_report.get("input_dirs_compare") or []
    if not isinstance(compare_dirs, list):
        compare_dirs = []

    dataset_name = plan.get("dataset_name", "")
    task_type = plan.get("task_type", "")
    step_preprocess = plan.get("step_preprocess") or {}

    main_cfg = plan.get("step_model_main") or {}
    main_block = build_model_result_block(
        role_name="main_model",
        method=main_cfg.get("method", ""),
        family=main_cfg.get("family", ""),
        task_type=main_cfg.get("task_type", task_type),
        result_dir=main_dir,
    )

    raw_compare_cfg = plan.get("step_model_compare")
    compare_model_cfgs = []
    if isinstance(raw_compare_cfg, dict):
        if raw_compare_cfg.get("enabled") and isinstance(raw_compare_cfg.get("models"), list):
            compare_model_cfgs = raw_compare_cfg.get("models") or []
    elif isinstance(raw_compare_cfg, list):
        compare_model_cfgs = raw_compare_cfg or []

    compare_blocks = []
    for idx, cfg in enumerate(compare_model_cfgs):
        c_dir = compare_dirs[idx] if idx < len(compare_dirs) else cfg.get("out_dir")
        compare_blocks.append(
            build_model_result_block(
                role_name=f"compare_model_{idx + 1}",
                method=cfg.get("method", ""),
                family=cfg.get("family", ""),
                task_type=cfg.get("task_type", task_type),
                result_dir=c_dir,
            )
        )

    task_level_prior = choose_task_level_prior(dataset_name, main_block, compare_blocks)
    predicted_substance = task_level_prior.get("predicted_substance") or dataset_to_target_name(dataset_name)

    task_statement = {
        "dataset_name": dataset_name,
        "dataset_description": DATASET_DESC.get(dataset_name, dataset_name),
        "predicted_substance": predicted_substance,
        "task_type": task_type,
        "preprocess": {
            "enabled": bool(step_preprocess.get("enabled", False)),
            "method": step_preprocess.get("method"),
        },
        "main_model": main_cfg.get("method"),
        "compare_models": [b.get("method") for b in compare_blocks],
    }

    return {
        "task_overview": task_statement,
        "task_level_prior": task_level_prior,
        "main_model_result": main_block,
        "compare_model_results": compare_blocks,
    }

def build_response_json(raw_json_str: str, report_raw: str, report_md: str) -> Dict[str, Any]:
    response_obj: Dict[str, Any] = {
        "report_raw": report_raw,
        "report_md": report_md,
        "raw_json_str": raw_json_str,
    }

    if raw_json_str:
        try:
            parsed = json.loads(raw_json_str)
            response_obj["raw_json_parsed"] = parsed

            choices = parsed.get("choices", [])
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message", {}) or {}
                content = msg.get("content") or ""
                response_obj["response_lines"] = {
                    "role": msg.get("role"),
                    "model": parsed.get("model"),
                    "content": content,
                    "content_lines": content.splitlines(),
                }
        except Exception:
            pass

    return response_obj

def main():
    agents_dir = os.path.dirname(os.path.abspath(__file__))
    plan_path = os.path.join(agents_dir, "plan.json")

    if not os.path.isfile(plan_path):
        print(f"未找到 plan.json：{plan_path}")
        sys.exit(1)

    try:
        with open(plan_path, "r", encoding="utf-8") as f:
            plan = json.load(f)
    except Exception as e:
        print(f"读取 plan.json 失败：{e}")
        sys.exit(1)

    step_report = plan.get("step_report") or {}
    if not step_report.get("enabled", False):
        print("step_report 未启用，跳过总结生成。")
        return

    output_root = step_report.get("output_path") or agents_dir
    if not os.path.isabs(output_root):
        output_root = os.path.abspath(os.path.join(agents_dir, output_root))
    ensure_dir(output_root)

    print("开始生成总结报告...")

    prompt_payload = build_prompt_payload(plan)

    user_content = (
        "下面给出一份结构化实验摘要。请基于这些摘要写中文 Markdown 报告，不要编造不存在的信息。\n\n"
        + json.dumps(prompt_payload, ensure_ascii=False, indent=2)
    )

    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    send_path = os.path.join(output_root, "send.json")
    try:
        with open(send_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "messages": messages,
                    "prompt_payload": prompt_payload,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:
        print(f"写入 send.json 失败：{e}")

    report_raw, raw_json_str = call_llm(messages)
    report_md = strip_md_fences(report_raw) if report_raw else ""

    report_path = os.path.join(output_root, "summary_report.md")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
    except Exception as e:
        print(f"写入 summary_report.md 失败：{e}")
        sys.exit(1)

    response_path = os.path.join(output_root, "response.json")
    try:
        response_obj = build_response_json(raw_json_str, report_raw, report_md)
        with open(response_path, "w", encoding="utf-8") as f:
            json.dump(response_obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"写入 response.json 失败：{e}")

    print(f"总结报告已生成：{report_path}")

if __name__ == "__main__":
    main()
