import os
import re
import json
from typing import Dict, List, Any
import traceback
import numpy as np
import pandas as pd
from openai import OpenAI

PRIOR_KB_PATH = r"your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\RAG_Prior knowledge\Data\Functional Group.xlsx"

API_KEY = "your key"

client = OpenAI(
    api_key=API_KEY,
    base_url="your URL",
    timeout=600,
    max_retries=0,
)

_GLOBAL_PRIOR_CACHE: Dict[str, str] = {}

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def _clean_json_like_text(text: str) -> str:
    if text is None:
        return ""
    s = text.strip()
    s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def call_llm_agent35(messages: List[Dict[str, str]]) -> str:
    try:
        resp = client.chat.completions.create(
            model="your model name",
            messages=messages,
            temperature=0,

            max_completion_tokens=64000,
            seed=0,
            response_format={"type": "text"},
            stream=False,
            reasoning_effort="high",

        )

    except Exception as e:

        cause = getattr(e, "__cause__", None)
        context = getattr(e, "__context__", None)

        detail = {
            "exc_type": type(e).__name__,
            "str": str(e),
            "repr": repr(e),
            "cause": repr(cause) if cause else None,
            "context": repr(context) if context else None,

            "status_code": getattr(e, "status_code", None),
            "body": getattr(e, "body", None),
            "traceback": traceback.format_exc(),
        }

        return "调用大模型失败（raw）:\n" + json.dumps(detail, ensure_ascii=False, indent=2)

    try:
        content = resp.choices[0].message.content
        if not isinstance(content, str):
            content = str(content)
    except Exception as e:
        return f"解析响应失败：{e}"

    return content

def call_llm(messages: List[Dict[str, str]]) -> str:
    return call_llm_agent35(messages)

def build_global_prior_prompt(
    dataset_col: str,
    predicted_substance: str,
    prior_df: pd.DataFrame,
) -> str:

    target = (predicted_substance or "").strip()

    base_cols = [
        "Functional Group",
        "Nanometers (nm)",
        "Wavenumber in (cm⁻¹)",
        "Spectra Structure",
        "Material Type",
        "range_nm",
        "component",
    ]
    cols_exist = [c for c in base_cols if c in prior_df.columns]
    final_cols = cols_exist + ([dataset_col] if dataset_col in prior_df.columns else [])

    lines = []

    lines.append(
        "你正在为一个近红外光谱（NIR）定量回归任务生成“全局先验知识”，"
        "它将被另一个智能体用于后续波段排序与选择。"
        "请把回答写得“可直接指导选段”，而不是泛泛科普。"
    )
    lines.append("\n【任务信息（必须在回答开头明确复述）】")
    lines.append(f"- 当前数据集名称: {dataset_col}（注意：这就是数据集名，不只是列名）")
    if target:
        lines.append(f"- 预测目标: {target}（定量回归其含量/指标）")
    else:
        lines.append(f"- 预测目标: （未提供，请根据表中信息尽量推断其可能对应的目标物质/指标，并说明不确定性）")

    lines.append(
        "\n【effect 字段解释（请严格按此理解并在回答中用到）】\n"
        f"- 表格最后一列“{dataset_col}”是该数据集任务下的 effect。\n"
        "- effect 可能是 + / - 以及数字的组合（例如 +2-1）。\n"
        "- “+”表示该条目对应的官能团/组分及其波段对预测目标通常提供正向信息（含信号、可用）。\n"
        "- “-”表示更可能是干扰/负影响/噪声来源（需谨慎或降低权重）。\n"
        "- 数字的绝对值越大，表示影响强度/证据强度越高。\n"
        "- 若出现“+2-1”这类混合符号，表示同一波段可能同时包含有用信息与干扰，需结合 component / Material Type 判断主导机制，并在结论里标为“谨慎/需验证”。"
    )

    lines.append(
        "\n【输出要求（必须全部满足）】\n"
        "1）必须用中文、连续自然段输出；不要使用项目符号/编号列表/表格/JSON/代码块。\n"
        "2）字数不少于 600 字，建议 4–6 段，每段 2–5 句。\n"
        "3）回答必须包含以下内容（都要写到）：\n"
        "   - 段落1：一句话概括“数据集是什么 + 预测什么含量/指标”（必须点名数据集与目标）。\n"
        "   - 段落2：把表中信息按 component / Material Type / Functional Group 归纳成一些“机制簇”，说明哪些是目标相关信号来源、哪些是基质/共线/水/散射等干扰来源（不要逐行复述）。\n"
        "   - 段落3：给出“高置信有用/正影响”的关键波段范围（nm），至少写出 5 个连续区间，并说明对应的官能团/组分与 effect 依据。\n"
        "   - 段落4：给出“低价值/负影响/高噪声风险”的波段范围（nm），至少写出 3 个连续区间，并说明原因（effect 为负、典型干扰、水峰/散射/仪器边缘等）。\n"
        "   - 段落5（可选但强烈建议）：写出“混合/不确定/需验证”的波段范围（nm）与原因（例如 + 与 - 同时出现、component 冲突、材料类型复杂）。\n"
        "4）波段描述优先使用 range_nm；若只有单点 Nanometers (nm)，请用“中心±若干 nm”合理扩展成一个连续区间再表达。\n"
        "5）最后用 1–2 句做行动性总结：明确告诉后续选段应“优先关注哪些区间、谨慎哪些区间、尽量避开哪些区间”。"
    )

    lines.append(
        "\n【筛选后的先验条目表】\n"
        "说明：该表已按当前数据集列筛选为“非空行”。请综合归纳，不要逐行复述。\n"
    )
    header = " | ".join(final_cols)
    lines.append(header)

    for _, row in prior_df.iterrows():
        vals = []
        for col in final_cols:
            v = row.get(col, "")
            if pd.isna(v):
                v = ""
            v = str(v).strip()
            vals.append(v if v else "")
        lines.append(" | ".join(vals))

    lines.append("\n请开始生成该任务的全局先验（按上述输出要求）。")

    return "\n".join(lines)

def call_ai_for_global_prior(message: str) -> str:
    system_prompt = (
    "你是一名熟悉近红外光谱机理与化学计量学的专家。"
    "你要基于给定的结构化先验表，为特定数据集的定量回归任务生成“可直接指导波段选择”的全局先验。"
    "必须显式复述数据集名称与预测目标，并且必须输出：正影响（有用）波段范围、负影响/噪声波段范围、以及必要时的不确定波段。"
    "只用中文、连续自然段输出；严禁项目符号/编号列表/表格/JSON/Markdown 代码块。"
)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]

    print("[GlobalPrior] 正在调用大模型接口生成全局先验描述...")
    content = call_llm(messages)
    print("[GlobalPrior] 大模型返回的全局先验描述如下：")
    print("------------------------------------------------------------")
    print(content)
    print("------------------------------------------------------------")
    return content

def get_global_prior_knowledge(dataset_key_norm: str, predicted_substance: str) -> str:
    global _GLOBAL_PRIOR_CACHE

    if dataset_key_norm in _GLOBAL_PRIOR_CACHE:
        return _GLOBAL_PRIOR_CACHE[dataset_key_norm]

    if not PRIOR_KB_PATH or not os.path.exists(PRIOR_KB_PATH):
        print(f"[GlobalPrior] 先验知识库 CSV 不存在或未配置: {PRIOR_KB_PATH}")
        _GLOBAL_PRIOR_CACHE[dataset_key_norm] = ""
        return ""

    dataset_col_map = {
    "cassav": "cassav",
    "cassava": "cassav",
    "gasoline": "gasoline",
    "tecator": "tecator",
    "soil": "soil",
    "shootout": "shootout",
    "corn": "corn",
}

    dataset_col = dataset_col_map.get(dataset_key_norm, dataset_key_norm)

    print("[GlobalPrior] ===== 全局先验生成开始 =====")
    print(f"[GlobalPrior] dataset_key_norm: {dataset_key_norm}")
    print(f"[GlobalPrior] 使用的任务列名 : {dataset_col}")
    print(f"[GlobalPrior] predicted_substance: {predicted_substance}")

    xls = pd.ExcelFile(PRIOR_KB_PATH)
    sheet_name = "Sheet1" if "Sheet1" in xls.sheet_names else xls.sheet_names[0]
    prior_df = pd.read_excel(PRIOR_KB_PATH, sheet_name=sheet_name)

    if dataset_col not in prior_df.columns:
        print(f"[GlobalPrior] 先验知识库中不存在列: {dataset_col}")
        _GLOBAL_PRIOR_CACHE[dataset_key_norm] = ""
        return ""

    col_series = prior_df[dataset_col]
    mask = col_series.notna() & (col_series.astype(str).str.strip() != "")
    prior_hit = prior_df[mask].copy()

    if prior_hit.empty:
        print(f"[GlobalPrior] 在列 {dataset_col} 下没有筛选到任何非空先验行。")
        _GLOBAL_PRIOR_CACHE[dataset_key_norm] = ""
        return ""

    print(f"[GlobalPrior] 在列 {dataset_col} 下筛选到 {len(prior_hit)} 条先验行。")

    prompt = build_global_prior_prompt(
        dataset_col=dataset_col,
        predicted_substance=predicted_substance,
        prior_df=prior_hit,
    )

    global_prior_text = call_ai_for_global_prior(prompt)

    _GLOBAL_PRIOR_CACHE[dataset_key_norm] = global_prior_text
    print("[GlobalPrior] ===== 全局先验生成结束 =====")
    return global_prior_text

def rank_intervals_with_llm(ep_result: Dict, outer_fold_id: int, ep_dir: str) -> Dict:
    ensure_dir(ep_dir)

    interval_content_pretty = json.dumps(ep_result, ensure_ascii=False, indent=2)

    system_prompt = r"""你是“智能体3.5：NIR 光谱波段排序专家（Evidence + Prior Guided）”。

我会给你一份 interval_metrics_xxx.json，它包含：
- info：数据集与切段信息、参数说明、字段解释等；
- global_prior_knowledge：针对该数据集/预测目标生成的“全局化学机理先验”（中文自然段）；
- intervals：一个数组，每个元素是一段光谱区间的指标（例如 start_nm, end_nm, local_r2/var_explained_delta 或 local_bal_acc/local_f1_macro、stability、cv_std、snr_band、artifact_risk、red_spec_max、red_target_max/red_pred_max 等）。

【你的真正目标（请按这个目标来排序）】
我们后续会按你的排序取“前 K 个区间的并集”去训练传统模型（如 PLSR/PLS-DA 等），并在不同 K 上遍历比较。
因此，这里的排序不是抽象的“重要性”，而是：
把“更可能提供可用信号、并且在多区间联合建模时更可能带来增益”的区间尽量排在前面；
把“主要是噪声/边缘伪像/不稳定/与主力区间高度冗余、对联合建模贡献小”的区间排在后面。
算是一种“联合建模信号优先级排序”，（前排=更值得进入 top-k 组合建模）

【如何综合判断（强调综合，不要机械按单一指标）】
你需要综合考虑以下维度（没有硬权重，按常识与数据表现综合）：
1）信号强度/预测性：
- 回归：local_r2、var_explained_delta、corr_max/corr_mean 等；
- 分类：local_bal_acc、local_f1_macro、class_fisher、anova_f_max 等。
2）稳定性/泛化可靠性：
- stability 或 stability_cls 高、cv_std 或 cv_std_cls 低，通常更可靠；
3）噪声与伪像风险：
- snr_band 低、artifact_risk 高，且预测性不强时，应明显后置；
- 处在谱段边缘且风险高的区间，通常更像伪像来源；
4）冗余与互补性（对“前K并集”尤为关键）：
- red_spec_max / red_target_max(red_pred_max) 很高时，说明与其它段高度相似；
  如果它自身预测性又不顶尖，则对联合建模的边际贡献更小，应后置；
  若自身预测性非常强，也可靠前，但理由中要说明“虽冗余但强、可作为主力之一”或“与主力互补不足所以稍靠后”等更细的判断。
5）全局先验 global_prior_knowledge（软约束，只作为参考）：
- 先验中提到的“可能包含目标官能团吸收/组合带”的波段范围，如果与该区间的数值证据（预测性、稳定性、低风险）一致，可作为加分理由；
- 如果先验强烈提示某些区域易受水峰/散射/基质干扰等，且该区间又表现出高风险/低稳定/低预测性，应作为后置理由；
- 当先验与数值指标冲突时：不要盲从先验。以 interval 指标为主，但在 reason 里用一句话说明“先验提示XX但本折证据显示YY，因此暂按YY排序”。

【输出的 reason 怎么写（避免模板化、避免绝对化）】
- 每个区间的 reason 请写 2–3 句中文（不要太长），尽量包含：
  （a）该段是否“含预测信息/信号”以及依据哪些指标；
  （b）该段是否可能“含噪声/不稳定/边缘伪像”以及依据哪些指标；
  （c）该段与其它段的“冗余/互补性”以及依据 red_* 指标；
  （d）是否与 global_prior_knowledge 的机理描述一致（如果 relevant 就提一句，不相关可不提）。
- 语气要留有余地：用“可能/倾向于/较可能/需要结合”而不是“必然/绝对”。
- 避免重复句式：不要所有段都写“指标高所以重要”。请根据每段的“短板/优势”写出差异化理由。

【必须输出的唯一格式】
你只能输出下面这种 JSON，不能多一个字、不能有注释、不能有 markdown 代码块：

{
  "interval_ranking": [
    {
      "id": "fold01_seg_000",
      "start": 1100.0,
      "end": 1120.0,
      "rank": 1,
      "importance_level": "strong",
      "reason": "..."
    }
  ]
}

严格要求：
1. 顶层只能有一个键：interval_ranking。
2. interval_ranking 的值必须是一个数组，数组长度必须等于输入 intervals 的长度。
3. 数组中每个元素必须有并且只能有下面这 6 个键：
   - id：优先使用该段原有的 interval_id / id / name。如果原始数据里没有，就用 "band_1"、"band_2"……。
   - start：优先用 start_nm；如果没有就用 start；都没有就写 null。
   - end：优先用 end_nm；如果没有就用 end；都没有就写 null。
   - rank：该段的排序名次，从 1 开始，1 表示最推荐优先进入“前K并集建模”的区间。rank 必须是 1,2,3,... 连续整数。
   - importance_level：仍使用 "strong" / "medium" / "weak"（语义改为：strong=高可用信号且更可能带来建模增益；medium=中等信号或偏互补；weak=低信号/高风险/高冗余导致边际贡献小）。
   - reason：2–3 句中文，按上面要求写。

4. 数组必须按 rank 从小到大排序（rank=1 的元素排在数组第一个）。

不要输出其他任何内容，只输出这个 JSON。
"""

    user_msg = (
    "下面是某一外层折训练集的 interval_metrics JSON（其中包含 global_prior_knowledge，作为软参考）。"
    "请按 system 指令输出所有区间的排序结果（只能输出 JSON）：\n"
    f"{interval_content_pretty}"
)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    print(f"[LLM-Rank] 调用大模型对 Outer fold {outer_fold_id} 的区间做排序...")
    content = call_llm(messages)
    content_clean = _clean_json_like_text(content)

    rank_path = os.path.join(ep_dir, f"interval_ranking_outerfold{outer_fold_id}.json")
    with open(rank_path, "w", encoding="utf-8") as f:
        f.write(content_clean)
    print(f"[LLM-Rank] Outer fold {outer_fold_id} 排序结果已写入: {rank_path}")

    try:
        ranking_obj = json.loads(content_clean)
    except Exception as e:
        print(f"[LLM-Rank] 解析排序 JSON 失败：{e}")
        ranking_obj = {"interval_ranking": []}

    return ranking_obj

def apply_global_prior_and_rank_intervals(
    ep_result: Dict,
    dataset_key: str,
    outer_fold_id: int,
    out_dir: str,
) -> Dict:
    ensure_dir(out_dir)

    info = ep_result.get("info", {})
    predicted_substance = info.get("predicted_substance", "")

    dataset_key_norm = dataset_key.lower().replace("_preprocessed", "")

    global_prior_text = get_global_prior_knowledge(
        dataset_key_norm=dataset_key_norm,
        predicted_substance=predicted_substance,
    )
    ep_result["global_prior_knowledge"] = global_prior_text

    json_name = f"{dataset_key}_interval_metrics_outerfold{outer_fold_id:02d}.json"
    json_path = os.path.join(out_dir, json_name)
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(ep_result, f, indent=2, ensure_ascii=False)
        print(f"[GlobalPrior] 已将带全局先验的 EP JSON 写回: {json_path}")
    except Exception as e:
        print(f"[GlobalPrior] 写回 EP JSON 失败（不影响排序）：{e}")

    ranking_obj = rank_intervals_with_llm(
        ep_result=ep_result,
        outer_fold_id=outer_fold_id,
        ep_dir=out_dir,
    )

    return ranking_obj

if __name__ == "__main__":
    print("This module is intended to be imported by EPSpec.")
