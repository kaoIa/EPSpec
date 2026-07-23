import requests
import pandas as pd
import json
import time
from typing import List, Dict, Any

CSV_PATH = r"your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\RAG_Prior knowledge\nir_spectroscopy_verified_sources.csv"

MATERIAL_PATH = r"your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\RAG_Prior knowledge\Material.xlsx"

API_URL = "your URL"
API_KEY = "your key"
MODEL_NAME = "your model name"

START_ROW = 28

SKIP_NONEMPTY = True

SAVE_EVERY_N_ROWS = 1

MAX_RETRY = 2

DEBUG_PRINT_PROMPT = True

def call_ai(message: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    data = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "你是一名熟悉近红外光谱和化学成分分类的专家助手。"
            },
            {"role": "user", "content": message},
        ],
        "temperature": 0,
        "stream": False,
    }

    response = requests.post(API_URL, headers=headers, json=data, timeout=360)

    print(f"[HTTP] status={response.status_code}")
    print(f"[HTTP] raw response: {response.text[:500]}...")

    if response.status_code != 200:
        raise RuntimeError(f"请求失败: {response.status_code} - {response.text}")

    result = response.json()
    choices = result.get("choices", [])
    if not choices:
        raise RuntimeError("API响应格式错误：choices 为空")

    content = choices[0]["message"]["content"]
    return content

def build_component_catalog_text(material_df: pd.DataFrame) -> str:
    required_cols = ["component_name", "definition_en"]
    for col in required_cols:
        if col not in material_df.columns:
            raise ValueError(f"Material.xlsx 缺少必要列: {col}")

    lines = []
    for _, row in material_df.iterrows():
        name = str(row["component_name"]).strip()
        definition = str(row["definition_en"]).strip()
        if not name:
            continue

        lines.append(f"- {name}: {definition}")

    catalog_text = "\n".join(lines)
    return catalog_text

def build_prompt_for_row(
    row_dict: Dict[str, Any],
    component_catalog_text: str,
) -> str:
    row_json = json.dumps(row_dict, ensure_ascii=False, indent=2)

    prompt = f"""
你是一名熟悉近红外光谱和化学机理的专家助手，任务是为一条“先验知识行”打上成分类别标签。

下面是你可以选择的成分类别（component_name 以及对应的英文定义）：
{component_catalog_text}

请你只做一件事：
根据这条先验知识行的内容，判断它主要涉及哪些成分类别（0~3 个），从上面列出的 component_name 中选择。

请注意，这里“成分类别”指的是这一行**主要描述的物质或物质家族**，而不是所有可能出现过的官能团。

具体要求：
1. 只能使用上面给出的 component_name 的值，不能自己发明新类别。
2. 请综合考虑 section、category、subcategory、example_compounds、notes 等字段，
   按照化学和光谱机理，选择最可能的 0~3 个成分类别。
3. 大多数行只会对应 0 个或 1 个成分类别。
   - 如果你认为“没有哪个类别明显比其他更合适”，请直接选择 ["none"]。
   - 不要为了“凑个数”而勉强选择，也不要经过联想选择，必须是从给你的先验知识中有描述或者合理推理得出的。
4. 只有在你认为这一行确实在同时描述多种成分，能看出这种复合特征时，才可以选择 2~3 个类别。
    也就是只有在你非常有把握时，才选择 2~3 个类别；
5. 不要仅仅因为出现了常见官能团（比如 O–H、C–H、N–H 等）就归入某个具体类别。
   例如：
   - 如果只是写“primary alcohols, secondary alcohols”，但没有任何关于淀粉/糖/纤维素/蛋白/脂肪等的提示，
     就不应该归为 starch_polysaccharides、simple_sugars、cellulose_hemicellulose、protein_general 或 lipids_triglycerides。
   - 请优先依据 category/subcategory、example_compounds 字段中出现的具体物质名称来判断属于哪一类。
6. 输出必须是一个严格的 JSON，对象形式为：
   {{ "components": ["component_name1", "component_name2"] }}
   如果你判断“没有合适的类别”，请输出：
   {{ "components": ["none"] }}
7. 不要输出多余解释或文字，整个回复只包含这一段 JSON。

现在这条先验知识行的字段如下（来自 nir_spectroscopy_verified_sources.csv）：
{row_json}

请按要求输出 JSON：
"""
    return prompt.strip()

def parse_components_from_response(resp_text: str) -> List[str]:
    try:
        data = json.loads(resp_text)
    except json.JSONDecodeError:
        first = resp_text.find("{")
        last = resp_text.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                snippet = resp_text[first: last + 1]
                data = json.loads(snippet)
            except Exception as e2:
                raise ValueError(f"无法解析为 JSON，截取后仍失败: {e2}\n原始回复：{resp_text}")
        else:
            raise ValueError(f"无法找到 JSON 结构：{resp_text}")

    if not isinstance(data, dict) or "components" not in data:
        raise ValueError(f"JSON 中缺少 'components' 字段：{data}")

    comps = data["components"]
    if comps is None:
        return []
    if not isinstance(comps, list):
        raise ValueError(f"'components' 字段不是列表：{comps}")

    result = []
    for c in comps:
        if c is None:
            continue
        name = str(c).strip()
        if name:
            result.append(name)

    return result

def main():
    print(f"[INFO] Loading material from: {MATERIAL_PATH}")
    material_df = pd.read_excel(MATERIAL_PATH)
    component_catalog_text = build_component_catalog_text(material_df)
    print("[INFO] Component catalog loaded.")

    print(f"[INFO] Loading CSV from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    if "components" not in df.columns:
        df["components"] = ""

    n_rows = len(df)
    print(f"[INFO] Total rows: {n_rows}, START_ROW={START_ROW}, SKIP_NONEMPTY={SKIP_NONEMPTY}")

    processed_since_save = 0

    for idx in range(START_ROW, n_rows):
        row = df.iloc[idx]

        excel_row = idx + 2

        if SKIP_NONEMPTY:
            val = str(row.get("components", "")).strip()
            if val not in ("", "nan", "NaN", "None"):
                print(f"[SKIP] Data row index={idx}, excel_row={excel_row}, components already: {val}")
                continue

        row_dict = row.to_dict()
        row_dict.pop("components", None)

        print(f"\n[ROW] Processing data row index={idx}, excel_row={excel_row} ...")

        prompt = build_prompt_for_row(row_dict, component_catalog_text)

        if DEBUG_PRINT_PROMPT:
            print("=" * 80)
            print(f"[PROMPT] For data row index={idx}, excel_row={excel_row}:")
            print(prompt)
            print("=" * 80)

        for attempt in range(1, MAX_RETRY + 1):
            try:
                resp_text = call_ai(prompt)
                components = parse_components_from_response(resp_text)

                if not components:
                    components = ["none"]

                print(f"[OK] Parsed components for data row index={idx}, excel_row={excel_row}: {components}")

                df.at[idx, "components"] = ", ".join(components)
                break
            except Exception as e:
                print(f"[ERROR] Row index={idx}, excel_row={excel_row}, attempt {attempt}/{MAX_RETRY} failed: {e}")
                if attempt >= MAX_RETRY:
                    print("[FATAL] Max retry reached, aborting.")

                    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
                    return
                else:
                    time.sleep(3)

        df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
        processed_since_save += 1
        if processed_since_save >= SAVE_EVERY_N_ROWS:
            print(f"[INFO] Auto-saved after {processed_since_save} processed rows.")
            processed_since_save = 0

    print("[DONE] All rows processed.")
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"[DONE] Final CSV saved to: {CSV_PATH}")

if __name__ == "__main__":
    main()
