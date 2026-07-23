import json
import time
import requests
import pandas as pd
from typing import List, Dict, Any, Optional

COMPONENTS_PATH = r"your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\RAG_Prior knowledge\Data\Components.xlsx"

FG_PATH = r"your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\RAG_Prior knowledge\Data\Functional Group.xlsx"

SHEET_NAME: Optional[str] = None

API_URL = "your URL"
API_KEY = "your key"
MODEL_NAME = "your model name"

START_ROW = 0
SKIP_NONEMPTY = True

SAVE_EVERY_N_ROWS = 1

MAX_RETRY = 2
RETRY_SLEEP_SEC = 3

DEBUG_PRINT_PROMPT = True
DEBUG_PRINT_HTTP = True

TARGET_COL = "component"

COMPONENT_JOINER = " (or) "

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
                "content": (
                    "You are a domain expert in near-infrared spectroscopy and "
                    "chemical/material categorization. You must follow the user's "
                    "output format strictly and return only valid JSON."
                ),
            },
            {"role": "user", "content": message},
        ],
        "temperature": 0,
        "stream": False,
        "max_tokens": 65535,
        "thinking": {
            "type": "enabled"
        },
    }

    response = requests.post(API_URL, headers=headers, json=data, timeout=360)

    if DEBUG_PRINT_HTTP:
        print(f"[HTTP] status={response.status_code}")
        print(f"[HTTP] raw response: {response.text}")

    if response.status_code != 200:
        raise RuntimeError(f"Request failed: {response.status_code} - {response.text}")

    result = response.json()
    choices = result.get("choices", [])
    if not choices:
        raise RuntimeError("Bad API response format: empty choices")

    content = choices[0]["message"]["content"]
    return content

def load_components_catalog(components_path: str) -> pd.DataFrame:
    df = pd.read_excel(components_path)

    if "component_name" not in df.columns:
        raise ValueError("Components.xlsx must contain column: component_name")

    df = df[df["component_name"].notna()].copy()
    df["component_name"] = df["component_name"].astype(str).str.strip()

    if "definition_en" not in df.columns:
        df["definition_en"] = ""

    df["definition_en"] = df["definition_en"].fillna("").astype(str).str.strip()

    keep_cols = [c for c in ["component_id", "component_name", "definition_en"] if c in df.columns]
    return df[keep_cols].copy()

def build_component_catalog_text(comp_df: pd.DataFrame) -> str:
    lines = []
    for _, row in comp_df.iterrows():
        name = str(row.get("component_name", "")).strip()
        if not name:
            continue

        def_en = str(row.get("definition_en", "")).strip()

        if def_en:
            lines.append(f"- {name}: {def_en}")
        else:
            lines.append(f"- {name}")

    return "\n".join(lines)

def choose_target_sheet(xls: pd.ExcelFile) -> str:
    for s in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=s, nrows=1)
            if TARGET_COL in df.columns:
                return s
        except Exception:
            continue
    return xls.sheet_names[0]

def build_prompt_for_row(
    row_dict: Dict[str, Any],
    component_catalog_text: str,
    allowed_component_names: List[str],
) -> str:
    row_json = json.dumps(row_dict, ensure_ascii=False, indent=2)
    allowed_list_text = ", ".join(allowed_component_names)

    prompt = f"""
You are an expert in near-infrared spectroscopy and chemical/material mechanisms.

Task:
Assign material super-class label(s) to ONE prior-knowledge row.

Context:
These super-classes form an intermediate semantic layer between
functional-group band knowledge and task-level chemical priors.
Your assignment should reflect the PRIMARY material family implied by this row,
not every possible functional group that could contain similar bonds.

You MUST choose from the following material super-classes (EN definitions):
{component_catalog_text}

Allowed component_name values (exact match):
{allowed_list_text}

Decision rules:
1) Select 0–3 super-classes. Most rows should map to exactly 0 or 1 class.
2) Use explicit cues first:
   - "Material Type"
   - specific compound families in the row text
   - clear contextual descriptors (food matrix, soil minerals, fuels, polymers, APIs, etc.)
3) Do NOT assign a class solely because common bonds appear
   (e.g., O–H, C–H, N–H). Bond presence alone is insufficient evidence.
4) If the row describes a generic functional group without a clear material family,
   output ["none"].
5) Only select 2–3 classes if the row explicitly indicates a composite
   or multiple distinct material families.
6) Output MUST be strict JSON in this schema:
   {{ "component": ["component_name1", "component_name2"] }}
   or
   {{ "component": ["none"] }}
7) Output ONLY the JSON, no extra explanation.

Current row (from Functional Group.xlsx):
{row_json}

Return JSON now:
"""
    return prompt.strip()

def parse_component_from_response(resp_text: str) -> List[str]:
    try:
        data = json.loads(resp_text)
    except json.JSONDecodeError:
        first = resp_text.find("{")
        last = resp_text.rfind("}")
        if first != -1 and last != -1 and last > first:
            snippet = resp_text[first:last + 1]
            data = json.loads(snippet)
        else:
            raise ValueError(f"Cannot locate JSON object in response: {resp_text}")

    if not isinstance(data, dict) or "component" not in data:
        raise ValueError(f"JSON must contain 'component' field. Got: {data}")

    comps = data["component"]
    if comps is None:
        return []
    if not isinstance(comps, list):
        raise ValueError(f"'component' must be a list. Got: {comps}")

    out = []
    for c in comps:
        if c is None:
            continue
        name = str(c).strip()
        if name:
            out.append(name)
    return out

def save_workbook(all_sheets: Dict[str, pd.DataFrame], path: str) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in all_sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

def main():
    print(f"[INFO] Loading Components.xlsx: {COMPONENTS_PATH}")
    comp_df = load_components_catalog(COMPONENTS_PATH)
    component_catalog_text = build_component_catalog_text(comp_df)
    allowed_component_names = comp_df["component_name"].tolist()

    print(f"[INFO] Loaded {len(allowed_component_names)} material super-classes.")

    print(f"[INFO] Loading Functional Group.xlsx: {FG_PATH}")
    xls = pd.ExcelFile(FG_PATH)
    target_sheet = SHEET_NAME or choose_target_sheet(xls)

    print(f"[INFO] Target sheet for mapping: {target_sheet}")

    all_sheets: Dict[str, pd.DataFrame] = {}
    for s in xls.sheet_names:
        all_sheets[s] = pd.read_excel(xls, sheet_name=s)

    df = all_sheets[target_sheet]

    if TARGET_COL not in df.columns:
        df[TARGET_COL] = ""

    n_rows = len(df)
    print(f"[INFO] Total rows in sheet '{target_sheet}': {n_rows}")
    print(f"[INFO] START_ROW={START_ROW}, SKIP_NONEMPTY={SKIP_NONEMPTY}")

    processed_since_save = 0

    for idx in range(START_ROW, n_rows):
        row = df.iloc[idx]
        excel_row = idx + 2

        if SKIP_NONEMPTY:
            val = str(row.get(TARGET_COL, "")).strip()
            if val not in ("", "nan", "NaN", "None"):
                print(f"[SKIP] Row index={idx}, excel_row={excel_row}, {TARGET_COL} already: {val}")
                continue

        row_dict = row.to_dict()
        row_dict.pop(TARGET_COL, None)

        print(f"\n[ROW] Processing index={idx}, excel_row={excel_row} ...")

        prompt = build_prompt_for_row(row_dict, component_catalog_text, allowed_component_names)

        if DEBUG_PRINT_PROMPT:
            print("=" * 80)
            print(prompt)
            print("=" * 80)

        components: List[str] = []
        for attempt in range(1, MAX_RETRY + 1):
            try:
                resp_text = call_ai(prompt)
                components = parse_component_from_response(resp_text)

                if not components:
                    components = ["none"]

                norm = []
                for c in components:
                    if c == "none":
                        norm.append("none")
                    elif c in allowed_component_names:
                        norm.append(c)

                if not norm:
                    norm = ["none"]

                components = norm

                print(f"[OK] Parsed component(s) for index={idx}, excel_row={excel_row}: {components}")

                df.at[idx, TARGET_COL] = COMPONENT_JOINER.join(components)
                break

            except Exception as e:
                print(f"[ERROR] index={idx}, excel_row={excel_row}, attempt {attempt}/{MAX_RETRY} failed: {e}")
                if attempt >= MAX_RETRY:
                    print("[FATAL] Max retry reached. Saving progress and aborting.")
                    all_sheets[target_sheet] = df
                    save_workbook(all_sheets, FG_PATH)
                    return
                time.sleep(RETRY_SLEEP_SEC)

        processed_since_save += 1

        if processed_since_save >= SAVE_EVERY_N_ROWS:
            all_sheets[target_sheet] = df
            save_workbook(all_sheets, FG_PATH)
            print(f"[INFO] Auto-saved after {processed_since_save} processed rows.")
            processed_since_save = 0

    all_sheets[target_sheet] = df
    save_workbook(all_sheets, FG_PATH)

    print("[DONE] All rows processed.")
    print(f"[DONE] Updated workbook saved to: {FG_PATH}")

if __name__ == "__main__":
    main()
