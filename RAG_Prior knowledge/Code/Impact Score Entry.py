import os
import shutil
from datetime import datetime

import pandas as pd

FILE_PATH = r"your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\RAG_Prior knowledge\Data\Functional Group.xlsx"

MODE = "all"

INCLUDE_ZERO = False
DEDUP_TOKEN = True

CATEGORIES = [
    "Water and polyhydroxy species",
    "Lipids and fats",
    "Carbohydrates and polysaccharides",
    "Proteins, peptides and amide species",
    "Hydrocarbon C–H components, aliphatic",
    "Aromatic and conjugated unsaturated systems",
    "Oxygenated organics (non-aqueous ethers, esters, carbonyl compounds, etc.)",
    "Nitrogen-containing organics and active pharmaceutical ingredients",
    "Inorganic minerals and carbonates",
    "Polymers, excipients and exogenous organic contaminants",
]

EFFECTS_REGRESSION = {
    "cassav": {
        "Water and polyhydroxy species": -3,
        "Lipids and fats": +2,
        "Carbohydrates and polysaccharides": -2,
        "Proteins, peptides and amide species": 0,
        "Hydrocarbon C–H components, aliphatic": +1,
        "Aromatic and conjugated unsaturated systems": +3,
        "Oxygenated organics (non-aqueous ethers, esters, carbonyl compounds, etc.)": 0,
        "Nitrogen-containing organics and active pharmaceutical ingredients": 0,
        "Inorganic minerals and carbonates": 0,
        "Polymers, excipients and exogenous organic contaminants": 0,
    },
    "gasoline": {
        "Water and polyhydroxy species": -3,
        "Lipids and fats": 0,
        "Carbohydrates and polysaccharides": 0,
        "Proteins, peptides and amide species": 0,
        "Hydrocarbon C–H components, aliphatic": +3,
        "Aromatic and conjugated unsaturated systems": +2,
        "Oxygenated organics (non-aqueous ethers, esters, carbonyl compounds, etc.)": +1,
        "Nitrogen-containing organics and active pharmaceutical ingredients": 0,
        "Inorganic minerals and carbonates": 0,
        "Polymers, excipients and exogenous organic contaminants": 0,
    },
    "tecator": {
        "Water and polyhydroxy species": -3,
        "Lipids and fats": +3,
        "Carbohydrates and polysaccharides": 0,
        "Proteins, peptides and amide species": -2,
        "Hydrocarbon C–H components, aliphatic": +2,
        "Aromatic and conjugated unsaturated systems": 0,
        "Oxygenated organics (non-aqueous ethers, esters, carbonyl compounds, etc.)": +2,
        "Nitrogen-containing organics and active pharmaceutical ingredients": 0,
        "Inorganic minerals and carbonates": 0,
        "Polymers, excipients and exogenous organic contaminants": 0,
    },
    "soil": {
        "Water and polyhydroxy species": -2,
        "Lipids and fats": +1,
        "Carbohydrates and polysaccharides": +2,
        "Proteins, peptides and amide species": 0,
        "Hydrocarbon C–H components, aliphatic": +1,
        "Aromatic and conjugated unsaturated systems": +3,
        "Oxygenated organics (non-aqueous ethers, esters, carbonyl compounds, etc.)": 0,
        "Nitrogen-containing organics and active pharmaceutical ingredients": 0,
        "Inorganic minerals and carbonates": -3,
        "Polymers, excipients and exogenous organic contaminants": -1,
    },
    "shootout": {
        "Water and polyhydroxy species": -2,
        "Lipids and fats": 0,
        "Carbohydrates and polysaccharides": -2,
        "Proteins, peptides and amide species": 0,
        "Hydrocarbon C–H components, aliphatic": 0,
        "Aromatic and conjugated unsaturated systems": +2,
        "Oxygenated organics (non-aqueous ethers, esters, carbonyl compounds, etc.)": +2,
        "Nitrogen-containing organics and active pharmaceutical ingredients": +3,
        "Inorganic minerals and carbonates": 0,
        "Polymers, excipients and exogenous organic contaminants": -3,
    },
    "corn": {
        "Water and polyhydroxy species": +2,
        "Lipids and fats": -2,
        "Carbohydrates and polysaccharides": +3,
        "Proteins, peptides and amide species": -2,
        "Hydrocarbon C–H components, aliphatic": -1,
        "Aromatic and conjugated unsaturated systems": 0,
        "Oxygenated organics (non-aqueous ethers, esters, carbonyl compounds, etc.)": 0,
        "Nitrogen-containing organics and active pharmaceutical ingredients": 0,
        "Inorganic minerals and carbonates": 0,
        "Polymers, excipients and exogenous organic contaminants": 0,
    },
}

EFFECTS_CLASSIFICATION = {
    "forages2": {
        "Water and polyhydroxy species": -1,
        "Lipids and fats": 0,
        "Carbohydrates and polysaccharides": +3,
        "Proteins, peptides and amide species": +3,
        "Hydrocarbon C–H components, aliphatic": 0,
        "Aromatic and conjugated unsaturated systems": +2,
        "Oxygenated organics (non-aqueous ethers, esters, carbonyl compounds, etc.)": 0,
        "Nitrogen-containing organics and active pharmaceutical ingredients": 0,
        "Inorganic minerals and carbonates": 0,
        "Polymers, excipients and exogenous organic contaminants": -1,
    },
    "milk": {
        "Water and polyhydroxy species": +2,
        "Lipids and fats": -2,
        "Carbohydrates and polysaccharides": +3,
        "Proteins, peptides and amide species": -1,
        "Hydrocarbon C–H components, aliphatic": -1,
        "Aromatic and conjugated unsaturated systems": 0,
        "Oxygenated organics (non-aqueous ethers, esters, carbonyl compounds, etc.)": 0,
        "Nitrogen-containing organics and active pharmaceutical ingredients": 0,
        "Inorganic minerals and carbonates": 0,
        "Polymers, excipients and exogenous organic contaminants": 0,
    },
    "grape": {
        "Water and polyhydroxy species": +1,
        "Lipids and fats": 0,
        "Carbohydrates and polysaccharides": +2,
        "Proteins, peptides and amide species": 0,
        "Hydrocarbon C–H components, aliphatic": 0,
        "Aromatic and conjugated unsaturated systems": +1,
        "Oxygenated organics (non-aqueous ethers, esters, carbonyl compounds, etc.)": +1,
        "Nitrogen-containing organics and active pharmaceutical ingredients": 0,
        "Inorganic minerals and carbonates": 0,
        "Polymers, excipients and exogenous organic contaminants": 0,
    },
}

def fmt_effect(v: int) -> str:
    if v == 0 and not INCLUDE_ZERO:
        return ""
    if v == 0 and INCLUDE_ZERO:
        return "0"
    return f"{v:+d}"

def safe_append(existing, token: str) -> str:
    if token == "":
        return existing

    if pd.isna(existing):
        existing_str = ""
    else:
        existing_str = str(existing).strip()

    if existing_str == "":
        return token

    if DEDUP_TOKEN:
        if token in existing_str:
            return existing_str

    return existing_str + token

def find_col_case_insensitive(df, wanted_name: str):
    lower_map = {c.lower(): c for c in df.columns}
    return lower_map.get(wanted_name.lower(), None)

def get_effects_by_mode(mode: str):
    mode = (mode or "").lower().strip()
    if mode == "regression":
        return EFFECTS_REGRESSION
    if mode == "classification":
        return EFFECTS_CLASSIFICATION
    if mode == "all":
        merged = {}
        merged.update(EFFECTS_REGRESSION)
        merged.update(EFFECTS_CLASSIFICATION)
        return merged
    raise ValueError("MODE must be one of: 'regression' / 'classification' / 'all'")

def main():
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"File not found: {FILE_PATH}")

    effects = get_effects_by_mode(MODE)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = FILE_PATH.replace(".xlsx", f".backup_{MODE}_{ts}.xlsx")
    shutil.copy2(FILE_PATH, backup_path)
    print(f"[OK] Backup created: {backup_path}")

    xls = pd.ExcelFile(FILE_PATH)
    sheet_name = "Sheet1" if "Sheet1" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(FILE_PATH, sheet_name=sheet_name)

    comp_col = find_col_case_insensitive(df, "component")
    if comp_col is None:
        raise KeyError("Cannot find 'component' column (case-insensitive).")

    dataset_cols = {}
    for ds in effects.keys():
        col = find_col_case_insensitive(df, ds)
        if col is None:
            df[ds] = ""
            dataset_cols[ds] = ds
        else:
            dataset_cols[ds] = col

    comp_series = df[comp_col].astype(str)

    for ds, cat_map in effects.items():
        ds_col = dataset_cols[ds]

        for cat in CATEGORIES:
            if cat not in cat_map:
                continue

            effect_val = cat_map[cat]
            token = fmt_effect(effect_val)
            if token == "":
                continue

            mask = comp_series.str.contains(cat, regex=False, na=False)

            df.loc[mask, ds_col] = df.loc[mask, ds_col].apply(
                lambda x: safe_append(x, token)
            )

    with pd.ExcelWriter(FILE_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"[OK] Updated sheet '{sheet_name}' in: {FILE_PATH}")
    print(f"[DONE] Effects appended in MODE='{MODE}' for datasets: {', '.join(effects.keys())}")

if __name__ == "__main__":
    main()
