import os
import pandas as pd
import numpy as np

INPUT_PATH  = r'your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Data\Raw Data\shootout.csv'
OUTPUT_PATH = r'your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Data\Preprocessed Data\shootout_snv.csv'

def snv(X: np.ndarray, robust: bool = False, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)

    if robust:
        center = np.nanmedian(X, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(X - center), axis=1, keepdims=True)
        scale = 1.4826 * mad
    else:
        center = np.nanmean(X, axis=1, keepdims=True)
        scale = np.nanstd(X, axis=1, ddof=0, keepdims=True)

    bad = (~np.isfinite(scale)) | (scale < eps)
    if np.any(bad):
        scale[bad] = 1.0

    return (X - center) / scale

def preprocess_file(in_path: str, out_path: str, robust: bool = False):
    out_dir = os.path.dirname(out_path) or '.'
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(in_path, encoding='utf-8-sig')

    if df.shape[1] < 3:
        raise ValueError("CSV 至少需要三列：sample_id、若干光谱列、y。")

    spec_cols = df.columns[1:-1]

    X = df.loc[:, spec_cols].to_numpy(dtype=np.float64)

    X_snv = snv(X, robust=robust)

    df.loc[:, spec_cols] = X_snv
    df.to_csv(out_path, index=False, encoding='utf-8-sig', float_format='%.8g')

    print(f"[SNV] {os.path.basename(in_path)} -> {os.path.basename(out_path)} | "
          f"samples={df.shape[0]}, bands={len(spec_cols)}, robust={robust}")

if __name__ == "__main__":

    preprocess_file(INPUT_PATH, OUTPUT_PATH, robust=False)
