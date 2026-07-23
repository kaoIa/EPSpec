import os
import re
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

INPUT_PATH  = r'your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Data\Raw Data\shootout.csv'
OUTPUT_PATH = r'your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Data\Preprocessed Data\shootout_sg.csv'

DERIV_ORDER = 1
POLY_ORDER  = 3
WINDOW_PTS  = 11
EDGE_MODE   = 'interp'
FLOAT_FMT   = '%.8g'

def _parse_numeric(name: str):
    if isinstance(name, (int, float)):
        return float(name)
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(name))
    return float(m.group(0)) if m else None

def infer_delta_from_cols(cols):
    xs = [_parse_numeric(c) for c in cols]
    xs = [x for x in xs if x is not None]
    if len(xs) >= 3:
        xs = np.array(sorted(xs), dtype=float)
        diffs = np.diff(xs)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(np.median(diffs))
    return 1.0

def make_valid_window(n_points, want, poly):

    w = int(want)
    if w % 2 == 0:
        w += 1
    w = max(w, poly + 2 + (poly % 2 == 0))
    w = max(w, 5)
    if w > n_points:

        w = n_points if n_points % 2 == 1 else n_points - 1
        if w <= poly:

            w = poly + 1 + (poly % 2 == 0)
            w = min(w, max(3, n_points if n_points % 2 == 1 else n_points - 1))
    return max(3, w)

def interpolate_nans_rowwise(X, x_axis=None):
    n, p = X.shape
    if x_axis is None:
        x_axis = np.arange(p, dtype=float)
    X = X.copy()
    for i in range(n):
        y = X[i, :]
        mask = np.isfinite(y)
        if mask.sum() == 0:
            continue
        if mask.sum() < p:

            X[i, ~mask] = np.interp(x_axis[~mask], x_axis[mask], y[mask])

            if not np.isfinite(X[i, 0]):
                X[i, 0] = X[i, np.where(np.isfinite(X[i, :]))[0][0]]
            if not np.isfinite(X[i, -1]):
                X[i, -1] = X[i, np.where(np.isfinite(X[i, :]))[0][-1]]
    return X

def apply_sg(X: np.ndarray, window_pts: int, poly_order: int, deriv_order: int, delta: float, mode='interp'):
    n, p = X.shape
    w = make_valid_window(p, window_pts, poly_order)

    return savgol_filter(X, window_length=w, polyorder=poly_order,
                         deriv=deriv_order, delta=delta, axis=1, mode=mode)

def preprocess_file(in_path: str, out_path: str,
                    deriv_order: int = DERIV_ORDER,
                    poly_order: int  = POLY_ORDER,
                    window_pts: int  = WINDOW_PTS,
                    edge_mode: str   = EDGE_MODE):
    out_dir = os.path.dirname(out_path) or '.'
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(in_path, encoding='utf-8-sig')
    if df.shape[1] < 3:
        raise ValueError("CSV 至少需要三列：sample_id、若干光谱列、y。")

    sample_col = df.columns[0]
    target_col = df.columns[-1]
    spec_cols   = df.columns[1:-1]

    delta = infer_delta_from_cols(spec_cols)
    for c in spec_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    X = df.loc[:, spec_cols].to_numpy(dtype=np.float64)

    X = interpolate_nans_rowwise(X)
    X_sg = apply_sg(X, window_pts=window_pts, poly_order=poly_order,
                    deriv_order=deriv_order, delta=delta, mode=edge_mode)

    df.loc[:, spec_cols] = X_sg
    df.to_csv(out_path, index=False, encoding='utf-8-sig', float_format=FLOAT_FMT)

    tag = "SG" if deriv_order == 0 else f"SG-D{deriv_order}"
    print(f"[{tag}] {os.path.basename(in_path)} -> {os.path.basename(out_path)} | "
          f"samples={df.shape[0]}, bands={len(spec_cols)}, window={make_valid_window(len(spec_cols), WINDOW_PTS, POLY_ORDER)}, "
          f"poly={poly_order}, deriv={deriv_order}, delta={delta}, mode={edge_mode}")

if __name__ == "__main__":
    preprocess_file(INPUT_PATH, OUTPUT_PATH)
