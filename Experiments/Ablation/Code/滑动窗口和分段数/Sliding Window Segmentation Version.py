import os
import re
import json
import math
import warnings
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score as sk_r2_score

from openai import OpenAI

INPUT_PATH = r'your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Data\Raw Data\corn.csv'
OUT_DIR = r'your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Experiments\Ablation\Results\滑动窗口和分段数\corn\plsr_sliding-demo8'

ID_COL = 'sample_id'
Y_COL = 'y'

N_OUTER = 5
N_INNER = 5
RANDOM_STATE = 42
MAX_PC_CAP = 30

USE_EPSPEC_PIPELINE = True

K_LIST = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20,22, 24, 26, 28, 30, 32, 34, 36, 38, 40,42, 44, 46, 48, 50,52, 54, 56, 58, 60,62, 64, 66, 68, 70,72, 74, 76, 78, 80,82, 84, 86, 88, 90,92, 94, 96, 98, 100]

SEGMENTATION_METHOD = "sliding"

BASE_INTERVALS = 40

WINDOW_LEN_POINTS: Optional[int] = None
STRIDE_POINTS: Optional[int] = None

DEFAULT_OVERLAP_RATIO = 0.5

USE_NMS_DEDUP = True

NMS_IOU_THRESHOLD = 0.60
NMS_OVERLAP_MIN_THRESHOLD = 0.80

PRIOR_KB_PATH = r"your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\RAG_Prior knowledge\Data\Functional Group.xlsx"

API_KEY =  "your key"

CLIENT_BASE_URL = "your URL"
CLIENT_MODEL = "your model name"

_GLOBAL_PRIOR_CACHE: Dict[str, str] = {}

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def read_dataset(path: str, id_col: str, y_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path)
    cols = list(df.columns)
    if cols[0] != id_col or cols[-1] != y_col:
        raise ValueError("列顺序需为 [sample_id | 波长... | y]，请检查表头。")
    feat_cols = cols[1:-1]

    ids = df[id_col].values
    y = df[y_col].astype(float).values.reshape(-1, 1)
    X = df[feat_cols].apply(pd.to_numeric, errors='raise').values.astype(float)

    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("检测到 NaN；为避免数据泄露，本脚本不做全局填补，请先清洗。")

    def clean(c):
        s = str(c).lower().replace('nm', '').replace(' ', '')
        try:
            return str(float(s))
        except Exception:
            return str(c)

    feature_names = [clean(c) for c in feat_cols]
    return ids, X, y, feature_names

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def r2_score(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

def bias(y_true, y_pred):
    return float(np.mean(y_pred - y_true))

def rpd(y_true, y_pred):
    sdy = float(np.std(y_true, ddof=1))
    e = rmse(y_true, y_pred)
    return float(sdy / e) if e > 0 else float('inf')

def rpiq(y_true, y_pred):
    q75, q25 = np.percentile(y_true, [75, 25])
    iqr = float(q75 - q25)
    e = rmse(y_true, y_pred)
    return float(iqr / e) if e > 0 else float('inf')

def pls_nipals(X: np.ndarray, Y: np.ndarray, n_components: int):
    Xk = X.copy()
    Yk = Y.copy()
    n, p = Xk.shape
    m = Yk.shape[1]

    A = n_components
    T = np.zeros((n, A))
    P = np.zeros((p, A))
    W = np.zeros((p, A))
    Q = np.zeros((m, A))
    U = np.zeros((n, A))

    for a in range(A):
        u = Yk[:, [0]]
        if np.allclose(u, 0):
            u = np.random.default_rng(0).standard_normal((n, 1))

        for _ in range(500):
            den1 = (u.T @ u).item() + 1e-15
            w = (Xk.T @ u) / den1
            w /= (np.linalg.norm(w) + 1e-15)

            t = Xk @ w
            den2 = (t.T @ t).item() + 1e-15
            q = (Yk.T @ t) / den2
            den3 = (q.T @ q).item() + 1e-15
            u_new = (Yk @ q) / den3

            if np.linalg.norm(u_new - u) < 1e-12:
                u = u_new
                break
            u = u_new

        den4 = (t.T @ t).item() + 1e-15
        p_vec = (Xk.T @ t) / den4

        Xk = Xk - t @ p_vec.T
        Yk = Yk - t @ q.T

        T[:, [a]] = t
        P[:, [a]] = p_vec
        W[:, [a]] = w
        Q[:, [a]] = q
        U[:, [a]] = u

    PTW = P.T @ W
    R = W @ np.linalg.inv(PTW + 1e-15 * np.eye(A))
    return T, P, R, Q, U

@dataclass
class PLSRResult:
    T: np.ndarray
    P: np.ndarray
    R: np.ndarray
    Q: np.ndarray
    U: np.ndarray
    beta: np.ndarray
    Xm: np.ndarray
    Ym: np.ndarray
    pc: int
    leverages: np.ndarray
    stdized_residual: np.ndarray
    is_studentized_residual: bool

class PLSR:

    def __init__(self, pc: int):
        self.pc = int(pc)
        self.res_: Optional[PLSRResult] = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> PLSRResult:
        Xm = X.mean(axis=0)
        Ym = Y.mean(axis=0)
        Xc = X - Xm
        Yc = Y - Ym

        T, P, R, Q, U = pls_nipals(Xc, Yc, self.pc)
        pc_eff = T.shape[1]
        beta = R @ Q.T

        H_mid = np.linalg.inv(T.T @ T + 1e-15 * np.eye(pc_eff))
        hatM = Xc @ R @ H_mid @ T.T
        leverages = (hatM ** 2).sum(axis=1)
        is_stud = not (np.any(leverages >= 1) or np.any(leverages < 0))

        res = Xc @ beta + Ym - Y
        std_res = res / (np.std(res, axis=0) + 1e-15)
        if is_stud:
            std_res = std_res / np.sqrt(1 - leverages[:, np.newaxis] + 1e-15)

        self.res_ = PLSRResult(
            T=T, P=P, R=R, Q=Q, U=U, beta=beta,
            Xm=Xm, Ym=Ym, pc=pc_eff,
            leverages=leverages, stdized_residual=std_res,
            is_studentized_residual=is_stud
        )
        return self.res_

    def predict(self, X: np.ndarray) -> np.ndarray:
        r = self.res_
        if r is None:
            raise RuntimeError("PLSR.predict() called before fit().")
        return (X - r.Xm) @ r.beta + r.Ym

def nested_cv_plsr(X: np.ndarray, y: np.ndarray, ids: np.ndarray, feature_names: List[str]) -> Dict:
    outer_cv = KFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = KFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE)

    all_fold_metrics = []
    all_fold_preds = []
    all_fold_coefs = []

    for fold_id, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(X), start=1):
        X_train_outer, y_train_outer = X[train_outer_idx], y[train_outer_idx]
        X_test_outer, y_test_outer = X[test_outer_idx], y[test_outer_idx]
        ids_test = ids[test_outer_idx]

        max_pc = int(min(MAX_PC_CAP, X_train_outer.shape[0] - 1, X_train_outer.shape[1]))
        max_pc = max(1, max_pc)
        pc_grid = list(range(1, max_pc + 1))

        pc_cv_records = []
        for pc in pc_grid:
            rmses = []
            for tr_idx_in, va_idx_in in inner_cv.split(X_train_outer):
                tr_abs = train_outer_idx[tr_idx_in]
                va_abs = train_outer_idx[va_idx_in]
                assert set(tr_abs).issubset(set(train_outer_idx)) and set(va_abs).issubset(set(train_outer_idx))
                assert len(set(tr_abs).intersection(set(va_abs))) == 0

                model = PLSR(pc=pc)
                model.fit(X[tr_abs], y[tr_abs])
                y_hat = model.predict(X[va_abs])
                rmses.append(rmse(y[va_abs], y_hat))

            avg_rmse = float(np.mean(rmses))
            se_rmse = float(np.std(rmses, ddof=1) / math.sqrt(len(rmses)))
            pc_cv_records.append({
                "pc": int(pc),
                "avg_rmse": avg_rmse,
                "se_rmse": se_rmse
            })

        min_record = min(pc_cv_records, key=lambda d: (d["avg_rmse"], d["pc"]))
        threshold = float(min_record["avg_rmse"] + min_record["se_rmse"])
        one_se_record = min(
            [d for d in pc_cv_records if d["avg_rmse"] <= threshold + 1e-12],
            key=lambda d: d["pc"]
        )
        best_pc = int(one_se_record["pc"])
        best_rmse = float(one_se_record["avg_rmse"])

        print(
            f"[Outer Fold {fold_id}] best_pc (inner CV) = {best_pc}, "
            f"inner_CV_RMSE={best_rmse:.4f}"
        )

        final_model = PLSR(pc=int(best_pc))
        res = final_model.fit(X_train_outer, y_train_outer)
        y_pred_test = final_model.predict(X_test_outer)

        m = {
            "fold": int(fold_id),
            "pc": int(res.pc),
            "R2": r2_score(y_test_outer, y_pred_test),
            "RMSE": rmse(y_test_outer, y_pred_test),
            "MAE": mae(y_test_outer, y_pred_test),
            "Bias": bias(y_test_outer, y_pred_test),
            "RPD": rpd(y_test_outer, y_pred_test),
            "RPIQ": rpiq(y_test_outer, y_pred_test),
            "n_train": int(len(train_outer_idx)),
            "n_test": int(len(test_outer_idx)),
        }
        all_fold_metrics.append(m)

        preds_df = pd.DataFrame({
            "fold": fold_id,
            "sample_id": ids_test,
            "y_true": y_test_outer.reshape(-1),
            "y_pred": y_pred_test.reshape(-1)
        })
        all_fold_preds.append(preds_df)

        intercept_val = (res.Ym - res.Xm @ res.beta).item()
        coef = res.beta.reshape(-1)
        names = ["intercept"] + feature_names
        coefs_df = pd.DataFrame({
            "fold": fold_id,
            "feature": names,
            "coef": np.concatenate([[intercept_val], coef])
        })
        all_fold_coefs.append(coefs_df)

        print(
            f"[Outer Fold {fold_id}] test_metrics: "
            f"R2={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, "
            f"Bias={m['Bias']:.4f}, RPD={m['RPD']:.4f}, RPIQ={m['RPIQ']:.4f}"
        )
        print(f"[Outer Fold {fold_id}] pc={res.pc:02d}  R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  RPD={m['RPD']:.3f}")

    metrics_df = pd.DataFrame(all_fold_metrics)
    preds_df = pd.concat(all_fold_preds, ignore_index=True)
    coefs_df = pd.concat(all_fold_coefs, ignore_index=True)

    summary = {}
    for k in ["R2", "RMSE", "MAE", "Bias", "RPD", "RPIQ"]:
        mean_v = float(metrics_df[k].mean())
        std_v = float(metrics_df[k].std(ddof=1))
        summary[k] = {"mean": mean_v, "std": std_v}
    summary["pc_per_fold"] = [int(x) for x in metrics_df["pc"].tolist()]

    return {
        "metrics_per_fold": metrics_df,
        "preds": preds_df,
        "coefs": coefs_df,
        "summary": summary
    }

EP_N_INTERVALS = 40
EP_N_SPLITS = 5
EP_N_REPEATS_STAB = 3
EP_RANDOM_STATE = 42
EP_EDGE_RATIO = 0.03
EP_SNR_LOW_THRESHOLD = 10.0
EP_MAX_PLS_COMPONENTS = 5
EP_STABILITY_THRESHOLD = 0.4

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def fit_pls_oof(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    max_components: int = 5,
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n_samples, _ = X.shape

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(n_samples, dtype=float)
    fold_r2s: List[float] = []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        n_comp = int(min(max_components, X_tr.shape[1], len(train_idx) - 1))
        if n_comp < 1:
            oof_pred[test_idx] = np.mean(y_tr)
            fold_r2s.append(0.0)
            continue

        pls = PLSRegression(n_components=n_comp, scale=True)
        pls.fit(X_tr, y_tr)
        y_hat = pls.predict(X_te).ravel()
        oof_pred[test_idx] = y_hat
        fold_r2s.append(sk_r2_score(y_te, y_hat))

    return oof_pred, fold_r2s

def compute_snr_band(X_seg: np.ndarray) -> float:
    X_seg = np.asarray(X_seg, dtype=float)
    if X_seg.size == 0:
        return 0.0

    signal = float(np.mean(np.abs(X_seg)))
    if X_seg.shape[1] > 1:
        noise = float(np.std(np.diff(X_seg, axis=1)))
    else:
        noise = float(np.std(X_seg))
    return signal / (noise + 1e-6)

def _infer_predicted_substance(dataset_key: str) -> str:
    dk = dataset_key.lower().replace("_preprocessed", "")
    predicted_substance_map = {
        "diesel": "cetane value",
        "cassava": "β-carotene",
        "cassav": "β-carotene",
        "gasoline": "octane number",
        "tecator": "fat",
        "soil": "soil organic matter",
        "shootout": "active pharmaceutical ingredient",
        "corn": "starch",
    }
    return predicted_substance_map.get(dk, "")

def _derive_sliding_params(n_waves: int) -> Tuple[int, int]:
    base_n = max(1, int(min(BASE_INTERVALS, n_waves)))
    win_len = int(round(n_waves / base_n))
    win_len = max(1, win_len)
    win_len = max(2, win_len)

    stride = int(round(win_len * (1.0 - DEFAULT_OVERLAP_RATIO)))
    stride = max(1, stride)
    return win_len, stride

def _generate_interval_indices(wl_sorted: np.ndarray) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    n_waves = int(len(wl_sorted))
    if n_waves <= 0:
        return [], {"method": SEGMENTATION_METHOD}

    method = (SEGMENTATION_METHOD or "uniform").lower().strip()

    if method == "uniform":
        n_intervals = int(min(EP_N_INTERVALS, n_waves))
        interval_indices: List[Tuple[int, int]] = []
        for i in range(n_intervals):
            start_idx = i * n_waves // n_intervals
            end_idx = (i + 1) * n_waves // n_intervals
            interval_indices.append((start_idx, end_idx))
        meta = {
            "method": "uniform",
            "n_intervals": n_intervals,
        }
        return interval_indices, meta

    if WINDOW_LEN_POINTS is None or STRIDE_POINTS is None:
        win_len, stride = _derive_sliding_params(n_waves)
    else:
        win_len = int(WINDOW_LEN_POINTS)
        stride = int(STRIDE_POINTS)
        win_len = max(1, win_len)
        stride = max(1, stride)

    if win_len >= n_waves:
        interval_indices = [(0, n_waves)]
    else:
        interval_indices = []
        for s in range(0, n_waves - win_len + 1, stride):
            interval_indices.append((s, s + win_len))
        if interval_indices and interval_indices[-1][1] != n_waves:
            interval_indices.append((n_waves - win_len, n_waves))
        elif not interval_indices:
            interval_indices = [(0, min(win_len, n_waves))]

    meta = {
        "method": "sliding",
        "base_intervals": int(BASE_INTERVALS),
        "window_len_points": int(win_len),
        "stride_points": int(stride),
        "overlap_ratio": float(1.0 - stride / max(1, win_len)),
        "n_windows": int(len(interval_indices)),
    }
    return interval_indices, meta

def compute_ep_for_fold(
    X_train_outer: np.ndarray,
    y_train_outer: np.ndarray,
    wavelengths: np.ndarray,
    dataset_key: str,
    outer_fold_id: int,
    out_dir: str,
) -> Dict:
    ensure_dir(out_dir)

    X = np.asarray(X_train_outer, dtype=float)
    y = np.asarray(y_train_outer, dtype=float).ravel()
    wavelengths = np.asarray(wavelengths, dtype=float)

    n_samples, n_waves = X.shape
    if wavelengths.shape[0] != n_waves:
        raise ValueError(
            f"wavelengths 长度({wavelengths.shape[0]})与 X 列数({n_waves})不一致。"
        )

    sort_idx = np.argsort(wavelengths)
    wl_sorted = wavelengths[sort_idx]
    X_full = X[:, sort_idx]

    interval_indices, seg_meta = _generate_interval_indices(wl_sorted)

    full_oof, _ = fit_pls_oof(
        X_full,
        y,
        n_splits=EP_N_SPLITS,
        random_state=EP_RANDOM_STATE,
        max_components=EP_MAX_PLS_COMPONENTS,
    )
    baseline_r2 = sk_r2_score(y, full_oof)

    interval_records: List[Dict] = []
    oof_preds_for_all_intervals: List[np.ndarray] = []

    wl_min, wl_max = float(wl_sorted[0]), float(wl_sorted[-1])
    wl_range = wl_max - wl_min

    for seg_id, (s_idx, e_idx) in enumerate(interval_indices):
        X_seg = X_full[:, s_idx:e_idx]
        if X_seg.shape[1] == 0:
            continue

        wls_seg = wl_sorted[s_idx:e_idx]
        start_nm = float(wls_seg[0])
        end_nm = float(wls_seg[-1])

        corrs = []
        for j in range(X_seg.shape[1]):
            c = safe_corr(X_seg[:, j], y)
            corrs.append(abs(c))
        if len(corrs) == 0:
            corr_max = 0.0
            corr_mean = 0.0
        else:
            corr_max = float(np.max(corrs))
            corr_mean = float(np.mean(corrs))

        oof_pred_seg, fold_r2s = fit_pls_oof(
            X_seg,
            y,
            n_splits=EP_N_SPLITS,
            random_state=EP_RANDOM_STATE,
            max_components=EP_MAX_PLS_COMPONENTS,
        )
        local_r2 = sk_r2_score(y, oof_pred_seg)
        cv_std = float(np.std(fold_r2s))
        cv_std = float(min(cv_std, 1.0))

        stab_scores = []
        for rp in range(EP_N_REPEATS_STAB):
            _, fold_r2s_rp = fit_pls_oof(
                X_seg,
                y,
                n_splits=EP_N_SPLITS,
                random_state=EP_RANDOM_STATE + rp,
                max_components=EP_MAX_PLS_COMPONENTS,
            )
            stab_scores.append(np.mean(fold_r2s_rp))

        if len(stab_scores) == 0:
            stability = 0.0
        else:
            stability = float(np.mean([s >= EP_STABILITY_THRESHOLD for s in stab_scores]))

        snr_band = compute_snr_band(X_seg)

        pos_ratio_start = (start_nm - wl_min) / (wl_range + 1e-6)
        pos_ratio_end = (wl_max - end_nm) / (wl_range + 1e-6)
        artifact_risk = 0.0
        if pos_ratio_start < EP_EDGE_RATIO or pos_ratio_end < EP_EDGE_RATIO:
            artifact_risk += 0.5
        if snr_band < EP_SNR_LOW_THRESHOLD:
            artifact_risk += 0.5
        artifact_risk = float(min(1.0, artifact_risk))

        if local_r2 <= 0:
            var_explained_delta = 0.0
        else:
            if baseline_r2 <= 1e-6:
                var_explained_delta = float(local_r2)
            else:
                var_explained_delta = float(local_r2 / (baseline_r2 + 1e-6))
            var_explained_delta = float(min(max(var_explained_delta, 0.0), 2.0))

        interval_id = f"fold{outer_fold_id:02d}_{'win' if seg_meta.get('method')=='sliding' else 'seg'}_{seg_id:03d}"

        interval_records.append({
            "interval_id": interval_id,
            "outer_fold": int(outer_fold_id),
            "start_nm": start_nm,
            "end_nm": end_nm,
            "corr_max": float(corr_max),
            "corr_mean": float(corr_mean),
            "local_r2": float(local_r2),
            "var_explained_delta": float(var_explained_delta),
            "stability": float(stability),
            "cv_std": float(cv_std),
            "snr_band": float(snr_band),
            "artifact_risk": float(artifact_risk),
            "start_idx_sorted": int(s_idx),
            "end_idx_sorted": int(e_idx),
        })
        oof_preds_for_all_intervals.append(oof_pred_seg)

    mean_spectra_per_interval: List[np.ndarray] = []
    for rec in interval_records:
        s_idx = int(rec["start_idx_sorted"])
        e_idx = int(rec["end_idx_sorted"])
        X_seg = X_full[:, s_idx:e_idx]
        if X_seg.shape[1] == 0:
            mean_spectra_per_interval.append(np.array([0.0], dtype=float))
        else:
            mean_spec = X_seg.mean(axis=0)
            mean_spec = (mean_spec - mean_spec.mean()) / (mean_spec.std() + 1e-6)
            mean_spectra_per_interval.append(mean_spec)

    n_intervals_effective = len(interval_records)
    red_spec_max_list: List[float] = []
    red_target_max_list: List[float] = []

    for i in range(n_intervals_effective):
        spec_i = mean_spectra_per_interval[i]
        best_spec_corr = 0.0
        for j in range(n_intervals_effective):
            if i == j:
                continue
            spec_j = mean_spectra_per_interval[j]
            mlen = min(len(spec_i), len(spec_j))
            if mlen == 0:
                continue
            c = safe_corr(spec_i[:mlen], spec_j[:mlen])
            best_spec_corr = max(best_spec_corr, abs(c))
        red_spec_max_list.append(float(best_spec_corr))

        pred_i = oof_preds_for_all_intervals[i]
        best_pred_corr = 0.0
        for j in range(n_intervals_effective):
            if i == j:
                continue
            pred_j = oof_preds_for_all_intervals[j]
            c = safe_corr(pred_i, pred_j)
            best_pred_corr = max(best_pred_corr, abs(c))
        red_target_max_list.append(float(best_pred_corr))

    for k in range(n_intervals_effective):
        interval_records[k]["red_spec_max"] = red_spec_max_list[k]
        interval_records[k]["red_target_max"] = red_target_max_list[k]

    predicted_substance = _infer_predicted_substance(dataset_key)

    info = {
        "description": "NIR interval metrics for LLM-based band selection (regression, per outer fold)",
        "dataset_key": dataset_key,
        "outer_fold_id": int(outer_fold_id),
        "input_file": f"{dataset_key}.csv",
        "predicted_substance": predicted_substance,
        "n_samples": int(n_samples),
        "n_wavelengths": int(n_waves),
        "n_intervals": int(n_intervals_effective),
        "params": {
            "n_splits": EP_N_SPLITS,
            "n_repeats_stab": EP_N_REPEATS_STAB,
            "stability_threshold": EP_STABILITY_THRESHOLD,
            "edge_ratio": EP_EDGE_RATIO,
            "snr_low_threshold": EP_SNR_LOW_THRESHOLD,
            "max_pls_components": EP_MAX_PLS_COMPONENTS,
            "segmentation": seg_meta,
        },
        "field_explanation": {
            "interval_id": "当前区间 ID（带 fold 编号）",
            "outer_fold": "所属外层折编号（从 1 开始）",
            "start_nm": "区间起始波长 (nm)",
            "end_nm": "区间结束波长 (nm)",
            "corr_max": "本段内与 y 的最大绝对相关系数，衡量是否存在强信号点",
            "corr_mean": "本段内所有点与 y 的绝对相关系数平均，衡量整体相关水平",
            "local_r2": "只用本段做 K 折 PLS 的整体 R²，衡量本段单独可预测性",
            "var_explained_delta": "本段解释方差相对全谱基线的比例，截断到 [0, 2]",
            "stability": "基于多次随机 CV 的“表现合格”比例，0~1 之间",
            "cv_std": "local_r2 在 K 折之间的标准差，截断到 <= 1.0，越小说明越稳定",
            "snr_band": "本段简单信噪比，越大越干净",
            "artifact_risk": "本段为边缘/低 SNR 的风险评分，0~1，越大越危险",
            "red_spec_max": "本段平均谱与其它段最相似谱的相关系数，越大说明光谱上越冗余",
            "red_target_max": "本段小模型预测与其它段预测最相近时的相关系数，越大说明在预测上冗余",
        },
    }

    ep_result: Dict[str, Any] = {
        "info": info,
        "global_prior_knowledge": "",
        "intervals": interval_records,
    }

    json_name = f"{dataset_key}_interval_metrics_outerfold{outer_fold_id:02d}.json"
    json_path = os.path.join(out_dir, json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ep_result, f, indent=2, ensure_ascii=False)

    print(
        f"[EP-{seg_meta.get('method','?')}] outer_fold={outer_fold_id} 完成 interval metrics："
        f"{len(interval_records)} intervals -> {json_path}"
    )

    return ep_result

def _clean_json_like_text(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _build_client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError(
            "未检测到 API_KEY。请设置环境变量 BIGMODEL_API_KEY 或 LLM_API_KEY。"
        )
    return OpenAI(
        api_key=API_KEY,
        base_url=CLIENT_BASE_URL,
        timeout=600,
        max_retries=0,
    )

_client: Optional[OpenAI] = None

def call_llm_agent35(messages: List[Dict[str, str]]) -> str:
    global _client
    if _client is None:
        _client = _build_client()

    try:
        resp = _client.chat.completions.create(
            model=CLIENT_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=64000,
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
        lines.append("- 预测目标: （未提供，请根据表中信息尽量推断其可能对应的目标物质/指标，并说明不确定性）")

    lines.append(
        "\n【effect 字段解释（请严格按此理解并在回答中用到）】\n"
        f"- 表格最后一列“{dataset_col}”是该数据集任务下的 effect。\n"
        "- effect 可能是 + / - 以及数字的组合（例如 +2-1）。\n"
        "- “+”表示该条目对应的官能团/组分及其波段对预测目标通常提供正向信息（含信号、可用）。\n"
        "- “-”表示更可能是干扰/负影响/噪声来源（需谨慎或降低权重）。\n"
        "- 数字的绝对值越大，表示影响强度/证据强度越高。\n"
        "- 若出现“+2-1”这类混合符号，表示同一波段可能同时包含有用信息与干扰，"
        "需结合 component / Material Type 判断主导机制，并在结论里标为“谨慎/需验证”。"
    )

    lines.append(
        "\n【输出要求（必须全部满足）】\n"
        "1）必须用中文、连续自然段输出；不要使用项目符号/编号列表/表格/JSON/代码块。\n"
        "2）字数不少于 600 字，建议 4–6 段，每段 2–5 句。\n"
        "3）回答必须包含以下内容（都要写到）：\n"
        "   - 段落1：一句话概括“数据集是什么 + 预测什么含量/指标”（必须点名数据集与目标）。\n"
        "   - 段落2：按 component / Material Type / Functional Group 归纳机制簇，说明目标相关信号与干扰来源（不要逐行复述）。\n"
        "   - 段落3：给出“高置信有用/正影响”的关键波段范围（nm），至少写出 5 个连续区间，并说明对应依据。\n"
        "   - 段落4：给出“低价值/负影响/高噪声风险”的波段范围（nm），至少写出 3 个连续区间，并说明原因。\n"
        "   - 段落5（可选但强烈建议）：写出“混合/不确定/需验证”的波段范围与原因。\n"
        "4）波段描述优先使用 range_nm；若只有单点 Nanometers (nm)，请用“中心±若干 nm”扩展成连续区间再表达。\n"
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
        print(f"[GlobalPrior] 先验知识库不存在或未配置: {PRIOR_KB_PATH}")
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

def _interval_iou_nm(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    a0, a1 = float(min(a)), float(max(a))
    b0, b1 = float(min(b)), float(max(b))
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    len_a = max(0.0, a1 - a0)
    len_b = max(0.0, b1 - b0)
    union = len_a + len_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)

def _interval_overlap_min(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    a0, a1 = float(min(a)), float(max(a))
    b0, b1 = float(min(b)), float(max(b))
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    len_a = max(0.0, a1 - a0)
    len_b = max(0.0, b1 - b0)
    denom = max(1e-12, min(len_a, len_b))
    return float(inter / denom)

def nms_dedup_ranked_ids(
    ranked_ids: List[str],
    interval_ranges: Dict[str, Tuple[float, float]],
    iou_th: float = NMS_IOU_THRESHOLD,
    overlap_min_th: float = NMS_OVERLAP_MIN_THRESHOLD,
) -> List[str]:
    kept: List[str] = []
    for iid in ranked_ids:
        rng = interval_ranges.get(iid)
        if rng is None:
            continue
        is_dup = False
        for kid in kept:
            krng = interval_ranges.get(kid)
            if krng is None:
                continue
            iou = _interval_iou_nm(rng, krng)
            ovm = _interval_overlap_min(rng, krng)
            if iou >= iou_th or ovm >= overlap_min_th:
                is_dup = True
                break
        if not is_dup:
            kept.append(iid)
    return kept

def nested_cv_plsr_with_band_selection(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    feature_names: List[str],
    dataset_key: str,
    out_dir: str,
) -> Dict:
    ensure_dir(out_dir)
    ep_dir = os.path.join(out_dir, "EP")
    ensure_dir(ep_dir)

    try:
        wavelengths = np.array([float(str(fn)) for fn in feature_names], dtype=float)
    except Exception as e:
        raise ValueError(f"feature_names 转换为波长失败: {e}")

    outer_cv = KFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = KFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE)

    all_fold_metrics = []
    all_fold_preds = []
    all_fold_coefs = []

    for fold_id, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(X), start=1):
        X_train_outer, y_train_outer = X[train_outer_idx], y[train_outer_idx]
        X_test_outer, y_test_outer = X[test_outer_idx], y[test_outer_idx]
        ids_test = ids[test_outer_idx]

        print(f"\n===== [Outer Fold {fold_id}] =====")
        print(f"训练样本数={len(train_outer_idx)}, 测试样本数={len(test_outer_idx)}")

        ep_result = compute_ep_for_fold(
            X_train_outer,
            y_train_outer,
            wavelengths=wavelengths,
            dataset_key=dataset_key,
            outer_fold_id=fold_id,
            out_dir=ep_dir,
        )

        intervals = ep_result.get("intervals", []) if isinstance(ep_result, dict) else []

        ranked_ids: List[str] = []
        ranked_ids_filtered: List[str] = []
        interval_masks: Dict[str, np.ndarray] = {}
        interval_ranges: Dict[str, Tuple[float, float]] = {}
        num_intervals = 0

        if intervals:
            ranking_obj = apply_global_prior_and_rank_intervals(
                ep_result=ep_result,
                dataset_key=dataset_key,
                outer_fold_id=fold_id,
                out_dir=ep_dir,
            )

            interval_dict: Dict[str, Dict] = {}
            for rec in intervals:
                iid = rec.get("interval_id") or rec.get("id")
                if iid is None:
                    continue
                interval_dict[str(iid)] = rec

            for iid, rec in interval_dict.items():
                start_nm = rec.get("start_nm", rec.get("start"))
                end_nm = rec.get("end_nm", rec.get("end"))
                try:
                    start_nm_f = float(start_nm) if start_nm is not None else None
                    end_nm_f = float(end_nm) if end_nm is not None else None
                except Exception:
                    start_nm_f = end_nm_f = None
                if start_nm_f is None or end_nm_f is None:
                    continue
                mask = (wavelengths >= start_nm_f) & (wavelengths <= end_nm_f)
                interval_masks[iid] = mask
                interval_ranges[iid] = (start_nm_f, end_nm_f)

            if isinstance(ranking_obj, dict) and "interval_ranking" in ranking_obj:
                rl = ranking_obj.get("interval_ranking", [])
                rl_sorted = sorted(rl, key=lambda x: float(x.get("rank", 1e9)))
                for item in rl_sorted:
                    iid = str(item.get("id"))
                    if iid in interval_masks:
                        ranked_ids.append(iid)

            if not ranked_ids:
                print(f"[Outer Fold {fold_id}] 使用 local_r2 作为兜底排序。")
                intervals_sorted = sorted(
                    intervals,
                    key=lambda r: float(r.get("local_r2", 0.0)),
                    reverse=True
                )
                for rec in intervals_sorted:
                    iid = rec.get("interval_id") or rec.get("id")
                    if iid is not None and str(iid) in interval_masks:
                        ranked_ids.append(str(iid))

            if USE_NMS_DEDUP and ranked_ids:
                ranked_ids_filtered = nms_dedup_ranked_ids(
                    ranked_ids=ranked_ids,
                    interval_ranges=interval_ranges,
                    iou_th=NMS_IOU_THRESHOLD,
                    overlap_min_th=NMS_OVERLAP_MIN_THRESHOLD,
                )
                print(
                    f"[Outer Fold {fold_id}] ranked_ids: {len(ranked_ids)} -> after NMS: {len(ranked_ids_filtered)}"
                )
                try:
                    nms_path = os.path.join(ep_dir, f"interval_ranking_outerfold{fold_id:02d}_nms.json")
                    with open(nms_path, "w", encoding="utf-8") as f:
                        json.dump({"ranked_ids_filtered": ranked_ids_filtered}, f, ensure_ascii=False, indent=2)
                    print(f"[Outer Fold {fold_id}] NMS 排序已写入: {nms_path}")
                except Exception as e:
                    print(f"[Outer Fold {fold_id}] 写入 NMS 排序失败（不影响运行）：{e}")
            else:
                ranked_ids_filtered = ranked_ids

            num_intervals = len(ranked_ids_filtered)
        else:
            print(f"[Outer Fold {fold_id}] interval metrics 为空，后续只会在 inner CV 中搜索 pc（全谱）。")

        valid_k_list = [k for k in K_LIST if k <= num_intervals] if num_intervals > 0 else []
        candidate_k_list = [0] + valid_k_list

        inner_splits = list(inner_cv.split(X_train_outer))

        best_pc = None
        best_k = 0
        best_score = math.inf
        best_mask = None
        n_feats_used = X_train_outer.shape[1]
        joint_cv_records = []

        for k_candidate in candidate_k_list:
            if k_candidate == 0:
                mask_k = None
                X_train_outer_sub = X_train_outer
            else:
                if num_intervals == 0:
                    continue

                top_ids = ranked_ids_filtered[:k_candidate]
                mask_k = np.zeros(X_train_outer.shape[1], dtype=bool)
                for iid in top_ids:
                    msk = interval_masks.get(iid)
                    if msk is None:
                        continue
                    mask_k |= msk

                if not mask_k.any():
                    print(f"[Outer Fold {fold_id}] k={k_candidate}: 没有任何特征列，跳过。")
                    continue

                X_train_outer_sub = X_train_outer[:, mask_k]

            max_pc_sub = int(min(
                MAX_PC_CAP,
                X_train_outer_sub.shape[0] - 1,
                X_train_outer_sub.shape[1]
            ))
            if max_pc_sub < 1:
                print(f"[Outer Fold {fold_id}] k={k_candidate}: max_pc_sub<1，跳过该K。")
                continue
            pc_grid_sub = list(range(1, max_pc_sub + 1))

            for pc_candidate in pc_grid_sub:
                inner_rmses = []
                for tr_idx, va_idx in inner_splits:
                    X_tr = X_train_outer_sub[tr_idx]
                    X_va = X_train_outer_sub[va_idx]
                    y_tr = y_train_outer[tr_idx]
                    y_va = y_train_outer[va_idx]

                    model = PLSR(pc=pc_candidate)
                    model.fit(X_tr, y_tr)
                    y_hat = model.predict(X_va)
                    inner_rmses.append(rmse(y_va, y_hat))

                avg_rmse = float(np.mean(inner_rmses))
                se_rmse = float(np.std(inner_rmses, ddof=1) / math.sqrt(len(inner_rmses)))
                n_feats_candidate = X_train_outer.shape[1] if mask_k is None else int(mask_k.sum())

                joint_cv_records.append({
                    "pc": int(pc_candidate),
                    "k": int(k_candidate),
                    "avg_rmse": avg_rmse,
                    "se_rmse": se_rmse,
                    "mask": None if mask_k is None else mask_k.copy(),
                    "n_features": int(n_feats_candidate),
                })

        if joint_cv_records:

            full_records = [d for d in joint_cv_records if d["k"] == 0]
            full_min_record = min(
                full_records,
                key=lambda d: (d["avg_rmse"], d["pc"])
            )
            full_threshold = float(full_min_record["avg_rmse"] + full_min_record["se_rmse"])
            full_one_se_record = min(
                [d for d in full_records if d["avg_rmse"] <= full_threshold + 1e-12],
                key=lambda d: d["pc"]
            )
            full_one_se_pc = int(full_one_se_record["pc"])

            constrained_records = [
                d for d in joint_cv_records
                if d["pc"] <= full_one_se_pc
            ]
            min_record = min(
                constrained_records,
                key=lambda d: (d["avg_rmse"], d["k"], d["pc"])
            )
            threshold = float(min_record["avg_rmse"] + min_record["se_rmse"])
            eligible_records = [
                d for d in constrained_records
                if d["avg_rmse"] <= threshold + 1e-12
            ]

            best_record = min(
                eligible_records,
                key=lambda d: (d["n_features"], d["pc"], d["avg_rmse"], d["k"])
            )

            best_score = float(best_record["avg_rmse"])
            best_pc = int(best_record["pc"])
            best_k = int(best_record["k"])
            best_mask = None if best_record["mask"] is None else best_record["mask"].copy()
            n_feats_used = int(best_record["n_features"])

        if best_pc is not None:
            if best_mask is not None and best_k > 0:
                print(
                    f"[Outer Fold {fold_id}] 内层CV 联合调参最优组合："
                    f"pc={best_pc}, K={best_k}, "
                    f"RMSE={best_score:.4f}，使用特征数={n_feats_used}"
                )
            else:
                print(
                    f"[Outer Fold {fold_id}] 内层CV 联合调参最优组合："
                    f"pc={best_pc}, K=0(全谱)，"
                    f"RMSE={best_score:.4f}，使用特征数={n_feats_used}"
                )
        else:
            print(
                f"[Outer Fold {fold_id}] 联合调参未找到有效 (pc, K) 组合，退化为全谱 pc=1。"
            )
            best_pc = 1
            best_k = 0
            best_mask = None
            n_feats_used = X_train_outer.shape[1]

        if best_mask is not None and best_k > 0:
            X_train_used = X_train_outer[:, best_mask]
            X_test_used = X_test_outer[:, best_mask]
            used_feature_names = [fn for i, fn in enumerate(feature_names) if best_mask[i]]
        else:
            X_train_used = X_train_outer
            X_test_used = X_test_outer
            used_feature_names = feature_names
            n_feats_used = X_train_outer.shape[1]
            best_k = 0

        final_model = PLSR(pc=int(best_pc))
        res = final_model.fit(X_train_used, y_train_outer)
        y_pred_test = final_model.predict(X_test_used)

        m = {
            "fold": fold_id,
            "pc": int(res.pc),
            "best_k": int(best_k),
            "n_features": int(n_feats_used),
            "R2": r2_score(y_test_outer, y_pred_test),
            "RMSE": rmse(y_test_outer, y_pred_test),
            "MAE": mae(y_test_outer, y_pred_test),
            "Bias": bias(y_test_outer, y_pred_test),
            "RPD": rpd(y_test_outer, y_pred_test),
            "RPIQ": rpiq(y_test_outer, y_pred_test),
            "n_train": int(len(train_outer_idx)),
            "n_test": int(len(test_outer_idx)),
        }
        all_fold_metrics.append(m)

        preds_df = pd.DataFrame({
            "fold": fold_id,
            "sample_id": ids_test,
            "y_true": y_test_outer.reshape(-1),
            "y_pred": y_pred_test.reshape(-1)
        })
        all_fold_preds.append(preds_df)

        intercept_val = (res.Ym - res.Xm @ res.beta).item()
        coef = res.beta.reshape(-1)
        names = ["intercept"] + used_feature_names
        coefs_df = pd.DataFrame({
            "fold": fold_id,
            "feature": names,
            "coef": np.concatenate([[intercept_val], coef])
        })
        all_fold_coefs.append(coefs_df)

        print(
            f"[Outer Fold {fold_id}] 最优组合 pc={res.pc:02d}, "
            f"best_k={m['best_k']}, n_features={m['n_features']}, "
            f"R2={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, RPD={m['RPD']:.3f}"
        )

    metrics_df = pd.DataFrame(all_fold_metrics)
    preds_df = pd.concat(all_fold_preds, ignore_index=True)
    coefs_df = pd.concat(all_fold_coefs, ignore_index=True)

    summary: Dict[str, Any] = {}
    for kk in ["R2", "RMSE", "MAE", "Bias", "RPD", "RPIQ"]:
        summary[kk] = {
            "mean": float(metrics_df[kk].mean()),
            "std": float(metrics_df[kk].std(ddof=1))
        }
    summary["pc"] = {"per_fold": [int(pc) for pc in metrics_df["pc"].tolist()]}
    summary["best_k"] = {"per_fold": [int(k) for k in metrics_df["best_k"].tolist()]}
    summary["n_features"] = {"per_fold": [int(n) for n in metrics_df["n_features"].tolist()]}

    return {
        "metrics_per_fold": metrics_df,
        "preds": preds_df,
        "coefs": coefs_df,
        "summary": summary,
    }

def main():
    ensure_dir(OUT_DIR)
    print(f"读取数据：{INPUT_PATH}")
    ids, X, y, feature_names = read_dataset(INPUT_PATH, ID_COL, Y_COL)
    print(f"样本数={X.shape[0]}，特征数={X.shape[1]}（波长列）")

    dataset_key = os.path.splitext(os.path.basename(INPUT_PATH))[0]

    if USE_EPSPEC_PIPELINE:
        print("\n>>> 使用：EP + 大模型排序 + 前K子波段 + pc 联合搜索版嵌套CV PLSR")
        print(f"    segmentation_method={SEGMENTATION_METHOD}  base_intervals={BASE_INTERVALS}")
        results = nested_cv_plsr_with_band_selection(
            X=X,
            y=y,
            ids=ids,
            feature_names=feature_names,
            dataset_key=dataset_key,
            out_dir=OUT_DIR,
        )
    else:
        print("\n>>> 使用：原始 5×5 嵌套CV PLSR（不做选段）")
        results = nested_cv_plsr(X, y, ids, feature_names)

    metrics_path = os.path.join(OUT_DIR, "metrics_per_fold.csv")
    preds_path = os.path.join(OUT_DIR, "predictions.csv")
    coefs_path = os.path.join(OUT_DIR, "coefficients.csv")
    summary_path = os.path.join(OUT_DIR, "summary.json")

    results["metrics_per_fold"].to_csv(metrics_path, index=False)
    results["preds"].to_csv(preds_path, index=False)
    results["coefs"].to_csv(coefs_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results["summary"], f, ensure_ascii=False, indent=2)

    print("\n==== 嵌套CV总体结果（外层5折的均值±标准差） ====")
    for k in ["R2", "RMSE", "MAE", "Bias", "RPD", "RPIQ"]:
        v = results["summary"][k]
        print(f"{k:>4s}: {v['mean']:.4f} ± {v['std']:.4f}")

    if USE_EPSPEC_PIPELINE:
        print(f"PC per fold: {results['summary']['pc']['per_fold']}")
        print(f"best K per fold: {results['summary']['best_k']['per_fold']}")
        print(f"n_features per fold: {results['summary']['n_features']['per_fold']}")
        print(f"mean n_features: {np.mean(results['summary']['n_features']['per_fold']):.1f}")
    else:
        print(f"PC per fold: {results['summary']['pc_per_fold']}")

    print(f"\n已保存：\n- {metrics_path}\n- {preds_path}\n- {coefs_path}\n- {summary_path}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()

def run_regression(input_path: str, out_dir: str, use_epspec_pipeline: bool = USE_EPSPEC_PIPELINE):
    global INPUT_PATH, OUT_DIR, USE_EPSPEC_PIPELINE
    INPUT_PATH = input_path
    OUT_DIR = out_dir
    USE_EPSPEC_PIPELINE = use_epspec_pipeline
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
