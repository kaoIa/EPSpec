import os
import json
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

INPUT_PATH = r'your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Data\Raw Data\soil.csv'
OUT_DIR = r'your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Experiments\ipls and cars\soil\ipls_plsr_cv_results'

ID_COL = 'sample_id'
Y_COL = 'y'
N_OUTER = 5
N_INNER = 5
RANDOM_STATE = 42
MAX_PC_CAP = 30

USE_IPLS = True

IPLS_INTERVAL_WIDTH = 20

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

def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
def mae(y_true, y_pred): return float(np.mean(np.abs(y_true - y_pred)))
def r2_score(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
def bias(y_true, y_pred): return float(np.mean(y_pred - y_true))
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
        p = (Xk.T @ t) / den4

        Xk = Xk - t @ p.T
        Yk = Yk - t @ q.T

        T[:, [a]] = t
        P[:, [a]] = p
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
        self.res_: PLSRResult = None

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

        self.res_ = PLSRResult(T=T, P=P, R=R, Q=Q, U=U, beta=beta,
                               Xm=Xm, Ym=Ym, pc=pc_eff,
                               leverages=leverages, stdized_residual=std_res,
                               is_studentized_residual=is_stud)
        return self.res_

    def predict(self, X: np.ndarray) -> np.ndarray:
        r = self.res_
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

        print(f"[Outer Fold {fold_id}] pc={res.pc:02d}  R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  RPD={m['RPD']:.3f}")

    metrics_df = pd.DataFrame(all_fold_metrics)
    preds_df = pd.concat(all_fold_preds, ignore_index=True)
    coefs_df = pd.concat(all_fold_coefs, ignore_index=True)

    summary = {}
    for k in ["R2", "RMSE", "MAE", "Bias", "RPD", "RPIQ"]:
        mean_v = float(metrics_df[k].mean())
        std_v  = float(metrics_df[k].std(ddof=1))
        summary[k] = {"mean": mean_v, "std": std_v}
    summary["pc_per_fold"] = [int(x) for x in metrics_df["pc"].tolist()]

    return {
        "metrics_per_fold": metrics_df,
        "preds": preds_df,
        "coefs": coefs_df,
        "summary": summary
    }

def build_ipls_intervals(n_features: int, window_size: int) -> List[Tuple[int, int]]:
    if window_size <= 0:
        raise ValueError("IPLS_INTERVAL_WIDTH 必须为正整数。")
    intervals = []
    start = 0
    while start < n_features:
        end = min(start + window_size, n_features)
        intervals.append((start, end))
        start = end
    return intervals

def evaluate_subset_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feat_idx: np.ndarray,
    inner_cv: KFold,
    max_pc_cap: int
) -> Tuple[float, int]:
    X_sub = X_train[:, feat_idx]
    n_samples, n_features = X_sub.shape
    max_pc = int(min(max_pc_cap, n_samples - 1, n_features))
    max_pc = max(1, max_pc)

    pc_cv_records = []

    for pc in range(1, max_pc + 1):
        rmses = []
        for tr_idx, va_idx in inner_cv.split(X_sub):
            model = PLSR(pc=pc)
            model.fit(X_sub[tr_idx], y_train[tr_idx])
            y_hat = model.predict(X_sub[va_idx])
            rmses.append(rmse(y_train[va_idx], y_hat))

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

    return float(one_se_record["avg_rmse"]), int(one_se_record["pc"])

def ipls_forward_select(
    X_train_outer: np.ndarray,
    y_train_outer: np.ndarray,
    inner_cv: KFold,
    interval_width: int,
    max_pc_cap: int
) -> Tuple[np.ndarray, int, List[Tuple[int, int]], List[int], float]:
    n_samples, n_features = X_train_outer.shape
    intervals = build_ipls_intervals(n_features, interval_width)
    n_intv = len(intervals)

    interval_results = []
    for idx, (s, e) in enumerate(intervals):
        feat_idx = np.arange(s, e)
        rmse_cv, best_pc = evaluate_subset_cv(
            X_train_outer, y_train_outer, feat_idx, inner_cv, max_pc_cap
        )
        interval_results.append({"idx": idx, "rmse": rmse_cv, "pc": best_pc})

    interval_results.sort(key=lambda d: d["rmse"])
    selected = [interval_results[0]["idx"]]
    best_rmse_global = interval_results[0]["rmse"]
    best_pc_global = interval_results[0]["pc"]

    improved = True

    while improved and len(selected) < n_intv:
        improved = False
        best_candidate = None
        remaining = [i for i in range(n_intv) if i not in selected]

        for j in remaining:
            subset = sorted(selected + [j])
            feat_idx = np.concatenate(
                [np.arange(intervals[k][0], intervals[k][1]) for k in subset]
            )
            rmse_cv, best_pc = evaluate_subset_cv(
                X_train_outer, y_train_outer, feat_idx, inner_cv, max_pc_cap
            )
            if (best_candidate is None) or (rmse_cv < best_candidate["rmse"] - 1e-12):
                best_candidate = {
                    "subset": subset,
                    "rmse": rmse_cv,
                    "pc": best_pc
                }

        if best_candidate is not None and best_candidate["rmse"] < best_rmse_global - 1e-12:
            selected = best_candidate["subset"]
            best_rmse_global = best_candidate["rmse"]
            best_pc_global = best_candidate["pc"]
            improved = True

    feat_idx_final = np.concatenate(
        [np.arange(intervals[k][0], intervals[k][1]) for k in selected]
    )
    return feat_idx_final, int(best_pc_global), intervals, selected, best_rmse_global

def nested_cv_ipls_plsr(X: np.ndarray, y: np.ndarray, ids: np.ndarray, feature_names: List[str]) -> Dict:
    outer_cv = KFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)

    all_fold_metrics = []
    all_fold_preds = []
    all_fold_coefs = []

    for fold_id, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(X), start=1):
        X_train_outer, y_train_outer = X[train_outer_idx], y[train_outer_idx]
        X_test_outer, y_test_outer = X[test_outer_idx], y[test_outer_idx]
        ids_test = ids[test_outer_idx]

        inner_cv = KFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE)

        _, full_one_se_pc = evaluate_subset_cv(
            X_train_outer,
            y_train_outer,
            np.arange(X_train_outer.shape[1]),
            inner_cv,
            MAX_PC_CAP
        )

        feat_idx_final, best_pc, intervals, selected_intv, best_cv_rmse = ipls_forward_select(
            X_train_outer,
            y_train_outer,
            inner_cv,
            IPLS_INTERVAL_WIDTH,
            int(full_one_se_pc)
        )

        n_selected_features = int(len(feat_idx_final))
        n_total_features = int(X.shape[1])

        X_train_sel = X_train_outer[:, feat_idx_final]
        X_test_sel = X_test_outer[:, feat_idx_final]

        final_model = PLSR(pc=int(best_pc))
        res = final_model.fit(X_train_sel, y_train_outer)
        y_pred_test = final_model.predict(X_test_sel)

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
            "n_selected_features": n_selected_features,
            "n_total_features": n_total_features,
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
        feat_names_sel = [feature_names[i] for i in feat_idx_final]
        names = ["intercept"] + feat_names_sel
        coefs_df = pd.DataFrame({
            "fold": fold_id,
            "feature": names,
            "coef": np.concatenate([[intercept_val], coef])
        })
        all_fold_coefs.append(coefs_df)

        interval_str = ",".join(str(i) for i in selected_intv)
        print(
            f"[Outer Fold {fold_id}] pc={res.pc:02d}  "
            f"R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  RPD={m['RPD']:.3f}  "
            f"selected_features={n_selected_features}/{n_total_features}  "
            f"selected_intervals=[{interval_str}]  (CV-RMSE={best_cv_rmse:.4f})"
        )

    metrics_df = pd.DataFrame(all_fold_metrics)
    preds_df = pd.concat(all_fold_preds, ignore_index=True)
    coefs_df = pd.concat(all_fold_coefs, ignore_index=True)

    summary = {}
    for k in ["R2", "RMSE", "MAE", "Bias", "RPD", "RPIQ"]:
        mean_v = float(metrics_df[k].mean())
        std_v  = float(metrics_df[k].std(ddof=1))
        summary[k] = {"mean": mean_v, "std": std_v}
    summary["pc_per_fold"] = [int(x) for x in metrics_df["pc"].tolist()]
    summary["n_selected_features_per_fold"] = [int(x) for x in metrics_df["n_selected_features"].tolist()]
    summary["ipls_interval_width"] = int(IPLS_INTERVAL_WIDTH)

    return {
        "metrics_per_fold": metrics_df,
        "preds": preds_df,
        "coefs": coefs_df,
        "summary": summary
    }

def main():
    ensure_dir(OUT_DIR)
    print(f"读取数据：{INPUT_PATH}")
    ids, X, y, feature_names = read_dataset(INPUT_PATH, ID_COL, Y_COL)
    print(f"样本数={X.shape[0]}，特征数={X.shape[1]}（波长列）")

    if USE_IPLS:
        print(f"\n>>> 模式：iPLS + PLSR（区间宽度={IPLS_INTERVAL_WIDTH} 个变量）")
        results = nested_cv_ipls_plsr(X, y, ids, feature_names)
    else:
        print("\n>>> 模式：全谱 PLSR")
        results = nested_cv_plsr(X, y, ids, feature_names)

    metrics_path = os.path.join(OUT_DIR, "metrics_per_fold.csv")
    preds_path   = os.path.join(OUT_DIR, "predictions.csv")
    coefs_path   = os.path.join(OUT_DIR, "coefficients.csv")
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
    print(f"PC per fold: {results['summary']['pc_per_fold']}")
    if USE_IPLS and "n_selected_features_per_fold" in results["summary"]:
        print(f"Selected features per fold: {results['summary']['n_selected_features_per_fold']}")
        print(f"IPLS interval width (variables): {results['summary']['ipls_interval_width']}")
    print(f"\n已保存：\n- {metrics_path}\n- {preds_path}\n- {coefs_path}\n- {summary_path}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()

def run_regression(input_path: str, out_dir: str, use_ipls: bool = False):
    global INPUT_PATH, OUT_DIR, USE_IPLS
    INPUT_PATH = input_path
    OUT_DIR = out_dir
    USE_IPLS = bool(use_ipls)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
