import os
import json
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

INPUT_PATH = r'your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Data\Raw Data\shootout.csv'
OUT_DIR = r'your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Experiments\ipls and cars\shootout\cars_plsr_cv_results_no_full_lv_cap'

ID_COL = 'sample_id'
Y_COL = 'y'
N_OUTER = 5
N_INNER = 5
RANDOM_STATE = 42
MAX_PC_CAP = 30

USE_CARS = True
CARS_N_MC = 50
CARS_RATIO_SAMPLES = 0.9
CARS_MIN_VARS = 2

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

        n_samples, n_features = X.shape
        A_eff = int(min(self.pc, n_samples - 1, n_features))
        A_eff = max(1, A_eff)

        Xm = X.mean(axis=0)
        Ym = Y.mean(axis=0)
        Xc = X - Xm
        Yc = Y - Ym

        T, P, R, Q, U = pls_nipals(Xc, Yc, A_eff)
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

def find_best_pc_full(X_train_outer: np.ndarray,
                      y_train_outer: np.ndarray,
                      inner_cv: KFold,
                      max_pc_cap: int) -> Tuple[int, float]:
    n_samples, n_features = X_train_outer.shape
    max_pc = int(min(max_pc_cap, n_samples - 1, n_features))
    max_pc = max(1, max_pc)

    pc_cv_records = []
    for pc in range(1, max_pc + 1):
        rmses = []
        for tr_idx, va_idx in inner_cv.split(X_train_outer):
            X_tr, y_tr = X_train_outer[tr_idx], y_train_outer[tr_idx]
            X_va, y_va = X_train_outer[va_idx], y_train_outer[va_idx]

            model = PLSR(pc=pc)
            model.fit(X_tr, y_tr)
            y_hat = model.predict(X_va)
            rmses.append(rmse(y_va, y_hat))

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

    return int(one_se_record["pc"]), float(one_se_record["avg_rmse"])

def evaluate_subset_cv(
    X_train_outer: np.ndarray,
    y_train_outer: np.ndarray,
    feat_idx: np.ndarray,
    inner_cv: KFold,
    max_pc_cap: int,
) -> Tuple[float, int]:
    X_sub = X_train_outer[:, feat_idx]
    n_samples, n_features = X_sub.shape
    max_pc = int(min(max_pc_cap, n_samples - 1, n_features))
    max_pc = max(1, max_pc)

    pc_cv_records = []
    for pc in range(1, max_pc + 1):
        rmses = []
        for tr_idx, va_idx in inner_cv.split(X_sub):
            model = PLSR(pc=pc)
            model.fit(X_sub[tr_idx], y_train_outer[tr_idx])
            y_hat = model.predict(X_sub[va_idx])
            rmses.append(rmse(y_train_outer[va_idx], y_hat))

        avg_rmse = float(np.mean(rmses))
        se_rmse = float(np.std(rmses, ddof=1) / math.sqrt(len(rmses)))
        pc_cv_records.append({
            "pc": int(pc),
            "avg_rmse": avg_rmse,
            "se_rmse": se_rmse,
        })

    min_record = min(pc_cv_records, key=lambda d: (d["avg_rmse"], d["pc"]))
    threshold = float(min_record["avg_rmse"] + min_record["se_rmse"])
    one_se_record = min(
        [d for d in pc_cv_records if d["avg_rmse"] <= threshold + 1e-12],
        key=lambda d: d["pc"],
    )
    return float(one_se_record["avg_rmse"]), int(one_se_record["pc"])

def cars_plsr_select(
    X_train_outer: np.ndarray,
    y_train_outer: np.ndarray,
    inner_cv: KFold,
    max_pc_cap: int,
    n_mc: int = 50,
    ratio_samples: float = 0.9,
    min_vars: int = 2,
    random_state: int = 42
) -> Tuple[np.ndarray, int, float]:
    rng = np.random.default_rng(random_state)
    n_samples, p_total = X_train_outer.shape

    p0 = p_total
    min_vars = max(2, min(int(min_vars), p0))

    current_vars = np.arange(p_total, dtype=int)
    subsets = []
    cv_errors = []
    subset_pcs = []

    for i in range(n_mc):
        n_current = len(current_vars)

        _, current_pc = evaluate_subset_cv(
            X_train_outer,
            y_train_outer,
            current_vars,
            inner_cv,
            max_pc_cap,
        )

        if n_current <= min_vars:
            subset = current_vars.copy()
            cv_rmse, subset_pc = evaluate_subset_cv(
                X_train_outer,
                y_train_outer,
                subset,
                inner_cv,
                max_pc_cap,
            )
            subsets.append(subset)
            cv_errors.append(cv_rmse)
            subset_pcs.append(subset_pc)
            print(
                f"  [CARS] MC={i+1:02d}, n_vars={n_current}, "
                f"pc={subset_pc}, RMSECV={cv_rmse:.4f} (reached min_vars)"
            )
            break

        n_mc_samples = max(int(ratio_samples * n_samples), current_pc + 2)
        n_mc_samples = min(n_mc_samples, n_samples)
        mc_idx = rng.choice(n_samples, size=n_mc_samples, replace=False)
        X_mc = X_train_outer[mc_idx][:, current_vars]
        y_mc = y_train_outer[mc_idx]

        model_mc = PLSR(pc=current_pc)
        model_mc.fit(X_mc, y_mc)
        beta_current = model_mc.res_.beta.reshape(-1)
        abs_beta = np.abs(beta_current)

        if not np.any(abs_beta > 0):
            subset = current_vars.copy()
            cv_rmse, subset_pc = evaluate_subset_cv(
                X_train_outer,
                y_train_outer,
                subset,
                inner_cv,
                max_pc_cap,
            )
            subsets.append(subset)
            cv_errors.append(cv_rmse)
            subset_pcs.append(subset_pc)
            print(
                f"  [CARS] MC={i+1:02d}, n_vars={n_current}, "
                f"pc={subset_pc}, RMSECV={cv_rmse:.4f} (all betas ~0)"
            )
            break

        r_i = (min_vars / p0) ** (i / max(1, n_mc - 1))
        n_keep_target = int(round(p0 * r_i))
        n_keep_target = max(min_vars, min(n_keep_target, n_current))

        weights = abs_beta.astype(float)
        if weights.sum() <= 0:
            weights = np.ones_like(weights, dtype=float)
        prob = weights / weights.sum()
        chosen_local = rng.choice(n_current, size=n_keep_target, replace=False, p=prob)
        selected_vars = np.sort(current_vars[chosen_local])

        cv_rmse, subset_pc = evaluate_subset_cv(
            X_train_outer,
            y_train_outer,
            selected_vars,
            inner_cv,
            max_pc_cap,
        )

        subsets.append(selected_vars)
        cv_errors.append(cv_rmse)
        subset_pcs.append(subset_pc)
        current_vars = selected_vars

        print(
            f"  [CARS] MC={i+1:02d}, n_vars={len(selected_vars):4d}, "
            f"pc={subset_pc}, RMSECV={cv_rmse:.4f}"
        )

    cv_errors = np.array(cv_errors)
    best_idx = int(np.argmin(cv_errors))
    best_subset = subsets[best_idx]
    best_pc = int(subset_pcs[best_idx])
    best_rmse = float(cv_errors[best_idx])

    print(
        f"  [CARS] Best subset size = {len(best_subset)}, "
        f"best_pc = {best_pc}, best RMSECV = {best_rmse:.4f}"
    )
    return best_subset, best_pc, best_rmse

def nested_cv_cars_plsr(X: np.ndarray, y: np.ndarray, ids: np.ndarray, feature_names: List[str]) -> Dict:
    outer_cv = KFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)

    all_fold_metrics = []
    all_fold_preds = []
    all_fold_coefs = []

    for fold_id, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(X), start=1):
        print(f"\n===== [Outer Fold {fold_id}] CARS + PLSR =====")
        X_train_outer, y_train_outer = X[train_outer_idx], y[train_outer_idx]
        X_test_outer, y_test_outer = X[test_outer_idx], y[test_outer_idx]
        ids_test = ids[test_outer_idx]

        inner_cv = KFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE)

        best_subset, best_pc, best_cv_rmse = cars_plsr_select(
            X_train_outer,
            y_train_outer,
            inner_cv,
            max_pc_cap=MAX_PC_CAP,
            n_mc=CARS_N_MC,
            ratio_samples=CARS_RATIO_SAMPLES,
            min_vars=CARS_MIN_VARS,
            random_state=RANDOM_STATE + fold_id
        )

        n_selected_features = int(len(best_subset))
        n_total_features = int(X.shape[1])

        final_model = PLSR(pc=int(best_pc))
        res = final_model.fit(X_train_outer[:, best_subset], y_train_outer)
        y_pred_test = final_model.predict(X_test_outer[:, best_subset])

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
        feat_names_sel = [feature_names[i] for i in best_subset]
        names = ["intercept"] + feat_names_sel
        coefs_df = pd.DataFrame({
            "fold": fold_id,
            "feature": names,
            "coef": np.concatenate([[intercept_val], coef])
        })
        all_fold_coefs.append(coefs_df)

        print(
            f"[Outer Fold {fold_id}] pc={res.pc:02d}  "
            f"R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  RPD={m['RPD']:.3f}  "
            f"selected_features={n_selected_features}/{n_total_features}"
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
    summary["cars_n_mc"] = int(CARS_N_MC)
    summary["cars_ratio_samples"] = float(CARS_RATIO_SAMPLES)

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

    if USE_CARS:
        print(f"\n>>> 模式：CARS + PLSR (N_MC={CARS_N_MC}, ratio_samples={CARS_RATIO_SAMPLES})")
        results = nested_cv_cars_plsr(X, y, ids, feature_names)
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
    if USE_CARS and "n_selected_features_per_fold" in results["summary"]:
        print(f"Selected features per fold: {results['summary']['n_selected_features_per_fold']}")
        print(f"CARS N_MC={results['summary']['cars_n_mc']}, ratio_samples={results['summary']['cars_ratio_samples']}")
    print(f"\n已保存：\n- {metrics_path}\n- {preds_path}\n- {coefs_path}\n- {summary_path}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()

def run_regression(input_path: str, out_dir: str, use_cars: bool = True):
    global INPUT_PATH, OUT_DIR, USE_CARS
    INPUT_PATH = input_path
    OUT_DIR = out_dir
    USE_CARS = bool(use_cars)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
