import os
import json
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

INPUT_PATH = r'your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Data\Raw Data\corn.csv'
OUT_DIR = r'your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Experiments\wavelength selection\corn\plsr_joink-new1'

ID_COL = 'sample_id'
Y_COL = 'y'

N_OUTER = 5
N_INNER = 5
RANDOM_STATE = 42
MAX_PC_CAP = 30

USE_EPSPEC_PIPELINE = True

K_LIST = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
          22, 24, 26, 28, 30, 32, 34, 36, 38, 40]

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
        std_v  = float(metrics_df[k].std(ddof=1))
        summary[k] = {"mean": mean_v, "std": std_v}
    summary["pc_per_fold"] = [int(x) for x in metrics_df["pc"].tolist()]

    return {
        "metrics_per_fold": metrics_df,
        "preds": preds_df,
        "coefs": coefs_df,
        "summary": summary
    }

from ep_interval_metrics_regression import compute_ep_for_fold
from ep_global_prior_and_ranking_regression import apply_global_prior_and_rank_intervals

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
        interval_masks: Dict[str, np.ndarray] = {}
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
                    start_nm = float(start_nm) if start_nm is not None else None
                    end_nm = float(end_nm) if end_nm is not None else None
                except Exception:
                    start_nm = end_nm = None
                if start_nm is None or end_nm is None:
                    continue
                mask = (wavelengths >= start_nm) & (wavelengths <= end_nm)
                interval_masks[iid] = mask

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

            num_intervals = len(ranked_ids)
        else:
            print(f"[Outer Fold {fold_id}] interval metrics 为空，后续只会在 inner CV 中搜索 pc（全谱）。")

        if num_intervals > 0:
            valid_k_list = [k for k in K_LIST if k <= num_intervals]
        else:
            valid_k_list = []

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
                top_ids = ranked_ids[:k_candidate]
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
                f"[Outer Fold {fold_id}] 联合调参未找到有效 (pc, K) 组合，"
                f"退化为全谱 pc=1。"
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

    summary: Dict[str, Dict] = {}
    for kk in ["R2", "RMSE", "MAE", "Bias", "RPD", "RPIQ"]:
        summary[kk] = {
            "mean": float(metrics_df[kk].mean()),
            "std": float(metrics_df[kk].std(ddof=1))
        }
    summary["pc"] = {
        "per_fold": [int(pc) for pc in metrics_df["pc"].tolist()]
    }
    summary["best_k"] = {
        "per_fold": [int(k) for k in metrics_df["best_k"].tolist()]
    }
    summary["n_features"] = {
        "per_fold": [int(n) for n in metrics_df["n_features"].tolist()]
    }

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
