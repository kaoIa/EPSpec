import os
import json
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

N_INTERVALS = 40
N_SPLITS = 5
N_REPEATS_STAB = 3
RANDOM_STATE = 42
EDGE_RATIO = 0.03
SNR_LOW_THRESHOLD = 10.0
MAX_PLS_COMPONENTS = 5
STABILITY_THRESHOLD = 0.4

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

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
        fold_r2s.append(r2_score(y_te, y_hat))

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
    snr = signal / (noise + 1e-6)
    return snr

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

    n_intervals = int(min(N_INTERVALS, n_waves))
    interval_indices: List[Tuple[int, int]] = []
    for i in range(n_intervals):
        start_idx = i * n_waves // n_intervals
        end_idx = (i + 1) * n_waves // n_intervals
        interval_indices.append((start_idx, end_idx))

    full_oof, _ = fit_pls_oof(
        X_full,
        y,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
        max_components=MAX_PLS_COMPONENTS,
    )
    baseline_r2 = r2_score(y, full_oof)

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
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE,
            max_components=MAX_PLS_COMPONENTS,
        )
        local_r2 = r2_score(y, oof_pred_seg)
        cv_std = float(np.std(fold_r2s))

        cv_std = float(min(cv_std, 1.0))

        stab_scores = []
        for rp in range(N_REPEATS_STAB):
            _, fold_r2s_rp = fit_pls_oof(
                X_seg,
                y,
                n_splits=N_SPLITS,
                random_state=RANDOM_STATE + rp,
                max_components=MAX_PLS_COMPONENTS,
            )
            stab_scores.append(np.mean(fold_r2s_rp))

        if len(stab_scores) == 0:
            stability = 0.0
        else:

            stability = float(np.mean([s >= STABILITY_THRESHOLD for s in stab_scores]))

        snr_band = compute_snr_band(X_seg)

        pos_ratio_start = (start_nm - wl_min) / (wl_range + 1e-6)
        pos_ratio_end = (wl_max - end_nm) / (wl_range + 1e-6)
        artifact_risk = 0.0
        if pos_ratio_start < EDGE_RATIO or pos_ratio_end < EDGE_RATIO:
            artifact_risk += 0.5
        if snr_band < SNR_LOW_THRESHOLD:
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

        interval_records.append({
            "interval_id": f"fold{outer_fold_id:02d}_seg_{seg_id:03d}",
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
        })
        oof_preds_for_all_intervals.append(oof_pred_seg)

    mean_spectra_per_interval: List[np.ndarray] = []
    for (s_idx, e_idx) in interval_indices:
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
            m = min(len(spec_i), len(spec_j))
            if m == 0:
                continue
            c = safe_corr(spec_i[:m], spec_j[:m])
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
            "n_splits": N_SPLITS,
            "n_repeats_stab": N_REPEATS_STAB,
            "stability_threshold": STABILITY_THRESHOLD,
            "edge_ratio": EDGE_RATIO,
            "snr_low_threshold": SNR_LOW_THRESHOLD,
            "max_pls_components": MAX_PLS_COMPONENTS,
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

    ep_result: Dict = {
        "info": info,
        "global_prior_knowledge": "",
        "intervals": interval_records,
    }

    json_name = f"{dataset_key}_interval_metrics_outerfold{outer_fold_id:02d}.json"
    json_path = os.path.join(out_dir, json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ep_result, f, indent=2, ensure_ascii=False)

    print(
        f"[EP] outer_fold={outer_fold_id} 完成 EP interval metrics 计算："
        f"{len(interval_records)} intervals -> {json_path}"
    )

    return ep_result

if __name__ == "__main__":

    print("This module is intended to be imported and used by EPSpec.")
