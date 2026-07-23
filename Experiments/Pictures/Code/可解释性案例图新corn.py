from pathlib import Path

CASE = {

    "dataset_name": "corn",
    "panel_label": "(a)",
    "panel_name": "Corn",

    "data_csv": Path(
        r"your address"
        r"\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection"
        r"\Data\Raw Data\corn.csv"
    ),

    "metrics_csv": Path(
        r"your address"
        r"\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection"
        r"\Experiments\wavelength selection\corn\plsr_joink\metrics_per_fold.csv"
    ),
    "ranking_dir": Path(
        r"your address"
        r"\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection"
        r"\Experiments\wavelength selection\corn\plsr_joink\EP"
    ),

    "ipls_coeff_csv": Path(
        r"your address"
        r"\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection"
        r"\Experiments\ipls and cars\corn\ipls_plsr_cv_results\coefficients.csv"
    ),
    "cars_coeff_csv": Path(
        r"your address"
        r"\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection"
        r"\Experiments\ipls and cars\corn\cars_plsr_cv_results_no_full_lv_cap\coefficients.csv"
    ),

    "n_outer_folds": 5,
    "ranking_pattern": "interval_ranking_outerfold{i}.json",

    "output_dir": Path(
        r"your address"
        r"\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection"
        r"\Experiments\Pictures\Result"
    ),
    "out_basename": "Fig6_corn_interpretability_case_final_consensus_only",

    "inkscape_exe": None,

    "plot_max_samples": None,
    "spectra_alpha": 0.12,
    "spectra_lw": 0.8,

    "epspec_facecolor": "#7A7A7A",
    "epspec_alpha": 0.16,
    "epspec_edgecolor": "#4A4A4A",
    "epspec_linewidth": 0.8,

    "ipls_color": "#4C78A8",
    "cars_color": "#F58518",
    "track_background_alpha": 0.08,
    "track_face_alpha": 0.38,
    "track_edge_alpha": 0.95,
    "track_linewidth": 1.1,

    "ipls_consensus_freq_threshold": 0.40,
    "ipls_max_gap_bins": 0,
    "ipls_min_segment_bins": 1,

    "cars_consensus_freq_threshold": 0.40,
    "cars_max_gap_bins": 0,
    "cars_min_segment_bins": 1,

    "prior_support": [
        (1426, 1465),
        (1760, 1795),
        (1915, 1975),
        (2085, 2115),
        (2265, 2365),
        (2470, 2515),
    ],

    "prior_risk": [
        (1179, 1430),
        (1448, 1545),
        (1680, 1740),
        (2035, 2195),
    ],

    "prior_caution": [
        (1535, 1565),
        (2068, 2098),
    ],
}

import json
import re
import statistics
import subprocess
import shutil

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba
plt.rcParams.update({
    "font.size": 17,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

def ikey(start, end, nd=1):
    return (round(float(start), nd), round(float(end), nd))

def clip_bands(bands, xmin, xmax):
    clipped = []
    for s, e in bands:
        s, e = float(s), float(e)
        if e <= xmin or s >= xmax:
            continue
        cs = max(s, xmin)
        ce = min(e, xmax)
        if ce > cs:
            clipped.append((cs, ce))
    return clipped

def load_wavelength_matrix(df: pd.DataFrame):
    wl_cols = []
    for c in df.columns:
        try:
            float(c)
            wl_cols.append(c)
        except Exception:
            pass

    if not wl_cols:
        for c in df.columns:
            m = re.search(r"(\d+(\.\d+)?)", str(c))
            if m:
                v = float(m.group(1))
                if 300 <= v <= 4000:
                    wl_cols.append(c)

    if not wl_cols:
        raise ValueError("Cannot detect wavelength columns in the dataset CSV.")

    wl = np.array(
        [
            float(re.search(r"(\d+(\.\d+)?)", str(c)).group(1))
            if not str(c).replace(".", "", 1).isdigit()
            else float(c)
            for c in wl_cols
        ]
    )
    order = np.argsort(wl)
    wl = wl[order]
    wl_cols_sorted = [wl_cols[i] for i in order]
    X = df[wl_cols_sorted].to_numpy(dtype=float)
    return wl, X

def wavelength_bin_edges(wl: np.ndarray):
    wl = np.asarray(wl, dtype=float)
    if wl.ndim != 1 or wl.size == 0:
        raise ValueError("wl must be a non-empty 1D array.")
    if wl.size == 1:
        return np.array([wl[0] - 0.5, wl[0] + 0.5], dtype=float)

    mids = (wl[:-1] + wl[1:]) / 2.0
    left = wl[0] - (wl[1] - wl[0]) / 2.0
    right = wl[-1] + (wl[-1] - wl[-2]) / 2.0
    return np.concatenate([[left], mids, [right]])

def find_inkscape(explicit=None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"INKSCAPE_EXE was set but not found: {p}")

    for name in ("inkscape.com", "inkscape"):
        found = shutil.which(name)
        if found:
            return Path(found)

    candidates = [
        Path(r"C:\Program Files\Inkscape\bin\inkscape.com"),
        Path(r"C:\Program Files\Inkscape\bin\inkscape.exe"),
        Path(r"C:\Program Files (x86)\Inkscape\bin\inkscape.com"),
        Path(r"C:\Program Files (x86)\Inkscape\bin\inkscape.exe"),
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Inkscape not found. Please install Inkscape and either:\n"
        "1) add it to PATH (recommended), or\n"
        "2) set CASE['inkscape_exe'] to the full path of inkscape.com"
    )

def svg_to_emf(svg_path: Path, emf_path: Path, inkscape_exe=None):
    inkscape = find_inkscape(inkscape_exe)

    cmd_variants = [
        [str(inkscape), str(svg_path), "--export-type=emf", f"--export-filename={emf_path}"],
        [str(inkscape), str(svg_path), f"--export-filename={emf_path}"],
        [str(inkscape), str(svg_path), f"--export-emf={emf_path}"],
        [str(inkscape), str(svg_path), "-M", str(emf_path)],
    ]

    last_err = None
    for cmd in cmd_variants:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
            return
        except Exception as e:
            last_err = e

    raise RuntimeError(f"SVG->EMF conversion failed. Last error: {last_err}")

def safe_read_best_k(metrics_csv: Path, n_folds: int):
    mp = pd.read_csv(metrics_csv)
    fold_col_candidates = ["outer_fold", "fold", "outer_fold_id", "fold_id"]
    fold_col = next((c for c in fold_col_candidates if c in mp.columns), None)

    if "best_k" not in mp.columns:
        raise ValueError("metrics_per_fold.csv must contain a column named 'best_k'.")

    if fold_col is not None:
        mp = mp.sort_values(by=fold_col)

    best_k = mp["best_k"].astype(int).tolist()
    if len(best_k) < n_folds:
        raise ValueError(f"best_k length must be >= {n_folds}, got {len(best_k)}")
    return best_k[:n_folds]

def parse_feature_value(v):
    s = str(v).strip()
    if s == "" or s.lower() == "intercept":
        return None
    try:
        return float(s)
    except Exception:
        m = re.search(r"(\d+(?:\.\d+)?)", s)
        if m:
            return float(m.group(1))
    return None

def load_method_feature_frequency(coeff_csv: Path, wl: np.ndarray, n_folds: int):
    if not coeff_csv.exists():
        raise FileNotFoundError(f"Missing coefficients CSV: {coeff_csv}")

    df = pd.read_csv(coeff_csv)
    need = {"fold", "feature"}
    if not need.issubset(df.columns):
        raise ValueError(
            f"{coeff_csv} must contain columns {sorted(need)}; got {list(df.columns)}"
        )

    df = df.copy()
    df["feature_parsed"] = df["feature"].map(parse_feature_value)
    df = df[df["feature_parsed"].notna()].copy()
    if df.empty:
        raise ValueError(f"No selectable wavelength features found in: {coeff_csv}")

    selected_per_fold = {}
    for fold, sub in df.groupby("fold"):
        try:
            fold_int = int(fold)
        except Exception:
            continue
        feats = np.sort(sub["feature_parsed"].astype(float).unique())
        selected_per_fold[fold_int] = feats.tolist()

    wl = np.asarray(wl, dtype=float)
    freq = np.zeros(wl.shape[0], dtype=float)
    wl_round_map = {round(float(v), 6): i for i, v in enumerate(wl)}
    med_step = float(np.median(np.diff(wl))) if wl.size > 1 else 1.0
    tol = max(1e-6, med_step * 0.51)

    missing_folds = [f for f in range(1, n_folds + 1) if f not in selected_per_fold]
    if missing_folds:
        print(
            f"[WARN] {coeff_csv.name}: missing folds {missing_folds}; "
            f"denominator still uses n_folds={n_folds}."
        )

    for fold in range(1, n_folds + 1):
        feats = selected_per_fold.get(fold, [])
        idxs = set()
        for feat in feats:
            key = round(float(feat), 6)
            if key in wl_round_map:
                idxs.add(wl_round_map[key])
                continue

            nearest = int(np.argmin(np.abs(wl - feat)))
            if abs(wl[nearest] - feat) <= tol:
                idxs.add(nearest)
            else:
                print(
                    f"[WARN] {coeff_csv.name}: feature {feat} in fold {fold} "
                    f"cannot be matched to wavelength grid; skipped."
                )

        for idx in idxs:
            freq[idx] += 1.0

    freq /= float(n_folds)
    return freq, selected_per_fold

def contiguous_true_segments(mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    segs = []
    i = 0
    n = mask.size
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        segs.append((i, j))
        i = j
    return segs

def fill_small_gaps(mask: np.ndarray, max_gap_bins: int):
    if max_gap_bins <= 0:
        return mask.copy()

    mask = np.asarray(mask, dtype=bool).copy()
    n = mask.size
    i = 0
    while i < n:
        if mask[i]:
            i += 1
            continue

        j = i + 1
        while j < n and not mask[j]:
            j += 1

        left_true = (i - 1 >= 0) and mask[i - 1]
        right_true = (j < n) and mask[j]
        gap_len = j - i

        if left_true and right_true and gap_len <= max_gap_bins:
            mask[i:j] = True

        i = j

    return mask

def drop_short_segments(mask: np.ndarray, min_segment_bins: int):
    if min_segment_bins <= 1:
        return mask.copy()

    mask = np.asarray(mask, dtype=bool).copy()
    for i0, i1 in contiguous_true_segments(mask):
        if (i1 - i0) < min_segment_bins:
            mask[i0:i1] = False
    return mask

def frequency_to_consensus_intervals(
    freq: np.ndarray,
    wl_edges: np.ndarray,
    threshold: float = 0.60,
    max_gap_bins: int = 0,
    min_segment_bins: int = 1,
):
    freq = np.asarray(freq, dtype=float)
    keep = freq >= float(threshold)

    keep = fill_small_gaps(keep, max_gap_bins=max_gap_bins)
    keep = drop_short_segments(keep, min_segment_bins=min_segment_bins)

    intervals = []
    for i0, i1 in contiguous_true_segments(keep):
        intervals.append((float(wl_edges[i0]), float(wl_edges[i1])))

    return intervals, keep

def style_strip_axis(ax, label, x=0.0):
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.text(
        x, 0.5, label,
        transform=ax.transAxes,
        ha="left", va="center",
        fontsize=17
    )
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.grid(False)

def draw_interval_track(
    ax,
    xmin,
    xmax,
    intervals,
    label,
    color,
    background_alpha=0.08,
    face_alpha=0.38,
    edge_alpha=0.95,
    linewidth=1.1,
):
    style_strip_axis(ax, label,x=-0.08)

    ax.add_patch(
        Rectangle(
            (xmin, 0.15),
            xmax - xmin,
            0.70,
            facecolor=to_rgba(color, background_alpha),
            edgecolor="none",
            zorder=0,
        )
    )

    for s, e in intervals:
        ax.add_patch(
            Rectangle(
                (s, 0.15),
                e - s,
                0.70,
                facecolor=to_rgba(color, face_alpha),
                edgecolor=to_rgba(color, edge_alpha),
                linewidth=linewidth,
                zorder=2,
            )
        )

def draw_topk_overlay_on_spectra(
    ax,
    intervals,
    facecolor="#7A7A7A",
    alpha=0.16,
    edgecolor="#4A4A4A",
    linewidth=0.8,
):
    ymin, ymax = ax.get_ylim()
    height = ymax - ymin

    for s, e in intervals:
        ax.add_patch(
            Rectangle(
                (s, ymin),
                e - s,
                height,
                facecolor=to_rgba(facecolor, alpha),
                edgecolor=edgecolor,
                linewidth=linewidth,
                zorder=0.2,
            )
        )

    ax.set_ylim(ymin, ymax)

def print_method_selection_summary(name, selected_per_fold):
    counts = {fold: len(v) for fold, v in sorted(selected_per_fold.items())}
    print(f"[{name}] selected feature counts per fold: {counts}")

def print_consensus_interval_summary(name, intervals):
    pretty = ", ".join([f"({s:.1f}, {e:.1f})" for s, e in intervals]) if intervals else "None"
    print(f"[{name}] consensus intervals: {pretty}")

def main():
    data_csv = CASE["data_csv"]
    metrics_csv = CASE["metrics_csv"]
    ranking_dir = CASE["ranking_dir"]
    ipls_coeff_csv = CASE["ipls_coeff_csv"]
    cars_coeff_csv = CASE["cars_coeff_csv"]
    n_folds = int(CASE["n_outer_folds"])

    out_dir = CASE["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    ranking_files = [
        ranking_dir / CASE["ranking_pattern"].format(i=i)
        for i in range(1, n_folds + 1)
    ]

    if not data_csv.exists():
        raise FileNotFoundError(f"Missing data CSV: {data_csv}")
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {metrics_csv}")
    for p in ranking_files:
        if not p.exists():
            raise FileNotFoundError(f"Missing ranking JSON: {p}")
    if not ipls_coeff_csv.exists():
        raise FileNotFoundError(f"Missing iPLS coefficients CSV: {ipls_coeff_csv}")
    if not cars_coeff_csv.exists():
        raise FileNotFoundError(f"Missing CARS coefficients CSV: {cars_coeff_csv}")

    df = pd.read_csv(data_csv)
    wl, X = load_wavelength_matrix(df)
    wl_edges = wavelength_bin_edges(wl)
    xmin, xmax = float(wl.min()), float(wl.max())

    rankings = []
    for p in ranking_files:
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if "interval_ranking" not in payload:
            raise KeyError(f"'interval_ranking' not found in: {p}")
        rankings.append(payload["interval_ranking"])

    all_keys = set()
    for r in rankings:
        for it in r:
            all_keys.add(ikey(it["start"], it["end"]))
    keys_sorted = sorted(all_keys, key=lambda k: k[0])

    rank_mat = {k: [] for k in keys_sorted}
    for r in rankings:
        rdict = {ikey(it["start"], it["end"]): it["rank"] for it in r}
        for k in keys_sorted:
            rank_mat[k].append(rdict.get(k, np.nan))
    avg_rank = {k: float(np.nanmean(rank_mat[k])) for k in keys_sorted}

    best_k = safe_read_best_k(metrics_csv, n_folds)
    k_star = int(statistics.median(best_k))

    selected_per_fold = []
    for fold_idx, r in enumerate(rankings):
        k = best_k[fold_idx]
        selected_per_fold.append([ikey(it["start"], it["end"]) for it in r[:k]])

    freq = {k: 0 for k in keys_sorted}
    for sel in selected_per_fold:
        for k in sel:
            freq[k] += 1
    freq = {k: v / len(selected_per_fold) for k, v in freq.items()}

    consensus_sorted = sorted(
        keys_sorted,
        key=lambda k: (-freq.get(k, 0), avg_rank.get(k, 1e9), k[0]),
    )
    consensus_topk = consensus_sorted[:k_star]
    print_consensus_interval_summary("EPSpec", consensus_topk)

    support_bands = clip_bands(CASE["prior_support"], xmin, xmax)
    caution_bands = clip_bands(CASE["prior_caution"], xmin, xmax)
    risk_bands = clip_bands(CASE["prior_risk"], xmin, xmax)

    ipls_freq, ipls_selected = load_method_feature_frequency(ipls_coeff_csv, wl, n_folds)
    cars_freq, cars_selected = load_method_feature_frequency(cars_coeff_csv, wl, n_folds)
    print_method_selection_summary("iPLS", ipls_selected)
    print_method_selection_summary("CARS", cars_selected)

    ipls_intervals, _ = frequency_to_consensus_intervals(
        ipls_freq,
        wl_edges,
        threshold=CASE["ipls_consensus_freq_threshold"],
        max_gap_bins=CASE["ipls_max_gap_bins"],
        min_segment_bins=CASE["ipls_min_segment_bins"],
    )
    cars_intervals, _ = frequency_to_consensus_intervals(
        cars_freq,
        wl_edges,
        threshold=CASE["cars_consensus_freq_threshold"],
        max_gap_bins=CASE["cars_max_gap_bins"],
        min_segment_bins=CASE["cars_min_segment_bins"],
    )
    print_consensus_interval_summary("iPLS", ipls_intervals)
    print_consensus_interval_summary("CARS", cars_intervals)

    fig = plt.figure(figsize=(11.2, 6.7))
    gs = plt.GridSpec(
        4,
        1,
        height_ratios=[2.45, 1.15, 0.92, 0.92],
        hspace=0.12,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    max_n = CASE["plot_max_samples"]
    idx = np.arange(X.shape[0])
    if isinstance(max_n, int) and max_n > 0 and max_n < X.shape[0]:
        idx = idx[:max_n]

    for i in idx:
        ax0.plot(
            wl,
            X[i, :],
            alpha=CASE["spectra_alpha"],
            linewidth=CASE["spectra_lw"],
            zorder=2,
        )

    ax0.set_xlim(xmin, xmax)
    ax0.set_ylabel("Absorbance", fontsize=17)
    ax0.set_title(
    f"{CASE['panel_label']} {CASE['panel_name']}",
    loc="center",
    pad=4,
    fontsize=18
)
    ax0.grid(False)

    draw_topk_overlay_on_spectra(
        ax0,
        consensus_topk,
        facecolor=CASE["epspec_facecolor"],
        alpha=CASE["epspec_alpha"],
        edgecolor=CASE["epspec_edgecolor"],
        linewidth=CASE["epspec_linewidth"],
    )

    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1.set_ylim(0, 3)
    ax1.set_yticks([2.5, 1.5, 0.5])
    ax1.set_yticklabels(["Support", "Caution", "Risk"], fontsize=16)
    ax1.set_ylabel("")
    ax1.tick_params(axis="y", pad=2, labelsize=16)
    ax1.text(
    -0.13, 1.02, "Prior",
    transform=ax1.transAxes,
    ha="left", va="bottom",
    fontsize=17
)

    lane_h = 0.75
    for s, e in support_bands:
        ax1.add_patch(
            Rectangle((s, 2.125), e - s, lane_h, fill=True, alpha=0.25, linewidth=0.8)
        )
    for s, e in caution_bands:
        ax1.add_patch(
            Rectangle((s, 1.125), e - s, lane_h, fill=False, hatch="xx", linewidth=0.9)
        )
    for s, e in risk_bands:
        ax1.add_patch(
            Rectangle((s, 0.125), e - s, lane_h, fill=False, hatch="///", linewidth=0.9)
        )

    ax1.spines[["right", "top"]].set_visible(False)
    ax1.grid(False)

    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    draw_interval_track(
        ax2,
        xmin,
        xmax,
        ipls_intervals,
        label="iPLS",
        color=CASE["ipls_color"],
        background_alpha=CASE["track_background_alpha"],
        face_alpha=CASE["track_face_alpha"],
        edge_alpha=CASE["track_edge_alpha"],
        linewidth=CASE["track_linewidth"],
    )

    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
    draw_interval_track(
        ax3,
        xmin,
        xmax,
        cars_intervals,
        label="CARS",
        color=CASE["cars_color"],
        background_alpha=CASE["track_background_alpha"],
        face_alpha=CASE["track_face_alpha"],
        edge_alpha=CASE["track_edge_alpha"],
        linewidth=CASE["track_linewidth"],
    )
    ax3.set_xlabel("Wavelength (nm)", fontsize=17)
    ax3.tick_params(axis="x", labelsize=16)

    for ax in [ax0, ax1, ax2]:
        plt.setp(ax.get_xticklabels(), visible=False)

    out_svg = out_dir / f"{CASE['out_basename']}.svg"
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    print(f"Saved SVG: {out_svg}")

    out_emf = out_dir / f"{CASE['out_basename']}.emf"
    try:
        svg_to_emf(out_svg, out_emf, inkscape_exe=CASE["inkscape_exe"])
        print(f"Saved EMF: {out_emf}")
    except Exception as e:
        print(f"[ERROR] SVG->EMF failed: {e}")
        print("        Please install Inkscape and either add it to PATH, or set CASE['inkscape_exe'].")

    plt.close(fig)

if __name__ == "__main__":
    main()
