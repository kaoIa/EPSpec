import os
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt

OUTPUT_DIR = (
    r"your address"
    r"\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection"
    r"\Experiments\Pictures\Result"
)
OUTPUT_STEM = "Ablation_4panel"

KEEP_TEMP_TEXTPATH_SVG = False

def set_paper_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.2,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.4,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.35,
        "lines.markersize": 4.8,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.default": "regular",
    })

BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#7A7A7A"
GRID = "#D9D9D9"

COLOR_EPSPEC = BLUE
COLOR_NOEP = ORANGE

COLOR_TECATOR = BLUE
COLOR_SOIL = ORANGE
COLOR_CORN = "#009E73"

HILITE = "#EAF2FB"

PANEL_LABEL_SIZE = 10
ANNOT_SIZE = 7.0

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SEGMENTATION_RESULT_ROOT = PROJECT_ROOT / "Experiments" / "Ablation" / "Results" / "滑动窗口和分段数"
MODEL_TEMP_RESULT_ROOT = PROJECT_ROOT / "Experiments" / "Ablation" / "Results" / "大模型和温度"

DATASETS = ("shootout", "soil", "corn")
DATASET_LABELS = {
    "shootout": "shootout",
    "soil": "soil",
    "corn": "corn",
}
DATASET_COLORS = {
    "shootout": COLOR_TECATOR,
    "soil": COLOR_SOIL,
    "corn": COLOR_CORN,
}
DATASET_MARKERS = {
    "shootout": "o",
    "soil": "s",
    "corn": "^",
}
DATASET_OFFSETS = {
    "shootout": 0.18,
    "soil": 0.0,
    "corn": -0.18,
}

def read_summary_r2(result_dir):
    summary_path = Path(result_dir) / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary.json: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return float(summary["R2"]["mean"])

def load_segmentation_r2():
    return {
        dataset: np.array([
            read_summary_r2(SEGMENTATION_RESULT_ROOT / dataset / f"plsr_n{int(n)}")
            for n in segments
        ], dtype=float)
        for dataset in DATASETS
    }

def load_llm_r2():
    return {
        dataset: np.array([
            read_summary_r2(MODEL_TEMP_RESULT_ROOT / dataset / f"plsr_joink-{suffix}")
            for suffix in llm_folder_suffixes
        ], dtype=float)
        for dataset in DATASETS
    }

def load_temperature_r2():
    return {
        dataset: np.array([
            read_summary_r2(MODEL_TEMP_RESULT_ROOT / dataset / f"plsr_joink-t{temp:.1f}")
            for temp in temps
        ], dtype=float)
        for dataset in DATASETS
    }

folds = ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5"]
x_fold = np.arange(1, len(folds) + 1)

tokens_epspec = np.array([19991, 20902, 25821, 23277, 22906], dtype=float)
tokens_noep   = np.array([65718, 66995, 66634, 66557, 66922], dtype=float)

tokens_epspec_k = tokens_epspec / 1000.0
tokens_noep_k   = tokens_noep / 1000.0

segments = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
r2_seg_by_dataset = load_segmentation_r2()

llm_names = ["GPT-5.2", "Gemini-3-pro", "DeepSeek-v3.2", "Qwen3-Max", "GLM-4.7"]
llm_folder_suffixes = ["gpt", "gemini", "deepseek", "qwen", "glm"]
r2_llm_by_dataset = load_llm_r2()

temps = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)
r2_temp_by_dataset = load_temperature_r2()

def add_panel_label(ax, label):
    ax.text(
        -0.14, 1.06, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold"
    )

def style_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, alpha=0.45, linewidth=0.8)
    ax.set_axisbelow(True)

def highlight_best_point(ax, x, y, color, text=None, dx=0.0, dy=0.0, ha="center", va="bottom"):
    ax.scatter(
        [x], [y],
        s=70,
        facecolors="none",
        edgecolors=color,
        linewidths=1.1,
        zorder=5
    )
    if text is not None:
        ax.text(
            x + dx, y + dy, text,
            fontsize=ANNOT_SIZE,
            color=color,
            ha=ha, va=va,
            zorder=6
        )

def add_vertical_band(ax, x0, x_values, color=HILITE, alpha=0.75):
    x_values = np.asarray(x_values, dtype=float)
    if len(x_values) >= 2:
        step = np.min(np.diff(np.unique(np.sort(x_values))))
        half_width = 0.42 * step
    else:
        half_width = 0.4
    ax.axvspan(x0 - half_width, x0 + half_width, color=color, alpha=alpha, zorder=0)

def add_horizontal_band(ax, y0, half_height=0.36, color=HILITE, alpha=0.75):
    ax.axhspan(y0 - half_height, y0 + half_height, color=color, alpha=alpha, zorder=0)

def add_corner_note(ax, text, xy=(0.98, 0.96), ha="right", va="top"):
    ax.text(
        xy[0], xy[1], text,
        transform=ax.transAxes,
        fontsize=ANNOT_SIZE,
        color=GRAY,
        ha=ha, va=va,
        zorder=6
    )

def find_inkscape_exe():
    candidates = [
        shutil.which("inkscape"),
        shutil.which("inkscape.exe"),
    ]

    env_pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    env_pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    env_local = os.environ.get("LOCALAPPDATA", "")

    common_paths = [
        os.path.join(env_pf, "Inkscape", "bin", "inkscape.exe"),
        os.path.join(env_pf, "Inkscape", "inkscape.exe"),
        os.path.join(env_pf86, "Inkscape", "bin", "inkscape.exe"),
        os.path.join(env_pf86, "Inkscape", "inkscape.exe"),
        os.path.join(env_local, "Programs", "Inkscape", "bin", "inkscape.exe"),
        os.path.join(env_local, "Programs", "Inkscape", "inkscape.exe"),
    ]

    candidates.extend(common_paths)

    for p in candidates:
        if p and os.path.isfile(p):
            return p

    return None

def export_emf_with_inkscape(svg_path, emf_path, keep_temp_svg=False):
    inkscape_exe = find_inkscape_exe()
    if inkscape_exe is None:
        raise RuntimeError("Cannot find Inkscape. Please install Inkscape or add it to PATH.")

    base, _ = os.path.splitext(emf_path)
    temp_svg = base + "_textpath_tmp.svg"

    cmd_textpath_svg = [
        inkscape_exe,
        svg_path,
        "--export-type=svg",
        "--export-text-to-path",
        f"--export-filename={temp_svg}",
    ]
    subprocess.check_call(cmd_textpath_svg)

    cmd_emf = [
        inkscape_exe,
        temp_svg,
        "--export-type=emf",
        f"--export-filename={emf_path}",
    ]
    subprocess.check_call(cmd_emf)

    if not keep_temp_svg and os.path.isfile(temp_svg):
        try:
            os.remove(temp_svg)
        except OSError:
            pass

def save_figure(fig, output_dir, output_stem):
    os.makedirs(output_dir, exist_ok=True)

    svg_path = os.path.join(output_dir, f"{output_stem}.svg")
    emf_path = os.path.join(output_dir, f"{output_stem}.emf")

    fig.savefig(svg_path, format="svg")

    print("Saved:")
    print(" -", svg_path)

    try:
        export_emf_with_inkscape(
            svg_path,
            emf_path,
            keep_temp_svg=KEEP_TEMP_TEXTPATH_SVG
        )
        print(" -", emf_path)
        if KEEP_TEMP_TEXTPATH_SVG:
            print(" -", os.path.splitext(emf_path)[0] + "_textpath_tmp.svg")
    except Exception as e:
        print("[Info] EMF export skipped:", str(e))
        print("[Info] SVG has been saved successfully.")

def panel_a_token_cost(ax):
    ax.plot(
        x_fold, tokens_epspec_k,
        color=COLOR_EPSPEC, marker="o", linewidth=1.35,
        label="EPSpec+PLSR", zorder=3
    )
    ax.plot(
        x_fold, tokens_noep_k,
        color=COLOR_NOEP, marker="s", linewidth=1.35, linestyle="--",
        label="no-EP", zorder=3
    )

    mean_epspec = float(tokens_epspec_k.mean())
    mean_noep = float(tokens_noep_k.mean())
    reduction_pct = (1.0 - mean_epspec / mean_noep) * 100.0
    ratio = mean_noep / mean_epspec

    ax.axhline(mean_epspec, color=COLOR_EPSPEC, linestyle=":", linewidth=1.0, alpha=0.9)
    ax.axhline(mean_noep, color=COLOR_NOEP, linestyle=":", linewidth=1.0, alpha=0.9)

    ax.text(
        0.98, 0.20,
        f"Mean: {mean_epspec:.1f}k vs {mean_noep:.1f}k\n"
        f"Reduction: {reduction_pct:.1f}%\n"
        f"Ratio: {ratio:.2f}×",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=ANNOT_SIZE,
        color=GRAY
    )

    ax.set_title("Token cost on corn")
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("Tokens (k)")
    ax.set_xticks(x_fold)
    ax.set_xticklabels(folds)

    ax.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.90),
        borderaxespad=0.0
    )

    style_axis(ax, grid_axis="y")
    add_panel_label(ax, "(a)")

def panel_b_segmentation(ax):
    seg_recommended = 40
    add_vertical_band(ax, seg_recommended, segments)

    all_y = []
    for dataset in DATASETS:
        values = r2_seg_by_dataset[dataset]
        mask = ~np.isnan(values)
        all_y.append(values[mask])
        ax.plot(
            segments[mask], values[mask],
            color=DATASET_COLORS[dataset],
            marker=DATASET_MARKERS[dataset],
            linewidth=1.35,
            label=DATASET_LABELS[dataset],
            zorder=3
        )

        max_val = np.nanmax(values)
        candidate_idx = np.where(np.isclose(values, max_val, atol=1e-6))[0]
        default_idx = np.where(segments == seg_recommended)[0]
        if len(default_idx) > 0 and default_idx[0] in candidate_idx:
            idx_best = int(default_idx[0])
        else:
            idx_best = int(candidate_idx[0])
        highlight_best_point(
            ax,
            segments[idx_best], values[idx_best],
            DATASET_COLORS[dataset],
            text=f"{values[idx_best]:.4f}",
            dy=0.0014
        )

    add_corner_note(ax, f"default: {int(seg_recommended)} segments", xy=(0.98, 0.96))

    all_y = np.concatenate(all_y)
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    pad = (y_max - y_min) * 0.18
    ax.set_ylim(y_min - pad, y_max + pad * 0.35)

    ax.set_title("Segmentation-number sensitivity")
    ax.set_xlabel("Number of segments")
    ax.set_ylabel(r"$R^2$ (outer CV)")
    ax.set_xticks(segments)
    ax.legend(frameon=False, loc="lower left")
    style_axis(ax, grid_axis="y")
    add_panel_label(ax, "(b)")

def panel_c_llm_backbone(ax):
    y_base = np.arange(len(llm_names))[::-1]

    recommended_name = "GPT-5.2"
    idx_rec = llm_names.index(recommended_name)
    recommended_y = y_base[idx_rec]

    add_horizontal_band(ax, recommended_y, half_height=0.36)

    all_x = []
    for dataset in DATASETS:
        values = r2_llm_by_dataset[dataset]
        y_values = y_base + DATASET_OFFSETS[dataset]
        all_x.append(values)
        ax.scatter(
            values, y_values,
            s=30,
            color=DATASET_COLORS[dataset],
            marker=DATASET_MARKERS[dataset],
            label=DATASET_LABELS[dataset],
            zorder=3
        )

        idx_best = int(np.argmax(values))
        ax.scatter(
            [values[idx_best]], [y_values[idx_best]],
            s=72, facecolors="none", edgecolors=DATASET_COLORS[dataset],
            linewidths=1.1, zorder=5
        )

    all_x = np.concatenate(all_x)
    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    pad = (x_max - x_min) * 0.22
    ax.set_xlim(x_min - pad, x_max + pad)

    add_corner_note(ax, f"default: {recommended_name}", xy=(0.98, 0.96))

    ax.set_yticks(y_base)
    ax.set_yticklabels(llm_names)
    ax.set_title("Backbone LLM comparison")
    ax.set_xlabel(r"$R^2$ (outer CV)")
    ax.set_ylabel("Backbone LLM")
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax, grid_axis="x")
    add_panel_label(ax, "(c)")

def panel_d_temperature(ax):
    temp_recommended = 0.0

    add_vertical_band(ax, temp_recommended, temps)

    all_y = []
    for dataset in DATASETS:
        values = r2_temp_by_dataset[dataset]
        all_y.append(values)
        ax.plot(
            temps, values,
            color=DATASET_COLORS[dataset],
            marker=DATASET_MARKERS[dataset],
            linewidth=1.35,
            label=DATASET_LABELS[dataset],
            zorder=3
        )

        idx_best = int(np.argmax(values))
        highlight_best_point(
            ax,
            temps[idx_best], values[idx_best],
            DATASET_COLORS[dataset],
            text=f"{values[idx_best]:.4f}",
            dx=0.03 if dataset == "soil" else 0.0,
            dy=0.0013,
            ha="left" if dataset == "soil" else "center"
        )

    add_corner_note(ax, f"default: T={temp_recommended:.1f}", xy=(0.98, 0.96))

    all_y = np.concatenate(all_y)
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    pad = (y_max - y_min) * 0.18
    ax.set_ylim(y_min - pad, y_max + pad * 0.30)

    ax.set_xlim(-0.03, 1.03)

    ax.set_title("Temperature sensitivity")
    ax.set_xlabel("Temperature")
    ax.set_ylabel(r"$R^2$ (outer CV)")
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax, grid_axis="y")
    add_panel_label(ax, "(d)")

def main():
    set_paper_style()

    fig, axes = plt.subplots(
        2, 2,
        figsize=(7.2, 5.4),
        constrained_layout=True
    )

    panel_a_token_cost(axes[0, 0])
    panel_b_segmentation(axes[0, 1])
    panel_c_llm_backbone(axes[1, 0])
    panel_d_temperature(axes[1, 1])

    save_figure(fig, OUTPUT_DIR, OUTPUT_STEM)
    plt.close(fig)

if __name__ == "__main__":
    main()
