import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, FormatStrFormatter, MultipleLocator

DATA_PATH = r"your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Data\Raw Data\shootout.csv"
OUT_DIR = r"your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Experiments\Pictures\Result"
OUT_SVG_NAME = "shootout_flat_600_1900_word_safe.svg"

FIGSIZE = (8.0, 1.35)
LINEWIDTH = 0.6
ALPHA = 0.16
DRAW_MEAN = True
MEAN_LINEWIDTH = 1.4
TRANSPARENT_BG = True

DOWNSAMPLE_STEP = 2

X_MIN = 600
X_MAX = 1900
X_MAJOR_STEP = 200
X_MINOR_STEP = 50

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["path.simplify"] = True
plt.rcParams["path.simplify_threshold"] = 0.7

def extract_wavelength_columns(columns):
    wl_pairs = []
    pattern = re.compile(r"^(\d+(?:\.\d+)?)nm$")
    for c in columns:
        m = pattern.match(str(c).strip())
        if m:
            wl_pairs.append((float(m.group(1)), c))
    wl_pairs.sort(key=lambda x: x[0])
    wls = [p[0] for p in wl_pairs]
    wl_cols = [p[1] for p in wl_pairs]
    return wl_cols, wls

def add_axis_arrowheads_and_xunit(ax, x_unit="(nm)", fontsize=7):

    ax.annotate(
        "", xy=(0.995, 0.0), xytext=(0.965, 0.0),
        xycoords=("axes fraction", "axes fraction"),
        arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black"),
    )

    ax.annotate(
        "", xy=(0.0, 0.995), xytext=(0.0, 0.965),
        xycoords=("axes fraction", "axes fraction"),
        arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black"),
    )

    ax.text(
        0.98, 0.02, x_unit,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=fontsize,
        clip_on=True
    )

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    wl_cols, wls = extract_wavelength_columns(df.columns)
    if not wl_cols:
        raise ValueError("未找到形如 '600nm' 的波长列，请检查 CSV 列名。")

    if DOWNSAMPLE_STEP > 1:
        wl_cols = wl_cols[::DOWNSAMPLE_STEP]
        wls = wls[::DOWNSAMPLE_STEP]

    keep_idx = [i for i, wl in enumerate(wls) if (wl >= X_MIN and wl <= X_MAX)]
    if not keep_idx:
        raise ValueError(f"裁剪后没有波长点，请检查 X_MIN={X_MIN}, X_MAX={X_MAX} 是否在数据范围内。")

    wl_cols = [wl_cols[i] for i in keep_idx]
    wls = [wls[i] for i in keep_idx]

    Y = df[wl_cols].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(wls, Y.T, linewidth=LINEWIDTH, alpha=ALPHA)
    if DRAW_MEAN:
        ax.plot(wls, Y.mean(axis=0), linewidth=MEAN_LINEWIDTH, alpha=1.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    ax.set_xlim(X_MIN, X_MAX)
    xticks = list(range(X_MIN, X_MAX + 1, X_MAJOR_STEP))
    if xticks[-1] != X_MAX:
        xticks.append(X_MAX)
    ax.set_xticks(xticks)
    ax.xaxis.set_minor_locator(MultipleLocator(X_MINOR_STEP))

    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.tick_params(axis="both", which="major", direction="out", length=3, width=0.9, labelsize=7, pad=1)
    ax.tick_params(axis="both", which="minor", direction="out", length=1.8, width=0.7)

    ax.set_xlabel("")
    ax.set_ylabel("")
    add_axis_arrowheads_and_xunit(ax, x_unit="(nm)", fontsize=7)

    ax.grid(False)

    ax.margins(x=0.01, y=0.08)

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

    fig.tight_layout(pad=0.06)

    out_path = os.path.join(OUT_DIR, OUT_SVG_NAME)
    fig.savefig(
        out_path,
        format="svg",
        transparent=TRANSPARENT_BG,
        bbox_inches="tight",
        pad_inches=0.12,
    )
    plt.close(fig)
    print(f"Done. Exported SVG to: {out_path}")

if __name__ == "__main__":
    main()
