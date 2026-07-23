import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, FormatStrFormatter

DATA_PATH = r"your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Data\Raw Data\corn.csv"
OUT_DIR = r"your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Experiments\Pictures\Result"
OUT_SVG_NAME = "corn_all_spectra_axes_arrows_xunit_no_yname.svg"

FIGSIZE = (3.8, 2.8)

LINEWIDTH = 0.7
ALPHA = 0.18
DRAW_MEAN = True
MEAN_LINEWIDTH = 2.0
TRANSPARENT_BG = True

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

def add_axis_arrowheads_and_xunit(ax, x_unit="(nm)", fontsize=8):

    ax.annotate(
        "", xy=(1.0, 0.0), xytext=(0.97, 0.0),
        xycoords=("axes fraction", "axes fraction"),
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black"),
        annotation_clip=False
    )

    ax.annotate(
        "", xy=(0.0, 1.0), xytext=(0.0, 0.97),
        xycoords=("axes fraction", "axes fraction"),
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black"),
        annotation_clip=False
    )

    ax.text(
        0.995, 0.02, x_unit,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=fontsize
    )

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    wl_cols, wls = extract_wavelength_columns(df.columns)
    if not wl_cols:
        raise ValueError("未找到形如 '1100nm' 的波长列，请检查 CSV 列名。")

    Y = df[wl_cols].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(wls, Y.T, linewidth=LINEWIDTH, alpha=ALPHA)

    if DRAW_MEAN:
        ax.plot(wls, Y.mean(axis=0), linewidth=MEAN_LINEWIDTH, alpha=1.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.set_xticks([1100, 1500, 2000, 2500])
    ax.set_xlim(1100, 2500)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.tick_params(axis="both", which="major", direction="out", length=4, width=1.0, labelsize=8)
    ax.tick_params(axis="both", which="minor", direction="out", length=2, width=0.8)

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.grid(False)

    add_axis_arrowheads_and_xunit(ax, x_unit="(nm)", fontsize=8)

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

    fig.tight_layout(pad=0.25)

    out_path = os.path.join(OUT_DIR, OUT_SVG_NAME)
    fig.savefig(
        out_path,
        format="svg",
        transparent=TRANSPARENT_BG,
        bbox_inches="tight",
        pad_inches=0.06,
    )
    plt.close(fig)
    print(f"Done. Exported SVG to: {out_path}")

if __name__ == "__main__":
    main()
