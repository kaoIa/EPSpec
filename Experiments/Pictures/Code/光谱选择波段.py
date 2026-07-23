import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = r"your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Data\Raw Data\corn.csv"
OUT_DIR = r"your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Experiments\Pictures\Result"
OUT_SVG_NAME = "corn_spectra_with_band_pillars_demo.svg"

FIGSIZE = (3.8, 2.6)

LINEWIDTH = 0.7
ALPHA = 0.18
DRAW_MEAN = True
MEAN_LINEWIDTH = 2.0
TRANSPARENT_BG = True

BANDS = [
    (1200, 1300),
    (1450, 1550),
    (1900, 2050),
    (2200, 2350),
]
BAND_ALPHA = 0.14
BAND_EDGE = False

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

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    wl_cols, wls = extract_wavelength_columns(df.columns)
    if not wl_cols:
        raise ValueError("未找到形如 '1100nm' 的波长列，请检查 CSV 列名。")

    Y = df[wl_cols].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.set_xlim(1100, 2500)

    xmin, xmax = ax.get_xlim()
    for (a, b) in BANDS:

        a2, b2 = max(a, xmin), min(b, xmax)
        if a2 >= b2:
            continue
        ax.axvspan(
            a2, b2,
            alpha=BAND_ALPHA,
            zorder=0,
            linewidth=0.8 if BAND_EDGE else 0.0
        )

    ax.plot(wls, Y.T, linewidth=LINEWIDTH, alpha=ALPHA, zorder=2)

    if DRAW_MEAN:
        ax.plot(wls, Y.mean(axis=0), linewidth=MEAN_LINEWIDTH, alpha=1.0, zorder=3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

    fig.tight_layout(pad=0.15)

    out_path = os.path.join(OUT_DIR, OUT_SVG_NAME)
    fig.savefig(
        out_path,
        format="svg",
        transparent=TRANSPARENT_BG,
        bbox_inches="tight",
        pad_inches=0.02
    )
    plt.close(fig)
    print(f"Done. Exported SVG to: {out_path}")

if __name__ == "__main__":
    main()
