from pathlib import Path
import copy
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

INPUT_SVGS = [
    Path(r"your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Experiments\Pictures\Result\Fig6_corn_interpretability_case_final_consensus_only.svg"),
    Path(r"your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Experiments\Pictures\Result\Fig6_soil_interpretability_case_final_consensus_only.svg"),
    Path(r"your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Experiments\Pictures\Result\Fig6_shootout_interpretability_case_final_consensus_only.svg"),
]

OUTPUT_DIR = Path(r"your address\EPSpec_ An Evidence-Guided, Prior-Retrieval Agent for NIR Band Selection\Experiments\Pictures\Result\可解释性案例图拼接结果")
OUT_BASENAME = "可解释性案例图拼接"

INKSCAPE_EXE = None

OUTER_MARGIN_PT = 0
COL_GAP_PT = 0
ROW_GAP_PT = 0
BACKGROUND_COLOR = "white"

KEEP_ORIGINAL_SIZE = True

TOP_LEFT_SCALE = 1.0
TOP_RIGHT_SCALE = 1.0
BOTTOM_SCALE = 1.0

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

def parse_svg_length(value: str):
    m = re.fullmatch(
        r"\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([A-Za-z%]*)\s*",
        str(value),
    )
    if not m:
        raise ValueError(f"Cannot parse SVG length: {value}")
    number = float(m.group(1))
    unit = m.group(2) or ""
    return number, unit

def length_to_pt(number: float, unit: str) -> float:
    unit = unit.lower()
    if unit in ("", "pt"):
        return number
    if unit == "px":
        return number * 72.0 / 96.0
    if unit == "in":
        return number * 72.0
    if unit == "cm":
        return number * 72.0 / 2.54
    if unit == "mm":
        return number * 72.0 / 25.4
    raise ValueError(f"Unsupported SVG unit: {unit}")

def get_svg_size_in_pt(root: ET.Element):
    width = root.get("width")
    height = root.get("height")
    if width is not None and height is not None:
        w_num, w_unit = parse_svg_length(width)
        h_num, h_unit = parse_svg_length(height)
        return length_to_pt(w_num, w_unit), length_to_pt(h_num, h_unit)

    view_box = root.get("viewBox")
    if view_box:
        vals = [float(x) for x in view_box.replace(",", " ").split()]
        if len(vals) != 4:
            raise ValueError(f"Invalid viewBox: {view_box}")
        return vals[2], vals[3]

    raise ValueError("SVG must have width/height or viewBox.")

def get_svg_viewbox_size(root: ET.Element):
    view_box = root.get("viewBox")
    if view_box:
        vals = [float(x) for x in view_box.replace(",", " ").split()]
        if len(vals) != 4:
            raise ValueError(f"Invalid viewBox: {view_box}")
        return vals[2], vals[3]
    return get_svg_size_in_pt(root)

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
        "1) add it to PATH, or\n"
        "2) set INKSCAPE_EXE to the full path of inkscape.com"
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
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return
        except Exception as e:
            last_err = e

    raise RuntimeError(f"SVG->EMF conversion failed. Last error: {last_err}")

IRI_ATTRS = {
    "fill",
    "stroke",
    "filter",
    "clip-path",
    "mask",
    "marker-start",
    "marker-mid",
    "marker-end",
}

def prefix_ids_and_refs(root: ET.Element, prefix: str):
    id_map = {}

    for elem in root.iter():
        old_id = elem.get("id")
        if old_id:
            new_id = f"{prefix}{old_id}"
            id_map[old_id] = new_id
            elem.set("id", new_id)

    url_pat = re.compile(r"url\(#([^)]+)\)")
    hash_pat = re.compile(r"^#(.+)$")

    def replace_refs(text: str):
        if not text:
            return text

        text = url_pat.sub(
            lambda m: f"url(#{id_map.get(m.group(1), m.group(1))})",
            text,
        )

        m = hash_pat.match(text)
        if m:
            old = m.group(1)
            if old in id_map:
                text = f"#{id_map[old]}"

        return text

    for elem in root.iter():
        for attr_name, attr_val in list(elem.attrib.items()):
            if attr_name == "id":
                continue
            if attr_name in IRI_ATTRS or attr_name.endswith("href") or attr_name == "style":
                elem.set(attr_name, replace_refs(attr_val))

        tag_name = elem.tag.split("}")[-1]
        if tag_name == "style" and elem.text:
            elem.text = replace_refs(elem.text)

    return root

def load_svg_for_merge(svg_path: Path, prefix: str):
    if not svg_path.exists():
        raise FileNotFoundError(f"Missing SVG: {svg_path}")

    tree = ET.parse(svg_path)
    root = tree.getroot()
    root = prefix_ids_and_refs(root, prefix)

    width_pt, height_pt = get_svg_size_in_pt(root)
    vb_width, vb_height = get_svg_viewbox_size(root)
    return root, width_pt, height_pt, vb_width, vb_height

def append_svg_group(parent: ET.Element, child_root: ET.Element, x_pt: float, y_pt: float, scale: float):
    g = ET.SubElement(
        parent,
        f"{{{SVG_NS}}}g",
        {
            "transform": f"translate({x_pt:.6f},{y_pt:.6f}) scale({scale:.8f})"
        },
    )
    for child in list(child_root):
        g.append(copy.deepcopy(child))

def compose_svgs_two_top_one_center(
    svg_paths,
    out_svg_path: Path,
    outer_margin_pt: float = 8.0,
    col_gap_pt: float = 18.0,
    row_gap_pt: float = 18.0,
    background_color: str = "white",
    keep_original_size: bool = True,
    top_left_scale: float = 1.0,
    top_right_scale: float = 1.0,
    bottom_scale: float = 1.0,
):
    if len(svg_paths) != 3:
        raise ValueError("svg_paths 必须正好给 3 张图。")

    loaded = []
    for i, svg_path in enumerate(svg_paths):
        loaded.append(load_svg_for_merge(svg_path, prefix=f"panel{i}_"))

    (root1, w1, h1, _, _), (root2, w2, h2, _, _), (root3, w3, h3, _, _) = loaded

    if keep_original_size:
        s1 = s2 = s3 = 1.0
    else:
        s1 = float(top_left_scale)
        s2 = float(top_right_scale)
        s3 = float(bottom_scale)

    top_row_width = w1 * s1 + col_gap_pt + w2 * s2
    bottom_row_width = w3 * s3
    total_width_pt = outer_margin_pt * 2 + max(top_row_width, bottom_row_width)

    top_row_height = max(h1 * s1, h2 * s2)
    bottom_row_height = h3 * s3
    total_height_pt = outer_margin_pt * 2 + top_row_height + row_gap_pt + bottom_row_height

    root_out = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "version": "1.1",
            "width": f"{total_width_pt:.6f}pt",
            "height": f"{total_height_pt:.6f}pt",
            "viewBox": f"0 0 {total_width_pt:.6f} {total_height_pt:.6f}",
        },
    )

    ET.SubElement(
        root_out,
        f"{{{SVG_NS}}}rect",
        {
            "x": "0",
            "y": "0",
            "width": f"{total_width_pt:.6f}",
            "height": f"{total_height_pt:.6f}",
            "fill": background_color,
        },
    )

    x1 = outer_margin_pt
    y1 = outer_margin_pt
    x2 = outer_margin_pt + w1 * s1 + col_gap_pt
    y2 = outer_margin_pt

    x3 = (total_width_pt - w3 * s3) / 2.0
    y3 = outer_margin_pt + top_row_height + row_gap_pt

    append_svg_group(root_out, root1, x1, y1, s1)
    append_svg_group(root_out, root2, x2, y2, s2)
    append_svg_group(root_out, root3, x3, y3, s3)

    out_svg_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root_out).write(out_svg_path, encoding="utf-8", xml_declaration=True)

def main():
    if len(INPUT_SVGS) != 3:
        raise ValueError("INPUT_SVGS 必须正好给 3 张子图路径。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_svg = OUTPUT_DIR / f"{OUT_BASENAME}.svg"
    out_emf = OUTPUT_DIR / f"{OUT_BASENAME}.emf"

    compose_svgs_two_top_one_center(
        svg_paths=INPUT_SVGS,
        out_svg_path=out_svg,
        outer_margin_pt=OUTER_MARGIN_PT,
        col_gap_pt=COL_GAP_PT,
        row_gap_pt=ROW_GAP_PT,
        background_color=BACKGROUND_COLOR,
        keep_original_size=KEEP_ORIGINAL_SIZE,
        top_left_scale=TOP_LEFT_SCALE,
        top_right_scale=TOP_RIGHT_SCALE,
        bottom_scale=BOTTOM_SCALE,
    )
    print(f"Saved merged SVG: {out_svg}")

    try:
        svg_to_emf(out_svg, out_emf, inkscape_exe=INKSCAPE_EXE)
        print(f"Saved merged EMF: {out_emf}")
    except Exception as e:
        print(f"[ERROR] SVG->EMF failed: {e}")
        print("        请安装 Inkscape，并加入 PATH，或手动指定 INKSCAPE_EXE。")

if __name__ == "__main__":
    main()
