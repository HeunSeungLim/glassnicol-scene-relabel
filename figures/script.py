#!/usr/bin/env python3
"""Label-noise figure: the released automatic glass type disagrees on the same
object between adjacent views.

Panel (a): three flips, one per confusable pair. Each column is one physical
glass seen in view f (top) and view f+1 (bottom), cropped from the released
training frames with the released box drawn; the word under each crop is the
released class name of that box. Panel (b): 6x6 count matrix over every matched
adjacent-view pair of the training split (off-diagonal = the two views carry
different types, diagonal = same type). Single-hue sequential colour, counts
printed in every cell; no red/green encoding.

Everything drawn is read from source_data.json (written by build_source.py).
The figure is built at the spconf column width (86 mm), so 8 pt here is 8 pt
on the page; the script measures the emitted MediaBox and the text actually
embedded in the PDF and refuses to finish if anything renders below MIN_PT.

    script.py                      # writes output/ and copies to ../figure_labelnoise.{pdf,png}
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib as _mpl
_mpl.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Liberation Serif"], "mathtext.fontset": "stix"})
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
from _style import apply  # noqa: E402

SOURCE = HERE / "source_data.json"
MATRIX_KEY = "train_registered"          # the paper's procedure (see build_source.py)
COLUMN_MM = 86.0                          # spconf: (178 - 6) / 2 mm
PLACED_PT = COLUMN_MM / 25.4 * 72.27      # TeX points
MIN_PT = 8.0
PT = 8.0
BOX_COLOR = "#0072B2"                     # Okabe-Ito blue; the only hue besides the sequential map
CMAP = "Blues"                            # single hue, CVD-safe

FIG_W, FIG_H = COLUMN_MM / 25.4, 2.05     # inches
CROP = 0.50; GAP = 0.05; ROWLAB = 0.14    # panel (a) geometry, inches
CELL = 0.19; YLAB = 0.36                  # panel (b) geometry, inches


def ax_inches(fig, x, y, w, h):
    return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])


def text_inches(fig, x, y, s, **kw):
    return fig.text(x / FIG_W, y / FIG_H, s, **kw)


def crop_view(v):
    img = cv2.cvtColor(cv2.imread(v["image"]), cv2.COLOR_BGR2RGB)
    x0, y0, x1, y1 = v["crop_xyxy_px"]
    return img[y0:y1, x0:x1], (x0, y0)


def main() -> None:
    apply()
    mpl.rcParams.update({"font.size": PT, "axes.titlesize": PT, "axes.labelsize": PT,
                         "xtick.labelsize": PT, "ytick.labelsize": PT})
    src = json.loads(SOURCE.read_text())
    names = src["inputs"]["class_names"]
    examples = [src["examples"][k] for k in src["example_selection"]["target_pairs"]]
    mat = np.array(src[MATRIX_KEY]["matrix_symmetric_rows_then_cols"])
    n_pairs, n_flips = src[MATRIX_KEY]["matched_pairs"], src[MATRIX_KEY]["flips"]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    top = FIG_H - 0.04
    # ---------------- panel (a) ----------------
    text_inches(fig, 0.0, top, "(a) Same glass, two adjacent views", ha="left", va="top", fontsize=PT)
    y_hdr = top - 0.15
    y_crop1 = y_hdr - 0.11 - CROP
    y_lab1 = y_crop1 - 0.02
    y_crop2 = y_lab1 - 0.12 - CROP
    y_lab2 = y_crop2 - 0.02
    for c, ex in enumerate(examples):
        x = ROWLAB + c * (CROP + GAP)
        text_inches(fig, x + CROP / 2, y_hdr, f"s{int(ex['scene'])} f{ex['frames'][0]}→{ex['frames'][1]}",
                    ha="center", va="top", fontsize=PT)
        for r, (v, y_ax, y_lab) in enumerate(zip(ex["views"], (y_crop1, y_crop2), (y_lab1, y_lab2))):
            ax = ax_inches(fig, x, y_ax, CROP, CROP)
            im, (ox, oy) = crop_view(v)
            ax.imshow(im, interpolation="lanczos")
            bx0, by0, bx1, by1 = v["box_xyxy_px"]
            ax.add_patch(Rectangle((bx0 - ox, by0 - oy), bx1 - bx0, by1 - by0, fill=False,
                                   lw=0.9, ec=BOX_COLOR))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            text_inches(fig, x + CROP / 2, y_lab, v["class_name"], ha="center", va="top", fontsize=PT)
    for lab, y_ax in (("view f", y_crop1), ("view f+1", y_crop2)):
        text_inches(fig, 0.05, y_ax + CROP / 2, lab, rotation=90, ha="center", va="center", fontsize=PT)

    # ---------------- panel (b) ----------------
    xb = ROWLAB + 3 * CROP + 2 * GAP + 0.08 + YLAB
    mw = 6 * CELL
    text_inches(fig, xb - YLAB, top, "(b) Matched-pair type counts", ha="left", va="top", fontsize=PT)
    axb = ax_inches(fig, xb, y_hdr - 0.11 - mw, mw, mw)
    shade = np.log1p(mat) / np.log1p(mat.max())          # colour only; the printed number is the count
    shade_m = np.ma.masked_where(np.tril(np.ones_like(mat, dtype=bool), -1), shade)   # each unordered pair once
    axb.imshow(shade_m, cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    cmap = plt.get_cmap(CMAP)
    for i in range(6):
        for j in range(6):
            if i > j: continue
            rgb = cmap(shade[i, j])[:3]
            lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            axb.text(j, i, str(int(mat[i, j])), ha="center", va="center", fontsize=PT,
                     color="white" if lum < 0.5 else "black")
    axb.set_xticks(range(6)); axb.set_yticks(range(6))
    axb.set_xticklabels(names, rotation=90); axb.set_yticklabels(names)
    axb.tick_params(length=1.5, pad=1.5, width=0.4)
    axb.set_xticks(np.arange(-.5, 6, 1), minor=True); axb.set_yticks(np.arange(-.5, 6, 1), minor=True)
    axb.grid(which="minor", color="white", lw=0.5); axb.tick_params(which="minor", length=0)
    for s in axb.spines.values():
        s.set_linewidth(0.4)

    # every artist must lie inside the nominal canvas, otherwise bbox_inches="tight" widens the
    # PDF and \includegraphics would shrink the text below PT
    fig.canvas.draw()
    fb = fig.bbox
    overflow = []
    for t in fig.texts + [t for ax in fig.axes for t in ax.get_xticklabels() + ax.get_yticklabels() + ax.texts]:
        bb = t.get_window_extent()
        if bb.x0 < fb.x0 - 1 or bb.x1 > fb.x1 + 1 or bb.y0 < fb.y0 - 1 or bb.y1 > fb.y1 + 1:
            overflow.append((t.get_text(), [round(v / fig.dpi, 3) for v in (bb.x0, bb.y0, bb.x1, bb.y1)]))
    if overflow:
        raise RuntimeError(f"text outside the {FIG_W:.3f} x {FIG_H:.3f} in canvas: {overflow}")

    out = HERE / "output"; out.mkdir(exist_ok=True)
    pdf, png = out / "figure_labelnoise.pdf", out / "figure_labelnoise.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.005)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.005, dpi=300)
    plt.close(fig)

    # ---------------- legibility on the page ----------------
    natural = float(PdfReader(str(pdf)).pages[0].mediabox.width)
    scale = PLACED_PT / natural
    embedded = []
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTChar
        def walk(o):
            if isinstance(o, LTChar):
                if o.get_text().strip():
                    # pdfminer's .size is the glyph advance for text rotated by 90 degrees; the
                    # glyph box is exactly one font size tall along the glyph's own vertical, which
                    # is .height for upright text and .width for rotated text (matrix a, b)
                    a, b = o.matrix[:2]
                    embedded.append(round(o.height if abs(a) >= abs(b) else o.width, 3))
            elif hasattr(o, "_objs"):
                for c in o:
                    walk(c)
        for page in extract_pages(str(pdf)):
            walk(page)
    except ImportError:
        pass
    smallest_nominal = PT
    smallest_embedded = min(embedded) if embedded else None
    effective = (smallest_embedded if smallest_embedded else smallest_nominal) * scale
    report = {"status": "OK" if effective >= MIN_PT else "FAIL",
              "natural_width_pt": round(natural, 3), "natural_height_in": round(float(PdfReader(str(pdf)).pages[0].mediabox.height) / 72, 3),
              "placed_width_pt": round(PLACED_PT, 3), "includegraphics_scale": round(scale, 4),
              "smallest_nominal_pt": smallest_nominal,
              "smallest_embedded_pt": smallest_embedded, "n_embedded_glyphs": len(embedded),
              "smallest_effective_pt_on_page": round(effective, 2),
              "matrix_key": MATRIX_KEY, "matched_pairs": n_pairs, "flips": n_flips,
              "examples": [(e["pair"], e["scene"], e["frames"]) for e in examples]}
    (out / "LEGIBILITY.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))
    if effective < MIN_PT:
        raise RuntimeError(f"smallest text renders at {effective:.2f} pt on the page")
    shutil.copy(pdf, PARENT / "figure_labelnoise.pdf")
    shutil.copy(png, PARENT / "figure_labelnoise.png")


if __name__ == "__main__":
    main()
