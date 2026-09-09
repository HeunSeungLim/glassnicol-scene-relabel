from __future__ import annotations

import matplotlib as mpl
from matplotlib import font_manager
from pathlib import Path


PALETTE = {
    "gt": "#2b7a3d",
    "baseline": "#243b5e",
    "ours": "#c0392b",
    "positive": "#2b7a3d",
    "negative": "#b03a2e",
}


def apply() -> None:
    # The submitted figure used the Times New Roman file of the machine it was drawn on, which is not ours
    # to redistribute.  Without it, fall back to the metric-compatible free faces the paper's LaTeX uses;
    # glyph shapes match, so the figure is the same drawing in a different cut of the same typeface.
    font_path = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "Times New Roman.ttf"
    family = "Times New Roman"
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
    else:
        have = {f.name for f in font_manager.fontManager.ttflist}
        family = next((n for n in ("TeX Gyre Termes", "Nimbus Roman", "Liberation Serif", "DejaVu Serif")
                       if n in have), "serif")
    mpl.rcParams.update({
        "font.family": family,
        "font.size": 7.5,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })
