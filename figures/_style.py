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
    font_path = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "Times New Roman.ttf"
    font_manager.fontManager.addfont(str(font_path))
    mpl.rcParams.update({
        "font.family": "Times New Roman",
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
