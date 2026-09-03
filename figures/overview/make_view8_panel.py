# renders view 8 of scene 133 with released boxes (same style as pA1/pA2) -> pB_top.png
import sys, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman"]})
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path(".").resolve()
P = HERE.parents[1]; sys.path.insert(0, str(P / "relabel_v1")); import relabel_scene_v1 as S
by, stem_of = S.load_split("train"); f = 8; img = cv2.cvtColor(cv2.imread(str(S.IMG / "train" / f"{stem_of[('133', f)]}.png")), cv2.COLOR_BGR2RGB); Hh, W = img.shape[:2]
fg = plt.figure(figsize=(7.6, 4.28), dpi=100); ax = fg.add_axes([0, 0, 1, 1]); ax.set_xticks([]); ax.set_yticks([]); ax.imshow(img)
for c, cx, cy, w, h in by["133"][f]:
    x0, y0 = (cx - w/2) * W, (cy - h/2) * Hh; ax.add_patch(Rectangle((x0, y0), w * W, h * Hh, fill=False, lw=3.6, ec="#1f77b4"))
    ax.text(x0 + 3, y0 + 3, S.NAMES[c], fontsize=62, color="white", bbox=dict(fc="#1f77b4", ec="none", pad=1.0, alpha=0.92), va="top")
ax.set_xlim(0, W); ax.set_ylim(Hh, 0); fg.savefig(HERE / "pB_top.png", dpi=100); print("pB_top written")
