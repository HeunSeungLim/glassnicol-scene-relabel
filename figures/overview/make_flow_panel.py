# view 8 with the background inliers of the 6->8 homography drawn as displacement vectors (source -> destination) -> pB_flow.png
import sys, json
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman"]})
HERE = Path(__file__).resolve().parent; P = HERE.parents[1]; sys.path.insert(0, str(P / "relabel_v1")); import relabel_scene_v1 as S
by, stem_of = S.load_split("train"); fa, fb = 6, 8
ka, da, sh = S.features(S.IMG / "train" / f"{stem_of[('133', fa)]}.png", by["133"][fa]); kb, db, _ = S.features(S.IMG / "train" / f"{stem_of[('133', fb)]}.png", by["133"][fb])
good = [m for m, n in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2) if m.distance < .75 * n.distance]
src = np.float32([ka[m.queryIdx].pt for m in good]); dst = np.float32([kb[m.trainIdx].pt for m in good])
H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0); inl = inl.ravel().astype(bool)
img = cv2.cvtColor(cv2.imread(str(S.IMG / "train" / f"{stem_of[('133', fb)]}.png")), cv2.COLOR_BGR2RGB); Hh, W = img.shape[:2]
fg = plt.figure(figsize=(7.6, 4.28), dpi=100); ax = fg.add_axes([0, 0, 1, 1]); ax.set_xticks([]); ax.set_yticks([]); ax.imshow(img)
for k in np.where(inl)[0][::3]:
    ax.annotate("", xy=(dst[k, 0], dst[k, 1]), xytext=(src[k, 0], src[k, 1]), arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.6, shrinkA=0, shrinkB=0))
    ax.plot(dst[k, 0], dst[k, 1], "o", ms=4, color="#2ca02c", mec="black", mew=0.4)
for _, cx, cy, w, h in by["133"][fb]: ax.add_patch(Rectangle(((cx - w/2)*W, (cy - h/2)*Hh), w*W, h*Hh, fill=False, lw=2.2, ec="white", ls=":"))
ax.text(10, Hh - 12, f"{int(inl.sum())} inliers, {fa} → {fb}", fontsize=44, color="white", bbox=dict(fc="black", ec="none", pad=1.2, alpha=0.6), va="bottom")
ax.set_xlim(0, W); ax.set_ylim(Hh, 0); fg.savefig(HERE / "pB_flow.png", dpi=100); print("pB_flow written", int(inl.sum()))
