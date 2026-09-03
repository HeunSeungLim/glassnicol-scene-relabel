#!/usr/bin/env python3
"""Workflow wireframe with REAL data panels (scene 133 of the training split, official test frame for the output).
Panels, left to right: (A) released views with automatic type ids; (B) table-plane registration: background ORB
inliers between two views; (C) base points carried into the next view and linked; (D) chains across views and the
majority vote; (E) relabelled label file (type id changed, box unchanged) -> detector -> real detections on a manual
test frame. Everything drawn here comes from released frames, released labels, our association code and a trained
scene-vote detector; nothing is generated."""
import json, sys, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Liberation Serif"], "mathtext.fontset": "stix"})
HERE = Path(__file__).resolve().parent; P = HERE.parents[1]; RL = P / "relabel_v1"
sys.path.insert(0, str(RL)); import relabel_scene_v1 as S
NAMES = S.NAMES; cmap = plt.get_cmap("tab10")
SCENE, VIEWS = "133", [6, 8, 10]
by, stem_of = S.load_split("train")
img_of = lambda f: cv2.cvtColor(cv2.imread(str(S.IMG / "train" / f"{stem_of[(SCENE, f)]}.png")), cv2.COLOR_BGR2RGB)
# chains for the scene (same code path as the labels)
fs = sorted(by[SCENE]); feats = {f: S.features(S.IMG / "train" / f"{stem_of[(SCENE, f)]}.png", by[SCENE][f]) for f in fs}
parent, members = {}, {}
def find(x):
    while parent.setdefault(x, x) != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(u, v):
    ru, rv = find(u), find(v)
    if ru == rv: return True
    fu, fv = members.setdefault(ru, {u[1]}), members.setdefault(rv, {v[1]})
    if fu & fv: return False
    parent[ru] = rv; members[rv] = fu | fv; members.pop(ru, None); return True
links = defaultdict(list)   # (f,g) -> [(i,j)]
for f in fs:
    for i in range(len(by[SCENE][f])): find((SCENE, f, i))
for f in fs:
    for k in (1, 2):
        g = f + k
        if g not in by[SCENE]: continue
        H = S.homography(feats[f], feats[g])
        if H is None: continue
        shape = feats[f][2]; Hh, W = shape
        tgt = [((bx[1]*W, (bx[2]+bx[4]/2)*Hh), bx[3]*W) for bx in by[SCENE][g]]; cand = []
        for i, bx in enumerate(by[SCENE][f]):
            p, wmap = S.base_and_width(bx, H, shape)
            for j, (q, wj) in enumerate(tgt):
                d = float(np.hypot(*(p - np.float32(q)))); tol = 0.5 * 0.5 * (wmap + wj)
                if d < tol: cand.append((d / tol, i, j))
        ui, uj = set(), set()
        for _, i, j in sorted(cand):
            if i in ui or j in uj: continue
            if union((SCENE, f, i), (SCENE, g, j)): ui.add(i); uj.add(j); links[(f, g)].append((i, j))
chain_of = {}; chains = defaultdict(list)
for f in fs:
    for i in range(len(by[SCENE][f])): r = find((SCENE, f, i)); chains[r].append((f, i)); chain_of[(f, i)] = r
roots = sorted(chains, key=lambda r: -len(chains[r])); cid = {r: k for k, r in enumerate(roots)}
maj = {r: Counter(by[SCENE][f][i][0] for f, i in chains[r]).most_common(1)[0][0] for r in roots}
# ORB inlier matches between views 6 and 8 (background only)
fa, fb = VIEWS[0], VIEWS[1]
ka, da, sh = feats[fa]; kb, db, _ = feats[fb]
good = [m for m, n in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2) if m.distance < .75 * n.distance]
src = np.float32([ka[m.queryIdx].pt for m in good]); dst = np.float32([kb[m.trainIdx].pt for m in good])
Hab, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0); inl = inl.ravel().astype(bool)
print("matches", len(good), "inliers", int(inl.sum()))
# real detections of a scene-vote detector on a manual test frame (official standard split)
from ultralytics import YOLO
det_model = YOLO(str(P / "seed_v1/runs/scene136_scenevote_s1337/train/weights/best.pt"))
test_dir = P.parent / "official_test_v1/images"; test_img = test_dir / "122_000026.png"   # chosen (OOD split): every manual box detected with the correct type, nothing else detected, boxes separated
res = det_model.predict(str(test_img), imgsz=640, conf=0.25, iou=0.7, device="cpu", verbose=False)[0]
dets = [(int(b.cls.item()), *b.xyxy[0].tolist(), float(b.conf.item())) for b in res.boxes]
print("test frame", test_img.name, "detections", len(dets))
# ---------------- panels (each rendered at a fixed pixel size, no axes) ----------------
from PIL import Image, ImageDraw, ImageFont
PW, PH = 760, 428                                   # 16:9 panel pixels
def fig_panel(w=PW, h=PH):
    f = plt.figure(figsize=(w/100, h/100), dpi=100); ax = f.add_axes([0, 0, 1, 1]); ax.set_xticks([]); ax.set_yticks([]); return f, ax
def save(f, name): f.savefig(HERE / name, dpi=100); plt.close(f); return HERE / name
def frame_panel(f, name, colour_by_chain=False, show_vote=False):
    fg, ax = fig_panel(); ax.imshow(img_of(f))
    for i, (c, cx, cy, w, h) in enumerate(by[SCENE][f]):
        r = chain_of[(f, i)]; col = cmap(cid[r] % 10) if colour_by_chain else "#1f77b4"
        cnt = Counter(by[SCENE][ff][ii][0] for ff, ii in chains[r]).most_common(2)
        changed = show_vote and maj[r] != c and len(chains[r]) > 1 and (len(cnt) == 1 or cnt[0][1] > cnt[1][1])
        x0, y0 = (cx - w/2) * W, (cy - h/2) * Hh
        ax.add_patch(Rectangle((x0, y0), w * W, h * Hh, fill=False, lw=3.6, ec=col, ls="--" if changed else "-"))
        txt = NAMES[c] + (f"$\\to${NAMES[maj[r]]}" if changed else "")
        ax.text(x0 + 3, y0 + 3, txt, fontsize=62, color="white", bbox=dict(fc=col, ec="none", pad=1.0, alpha=0.92), va="top")
        if colour_by_chain: ax.plot([cx * W], [(cy + h/2) * Hh], marker="o", ms=11, color=col, mec="black", mew=1.2)
    ax.set_xlim(0, W); ax.set_ylim(Hh, 0); return save(fg, name)
pA1 = frame_panel(VIEWS[0], "pA1.png"); pA2 = frame_panel(VIEWS[2], "pA2.png")
pD1 = frame_panel(VIEWS[0], "pD1.png", True, True); pD2 = frame_panel(VIEWS[2], "pD2.png", True, True)
# (B) matches: two frames stacked, inliers drawn
fg, ax = fig_panel(PW, 2 * PH + 8); pair = np.concatenate([img_of(fa), np.full((8, W, 3), 255, np.uint8), img_of(fb)], axis=0); ax.imshow(pair)
sel = np.where(inl)[0][::5]
for k in sel:
    xa, ya = src[k]; xb, yb = dst[k]; ax.plot([xa, xb], [ya, yb + Hh + 8], color="#2ca02c", lw=1.6, alpha=0.85)
ax.scatter(src[sel, 0], src[sel, 1], s=22, c="#2ca02c", edgecolors="black", linewidths=0.5); ax.scatter(dst[sel, 0], dst[sel, 1] + Hh + 8, s=22, c="#2ca02c", edgecolors="black", linewidths=0.5)
for _, cx, cy, w, h in by[SCENE][fa]: ax.add_patch(Rectangle(((cx - w/2)*W, (cy - h/2)*Hh), w*W, h*Hh, fill=False, lw=2.4, ec="white", ls=":"))
ax.text(10, 14, f"view {fa}", fontsize=34, color="white", bbox=dict(fc="black", ec="none", pad=1.2, alpha=0.6), va="top")
ax.text(10, Hh + 8 + 14, f"view {fb}", fontsize=34, color="white", bbox=dict(fc="black", ec="none", pad=1.2, alpha=0.6), va="top")
ax.text(W - 10, Hh - 12, f"{int(inl.sum())} background inliers", fontsize=32, color="white", bbox=dict(fc="black", ec="none", pad=1.2, alpha=0.6), va="bottom", ha="right")
ax.set_xlim(0, W); ax.set_ylim(2 * Hh + 8, 0); pB = save(fg, "pB.png")
# (C) base points carried into view fb and linked
fg, ax = fig_panel(); ax.imshow(img_of(fb))
for j, (c, cx, cy, w, h) in enumerate(by[SCENE][fb]):
    ax.add_patch(Rectangle(((cx - w/2)*W, (cy - h/2)*Hh), w*W, h*Hh, fill=False, lw=3.0, ec="#1f77b4"))
    ax.plot(cx*W, (cy + h/2)*Hh, marker="o", ms=16, color="#1f77b4", mec="black", mew=1.2)
for i, bx in enumerate(by[SCENE][fa]):
    p, wmap = S.base_and_width(bx, Hab, sh); ax.plot(p[0], p[1], marker="o", ms=22, mfc="none", mec="#ff7f0e", mew=4.5)
for i, j in links.get((fa, fb), []):
    p, _ = S.base_and_width(by[SCENE][fa][i], Hab, sh); c, cx, cy, w, h = by[SCENE][fb][j]
    ax.plot([p[0], cx*W], [p[1], (cy + h/2)*Hh], color="#ff7f0e", lw=3.5)
ax.set_xlim(0, W); ax.set_ylim(Hh, 0); pC = save(fg, "pC.png")
# (E2) detections on a manual test frame
fg, ax = fig_panel(); timg = cv2.cvtColor(cv2.imread(str(test_img)), cv2.COLOR_BGR2RGB); ax.imshow(timg)
# crop to the manual boxes (8 % margin, 16:9, clipped to the frame); the panel states it is cropped
_gtj = json.load(open(P.parent / "official_test_v1/ood_types_remap_1based.json")); _gid = [im_["id"] for im_ in _gtj["images"] if im_["file_name"] == test_img.name][0]
_bb = [a["bbox"] for a in _gtj["annotations"] if a["image_id"] == _gid and 1 <= a["category_id"] <= 6]
_x0 = min(b[0] for b in _bb); _y0 = min(b[1] for b in _bb); _x1 = max(b[0] + b[2] for b in _bb); _y1 = max(b[1] + b[3] for b in _bb)
_cw, _ch = (_x1 - _x0) * 1.16, (_y1 - _y0) * 1.16
if _cw / _ch < 16 / 9: _cw = _ch * 16 / 9
else: _ch = _cw * 9 / 16
_cx, _cy = (_x0 + _x1) / 2, (_y0 + _y1) / 2; _TW, _TH = timg.shape[1], timg.shape[0]
_cw, _ch = min(_cw, _TW), min(_ch, _TH); _lx = min(max(_cx - _cw / 2, 0), _TW - _cw); _ly = min(max(_cy - _ch / 2, 0), _TH - _ch)
CROP = (_lx, _ly, _lx + _cw, _ly + _ch); print("crop", [int(v) for v in CROP])
gtj = json.load(open(P.parent / "official_test_v1/ood_types_remap_1based.json")); gid = [im_["id"] for im_ in gtj["images"] if im_["file_name"] == test_img.name][0]
for a in gtj["annotations"]:
    if a["image_id"] == gid and 1 <= a["category_id"] <= 6:
        gx, gy, gw, gh = a["bbox"]; ax.add_patch(Rectangle((gx, gy), gw, gh, fill=False, lw=4.0, ec="white", ls="--"))
for c, x1, y1, x2, y2, cf in dets:
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=3.6, ec="#d62728"))
    if x2 > timg.shape[1] - 170: ax.text(x2 - 3, y1 + 3, f"{NAMES[c]} {cf:.2f}", fontsize=40, color="white", bbox=dict(fc="#d62728", ec="none", pad=1.0, alpha=0.92), va="top", ha="right")
    else: ax.text(x1 + 3, y1 + 3, f"{NAMES[c]} {cf:.2f}", fontsize=40, color="white", bbox=dict(fc="#d62728", ec="none", pad=1.0, alpha=0.92), va="top")
ax.set_xlim(CROP[0], CROP[2]); ax.set_ylim(CROP[3], CROP[1]); pE2 = save(fg, "pE2.png")
# ---------------- compose with PIL ----------------
FONT = "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf"; FONTB = "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-bold.otf"
import glob
if not Path(FONT).exists():
    c = glob.glob("/usr/share/**/texgyretermes-regular.otf", recursive=True) + glob.glob("/usr/share/**/NimbusRoman-Regular.otf", recursive=True); FONT = c[0]
    c = glob.glob("/usr/share/**/texgyretermes-bold.otf", recursive=True) + glob.glob("/usr/share/**/NimbusRoman-Bold.otf", recursive=True); FONTB = c[0]
fT = ImageFont.truetype(FONTB, 54); fS = ImageFont.truetype(FONT, 44); fM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 34) if Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf").exists() else fS
colw, gap, top, pad = 760, 70, 150, 14
CW = 5 * colw + 4 * gap + 2 * pad; CH = top + 2 * PH + 8 + pad
canvas = Image.new("RGB", (CW, CH), "white"); d = ImageDraw.Draw(canvas)
xs = [pad + k * (colw + gap) for k in range(5)]
titles = ["(A) Released views,\nautomatic type ids", f"(B) Table-plane homography\nview {fa} → {fb}, background ORB", f"(C) Base points carried\ninto view {fb} and linked", "(D) Chains across views,\nmajority type per chain", "(E) Relabelled ids, boxes kept;\ntrain and test"]
for k, t in enumerate(titles):
    d.multiline_text((xs[k], 10), t, font=fT, fill="black", spacing=6)
def paste(p, x, y): im = Image.open(p).convert("RGB"); canvas.paste(im, (x, y)); d.rectangle([x, y, x + im.width - 1, y + im.height - 1], outline="black", width=2)
paste(pA1, xs[0], top); paste(pA2, xs[0], top + PH + 8)
paste(pB, xs[1], top)
paste(pC, xs[2], top + (PH + 8) // 2)
paste(pD1, xs[3], top); paste(pD2, xs[3], top + PH + 8)
# (E1) label lines: released vs relabelled
x, y = xs[4], top; d.rectangle([x, y, x + colw - 1, y + PH - 1], outline="black", width=2, fill="#fafafa")
d.text((x + 16, y + 10), "released", font=fT, fill="black"); d.text((x + colw // 2 + 16, y + 10), "relabelled", font=fT, fill="black")
f0 = VIEWS[0]; raw_lines = (S.SRC / "train" / f"{stem_of[(SCENE, f0)]}.txt").read_text().splitlines(); new_lines = (RL / "labels_scene/train" / f"{stem_of[(SCENE, f0)]}.txt").read_text().splitlines()
yy = y + 84
for a, b in zip(raw_lines[:5], new_lines[:5]):
    ra, rb = a.split(), b.split(); ca, cb = NAMES[int(ra[0])], NAMES[int(rb[0])]; coords = " ".join(f"{float(v):.2f}" for v in ra[1:])
    coords = " ".join(f"{float(v):.2f}" for v in ra[1:3])
    d.text((x + 16, yy), f"{ca:6s} {coords}", font=fM, fill="black"); d.text((x + colw // 2 + 16, yy), f"{cb:6s} {coords}", font=fM, fill=("#d62728" if ca != cb else "black")); yy += 48
d.text((x + 16, y + PH - 56), f"view {f0}: type ids change, boxes do not", font=fS, fill="black")
paste(pE2, xs[4], top + PH + 8)
d.rectangle([xs[4] + 2, top + PH + 10, xs[4] + colw - 3, top + PH + 10 + 58], fill=(0, 0, 0)); d.text((xs[4] + 12, top + PH + 14), "YOLO11n (relabelled set) on a manual test frame", font=ImageFont.truetype(FONT, 36), fill="white")
# arrows between columns at mid height
def arrow(x0, x1, y, dashed=False):
    if dashed:
        for t in range(x0, x1 - 20, 22): d.line([t, y, min(t + 12, x1 - 20), y], fill="black", width=4)
    else: d.line([x0, y, x1 - 16, y], fill="black", width=7)
    d.polygon([(x1, y), (x1 - 30, y - 16), (x1 - 30, y + 16)], fill="black")
ym = top + PH + 4
for k in range(4): arrow(xs[k] + colw + 8, xs[k + 1] - 8, ym)
canvas.save(HERE / "wire_v2.png"); print("composite", canvas.size)
json.dump({"scene": SCENE, "views": VIEWS, "orb_matches": len(good), "inliers": int(inl.sum()), "links_6_8": links.get((fa, fb), []),
           "test_frame": test_img.name, "detector": "scene136_scenevote_s1337/best.pt", "detections": dets, "chains": len(roots)}, open(HERE / "source_data.json", "w"), indent=1)
print("wireframe written")
