#!/usr/bin/env python3
"""Validity check of the scene-level association on the hand-labelled official splits.
Manual types are constant per object, so every within-chain type disagreement is an association error
(two glasses merged) or a labelling error. Reports chains, disagreeing boxes and the offending pairs, for the
scene-level method (hops K, base-point tolerance T) and, for reference, for adjacent-frame IoU linking."""
from __future__ import annotations
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from relabel_scene_v1 import features, homography, base_and_width, iou_box, NAMES
import cv2
ROOT = Path(__file__).resolve().parents[2]; OFF = ROOT / "official_test_v1"
from emit_relabel_table_v1 import ANN
def imgdir(split):
    key = "standard" if split == "standard" else "odd"
    hits = [q for q in OFF.rglob("000_0.png") if key in str(q.relative_to(OFF)).lower()]
    if not hits: hits = [q for q in OFF.rglob("000_0.png")]
    return hits[0].parent
HOPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
TOL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
CUE = sys.argv[3] if len(sys.argv) > 3 else "base"
SUFFIX = (f"_h{HOPS}_t{TOL}" + (f"_{CUE}" if CUE != "base" else "")) if len(sys.argv) > 1 else ""

ORIG = {"standard": OFF / "annotations" / "coco_test_standard_glass_types.json", "ood": OFF / "annotations" / "coco_test_odd_glass_types.json"}
def load(annf, split):
    """Scene and frame come from the ORIGINAL annotation file names (scene_N ... frame), zipped by image id
    with the renumbered file, exactly as figures/labelnoise_v1/build_source.manual_split_matrices does."""
    d = json.loads(annf.read_text()); orig = json.loads(ORIG[split].read_text()); by = defaultdict(dict); path = {}; idir = imgdir(split)
    meta = {}
    for im_r, im_o in zip(sorted(d["images"], key=lambda x: x["id"]), sorted(orig["images"], key=lambda x: x["id"])):
        m = re.search(r"scene_(\d+)", im_o["file_name"]); fr = re.findall(r"(\d+)", im_o["file_name"])
        meta[im_r["id"]] = (m.group(1), int(fr[-1]))
    for im in d["images"]:
        s, f = meta[im["id"]]; by[s][f] = []; path[(s, f)] = (idir / im["file_name"], im["width"], im["height"], im["id"])
    byid = {im["id"]: im for im in d["images"]}
    for a in d["annotations"]:
        if not (1 <= a["category_id"] <= 6): continue
        im = byid[a["image_id"]]; s, f = meta[im["id"]]
        x, y, w, h = a["bbox"]; W, H = im["width"], im["height"]
        by[s][f].append([a["category_id"] - 1, (x + w / 2) / W, (y + h / 2) / H, w / W, h / H])
    return by, path

def chains_scene(by, path):
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
    for s in by:
        fs = sorted(by[s]); feats = {f: features(path[(s, f)][0], by[s][f]) for f in fs}
        for f in fs:
            for i in range(len(by[s][f])): find((s, f, i))
        for f in fs:
            for k in range(1, HOPS + 1):
                g = f + k
                if g not in by[s]: continue
                Hm = homography(feats[f], feats[g])
                if Hm is None: continue
                shape = feats[f][2]; Hh, W = shape
                tgt = [((bx[1] * W, (bx[2] + bx[4] / 2) * Hh), bx[3] * W) for bx in by[s][g]]
                cand = []
                for i, bx in enumerate(by[s][f]):
                    if CUE == "iou":
                        c_, cx, cy, w, h = bx
                        pts = np.float32([[[(cx - w/2)*W, (cy - h/2)*Hh], [(cx + w/2)*W, (cy + h/2)*Hh]]])
                        q = cv2.perspectiveTransform(pts, Hm)[0]; wb = [c_, float((q[0][0]+q[1][0])/2/W), float((q[0][1]+q[1][1])/2/Hh), float(abs(q[1][0]-q[0][0])/W), float(abs(q[1][1]-q[0][1])/Hh)]
                        for j, bj in enumerate(by[s][g]):
                            v = iou_box(wb, bj)
                            if v > 0.5: cand.append((1 - v, i, j))
                        continue
                    p, wmap = base_and_width(bx, Hm, shape)
                    for j, (q, wj) in enumerate(tgt):
                        d = float(np.hypot(*(p - np.float32(q)))); tol = TOL * 0.5 * (wmap + wj)
                        if d < tol: cand.append((d / tol, i, j))
                ui, uj = set(), set()
                for _, i, j in sorted(cand):
                    if i in ui or j in uj: continue
                    if union((s, f, i), (s, g, j)): ui.add(i); uj.add(j)
    ch = defaultdict(list)
    for s in by:
        for f in by[s]:
            for i, bx in enumerate(by[s][f]): ch[find((s, f, i))].append((s, f, i, bx[0]))
    return list(ch.values())

def report(chains):
    n = sum(len(c) for c in chains); dis = 0; pairs = Counter(); lens = Counter(len(c) for c in chains)
    for c in chains:
        cnt = Counter(l for *_, l in c); maj = cnt.most_common(1)[0][0]
        for *_, l in c:
            if l != maj: dis += 1; pairs[f"{NAMES[min(l, maj)]}/{NAMES[max(l, maj)]}"] += 1
    return {"boxes": n, "chains": len(chains), "boxes_in_chains_ge3": sum(len(c) for c in chains if len(c) >= 3),
            "median_len_of_boxes": float(np.median([len(c) for c in chains for _ in c])),
            "disagreeing_boxes": dis, "disagree_pct": round(100 * dis / n, 2), "pairs": pairs.most_common(6), "len_hist": sorted(lens.items())}

out = {"hops": HOPS, "tol": TOL}
for name in ("standard", "ood"):
    by, path = load(ANN[name], name)
    r = report(chains_scene(by, path)); out[name] = r
    print(name, json.dumps(r))
(Path(__file__).resolve().parent / f"SCENE_ASSOC_MANUAL_CHECK_V1{SUFFIX}.json").write_text(json.dumps(out, indent=1))
