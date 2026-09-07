#!/usr/bin/env python3
"""Resampling-unit audit for the per-cell type-accuracy intervals.

The paper quotes a paired bootstrap over test IMAGES.  The premise of the method, however, is that the 25 views of
a scene show the same objects, so images inside one scene are not independent draws.  This script recomputes each
per-cell interval resampling whole SCENES (a cluster bootstrap) and prints the two side by side, so the paper can
say what the interval is a statement about.  Scene identity comes from the ORIGINAL annotation file names, the
same mapping check_scene_assoc_manual_v1.py uses.  Writes AUDIT_SCENE_BOOTSTRAP_V1.json."""
import json, random, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bootstrap_typeacc_frames_v1 import per_box            # identical matching code to the paper's table
from emit_relabel_table_v1 import ANN
ROOT = HERE.parents[1]; OFF = ROOT / "official_test_v1"
ORIG = {"standard": OFF / "annotations/coco_test_standard_glass_types.json",
        "ood": OFF / "annotations/coco_test_odd_glass_types.json"}
DRAWS, SEED = 2000, 20260902
CELLS = [("scene", 1337), ("scene", 3407), ("scene", 4567), ("scene136", 1337), ("scene136", 5678)]

def scene_of(split):
    d = json.loads(ANN[split].read_text()); o = json.loads(ORIG[split].read_text())
    pair = zip(sorted(d["images"], key=lambda x: x["id"]), sorted(o["images"], key=lambda x: x["id"]))
    return {r["id"]: re.search(r"scene_(\d+)", g["file_name"]).group(1) for r, g in pair}

def acc(rows):
    loc = [r for r in rows if r[1]]
    return 100.0 * sum(r[2] for r in loc) / len(loc) if loc else float("nan")

out = {"draws": DRAWS, "seed": SEED, "cells": {}}
for split in ["standard", "ood"]:
    smap = scene_of(split); scenes = sorted(set(smap.values()))
    out.setdefault("scenes_per_split", {})[split] = {s: sum(1 for v in smap.values() if v == s) for s in scenes}
    for host, seed in CELLS:
        a = per_box(f"{host}_transparent_s{seed}", split); b = per_box(f"{host}_scenevote_s{seed}", split)
        if a is None or b is None: continue
        assert [x[0] for x in a] == [x[0] for x in b]
        both = [(a[i][0], a[i][1] and b[i][1], a[i][2], b[i][2]) for i in range(len(a))]
        by_img = defaultdict(list)
        for img, loc, ca, cb in both: by_img[img].append((loc, ca, cb))
        point = (100.0 * sum(r[2] for v in by_img.values() for r in v if r[0]) / max(1, sum(1 for v in by_img.values() for r in v if r[0]))
                 - 100.0 * sum(r[1] for v in by_img.values() for r in v if r[0]) / max(1, sum(1 for v in by_img.values() for r in v if r[0])))
        by_scene = defaultdict(list)
        for img in by_img: by_scene[smap[img]].append(img)
        def gain(imgs):
            rows = [r for im in imgs for r in by_img[im] if r[0]]
            if not rows: return float("nan")
            return 100.0 * sum(r[2] for r in rows) / len(rows) - 100.0 * sum(r[1] for r in rows) / len(rows)
        per_scene = {s: round(gain(v), 2) for s, v in sorted(by_scene.items())}
        rnd = random.Random(SEED); imgs = sorted(by_img); sc = sorted(by_scene)
        bf = [gain(rnd.choices(imgs, k=len(imgs))) for _ in range(DRAWS)]
        bs = [gain([im for s in rnd.choices(sc, k=len(sc)) for im in by_scene[s]]) for _ in range(DRAWS)]
        f = np.nanpercentile(bf, [2.5, 97.5]); s_ = np.nanpercentile(bs, [2.5, 97.5])
        key = f"{host}_s{seed}_{split}"
        out["cells"][key] = {"point": round(point, 2), "per_scene": per_scene,
                             "frame_ci": [round(float(f[0]), 2), round(float(f[1]), 2)],
                             "scene_ci": [round(float(s_[0]), 2), round(float(s_[1]), 2)],
                             "frame_width": round(float(f[1] - f[0]), 2), "scene_width": round(float(s_[1] - s_[0]), 2),
                             "scene_ci_contains_zero": bool(s_[0] <= 0 <= s_[1]),
                             "frame_ci_contains_zero": bool(f[0] <= 0 <= f[1])}
        print(f"{key:26s} point {point:+6.2f}  frame {f[0]:+6.2f},{f[1]:+6.2f}  scene {s_[0]:+7.2f},{s_[1]:+7.2f}  x{(s_[1]-s_[0])/(f[1]-f[0]):.1f}")
json.dump(out, open(HERE / "AUDIT_SCENE_BOOTSTRAP_V1.json", "w"), indent=1)
z = [v for v in out["cells"].values() if v["scene_ci_contains_zero"]]
print(f"\nscene-level 구간이 0을 포함하는 셀: {len(z)} / {len(out['cells'])}")
