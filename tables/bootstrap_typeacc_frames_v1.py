#!/usr/bin/env python3
"""Paired bootstrap of the relabel - raw type-accuracy gain for every finished (host, seed, split) cell.
Statistic: type accuracy over the manual boxes localised by BOTH arms (IoU >= 0.5, conf >= 0.25),
relabelled arm minus raw arm.  Resampling unit for the interval the paper quotes: the test image (frame);
the box-level interval is written next to it for comparison only.  2000 draws, seed 20260902.
Writes BOOT_TYPEACC_FRAMES_V1.json and prints the earlier receipts beside the new values."""
from __future__ import annotations
import json, random, sys, statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from emit_relabel_table_v1 import EV, ANN, IOU, CONF, iou, ROWS  # same matching code as the paper table

DRAWS, SEED = 2000, 20260902
ARM_A = sys.argv[1] if len(sys.argv) > 1 else "transparent"; ARM_B = sys.argv[2] if len(sys.argv) > 2 else "relabel"
OUTNAME = "BOOT_TYPEACC_FRAMES_V1.json" if len(sys.argv) < 3 else f"BOOT_TYPEACC_FRAMES_{ARM_A}_{ARM_B}.json"

def per_box(tag, split):
    d = EV / f"{split}_{tag}"
    if not (d / "RESULT.json").exists(): return None
    gt = json.loads(ANN[split].read_text()); by = defaultdict(list)
    for a in gt["annotations"]:
        if 1 <= a["category_id"] <= 6: by[a["image_id"]].append(a)
    pr = defaultdict(list)
    for p in json.loads((d / "predictions_six_class.json").read_text()):
        if p["score"] >= CONF: pr[p["image_id"]].append(p)
    out = []  # (image_id, localised, type correct) in the table's fixed order
    for img, anns in sorted(by.items()):
        used = set()
        for a in sorted(anns, key=lambda x: x["id"]):
            best, bi = IOU, None
            for i, p in enumerate(pr[img]):
                if i in used: continue
                v = iou(a["bbox"], p["bbox"])
                if v >= best: best, bi = v, i
            if bi is None: out.append((img, 0, 0))
            else: used.add(bi); out.append((img, 1, int(pr[img][bi]["category_id"] == a["category_id"])))
    return out

def cell(host, seed, split):
    oa = per_box(f"{host}_{ARM_A}_s{seed}", split); ob = per_box(f"{host}_{ARM_B}_s{seed}", split)
    if oa is None or ob is None: return None
    assert [x[0] for x in oa] == [x[0] for x in ob]
    frames = sorted({x[0] for x in oa})
    both = defaultdict(list)                       # frame -> [(raw ok, relabel ok)]
    for (img, la, ta), (_, lb, tb) in zip(oa, ob):
        if la and lb: both[img].append((ta, tb))
    boxes = [d for img in frames for d in both[img]]
    point = 100 * (st.mean(b for _, b in boxes) - st.mean(a for a, _ in boxes))
    rng = random.Random(SEED); fd = []
    for _ in range(DRAWS):
        s = [d for img in (rng.choice(frames) for _ in frames) for d in both[img]]
        if s: fd.append(100 * (sum(b for _, b in s) - sum(a for a, _ in s)) / len(s))
    rng = random.Random(SEED); bd = []
    for _ in range(DRAWS):
        s = [rng.choice(boxes) for _ in boxes]
        bd.append(100 * (sum(b for _, b in s) - sum(a for a, _ in s)) / len(s))
    q = lambda v: [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)]
    return {"point_pp": round(point, 3), "n_frames": len(frames), "n_frames_with_both": len(both),
            "n_boxes_both": len(boxes), "ci95_frame": q(fd), "p_le_zero_frame": round(sum(x <= 0 for x in fd) / len(fd), 4),
            "ci95_box": q(bd), "draws": DRAWS, "seed": SEED}

def main():
    old = json.load(open(HERE / "RELABEL_VERDICT_V1.json"))["cells"]
    rtx = json.load(open(HERE / "BOOT_relabel_136_s1337.json"))
    res = {"schema": "apsr-relabel-typeacc-bootstrap-v1", "unit": "frame (box level for comparison only)",
           "definition": "type accuracy on manual boxes localised by both arms, relabel - raw, points", "cells": {}}
    for host, seed in ROWS:
        for split in ("standard", "ood"):
            c = cell(host, seed, split)
            if c is None: continue
            k = f"{host}_s{seed}_{split}"; res["cells"][k] = c
            prev = old.get(f"scene_s{seed}_{split}", {}).get("ci95") if host == "scene" else (rtx[split]["ci95"] if seed == 1337 else None)
            print(f"{k:24s} point {c['point_pp']:+6.2f}  frame {c['ci95_frame']}  box {c['ci95_box']}  "
                  f"nF {c['n_frames']} nB {c['n_boxes_both']}  | earlier receipt {prev}")
    res["arms"] = [ARM_A, ARM_B]; (HERE / OUTNAME).write_text(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
