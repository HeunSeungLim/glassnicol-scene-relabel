#!/usr/bin/env python3
"""Choose the OOD test frames for the qualitative comparison, by a stated rule.

The paper compares five training-label arms quantitatively; a reader still wants to see what the
difference looks like on a real frame.  Picking frames by eye would let the choice flatter us, so the
frames are chosen by a rule computed here and printed in the caption:

  disagreement  frames where the five arms disagree most about type on the manual boxes they localise;
  failure       the frame where our arm mistypes the most boxes that the released-label arm gets right.

Matching repeats the paper's scoring rule exactly: manual boxes in ascending annotation-id order greedily
take the unused prediction of maximum IoU, at IoU >= 0.5 and confidence >= 0.25, irrespective of type.
Predictions are the frozen ones that produced the tables, not a fresh inference pass.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "paper_work_v4" / "scenes_v1" / "evals_official"
GT = ROOT / "official_test_v1" / "ood_types_remap_1based.json"
ARMS = [("transparent", "Released labels"), ("lsmooth", "Label smoothing"),
        ("confid", "Detector confidence"), ("relabel", "Adjacent-frame vote"),
        ("scenevote", "Scene vote (ours)")]
CELL = "scene"          # accelerator A
SEED = 1337
IOU_T, CONF_T = 0.5, 0.25


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    i = w * h
    return i / (aw * ah + bw * bh - i) if i > 0 else 0.0


def match(gt_anns, preds):
    """Paper's greedy rule.  Returns {annotation_id: (category_id, score) or None}."""
    free = sorted(preds, key=lambda p: -p["score"])
    used, out = set(), {}
    for a in sorted(gt_anns, key=lambda a: a["id"]):
        best, bi = None, -1.0
        for k, p in enumerate(free):
            if k in used or p["score"] < CONF_T:
                continue
            v = iou(a["bbox"], p["bbox"])
            if v >= IOU_T and v > bi:
                best, bi = k, v
        if best is None:
            out[a["id"]] = None
        else:
            used.add(best)
            out[a["id"]] = (free[best]["category_id"], free[best]["score"])
    return out


def main():
    gt = json.loads(GT.read_text())
    by_img = defaultdict(list)
    for a in gt["annotations"]:
        if 1 <= a["category_id"] <= 6:
            by_img[a["image_id"]].append(a)
    images = {im["id"]: im for im in gt["images"]}

    matched = {}
    for arm, _ in ARMS:
        preds = json.loads((EV / f"ood_{CELL}_{arm}_s{SEED}" / "predictions_six_class.json").read_text())
        pim = defaultdict(list)
        for p in preds:
            pim[p["image_id"]].append(p)
        matched[arm] = {i: match(by_img[i], pim.get(i, [])) for i in by_img}

    rows = []
    for i, anns in by_img.items():
        if not anns:
            continue
        disagree = 0          # manual boxes on which the arms do not all agree on type
        ours_fixes = 0        # ours right where the released-label arm is wrong
        ours_breaks = 0       # ours wrong where the released-label arm is right
        for a in anns:
            got = [matched[arm][i][a["id"]] for arm, _ in ARMS]
            types = {g[0] for g in got if g}
            if len(types) > 1:
                disagree += 1
            raw, ours = matched["transparent"][i][a["id"]], matched["scenevote"][i][a["id"]]
            rc = bool(raw) and raw[0] == a["category_id"]
            oc = bool(ours) and ours[0] == a["category_id"]
            ours_fixes += oc and not rc
            ours_breaks += rc and not oc
        rows.append({"image_id": i, "file_name": images[i]["file_name"], "boxes": len(anns),
                     "disagree": disagree, "ours_fixes": ours_fixes, "ours_breaks": ours_breaks})

    rows.sort(key=lambda r: (-r["disagree"], -r["boxes"], r["file_name"]))
    print("== most type disagreement among the five arms ==")
    for r in rows[:8]:
        print(f"  {r['file_name']}  boxes {r['boxes']}  disagree {r['disagree']}  "
              f"ours fixes {r['ours_fixes']}  ours breaks {r['ours_breaks']}")
    worst = sorted(rows, key=lambda r: (-r["ours_breaks"], r["ours_fixes"], r["file_name"]))
    print("== ours worst against the released labels ==")
    for r in worst[:5]:
        print(f"  {r['file_name']}  boxes {r['boxes']}  breaks {r['ours_breaks']}  fixes {r['ours_fixes']}")
    (Path(__file__).parent / "QUALITATIVE_SELECTION_V18.json").write_text(
        json.dumps({"cell": f"{CELL}_s{SEED}", "split": "ood", "iou": IOU_T, "conf": CONF_T,
                    "frames": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
