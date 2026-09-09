#!/usr/bin/env python3
"""Confidence relabelling of the training type ids, the standard label-noise baseline for this dataset.

A detector is asked which type each released box really is, and the released id is replaced when the
detector disagrees confidently.  The judgement comes from a model that never saw the box: the 70 training
scenes are split into two scene-disjoint halves, a model is trained on each half, and every box is judged
by the model of the other half.  Without that, the judge has memorised the label it is judging.

    relabel_confidence_v1.py FOLD_A_WEIGHTS FOLD_B_WEIGHTS OUT_LABELS_DIR

Rules, fixed before the labels were written:
  - a released box is matched to the highest-IoU unused detection of the same image, IoU >= 0.5,
    class-agnostic, greedy in the order the boxes appear (the matcher the paper's type accuracy uses);
  - the type id is replaced when the matched detection's class differs and its confidence >= 0.5;
  - an unmatched box keeps its released id; no box moves, none is added and none is removed.

Writes the label tree and CONFIDENCE_RELABEL_STATS_V1.json beside it.
"""
from __future__ import annotations
import json, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "pool"; CODE = ROOT / "code"
SRC = POOL / "yolo_head_v1"
SPLIT = json.loads((CODE / "SCENE_SPLIT_V1.json").read_text())
NAMES = ["shot", "whisky", "water", "beer", "wine", "high"]
IOU, CONF = 0.5, float(__import__("os").environ.get("APSR_CONFID_TAU", "0.5"))


def scene_of(p: str) -> str:
    m = re.search(r"_s(\d+)_f", Path(p).name)
    assert m is not None, p
    return m.group(1)


def folds():
    scenes = sorted(SPLIT["train_scenes"])
    return {"a": scenes[0::2], "b": scenes[1::2]}


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> None:
    wa, wb, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    from ultralytics import YOLO
    rows = [json.loads(x) for x in (CODE / "PAIR_MANIFEST.jsonl").read_text().splitlines()
            if json.loads(x)["split"] == "train"]
    by_stem = {r["stem"]: r for r in rows}
    fold_of = {}
    for name, scenes in folds().items():
        for s in scenes:
            fold_of[s] = name
    models = {"a": YOLO(str(wa)), "b": YOLO(str(wb))}

    (out / "train").mkdir(parents=True, exist_ok=True)
    stats = Counter()
    pairs = Counter()
    for stem, row in sorted(by_stem.items()):
        src = SRC / "labels" / "train" / f"{stem}.txt"
        text = src.read_text()
        scene = scene_of(row["transparent_image"])
        judge = fold_of.get(scene)
        if judge is None:                      # a held-out scene: never judged, never trained on
            (out / "train" / f"{stem}.txt").write_text(text)
            stats["held_out_files"] += 1
            continue
        other = models["b" if judge == "a" else "a"]
        img = SRC / "images" / "train" / f"{stem}.png"
        res = other.predict(str(img), imgsz=640, conf=0.05, iou=0.7, device=0, verbose=False)[0]
        h, w = res.orig_shape
        boxes = res.boxes if res.boxes is not None else []
        det = [(int(b.cls.item()), float(b.conf.item()), [float(v) for v in b.xyxy[0].tolist()])
               for b in boxes]
        det.sort(key=lambda d: -d[1])
        used = set()
        lines = []
        for line in text.splitlines():
            if not line.strip():
                continue
            c, cx, cy, bw, bh = line.split()
            c = int(c); cx, cy, bw, bh = (float(v) for v in (cx, cy, bw, bh))
            box = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
            best, bi = IOU, None
            for i, (dc, dconf, dbox) in enumerate(det):
                if i in used:
                    continue
                v = iou(box, dbox)
                if v >= best:
                    best, bi = v, i
            stats["boxes"] += 1
            if bi is None:
                stats["unmatched"] += 1
            else:
                used.add(bi)
                dc, dconf, _ = det[bi]
                stats["matched"] += 1
                if dc != c and dconf >= CONF:
                    pairs[f"{NAMES[c]}->{NAMES[dc]}"] += 1
                    stats["changed"] += 1
                    c = dc
                elif dc != c:
                    stats["disagreed_below_threshold"] += 1
            lines.append(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out / "train" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats["files"] += 1

    result = {"schema": "apsr-confidence-relabel-v1", "iou_threshold": IOU, "confidence_threshold": CONF,
              "fold_split": "training scenes sorted, alternating; each box judged by the other half's model",
              "weights": {"fold_a": str(wa), "fold_b": str(wb)},
              "counts": dict(stats), "changed_pairs": dict(pairs.most_common()),
              "changed_pct": round(100 * stats["changed"] / max(1, stats["boxes"]), 3)}
    (out.parent / "CONFIDENCE_RELABEL_STATS_V1.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True)[:900])


if __name__ == "__main__":
    main()
