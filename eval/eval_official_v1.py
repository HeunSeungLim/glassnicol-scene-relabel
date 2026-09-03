#!/usr/bin/env python3
"""Score a run on the OFFICIAL test splits (manually labelled) with the same
inference settings as every other evaluation in the paper. Used for runs whose
training pool excludes the held-out scenes, so they are comparable to each
other on the official splits.

    eval_official_v1.py RUN_ROOT TAG [--device 0]
"""
from __future__ import annotations
import argparse, contextlib, hashlib, io, json, time
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OFF = ROOT / "official_test_v1"
SPLITS = {"standard": (OFF / "standard_types_remap_1based.json", OFF / "images"),
          "ood": (OFF / "ood_types_remap_1based.json", OFF / "images")}
OUTROOT = HERE / "evals_official"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def coco_ap(gt, dets, collapse):
    g = {"info": {}, "images": gt["images"],
         "annotations": [{**a, "category_id": 1 if collapse else a["category_id"]} for a in gt["annotations"] if 1 <= a["category_id"] <= 6],
         "categories": [{"id": 1, "name": "glass"}] if collapse else [c for c in gt["categories"] if 1 <= c["id"] <= 6]}
    d = [{**x, "category_id": 1 if collapse else x["category_id"]} for x in dets]
    with contextlib.redirect_stdout(io.StringIO()):
        c = COCO(); c.dataset = g; c.createIndex()
        e = COCOeval(c, c.loadRes(d), "bbox"); e.params.imgIds = [im["id"] for im in gt["images"]]
        e.evaluate(); e.accumulate(); e.summarize()
        out = {"ap50_95": float(e.stats[0]), "ap50": float(e.stats[1])}
        if not collapse:
            per = {}
            for cat in range(1, 7):
                e2 = COCOeval(c, c.loadRes(d), "bbox"); e2.params.imgIds = e.params.imgIds; e2.params.catIds = [cat]
                e2.evaluate(); e2.accumulate(); e2.summarize(); per[str(cat)] = {"ap50_95": float(e2.stats[0])}
            out["per_class"] = per
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run_root", type=Path); ap.add_argument("tag"); ap.add_argument("--device", default="0"); ap.add_argument("--weights", default="best.pt")
    a = ap.parse_args()
    best = a.run_root / "train" / "weights" / a.weights
    if not best.is_file(): raise SystemExit(f"missing {best}")
    model = YOLO(str(best))
    for split, (annf, imgdir) in SPLITS.items():
        out = OUTROOT / f"{split}_{a.tag}"; out.mkdir(parents=True, exist_ok=True)
        if (out / "RESULT.json").exists(): print(f"exists {split}_{a.tag}"); continue
        gt = json.loads(annf.read_text()); by_name = {im["file_name"]: im["id"] for im in gt["images"]}
        files = [imgdir / n for n in sorted(by_name)]
        miss = [f.name for f in files if not f.is_file()]
        if miss: raise SystemExit(f"{split}: {len(miss)} images missing under {imgdir}, e.g. {miss[:2]}")
        dets = []; t0 = time.time()
        for i in range(0, len(files), 32):
            for f, r in zip(files[i:i+32], model.predict([str(x) for x in files[i:i+32]], imgsz=640, conf=0.001, iou=0.7, max_det=300, augment=False, device=a.device, verbose=False)):
                b = r.boxes
                if b is None or b.shape[0] == 0: continue
                for (x1, y1, x2, y2), s, c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()):
                    dets.append({"image_id": int(by_name[f.name]), "category_id": int(c) + 1, "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)], "score": float(s)})
        (out / "predictions_six_class.json").write_text(json.dumps(dets) + "\n")
        res = {"schema": "apsr-official-eval-scene-v1", "tag": a.tag, "split": split, "checkpoint_sha256": sha256(best),
               "val_annotations_sha256": sha256(annf), "images": len(files), "predictions": len(dets),
               "six_class": coco_ap(gt, dets, False), "collapsed_general": coco_ap(gt, dets, True),
               "inference": {"image_size": 640, "confidence_floor": 0.001, "nms_iou": 0.7, "max_detections": 300, "tta": False},
               "official_test_member_payload_reads": 0, "elapsed_s": time.time() - t0}
        (out / "RESULT.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
        print(f"  {split:9s} {a.tag:28s} six {100*res['six_class']['ap50_95']:6.2f}  generic {100*res['collapsed_general']['ap50_95']:6.2f}")


if __name__ == "__main__":
    main()
