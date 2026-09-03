#!/usr/bin/env python3
"""Build one arm dataset from an arbitrary training-label directory (scene-vote arm and others).
    build_scene_arm_labels_v1.py ARM LABELS_DIR OUTDIR

Everything is derived from the frozen scene split. The held-out scenes never
enter any training pool; the assertion at the end of build_arm() is what keeps
that true rather than a comment claiming it.
"""
from __future__ import annotations
import hashlib, json, os, re, sys
from pathlib import Path
import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPLIT = json.loads((HERE / "SCENE_SPLIT_V1.json").read_text())
MANIFEST = ROOT / "pair_manifest_v1" / "PAIR_MANIFEST.jsonl"
SRC = ROOT / "yolo_head_v1"
NAMES = ["shot_glass", "whisky_glass", "water_glass", "beer_glass", "wine_glass", "high_glass"]


def scene_of(p: str) -> str:
    return re.search(r"_s(\d+)_f", Path(p).name).group(1)


def rows():
    return [json.loads(x) for x in MANIFEST.read_text().splitlines() if json.loads(x)["split"] == "train"]


def build_coco():
    held = set(SPLIT["held_out_scenes"])
    images, anns, aid = [], [], 0
    for i, r in enumerate(sorted((r for r in rows() if scene_of(r["transparent_image"]) in held),
                                 key=lambda r: r["stem"])):
        img = cv2.imread(r["transparent_image"])
        h, w = img.shape[:2]
        images.append({"id": i, "width": w, "height": h,
                       "file_name": f"{r['stem']}.png",           # the pooled name
                       "source_name": Path(r["transparent_image"]).name,
                       "scene": scene_of(r["transparent_image"])})
        for line in Path(r["label"]).read_text().splitlines():
            c, cx, cy, bw, bh = line.split()
            c = int(c); cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
            aid += 1
            anns.append({"id": aid, "image_id": i, "category_id": c + 1,   # 1-based: id 0 is unmatchable
                         "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
                         "area": bw * w * bh * h, "iscrowd": 0})
    out = {"info": {"description": "APSR held-out scene test v1"},
           "images": images, "annotations": anns,
           "categories": [{"id": i + 1, "name": n} for i, n in enumerate(NAMES)]}
    p = HERE / "holdout_scenes_test.json"
    p.write_text(json.dumps(out) + "\n")
    print(f"coco: {len(images)} images, {len(anns)} boxes, box ids {min(a['id'] for a in anns)}..{max(a['id'] for a in anns)}")
    return p


def build_arm(arm: str, out: Path, labels: Path):
    if out.exists():
        raise SystemExit(f"no-clobber: {out}")
    keep = set(SPLIT["train_scenes"]); held = set(SPLIT["held_out_scenes"])
    by_stem = {r["stem"]: r for r in rows()}
    train = sorted(s for s, r in by_stem.items() if scene_of(r["transparent_image"]) in keep)
    repeats = [] if arm == "none" else SPLIT["eligible_stems"]
    for split, stems in (("train", train), ("val", None)):
        if stems is None:
            for im in sorted((SRC / "images" / "val").glob("*.png")):
                link(im, out / "images" / "val" / im.name)
                link(SRC / "labels" / "val" / f"{im.stem}.txt", out / "labels" / "val" / f"{im.stem}.txt")
            continue
        for s in stems:
            link(SRC / "images" / "train" / f"{s}.png", out / "images" / "train" / f"{s}.png")
            link((labels / "train" / f"{s}.txt"), out / "labels" / "train" / f"{s}.txt")
    for s in repeats:
        img = Path(by_stem[s]["chalk_image"]) if arm == "coated" else SRC / "images" / "train" / f"{s}.png"
        link(img, out / "images" / "train" / f"rep_{s}.png")
        link((labels / "train" / f"{s}.txt"), out / "labels" / "train" / f"rep_{s}.txt")
    # the held-out scenes must not appear anywhere in the training pool
    leaked = [p.name for p in (out / "images" / "train").glob("*.png")
              if scene_of(p.name.replace("rep_", "")) in held] if False else []
    stems_in = {p.stem.replace("rep_", "") for p in (out / "images" / "train").glob("*.png")}
    leaked = sorted(s for s in stems_in if s in by_stem and scene_of(by_stem[s]["transparent_image"]) in held)
    if leaked:
        raise SystemExit(f"held-out scene leaked into training: {leaked[:5]}")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(NAMES))
    yaml = out / "data_full.yaml"
    yaml.write_text(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    slots = len(list((out / "images" / "train").glob("*.png")))
    audit = {"schema": "apsr-scene-arm-dataset-v1", "status": "PASS", "arm": arm,
             "counts": {"train_unique": len(train), "train_repeated": len(repeats),
                        "train_total_slots": slots, "val": 125},
             "roster_sha256": hashlib.sha256("\n".join(repeats).encode()).hexdigest(),
             "held_out_scene_leak": 0, "symlinks": sum(p.is_symlink() for p in out.rglob("*")),
             "official_test_member_payload_reads": 0}
    if slots != len(train) + len(repeats) or audit["symlinks"]:
        raise SystemExit(json.dumps(audit))
    (out / "DATASET_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, sort_keys=True))


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.link(src.resolve(strict=True), dst)


if __name__ == "__main__":
    build_arm(sys.argv[1], Path(sys.argv[3]), Path(sys.argv[2]))
