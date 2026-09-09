#!/usr/bin/env python3
"""Build one half of the training roster, for the out-of-fold predictions the confidence baseline needs.

Confidence relabelling asks a detector which type a box really is.  A detector trained on the box it is
judging has memorised that box, so the judgement has to come from a model that never saw it.  This builds
the two scene-disjoint halves of the 70 training scenes; each half trains a model that later predicts on
the other half only.  Fold models are never scored as arms.

    build_fold_dataset_v1.py {a|b} LABELS_DIRNAME OUTDIR

The split is by scene, sorted, alternating, so it is fixed by the roster alone and carries no choice.
Labels come from LABELS_DIRNAME (the raw released labels for this baseline).  Repeated-exposure slots are
dropped: fold models exist to predict, not to match the arms' optimiser budget.
"""
from __future__ import annotations
import hashlib, json, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "pool"; CODE = ROOT / "code"
SRC = POOL / "yolo_head_v1"
SPLIT = json.loads((CODE / "SCENE_SPLIT_V1.json").read_text())
NAMES = ["shot_glass", "whisky_glass", "water_glass", "beer_glass", "wine_glass", "high_glass"]


def scene_of(p: str) -> str:
    m = re.search(r"_s(\d+)_f", Path(p).name)
    assert m is not None, p
    return m.group(1)


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.link(src.resolve(strict=True), dst)


def folds():
    scenes = sorted(SPLIT["train_scenes"])
    return {"a": scenes[0::2], "b": scenes[1::2]}


def main() -> None:
    fold, name, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    # "POOL" means the released labels that ship with the image pool; anything else is a directory under code/.
    labels = (SRC / "labels") if name == "POOL" else CODE / name
    assert fold in ("a", "b"), fold
    if out.exists():
        raise SystemExit(f"no-clobber: {out}")
    rows = [json.loads(x) for x in (CODE / "PAIR_MANIFEST.jsonl").read_text().splitlines()
            if json.loads(x)["split"] == "train"]
    by_stem = {r["stem"]: r for r in rows}
    keep = set(folds()[fold]); held = set(SPLIT["held_out_scenes"])
    train = sorted(s for s, r in by_stem.items() if scene_of(r["transparent_image"]) in keep)
    if not train:
        raise SystemExit("empty fold")

    for s in train:
        link(SRC / "images" / "train" / f"{s}.png", out / "images" / "train" / f"{s}.png")
        link(labels / "train" / f"{s}.txt", out / "labels" / "train" / f"{s}.txt")
    for im in sorted((SRC / "images" / "val").glob("*.png")):
        link(im, out / "images" / "val" / im.name)
        link(SRC / "labels" / "val" / f"{im.stem}.txt", out / "labels" / "val" / f"{im.stem}.txt")

    stems_in = {p.stem for p in (out / "images" / "train").glob("*.png")}
    leaked = sorted(s for s in stems_in if s in by_stem and scene_of(by_stem[s]["transparent_image"]) in held)
    if leaked:
        raise SystemExit(f"held-out scene leaked into a fold: {leaked[:5]}")
    other = set(folds()["b" if fold == "a" else "a"])
    crossed = sorted(s for s in stems_in if scene_of(by_stem[s]["transparent_image"]) in other)
    if crossed:
        raise SystemExit(f"other fold leaked in: {crossed[:5]}")

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(NAMES))
    (out / "data_full.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    slots = len(list((out / "images" / "train").glob("*.png")))
    audit = {"schema": "apsr-fold-dataset-v1", "status": "PASS", "arm": f"fold{fold}",
             "counts": {"train_unique": len(train), "train_repeated": 0,
                        "train_total_slots": slots, "val": 125, "scenes": len(keep)},
             "roster_sha256": hashlib.sha256("\n".join(train).encode()).hexdigest(),
             "held_out_scenes": len(held), "held_out_scene_leak": 0,
             "symlinks": sum(p.is_symlink() for p in out.rglob("*")),
             "official_test_member_payload_reads": 0}
    if slots != len(train) or audit["symlinks"]:
        raise SystemExit(json.dumps(audit))
    (out / "DATASET_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    batches = -(-slots // 64)
    print(json.dumps({**audit, "batches_per_epoch": batches,
                      "updates_at_300_epochs": batches * 300}, sort_keys=True))


if __name__ == "__main__":
    main()
