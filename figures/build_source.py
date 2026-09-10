#!/usr/bin/env python3
"""Source receipt for the label-noise figure (adjacent-view type flips).

Everything the figure shows is derived here from released frames and released
labels; nothing is drawn or synthesised. The box matching is the procedure of
``paper_work_v4/relabel_v1/relabel_multiview_v1.py`` and is executed by
importing that module's functions (``frame_affine``, ``warp_box``, ``iou``),
so the figure cannot drift from the relabelling code:

  * consecutive frames f, f+1 of one scene;
  * frame f registered to f+1 by ORB(1500) on the background (label boxes
    dilated 1.2x and masked out), ratio test 0.75, partial affine with
    RANSAC threshold 3, inlier share >= 0.5, else boxes are used unwarped;
  * every box of f carried through the affine and matched greedily to the
    unused box of f+1 with the highest IoU, accepted if IoU > 0.5.

A matched pair whose two released class ids differ is a flip.

Two matchings are recorded. ``train_registered`` is the one above (the paper's
procedure). ``train_unregistered`` is the earlier check that produced the
"183 of 902" figure quoted in relabel_v1/DRAFT_sections_v10.tex: identical
except that no registration is applied. Both are kept so the text can be
reconciled against whichever is cited.

    build_source.py            # writes source_data.json next to this file
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RELABEL = ROOT / "paper_work_v4" / "relabel_v1"
if not RELABEL.is_dir():
    raise SystemExit("build_source.py rebuilds source_data.json from the workspace that ran steps 1-5; "
                     "it is not runnable from this repository alone. The file it writes is shipped here.")
sys.path.insert(0, str(RELABEL))
import relabel_multiview_v1 as R  # noqa: E402  (frame_affine, warp_box, iou, NAMES, IOU_LINK, paths)

NAMES = R.NAMES
OUT = HERE / "source_data.json"
# Which confusable pairs panel A must cover, in display order.
TARGET_PAIRS = [("water", "whisky"), ("beer", "water"), ("wine", "high")]
# Candidate filter for a visually clear example: released box fully inside the
# frame with a border margin, not tiny, and clearly the same object.
EDGE_MARGIN = 0.02          # normalised distance from any frame border
MIN_AREA = 0.010            # normalised box area (0.010 * 1280*720 = 9216 px)
MIN_IOU = 0.60              # registered IoU between the two released boxes
CROP_SCALE = 1.35           # square crop side = CROP_SCALE * max(w, h) in pixels
# Manual choice among the ranked candidates (index into example_candidates[pair]) after
# looking at output/candidates_contact.png; the reason is written into source_data.json.
OVERRIDE = {
    "water/whisky": (3, "whole glass visible and uncluttered in both views; ranked #0 (s35 f6-7) has the "
                        "released box of view f stopping short of the mug body and a cluttered crop"),
    "beer/water": (2, "whole mug including its handle inside both crops; ranked #0 (s13 f15-16) sits at the "
                      "frame edge so the crop is shifted and the mug rim touches the crop border"),
    "wine/high": (2, "visible viewpoint change between the two frames; ranked #0 (s15 f0-1) has identical "
                     "box coordinates in both frames (IoU 1.00), which reads as a duplicate frame"),
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_split(split: str):
    rows = [json.loads(x) for x in R.MANIFEST.read_text().splitlines()]
    by, stem_of = defaultdict(dict), {}
    for r in rows:
        if r["split"] != split:
            continue
        m = re.search(r"_s(\d+)_f(\d+)", Path(r["transparent_image"]).name)
        s, f = m.group(1), int(m.group(2))
        bx = []
        for l in (R.SRC / split / f"{r['stem']}.txt").read_text().splitlines():
            c, cx, cy, w, h = l.split()
            bx.append([int(c), float(cx), float(cy), float(w), float(h)])
        by[s][f] = bx
        stem_of[(s, f)] = r["stem"]
    return by, stem_of


def match_split(split: str, registered: bool):
    """Adjacent-view box matching, verbatim order of relabel_multiview_v1.main."""
    by, stem_of = load_split(split)
    pairs, n_reg, n_adj = [], 0, 0
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for s in by:
        fs = sorted(by[s])
        for f in fs:
            for i in range(len(by[s][f])):
                find((s, f, i))
        for a, b in zip(fs, fs[1:]):
            if b != a + 1:
                continue
            n_adj += 1
            A, B = by[s][a], by[s][b]
            used = set()
            reg = None
            if registered:
                reg = R.frame_affine(R.IMG / split / f"{stem_of[(s, a)]}.png",
                                     R.IMG / split / f"{stem_of[(s, b)]}.png", A)
                if reg is not None:
                    n_reg += 1
            Aw = [R.warp_box(x, *reg) for x in A] if reg is not None else A
            for i, x in enumerate(Aw):
                best, bj = R.IOU_LINK, None
                for j, y in enumerate(B):
                    if j in used:
                        continue
                    v = R.iou(x, y)
                    if v > best:
                        best, bj = v, j
                if bj is not None:
                    used.add(bj)
                    union((s, a, i), (s, b, bj))
                    pairs.append({
                        "scene": s, "frame_a": a, "frame_b": b,
                        "stem_a": stem_of[(s, a)], "stem_b": stem_of[(s, b)],
                        "box_index_a": i, "box_index_b": bj,
                        "box_a": A[i], "box_a_carried": [float(v) for v in x], "box_b": B[bj],
                        "iou": float(best), "cls_a": A[i][0], "cls_b": B[bj][0],
                        "frame_registered": reg is not None})
    tracks = len({find(k) for k in list(parent)})
    n_boxes = sum(len(v) for s in by for v in by[s].values())
    return pairs, {"adjacent_frame_pairs": n_adj, "frame_pairs_registered": n_reg,
                   "boxes": n_boxes, "tracks_union_find": tracks, "scenes": len(by)}


def matrices(pairs):
    ordered = np.zeros((6, 6), int)
    for p in pairs:
        ordered[p["cls_a"], p["cls_b"]] += 1
    sym = ordered + ordered.T - np.diag(np.diag(ordered))
    flips = int(sum(1 for p in pairs if p["cls_a"] != p["cls_b"]))
    unordered = Counter()
    for p in pairs:
        if p["cls_a"] != p["cls_b"]:
            unordered[" / ".join(sorted((NAMES[p["cls_a"]], NAMES[p["cls_b"]]),
                                        key=NAMES.index))] += 1
    return {
        "matched_pairs": len(pairs), "flips": flips,
        "flip_rate_pct": round(100.0 * flips / len(pairs), 2) if pairs else None,
        "agreeing_pairs_per_type_diagonal": {NAMES[i]: int(ordered[i, i]) for i in range(6)},
        "flip_pairs_unordered": dict(unordered.most_common()),
        "matrix_symmetric_rows_then_cols": sym.tolist(),
        "matrix_ordered_f_rows_fplus1_cols": ordered.tolist(),
        "matrix_note": ("symmetric[i][j] (i != j) = number of matched adjacent-view pairs "
                        "labelled type i in one view and type j in the other; symmetric[i][i] = "
                        "matched pairs labelled i in both views. Row sum = boxes of type i that "
                        "took part in a matched pair (counted once per pair)."),
        "class_names_in_order": NAMES,
    }


def crop_window(box, W, H):
    """Square crop around a released box, shifted (never scaled) to stay in the frame."""
    _, cx, cy, w, h = box
    side = int(round(CROP_SCALE * max(w * W, h * H)))
    side = min(side, W, H)
    x0 = int(round(cx * W - side / 2)); y0 = int(round(cy * H - side / 2))
    x0 = min(max(0, x0), W - side); y0 = min(max(0, y0), H - side)
    return [x0, y0, x0 + side, y0 + side]


def candidates(pairs, W, H):
    """Visually clear flips per unordered target pair, ranked by the smaller of the two box areas."""
    out = {}
    for a, b in TARGET_PAIRS:
        ia, ib = NAMES.index(a), NAMES.index(b)
        rows = []
        for p in pairs:
            if {p["cls_a"], p["cls_b"]} != {ia, ib} or not p["frame_registered"]:
                continue
            ok = True
            for bx in (p["box_a"], p["box_b"]):
                _, cx, cy, w, h = bx
                if (cx - w / 2 < EDGE_MARGIN or cy - h / 2 < EDGE_MARGIN or
                        cx + w / 2 > 1 - EDGE_MARGIN or cy + h / 2 > 1 - EDGE_MARGIN or w * h < MIN_AREA):
                    ok = False
            if ok and p["iou"] >= MIN_IOU:
                rows.append(p)
        rows.sort(key=lambda p: (-min(p["box_a"][3] * p["box_a"][4], p["box_b"][3] * p["box_b"][4]),
                                 -p["iou"], p["scene"], p["frame_a"]))
        out[f"{a}/{b}"] = rows[:6]
    return out


def example_record(p, split, W, H):
    rec = {"pair": f"{NAMES[p['cls_a']]}/{NAMES[p['cls_b']]}", "scene": p["scene"],
           "frames": [p["frame_a"], p["frame_b"]], "iou_after_registration": round(p["iou"], 4),
           "views": []}
    for tag, stem, bx, idx in (("f", p["stem_a"], p["box_a"], p["box_index_a"]),
                               ("f+1", p["stem_b"], p["box_b"], p["box_index_b"])):
        img = R.IMG / split / f"{stem}.png"
        lab = R.SRC / split / f"{stem}.txt"
        rec["views"].append({
            "view": tag, "stem": stem, "image": str(img), "image_sha256": sha(img),
            "label_file": str(lab), "label_sha256": sha(lab), "box_index_in_label_file": idx,
            "released_label_line": lab.read_text().splitlines()[idx],
            "class_id": bx[0], "class_name": NAMES[bx[0]],
            "box_yolo_normalised": bx[1:],
            "box_xyxy_px": [round((bx[1] - bx[3] / 2) * W, 1), round((bx[2] - bx[4] / 2) * H, 1),
                            round((bx[1] + bx[3] / 2) * W, 1), round((bx[2] + bx[4] / 2) * H, 1)],
            "crop_xyxy_px": crop_window(bx, W, H)})
    return rec


def manual_split_matrices():
    """Optional: the same test on the hand-labelled official splits (read-only)."""
    OFF = ROOT / "official_test_v1"
    out = {}
    orb = cv2.ORB_create(nfeatures=1500)

    def iou_xywh(a, b):
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by); x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        i = max(0, x2 - x1) * max(0, y2 - y1)
        return i / (aw * ah + bw * bh - i + 1e-9)

    def affine(pa, pb, boxes):
        a = cv2.imread(str(pa), 0); b = cv2.imread(str(pb), 0)
        if a is None or b is None or a.shape != b.shape:
            return None
        m = np.full(a.shape, 255, np.uint8)
        for x, y, w, h in boxes:
            m[max(0, int(y - .1 * h)):int(y + 1.1 * h), max(0, int(x - .1 * w)):int(x + 1.1 * w)] = 0
        ka, da = orb.detectAndCompute(a, m); kb, db = orb.detectAndCompute(b, None)
        if da is None or db is None or len(ka) < 8 or len(kb) < 8:
            return None
        g = [q for q, n in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2) if q.distance < .75 * n.distance]
        if len(g) < 8:
            return None
        M, inl = cv2.estimateAffinePartial2D(np.float32([ka[q.queryIdx].pt for q in g]),
                                             np.float32([kb[q.trainIdx].pt for q in g]),
                                             method=cv2.RANSAC, ransacReprojThreshold=3)
        return None if M is None or inl.mean() < .5 else M

    for name, annf, src in (("official_standard_manual", OFF / "standard_types_remap_1based.json",
                             OFF / "annotations" / "coco_test_standard_glass_types.json"),
                            ("official_ood_manual", OFF / "ood_types_remap_1based.json",
                             OFF / "annotations" / "coco_test_odd_glass_types.json")):
        if not (annf.exists() and src.exists() and (OFF / "images").exists()):
            out[name] = {"status": "skipped, files not present"}
            continue
        gt = json.loads(annf.read_text()); orig = json.loads(src.read_text())
        meta = {}
        for im_r, im_o in zip(sorted(gt["images"], key=lambda x: x["id"]),
                              sorted(orig["images"], key=lambda x: x["id"])):
            m = re.search(r"scene_(\d+)", im_o["file_name"])
            fr = re.findall(r"(\d+)", im_o["file_name"])
            meta[im_r["id"]] = (m.group(1), int(fr[-1]), im_r["file_name"])
        by = defaultdict(list)
        for a in gt["annotations"]:
            if 1 <= a["category_id"] <= 6:
                by[a["image_id"]].append(a)
        scenes = defaultdict(list)
        for iid, (s, f, fn) in meta.items():
            scenes[s].append((f, iid, fn))
        ordered = np.zeros((6, 6), int); nreg = 0
        for s, lst in scenes.items():
            lst.sort()
            for (fa, ia, na), (fb, ib, nb) in zip(lst, lst[1:]):
                A = [(a["bbox"], a["category_id"]) for a in by[ia]]
                B = [(a["bbox"], a["category_id"]) for a in by[ib]]
                if not A or not B:
                    continue
                M = affine(OFF / "images" / na, OFF / "images" / nb, [x[0] for x in A])
                if M is not None:
                    nreg += 1; Aw = []
                    for (x, y, w, h), c in A:
                        p = np.float32([[x + w / 2, y + h / 2]]) @ M[:, :2].T + M[:, 2]
                        sc = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
                        Aw.append(([float(p[0, 0]) - w * sc / 2, float(p[0, 1]) - h * sc / 2, w * sc, h * sc], c))
                else:
                    Aw = A
                used = set()
                for bx, c in Aw:
                    best, bj = 0.5, None
                    for j, (by_, cb) in enumerate(B):
                        if j in used:
                            continue
                        v = iou_xywh(bx, by_)
                        if v > best:
                            best, bj = v, j
                    if bj is None:
                        continue
                    used.add(bj); ordered[c - 1, B[bj][1] - 1] += 1
        sym = ordered + ordered.T - np.diag(np.diag(ordered))
        out[name] = {"matched_pairs": int(ordered.sum()), "flips": int(ordered.sum() - np.trace(ordered)),
                     "registered_frame_pairs": nreg, "matrix_symmetric": sym.tolist(),
                     "note": "same construction as LABEL_NOISE_DIAGNOSTIC_V1.json (COCO xywh boxes)"}
    return out


def main() -> None:
    t0 = time.time()
    W, H = 1280, 720
    probe = cv2.imread(str(R.IMG / "train" / "train_000000.png"))
    assert probe.shape[:2] == (H, W), probe.shape
    reg_pairs, reg_stats = match_split("train", registered=True)
    print(f"registered   : {reg_stats}  matched {len(reg_pairs)}  ({time.time()-t0:.0f}s)", flush=True)
    unreg_pairs, unreg_stats = match_split("train", registered=False)
    print(f"unregistered : {unreg_stats}  matched {len(unreg_pairs)}", flush=True)
    reg_m, unreg_m = matrices(reg_pairs), matrices(unreg_pairs)
    print("registered   flips", reg_m["flips"], "/", reg_m["matched_pairs"], reg_m["flip_pairs_unordered"])
    print("unregistered flips", unreg_m["flips"], "/", unreg_m["matched_pairs"], unreg_m["flip_pairs_unordered"])

    cands = candidates(reg_pairs, W, H)
    examples = {k: [example_record(p, "train", W, H) for p in v] for k, v in cands.items()}
    chosen = {k: v[OVERRIDE[k][0] if k in OVERRIDE else 0] for k, v in examples.items() if v}
    for k, (idx, why) in OVERRIDE.items():
        chosen[k]["chosen_candidate_index"] = idx; chosen[k]["override_reason"] = why

    relabel_stats = json.loads((RELABEL / "RELABEL_STATS_V1.json").read_text())["splits"]["train"]
    cross = {"relabel_stats_v1_train": {"frame_pairs_registered": relabel_stats["frame_pairs_registered"],
                                        "boxes": relabel_stats["boxes"], "tracks": relabel_stats["tracks"]},
             "boxes_minus_tracks_from_relabel_stats": relabel_stats["boxes"] - relabel_stats["tracks"],
             "registered_matched_pairs_here": len(reg_pairs),
             "registered_frame_pairs_here": reg_stats["frame_pairs_registered"],
             "tracks_here": reg_stats["tracks_union_find"],
             "consistent_with_relabel_stats_v1": (
                 reg_stats["frame_pairs_registered"] == relabel_stats["frame_pairs_registered"]
                 and reg_stats["tracks_union_find"] == relabel_stats["tracks"]
                 and len(reg_pairs) == relabel_stats["boxes"] - relabel_stats["tracks"]),
             "draft_v10_quote": "183 of 902 matched pairs (20.3%)",
             "unregistered_matches_draft_v10": (unreg_m["matched_pairs"] == 902 and unreg_m["flips"] == 183)}

    payload = {
        "schema": "labelnoise-figure-source-v1",
        "generated_by": str(Path(__file__).resolve()),
        "matching_code": {"module": str(RELABEL / "relabel_multiview_v1.py"),
                          "sha256": sha(RELABEL / "relabel_multiview_v1.py"),
                          "functions_imported": ["frame_affine", "warp_box", "iou"],
                          "orb_features": 1500, "ratio_test": 0.75, "ransac_reproj_threshold": 3,
                          "box_mask_dilation": 1.2, "min_inlier_share": 0.5, "iou_link": R.IOU_LINK,
                          "iou_rule": "strictly greater than iou_link, greedy over frame-f boxes in file order"},
        "inputs": {"manifest": str(R.MANIFEST), "manifest_sha256": sha(R.MANIFEST),
                   "images": str(R.IMG / "train"), "labels": str(R.SRC / "train"),
                   "frame_size_px": [W, H], "class_names": NAMES},
        "train_registered": {**reg_stats, **reg_m},
        "train_unregistered": {**unreg_stats, **unreg_m,
                               "note": "no registration; boxes of f compared to f+1 as released. "
                                       "This is the check quoted as '183 of 902' in DRAFT_sections_v10.tex."},
        "cross_check": cross,
        "example_selection": {
            "rule": (f"for each target pair, registered flips whose two released boxes lie >= {EDGE_MARGIN} "
                     f"(normalised) from every border, have area >= {MIN_AREA} (normalised) and registered "
                     f"IoU >= {MIN_IOU}; ranked by the smaller of the two box areas, then IoU; the first is used "
                     "unless 'override' names another candidate index with a reason"),
            "target_pairs": [f"{a}/{b}" for a, b in TARGET_PAIRS],
            "crop_rule": f"square window of side {CROP_SCALE} x max(box w, box h) px centred on the box, "
                         "shifted inside the frame if needed, never rescaled",
            "override": {k: {"candidate_index": i, "reason": r} for k, (i, r) in OVERRIDE.items()},
        },
        "example_candidates": examples,
        "examples": chosen,
        "manual_official_splits": manual_split_matrices(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(cross, indent=1))
    for k, v in chosen.items():
        print(k, "scene", v["scene"], "frames", v["frames"], "iou", v["iou_after_registration"],
              [(x["class_name"], x["box_yolo_normalised"]) for x in v["views"]])
    print(f"wrote {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
