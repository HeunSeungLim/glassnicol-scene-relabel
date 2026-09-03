#!/usr/bin/env python3
"""Multi-view type-consistent relabelling.

GlassNICOL assigns each box its glass type by matching a depth-derived height to
the nearest known glass. Measured on adjacent views of the same scene, 20.3% of
boxes that are unambiguously the same physical object (IoU > 0.5 across a small
neck rotation) carry a different type. The flips are between glasses of similar
height. That is per-view measurement noise on a quantity that is constant for
the object.

The scene structure removes it: every object is seen in up to 25 views. Chain
boxes across consecutive views by overlap, and give every box in a chain the
chain's majority type. No model is involved, no new capture, no new annotation.

    relabel_multiview_v1.py            # writes labels_relabelled/{train,val}/*.txt + stats
"""
from __future__ import annotations
import json, re, statistics as st
import cv2, numpy as np
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = ROOT / "pair_manifest_v1" / "PAIR_MANIFEST.jsonl"
SRC = ROOT / "yolo_head_v1" / "labels"
OUT = HERE / "labels_relabelled"
NAMES = ["shot", "whisky", "water", "beer", "wine", "high"]
IOU_LINK = 0.5
IMG = ROOT / "yolo_head_v1" / "images"
_ORB = cv2.ORB_create(nfeatures=1500)


def frame_affine(a_path, b_path, boxes_a):
    """Partial-affine from frame a to frame b using background features only,
    the same construction as the pair-eligibility test. None if it fails."""
    a = cv2.imread(str(a_path), cv2.IMREAD_GRAYSCALE); b = cv2.imread(str(b_path), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None or a.shape != b.shape: return None
    mask = np.full(a.shape, 255, np.uint8)
    for _, cx, cy, w, h in boxes_a:
        x1 = max(0, int((cx - .6*w)*a.shape[1])); x2 = min(a.shape[1], int((cx + .6*w)*a.shape[1]))
        y1 = max(0, int((cy - .6*h)*a.shape[0])); y2 = min(a.shape[0], int((cy + .6*h)*a.shape[0]))
        mask[y1:y2, x1:x2] = 0
    ka, da = _ORB.detectAndCompute(a, mask); kb, db = _ORB.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8: return None
    good = [m for m, n in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2) if m.distance < .75*n.distance]
    if len(good) < 8: return None
    src = np.float32([ka[m.queryIdx].pt for m in good]); dst = np.float32([kb[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3)
    return None if M is None or inl.mean() < 0.5 else (M, a.shape)


def warp_box(bx, M, shape):
    H, W = shape; c, cx, cy, w, h = bx
    p = np.float32([[cx*W, cy*H]]) @ M[:, :2].T + M[:, 2]
    sc = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    return [c, float(p[0,0])/W, float(p[0,1])/H, w*sc, h*sc]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    ix = max(0, min(ax2, bx2) - max(ax1, bx1)); iy = max(0, min(ay2, by2) - max(ay1, by1)); i = ix * iy
    return i / (a[3] * a[4] + b[3] * b[4] - i + 1e-9)


def main() -> None:
    rows = [json.loads(x) for x in MANIFEST.read_text().splitlines()]
    stats = {"schema": "apsr-multiview-relabel-v1", "iou_link": IOU_LINK, "splits": {}}
    for split in ("train", "val"):
        by = defaultdict(dict)   # scene -> frame -> list of [cls, cx, cy, w, h]
        stem_of = {}
        for r in rows:
            if r["split"] != split:
                continue
            m = re.search(r"_s(\d+)_f(\d+)", Path(r["transparent_image"]).name)
            s, f = m.group(1), int(m.group(2))
            bx = []
            for l in (SRC / split / f"{r['stem']}.txt").read_text().splitlines():
                c, cx, cy, w, h = l.split()
                bx.append([int(c), float(cx), float(cy), float(w), float(h)])
            by[s][f] = bx; stem_of[(s, f)] = r["stem"]

        # union-find over (scene, frame, box index)
        parent = {}
        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[ra] = rb

        n_reg = 0
        for s in by:
            fs = sorted(by[s])
            for f in fs:
                for i in range(len(by[s][f])): find((s, f, i))
            for a, b in zip(fs, fs[1:]):
                if b != a + 1: continue
                A, B = by[s][a], by[s][b]; used = set()
                reg = frame_affine(IMG / split / f"{stem_of[(s,a)]}.png", IMG / split / f"{stem_of[(s,b)]}.png", A)
                if reg is not None: n_reg += 1
                Aw = [warp_box(x, *reg) for x in A] if reg is not None else A
                for i, x in enumerate(Aw):
                    best, bj = IOU_LINK, None
                    for j, y in enumerate(B):
                        if j in used: continue
                        v = iou(x, y)
                        if v > best: best, bj = v, j
                    if bj is not None:
                        used.add(bj); union((s, a, i), (s, b, bj))

        tracks = defaultdict(list)
        for k in list(parent): tracks[find(k)].append(k)

        changed = total = ambiguous = 0
        lengths = []; flip_matrix = Counter()
        new_labels = {k: [b[:] for b in v] for s in by for k, v in [((s, f), by[s][f]) for f in by[s]]}
        for members in tracks.values():
            lengths.append(len(members))
            votes = Counter(by[s][f][i][0] for (s, f, i) in members)
            top, n = votes.most_common(1)[0]
            tie = sum(1 for c, m in votes.items() if m == n) > 1
            for (s, f, i) in members:
                total += 1
                old = by[s][f][i][0]
                if tie:
                    ambiguous += 1; continue          # no majority: leave the box as it was
                if old != top:
                    changed += 1; flip_matrix[(NAMES[old], NAMES[top])] += 1
                    new_labels[(s, f)][i][0] = top

        (OUT / split).mkdir(parents=True, exist_ok=True)
        for (s, f), bx in new_labels.items():
            (OUT / split / f"{stem_of[(s, f)]}.txt").write_text(
                "".join(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n" for b in bx))
        stats["splits"][split] = {
            "frame_pairs_registered": n_reg, "boxes": total, "tracks": len(tracks),
            "track_length": {"median": st.median(lengths), "mean": round(st.mean(lengths), 2),
                             "max": max(lengths), "singletons": sum(1 for x in lengths if x == 1)},
            "boxes_in_tracks_ge3": sum(len(m) for m in tracks.values() if len(m) >= 3),
            "relabelled": changed, "relabelled_pct": round(100 * changed / total, 2),
            "ambiguous_left_as_is": ambiguous,
            "top_relabel_directions": [(f"{a}->{b}", n) for (a, b), n in flip_matrix.most_common(8)]}
        print(f"[{split}] registered pairs {n_reg}  boxes {total}  tracks {len(tracks)}  median track len {st.median(lengths):.0f}  "
              f"relabelled {changed} ({100*changed/total:.1f}%)  ambiguous {ambiguous}")
    (HERE / "RELABEL_STATS_V1.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats["splits"]["train"]["top_relabel_directions"]))


if __name__ == "__main__":
    main()
