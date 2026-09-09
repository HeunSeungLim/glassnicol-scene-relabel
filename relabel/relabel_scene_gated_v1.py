#!/usr/bin/env python3
"""Scene-level type-consistent relabelling (the static-scene method).

GlassNICOL scenes are static: the glasses stand on the table and only the head camera moves. Every view
of a glass is therefore a view of one point of the table plane, its base. The adjacent-frame tracklet
vote (relabel_multiview_v1.py) links boxes only between consecutive frames, by box overlap after a
partial-affine warp; a single missed frame, a failed registration, or the parallax of a tall glass
breaks the chain, and the chains have median length 1.

Here the association is geometric and multi-hop: frame f is registered to f+k for k = 1..K by a
homography of the table plane (background features only), each box is carried by its BASE point
(bottom-centre, which lies on the plane and so maps exactly), and a box is linked to the box of f+k
whose base point lies within a fraction of the box width. Links across hops are merged by union-find
under the constraint that a chain holds at most one box per frame. Each chain then votes its majority
type; ties are left unchanged; boxes never move.

    relabel_scene_v1.py [--hops K] [--tol T]     # writes labels_scene/{train,val}/*.txt + SCENE_RELABEL_STATS_V1.json
"""
from __future__ import annotations
import argparse, json, re, statistics as st, time
import cv2, numpy as np
from collections import Counter, defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = ROOT / "pair_manifest_v1" / "PAIR_MANIFEST.jsonl"
SRC = ROOT / "yolo_head_v1" / "labels"
IMG = ROOT / "yolo_head_v1" / "images"
OUT = HERE / "labels_scene_gated"
NAMES = ["shot", "whisky", "water", "beer", "wine", "high"]
_ORB = cv2.ORB_create(nfeatures=1500)

def load_split(split):
    by, stem_of = defaultdict(dict), {}
    for line in MANIFEST.read_text().splitlines():
        r = json.loads(line)
        if r["split"] != split: continue
        m = re.search(r"_s(\d+)_f(\d+)", Path(r["transparent_image"]).name); s, f = m.group(1), int(m.group(2))
        bx = []
        for l in (SRC / split / f"{r['stem']}.txt").read_text().splitlines():
            c, cx, cy, w, h = l.split(); bx.append([int(c), float(cx), float(cy), float(w), float(h)])
        by[s][f] = bx; stem_of[(s, f)] = r["stem"]
    return by, stem_of

def features(path, boxes):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    mask = np.full(img.shape, 255, np.uint8)
    for _, cx, cy, w, h in boxes:
        x1 = max(0, int((cx - .6*w)*img.shape[1])); x2 = min(img.shape[1], int((cx + .6*w)*img.shape[1]))
        y1 = max(0, int((cy - .6*h)*img.shape[0])); y2 = min(img.shape[0], int((cy + .6*h)*img.shape[0]))
        mask[y1:y2, x1:x2] = 0
    k, d = _ORB.detectAndCompute(img, mask)
    return k, d, img.shape

def homography(fa, fb, min_inliers=30, min_share=0.5):
    ka, da, _ = fa; kb, db, _ = fb
    if da is None or db is None or len(ka) < 8 or len(kb) < 8: return None
    good = [m for m, n in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2) if m.distance < .75*n.distance]
    if len(good) < 8: return None
    src = np.float32([ka[m.queryIdx].pt for m in good]); dst = np.float32([kb[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or inl.sum() < min_inliers or inl.mean() < min_share: return None
    return H

def iou_box(a, b):
    ax1, ay1, ax2, ay2 = a[1]-a[3]/2, a[2]-a[4]/2, a[1]+a[3]/2, a[2]+a[4]/2; bx1, by1, bx2, by2 = b[1]-b[3]/2, b[2]-b[4]/2, b[1]+b[3]/2, b[2]+b[4]/2
    ix = max(0, min(ax2, bx2) - max(ax1, bx1)); iy = max(0, min(ay2, by2) - max(ay1, by1)); i = ix * iy
    return i / (a[3]*a[4] + b[3]*b[4] - i + 1e-9)

def base_and_width(bx, H, shape):
    """Map the box's base point and its bottom edge through H; return base (px), width (px)."""
    Hh, W = shape; c, cx, cy, w, h = bx
    pts = np.float32([[[ (cx - w/2)*W, (cy + h/2)*Hh ], [ (cx + w/2)*W, (cy + h/2)*Hh ]]])
    q = cv2.perspectiveTransform(pts, H)[0]
    return (q[0] + q[1]) / 2, float(np.linalg.norm(q[1] - q[0]))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--hops", type=int, default=4); ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--out", default=str(OUT)); ap.add_argument("--cue", choices=["base", "iou"], default="base", help="link by base-point distance (default) or by warped-box IoU > 0.5")
    ap.add_argument("--min-votes", type=int, default=3, dest="min_votes")
    ap.add_argument("--margin", type=int, default=2)
    ap.add_argument("--stats", default=str(HERE / "SCENE_RELABEL_STATS_GATED_V1.json")); a = ap.parse_args(); out = Path(a.out)
    stats = {"schema": "apsr-scene-relabel-v1", "hops": a.hops, "tol_of_width": a.tol, "cue": a.cue, "splits": {}}
    t0 = time.time()
    for split in ("train", "val"):
        by, stem_of = load_split(split)
        parent = {}
        def find(x):
            while parent.setdefault(x, x) != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        members = {}   # root -> set of frames (one box per frame per chain)
        def union(u, v):
            ru, rv = find(u), find(v)
            if ru == rv: return True
            fu, fv = members.setdefault(ru, {u[1]}), members.setdefault(rv, {v[1]})
            if fu & fv: return False            # would put two boxes of one frame in a chain: refuse
            parent[ru] = rv; members[rv] = fu | fv; members.pop(ru, None); return True
        n_pairs = n_reg = n_links = n_refused = 0
        for s in by:
            fs = sorted(by[s]); feats = {f: features(IMG / split / f"{stem_of[(s, f)]}.png", by[s][f]) for f in fs}
            for f in fs:
                for i in range(len(by[s][f])): find((s, f, i))
            for f in fs:
                for k in range(1, a.hops + 1):
                    g = f + k
                    if g not in by[s]: continue
                    n_pairs += 1
                    H = homography(feats[f], feats[g])
                    if H is None: continue
                    n_reg += 1
                    shape = feats[f][2]; Hh, W = shape
                    tgt = [((bx[1]*W, (bx[2] + bx[4]/2)*Hh), bx[3]*W) for bx in by[s][g]]   # base point and width in g
                    cand = []
                    for i, bx in enumerate(by[s][f]):
                        if a.cue == "iou":      # ablation: carry the whole box through H and link by overlap
                            c_, cx, cy, w, h = bx
                            pts = np.float32([[[(cx - w/2)*W, (cy - h/2)*Hh], [(cx + w/2)*W, (cy + h/2)*Hh]]])
                            q = cv2.perspectiveTransform(pts, H)[0]; wb = [c_, float((q[0][0]+q[1][0])/2/W), float((q[0][1]+q[1][1])/2/Hh), float(abs(q[1][0]-q[0][0])/W), float(abs(q[1][1]-q[0][1])/Hh)]
                            for j, bj in enumerate(by[s][g]):
                                v = iou_box(wb, bj)
                                if v > 0.5: cand.append((1 - v, i, j))
                            continue
                        p, wmap = base_and_width(bx, H, shape)
                        for j, (q, wj) in enumerate(tgt):
                            d = float(np.hypot(*(p - np.float32(q)))); tol = a.tol * 0.5 * (wmap + wj)
                            if d < tol: cand.append((d / tol, i, j))
                    used_i, used_j = set(), set()
                    for _, i, j in sorted(cand):
                        if i in used_i or j in used_j: continue
                        if union((s, f, i), (s, g, j)): used_i.add(i); used_j.add(j); n_links += 1
                        else: n_refused += 1
            print(f"  {split} scene {s}: done ({time.time()-t0:.0f}s)", flush=True) if s in ("001", "050", "095") else None
        chains = defaultdict(list)
        for s in by:
            for f in by[s]:
                for i, bx in enumerate(by[s][f]): chains[find((s, f, i))].append((s, f, i, bx[0]))
        # vote
        new = {}; changed = 0; tied_boxes = 0; directions = Counter()
        gated_short = gated_thin = 0
        for c in chains.values():
            cnt = Counter(l for *_, l in c).most_common()
            if len(cnt) > 1 and cnt[0][1] == cnt[1][1]: tied_boxes += len(c); continue
            # Gate: a plurality of two views, or a plurality that one view would overturn, is not evidence.
            if len(c) < a.min_votes: gated_short += len(c); continue
            second = cnt[1][1] if len(cnt) > 1 else 0
            if cnt[0][1] - second < a.margin: gated_thin += len(c); continue
            maj = cnt[0][0]
            for s, f, i, l in c:
                if l != maj: changed += 1; directions[f"{NAMES[l]}->{NAMES[maj]}"] += 1
                new[(s, f, i)] = maj
        for s in by:
            for f in by[s]:
                lines = []
                for i, bx in enumerate(by[s][f]):
                    c = new.get((s, f, i), bx[0]); lines.append(f"{c} {bx[1]:.6f} {bx[2]:.6f} {bx[3]:.6f} {bx[4]:.6f}")
                (out / split / f"{stem_of[(s, f)]}.txt").parent.mkdir(parents=True, exist_ok=True)
                (out / split / f"{stem_of[(s, f)]}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        lens = sorted(Counter(len(c) for c in chains.values()).items())
        n_boxes = sum(len(c) for c in chains.values())
        stats["splits"][split] = {"boxes": n_boxes, "chains": len(chains), "chain_length_hist": lens,
            "median_chain_len_of_boxes": st.median([len(c) for c in chains.values() for _ in c]),
            "boxes_in_chains_ge3": sum(len(c) for c in chains.values() if len(c) >= 3),
            "frame_pairs_tried": n_pairs, "frame_pairs_registered": n_reg, "links": n_links, "links_refused_same_frame": n_refused,
            "gate": {"min_votes": a.min_votes, "margin": a.margin},
            "gated_short_chain": gated_short, "gated_thin_margin": gated_thin,
            "changed_boxes": changed, "changed_pct": round(100 * changed / n_boxes, 2), "tied_boxes": tied_boxes,
            "top_directions": directions.most_common(10)}
        print(split, json.dumps(stats["splits"][split]))
    stats["elapsed_s"] = round(time.time() - t0, 1)
    Path(a.stats).write_text(json.dumps(stats, indent=2) + "\n")

if __name__ == "__main__":
    main()
