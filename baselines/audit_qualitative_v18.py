#!/usr/bin/env python3
"""Independently recheck every number printed in the qualitative figure.

House rule for this workspace: a result is checked against a separate implementation, not against the code
that produced it.  Nothing here imports the figure's matcher.  The greedy assignment is rewritten from the
paper's wording (Section 3.1) in a different order of operations -- a per-ground-truth scan over an
explicit IoU matrix instead of a sorted candidate list -- and the outcome must agree exactly.

Checks:
  1. split-level type accuracy per arm, against the figure's receipt;
  2. the per-panel k/n counts of the two rendered frames;
  3. the two selection rules actually pick those frames.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EV = ROOT / "paper_work_v4" / "scenes_v1" / "evals_official"
GTF = ROOT / "official_test_v1" / "ood_types_remap_1based.json"
RECEIPT = HERE / "QUALITATIVE_FIGURE_V18.json"
ARMS = ["transparent", "lsmooth", "confid", "relabel", "scenevote"]
TAG, IOU_T, CONF_T = "scene", 0.5, 0.25
SEED = 1337


def overlap(a, b):
    ax0, ay0, aw, ah = a; bx0, by0, bw, bh = b
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def assign(gts, preds):
    """Same rule, different implementation: build the IoU matrix, then walk the ground truth in id order."""
    cand = [p for p in preds if p["score"] >= CONF_T]
    order = sorted(range(len(cand)), key=lambda k: cand[k]["score"], reverse=True)
    rank = {k: r for r, k in enumerate(order)}          # ties broken exactly as a score sort would
    grid = [[overlap(g["bbox"], c["bbox"]) for c in cand] for g in gts]
    taken, out = set(), {}
    for gi, g in enumerate(sorted(range(len(gts)), key=lambda i: gts[i]["id"])):
        row = grid[g]
        pick, best = None, IOU_T - 1e-12
        for k, v in enumerate(row):
            if k in taken or v < IOU_T:
                continue
            if v > best or (v == best and pick is not None and rank[k] < rank[pick]):
                pick, best = k, v
        if pick is None:
            out[gts[g]["id"]] = None
        else:
            taken.add(pick)
            out[gts[g]["id"]] = cand[pick]["category_id"]
    return out


def main():
    gt = json.loads(GTF.read_text())
    per_img = defaultdict(list)
    for a in gt["annotations"]:
        if 1 <= a["category_id"] <= 6:
            per_img[a["image_id"]].append(a)
    names = {im["id"]: im["file_name"] for im in gt["images"]}
    rec = json.loads(RECEIPT.read_text())

    res = {}
    for arm in ARMS:
        preds = json.loads((EV / f"ood_{TAG}_{arm}_s{SEED}" / "predictions_six_class.json").read_text())
        pim = defaultdict(list)
        for p in preds:
            pim[p["image_id"]].append(p)
        res[arm] = {i: assign(per_img[i], pim.get(i, [])) for i in per_img}

    fails = []
    print("== 1. split-level type accuracy ==")
    for arm in ARMS:
        loc = cor = 0
        for i, anns in per_img.items():
            for a in anns:
                t = res[arm][i][a["id"]]
                if t is not None:
                    loc += 1
                    cor += t == a["category_id"]
        mine = 100.0 * cor / max(loc, 1)
        printed = rec["split_type_accuracy"][arm]
        ok = abs(mine - printed) < 0.05
        fails += [] if ok else [f"type accuracy {arm}: audit {mine:.2f} vs printed {printed}"]
        print(f"  {arm:12s} audit {mine:6.2f}  printed {printed:6.2f}  {'ok' if ok else 'MISMATCH'}")

    print("== 2. per-panel counts of the rendered frames ==")
    for row in rec["rows"]:
        anns = per_img[row["id"]]
        line = []
        for arm in ARMS:
            k = sum(1 for a in anns if res[arm][row["id"]][a["id"]] == a["category_id"])
            line.append(f"{arm[:5]} {k}/{len(anns)}")
        print(f"  {row['name']:18s} " + "  ".join(line))

    print("== 3. the stated selection rules ==")
    raw_err, breaks = {}, {}
    for i, anns in per_img.items():
        raw_err[i] = sum(1 for a in anns if res["transparent"][i][a["id"]] != a["category_id"])
        breaks[i] = sum(1 for a in anns
                        if res["transparent"][i][a["id"]] == a["category_id"]
                        and res["scenevote"][i][a["id"]] != a["category_id"])
    top_hard = max(per_img, key=lambda i: (raw_err[i], len(per_img[i]), [-ord(c) for c in names[i]]))
    top_break = max(per_img, key=lambda i: (breaks[i], len(per_img[i]), [-ord(c) for c in names[i]]))
    for label, got, want in (("row 1 (released labels mistype most)", names[top_hard], rec["rows"][0]["name"]),
                             ("row 2 (our worst frame)", names[top_break], rec["rows"][1]["name"])):
        ok = got == want
        fails += [] if ok else [f"{label}: audit picks {got}, figure shows {want}"]
        print(f"  {label:38s} audit {got}  figure {want}  {'ok' if ok else 'MISMATCH'}")

    print("\n" + ("PASS: 감사 재계산이 인쇄값과 일치" if not fails else "FAIL:\n  " + "\n  ".join(fails)))
    (HERE / "AUDIT_QUALITATIVE_V18.json").write_text(json.dumps(
        {"schema": "apsr-qualitative-audit-v18", "failures": fails}, indent=1) + "\n")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
