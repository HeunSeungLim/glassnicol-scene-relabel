#!/usr/bin/env python3
"""Macros for the gated-vote ablation.

The paper concedes a standard-split loss and reports an out-of-distribution gain, and a reader asks whether
the loss is simply the vote's low-confidence edits.  This ablation answers it: the same association, the
same roster and recipe, but a chain votes only when it has at least three views and its plurality leads by
two.  Trained on accelerator A's three seeds after the sealed protocol, so it is post hoc.

Reads the gated relabelling statistics and the six scored cells; writes GATE_ABLATION_V18F.json.
"""
from __future__ import annotations
import json, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent


def _resolve():
    """Workspace layout, or the six scored cells shipped beside this script."""
    local = HERE.parent / "receipts" / "gate_ablation_cells"
    if local.is_dir():
        return local
    return PAPER / "scenes_v1" / "evals_official"


EV = _resolve()
CELLS = [1337, 3407, 4567]           # accelerator A


def six(tag, split, seed):
    p = EV / f"{split}_scene_{tag}_s{seed}" / "RESULT.json"
    return 100 * json.loads(p.read_text())["six_class"]["ap50_95"]


def sign(x):
    # a contrast that rounds to nothing gets no sign: "$-$0.0" reads as a loss that is not there
    if abs(x) < 0.05:
        return "0.0"
    return ("+" if x >= 0 else "$-$") + f"{abs(x):.1f}"


def main():
    _r = HERE if (HERE / "SCENE_RELABEL_STATS_GATED_V1.json").is_file() else HERE.parent / "receipts"
    stats = json.loads((_r / "SCENE_RELABEL_STATS_GATED_V1.json").read_text())["splits"]["train"]
    ungated = json.loads((_r / "SCENE_RELABEL_STATS_V1.json").read_text())["splits"]["train"]
    m = {"GateMinVotes": str(stats["gate"]["min_votes"]), "GateMargin": str(stats["gate"]["margin"]),
         "GateChanged": f"{stats['changed_boxes']:,}", "GateChangedPct": f"{stats['changed_pct']:.1f}",
         "GateCells": str(len(CELLS))}
    out = {"schema": "apsr-gate-ablation-v18f", "gate": stats["gate"], "cells": CELLS,
           "changed": {"gated": stats["changed_boxes"], "ungated": ungated["changed_boxes"],
                       "boxes": stats["boxes"]}, "six_class": {}}
    for split, key in (("standard", "Std"), ("ood", "OOD")):
        raw = [six("transparent", split, s) for s in CELLS]
        gat = [six("gatedvote", split, s) for s in CELLS]
        sv = [six("scenevote", split, s) for s in CELLS]
        d_raw = [a - b for a, b in zip(gat, raw)]
        d_sv = [a - b for a, b in zip(gat, sv)]
        m[f"Gate{key}VsRaw"] = sign(st.mean(d_raw))
        m[f"Gate{key}VsRawPos"] = str(sum(x > 0 for x in d_raw))
        m[f"Gate{key}VsSv"] = sign(st.mean(d_sv))
        m[f"Gate{key}VsSvPos"] = str(sum(x > 0 for x in d_sv))
        out["six_class"][split] = {"raw": raw, "gated": gat, "scenevote": sv,
                                   "gated_minus_raw": d_raw, "gated_minus_scenevote": d_sv}
    # type accuracy of the gated arm, on the boxes it and the raw arm both localise: the paper's pairwise
    # convention.  Six-class AP alone would say the gate costs the whole out-of-distribution gain.
    import collections
    OFF = (HERE.parent / "receipts" / "table1_cells") if (HERE.parent / "receipts" / "table1_cells" / "ood_types_remap_1based.json").is_file() else (PAPER.parent / "official_test_v1")
    IOU_T, CONF_T = 0.5, 0.25

    def _iou(a, b):
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
        i = ix * iy; u = aw * ah + bw * bh - i
        return i / u if u > 0 else 0.0

    def _match(gts, preds):
        free = sorted([q for q in preds if q["score"] >= CONF_T], key=lambda q: -q["score"])
        used, res = set(), {}
        for g in sorted(gts, key=lambda g: g["id"]):
            best, bi = None, IOU_T - 1e-12
            for k, q in enumerate(free):
                if k in used: continue
                v = _iou(g["bbox"], q["bbox"])
                if v > bi: best, bi = k, v
            if best is None: res[g["id"]] = None
            else: used.add(best); res[g["id"]] = free[best]["category_id"]
        return res

    def _type_gain(tag, split, seed, by):
        out2 = {}
        for name in (tag, "transparent"):
            pr = json.loads((EV / f"{split}_scene_{name}_s{seed}" / "predictions_six_class.json").read_text())
            pim = collections.defaultdict(list)
            for q in pr: pim[q["image_id"]].append(q)
            out2[name] = {i: _match(by[i], pim.get(i, [])) for i in by}
        both = [(i, g) for i, anns in by.items() for g in anns
                if out2[tag][i][g["id"]] is not None and out2["transparent"][i][g["id"]] is not None]
        acc = lambda n: 100.0 * sum(1 for i, g in both if out2[n][i][g["id"]] == g["category_id"]) / len(both)
        return acc(tag) - acc("transparent")

    gt = json.loads((OFF / "ood_types_remap_1based.json").read_text())
    by = collections.defaultdict(list)
    for an in gt["annotations"]:
        if 1 <= an["category_id"] <= 6: by[an["image_id"]].append(an)
    for tag, key in (("gatedvote", "Gate"), ("scenevote", "Sv")):
        g = [_type_gain(tag, "ood", s, by) for s in CELLS]
        m[f"{key}OODTypeA"] = sign(st.mean(g))
        m[f"{key}OODTypeAPos"] = str(sum(x > 0 for x in g))
        out.setdefault("ood_type_accuracy_vs_raw", {})[tag] = g
    (HERE / "GATE_ABLATION_V18F.json").write_text(json.dumps(out, indent=1) + "\n")
    nb = PAPER / "numbers.tex" if (PAPER / "numbers.tex").is_file() else PAPER / "paper" / "numbers.tex"
    with open(nb, "a") as f:
        f.write("\n% --- V18f gated-vote ablation (emit_v18f_gate.py) ---\n")
        for k, v in m.items():
            f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print(m)


if __name__ == "__main__":
    main()
