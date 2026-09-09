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
EV = PAPER / "scenes_v1" / "evals_official"
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
    stats = json.loads((HERE / "SCENE_RELABEL_STATS_GATED_V1.json").read_text())["splits"]["train"]
    ungated = json.loads((HERE / "SCENE_RELABEL_STATS_V1.json").read_text())["splits"]["train"]
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
    (HERE / "GATE_ABLATION_V18F.json").write_text(json.dumps(out, indent=1) + "\n")
    with open(PAPER / "numbers.tex", "a") as f:
        f.write("\n% --- V18f gated-vote ablation (emit_v18f_gate.py) ---\n")
        for k, v in m.items():
            f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print(m)


if __name__ == "__main__":
    main()
