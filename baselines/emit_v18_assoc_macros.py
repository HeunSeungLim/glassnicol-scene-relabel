#!/usr/bin/env python3
"""Macros for the association ablation, which V18 states in prose instead of a table.

The table is released with the code; the sentence in Section 2.3 still has to print the two ingredients'
numbers, and they come from the same receipts the table row emitter reads, not from a literal typed into
the manuscript.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
SRC = {"AblOvTwo": ("ablation_assoc/STATS_h2_iou.json", "SCENE_ASSOC_MANUAL_CHECK_V1_h2_t0.5_iou.json"),
       "AblBaseOne": ("ablation_assoc/STATS_h1_base.json", "SCENE_ASSOC_MANUAL_CHECK_V1_h1_t0.5.json")}


def main():
    m = {}
    for key, (sf, mf) in SRC.items():
        d = json.loads((HERE / sf).read_text())["splits"]["train"]
        mm = json.loads((HERE / mf).read_text())
        m[key + "Votes"] = f"{int(d['median_chain_len_of_boxes'])}"
        m[key + "ChangedPct"] = f"{d['changed_pct']:.1f}"
        m[key + "OODDis"] = f"{mm['ood']['disagreeing_boxes']}"
        m[key + "StdDis"] = f"{mm['standard']['disagreeing_boxes']}"
    with open(PAPER / "numbers.tex", "a") as f:
        f.write("\n% --- V18 association-ablation sentence (emit_v18_assoc_macros.py) ---\n")
        for k, v in m.items():
            f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print(m)


if __name__ == "__main__":
    main()
