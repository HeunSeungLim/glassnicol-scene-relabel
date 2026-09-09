#!/usr/bin/env python3
"""Macros the V18b revision needs, each from a receipt rather than a literal.

Three quantities entered the text during this revision and none of them had a macro:

  VoteUpPct, VoteOneStepPct   which way the vote moves a type id.  The labeller is estimated to read
                              glasses as shorter than they are, so a repair should move ids up the height
                              ladder; counted here by walking the released and voted label files in file
                              and line order, which is the order the relabeller preserves.
  ConfidFloorPct              the share of the confidence baseline's rewrites that must differ from our
                              vote no matter what it decided: it edits more boxes than we do, and every
                              box it edits that we leave alone disagrees by construction.  Printing 81.8%
                              without this floor overstates the contrast.
  AblOvOneVotes               votes per box for overlap linking over one hop, needed by the corrected
                              ablation sentence.

Also writes VOTE_DIRECTION_V18.json.
"""
from __future__ import annotations
import collections, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parent
RAW = ROOT / "yolo_head_v1" / "labels" / "train"
VOTED = HERE / "labels_scene" / "train"
LADDER = ["shot", "whisky", "water", "beer", "wine", "high"]


def rows(p):
    return [l.split() for l in p.read_text().split("\n") if len(l.split()) >= 5]


def direction():
    m, boxes, moved = collections.Counter(), 0, 0.0
    for f in sorted(RAW.glob("*.txt")):
        a, b = rows(f), rows(VOTED / f.name)
        if len(a) != len(b):
            raise SystemExit(f"{f.name}: {len(a)} vs {len(b)} boxes")
        for x, y in zip(a, b):
            boxes += 1
            moved = max(moved, max(abs(float(p) - float(q)) for p, q in zip(x[1:5], y[1:5])))
            m[(int(x[0]), int(y[0]))] += 1
    changed = sum(n for (i, j), n in m.items() if i != j)
    up = sum(n for (i, j), n in m.items() if j > i)
    one = sum(n for (i, j), n in m.items() if abs(j - i) == 1)
    return {"boxes": boxes, "changed": changed, "changed_pct": round(100 * changed / boxes, 2),
            "up": up, "up_pct": round(100 * up / changed, 1),
            "down": changed - up, "one_step_pct": round(100 * one / changed, 1),
            "max_coord_shift_normalised": moved,
            "pairs": {f"{LADDER[i]}->{LADDER[j]}": n for (i, j), n in
                      sorted(m.items(), key=lambda t: -t[1]) if i != j}}


def main():
    d = direction()
    r = json.loads((HERE / "ROSTER_CHANGE_RATES_V1.json").read_text())["arms"]
    floor = 100.0 * (r["confid"]["changed"] - r["scene"]["changed"]) / r["confid"]["changed"]
    abl = json.loads((HERE / "ablation_assoc" / "STATS_h1_iou.json").read_text())["splits"]["train"]
    m = {"VoteUpPct": f"{d['up_pct']:.1f}", "VoteOneStepPct": f"{d['one_step_pct']:.1f}",
         "VoteChangedBoxes": f"{d['changed']:,}", "ConfidFloorPct": f"{floor:.1f}",
         "AblOvOneVotes": f"{int(abl['median_chain_len_of_boxes'])}"}
    # per-class AP changes: six-class AP is the mean over the six, so the split's headline number is the
    # average of these and the sentence in Section 4 must not contradict them
    s = json.loads((HERE / "STD_LOSS_ANALYSIS_V18.json").read_text())["splits"]
    for split, tag in (("standard", "Std"), ("ood", "OOD")):
        for name, v in s[split]["per_class"].items():
            g = v["gain"]
            m[f"Cls{tag}{name.capitalize()}"] = ("+" if g >= 0 else "$-$") + f"{abs(g):.1f}"
    # the extreme types can only be edited one way, so the raw upward share has a floor built into it
    fl = json.loads((HERE / "VOTE_DIRECTION_FLOOR_V18.json").read_text())
    m["VoteInnerUpPct"] = f"{fl['interior_up_pct']:.1f}"
    m["VoteInnerEdits"] = f"{fl['interior']:,}"
    # per-accelerator OOD gain: the merged table pools the cells, but the contribution claims two
    # accelerators, so the split has to be printed somewhere
    import statistics as _st
    EV = PAPER / "scenes_v1" / "evals_official"
    def _six(arm, h, s):
        return 100 * json.loads((EV / f"ood_{h}_{arm}_s{s}" / "RESULT.json").read_text())["six_class"]["ap50_95"]
    for tag, cells in (("A", [("scene", 1337), ("scene", 3407), ("scene", 4567)]),
                       ("B", [("scene136", 1337), ("scene136", 5678)])):
        gains = [_six("scenevote", h, s) - _six("transparent", h, s) for h, s in cells]
        m[f"SvOODGain{tag}"] = ("+" if _st.mean(gains) >= 0 else "$-$") + f"{abs(_st.mean(gains)):.1f}"
    q = json.loads((HERE / "QUALITATIVE_FIGURE_V18.json").read_text())
    m["QualCommonBoxes"] = str(q["denominator"]["boxes_localised_by_all_arms"])
    (HERE / "VOTE_DIRECTION_V18.json").write_text(json.dumps(
        {"schema": "apsr-vote-direction-v18", **d, "confid_disagreement_floor_pct": round(floor, 1)},
        indent=1) + "\n")
    with open(PAPER / "numbers.tex", "a") as f:
        f.write("\n% --- V18b revision (emit_v18b_fixes.py) ---\n")
        for k, v in m.items():
            f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print(m)
    print("changed", d["changed"], f"({d['changed_pct']}%)  up {d['up']}  down {d['down']}")
    print("max normalised coordinate difference", d["max_coord_shift_normalised"])


if __name__ == "__main__":
    main()
