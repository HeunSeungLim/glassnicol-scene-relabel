#!/usr/bin/env python3
"""Table rows and macros for the comparison against existing label-noise methods.

The paper compared three label variants that are all ours: the released ids, our earlier adjacent-frame
vote, and the scene vote.  A reader asks the obvious question first: how does this compare with what people
already do about noisy labels?  Two standard answers are trained here on the same roster, budget, seeds and
accelerators, and scored by the same protocol:

  lsmooth  train on the released labels with classification label smoothing, i.e. absorb the noise
           instead of repairing it;
  confid   replace a type id when a detector that never saw the box disagrees confidently, the
           confident-learning family, with out-of-fold judgements from two scene-disjoint halves.

Every cell is a (accelerator, seed) pair, exactly as the existing arms.  Rows carry the mean over cells and
the number of cells above the raw labels, so a method that wins on average by winning one cell is visible.

    emit_v18_baselines.py [--rows OUT.tex] [--macros]
"""
from __future__ import annotations
import argparse, json, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
EV = HERE.parent / "scenes_v1" / "evals_official"
PAPER = HERE.parent

CELLS = [("scene", 1337), ("scene", 3407), ("scene", 4567), ("scene136", 1337), ("scene136", 5678)]
ARMS = [("transparent", "Released labels"),
        ("lsmooth", "Label smoothing~\\cite{muller2019labelsmoothing}"),
        ("confid", "Detector confidence~\\cite{northcutt2021confident}"),
        ("relabel", "Adjacent-frame vote (earlier)"),
        ("scenevote", "Scene vote (ours)")]


def read(host, arm, seed, split):
    p = EV / f"{split}_{host}_{arm}_s{seed}" / "RESULT.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {"six": 100 * d["six_class"]["ap50_95"], "gen": 100 * d["collapsed_general"]["ap50_95"]}


def collect(arm):
    out = {}
    for split in ("standard", "ood"):
        vals = [read(h, arm, s, split) for h, s in CELLS]
        if any(v is None for v in vals):
            return None
        out[split] = vals
    return out


def sign(x):
    return ("+" if x >= 0 else "$-$") + f"{abs(x):.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(PAPER / "baselines_v18_rows.tex"))
    ap.add_argument("--macros", action="store_true")
    a = ap.parse_args()

    found = {arm: collect(arm) for arm, _ in ARMS}
    missing = [arm for arm, v in found.items() if v is None]
    if missing:
        raise SystemExit("cells not scored yet: " + ", ".join(missing))
    data = {arm: v for arm, v in found.items() if v is not None}
    raw = data["transparent"]
    lines, macros = [], {}
    for arm, label in ARMS:
        d = data[arm]
        cells = []
        for split in ("standard", "ood"):
            six = [v["six"] for v in d[split]]
            base = [v["six"] for v in raw[split]]
            gain = [x - b for x, b in zip(six, base)]
            cells.append((st.mean(six), st.mean(gain), sum(1 for g in gain if g > 0)))
        (s_mean, s_gain, s_pos), (o_mean, o_gain, o_pos) = cells
        if arm == "transparent":
            lines.append(f"{label} & {s_mean:.1f} & --- & {o_mean:.1f} & --- \\\\")
        else:
            lines.append(f"{label} & {s_mean:.1f} & {sign(s_gain)} ({s_pos}/5) & "
                         f"{o_mean:.1f} & {sign(o_gain)} ({o_pos}/5) \\\\")
        key = {"transparent": "Raw", "lsmooth": "Ls", "confid": "Cf",
               "relabel": "Trk", "scenevote": "Sv"}[arm]
        macros[f"Base{key}StdSix"] = f"{s_mean:.1f}"
        macros[f"Base{key}OODSix"] = f"{o_mean:.1f}"
        if arm != "transparent":
            macros[f"Base{key}OODGain"] = sign(o_gain)
            macros[f"Base{key}OODPos"] = str(o_pos)
            macros[f"Base{key}StdGain"] = sign(s_gain)
            macros[f"Base{key}StdPos"] = str(s_pos)

    Path(a.rows).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if a.macros:
        with open(PAPER / "numbers.tex", "a") as f:
            f.write("\n% --- V18 external-baseline comparison (emit_v18_baselines.py) ---\n")
            for k, v in macros.items():
                f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
        print("appended", len(macros), "macros")
    json.dump({"cells": [f"{h}_s{s}" for h, s in CELLS],
               "arms": {arm: {sp: [v["six"] for v in data[arm][sp]] for sp in ("standard", "ood")}
                        for arm, _ in ARMS}},
              open(HERE / "BASELINE_COMPARISON_V18.json", "w"), indent=1)


if __name__ == "__main__":
    main()
