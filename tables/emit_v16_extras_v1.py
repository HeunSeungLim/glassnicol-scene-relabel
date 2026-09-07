#!/usr/bin/env python3
"""Macros added for the V16 revision: positive-cell counts the text was missing, the tracklet vote's
last-epoch OOD six-class contrast (the comparison that makes the checkpoint rule matter), and the
scene-level resampling audit.  Values come from the verdict receipts and AUDIT_SCENE_BOOTSTRAP_V1.json."""
import json, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent
best = json.load(open(HERE / "SCENEVOTE_VERDICT_V1.json"))["contrasts"]
last = json.load(open(HERE / "SCENEVOTE_VERDICT_V1_LAST.json"))["contrasts"]
audit = json.load(open(HERE / "AUDIT_SCENE_BOOTSTRAP_V1.json")); boot = audit["cells"]
def sign(x): return ("+" if x >= 0 else "$-$") + f"{abs(x):.1f}"
m = {}
m["SvSixStdVsRawPos"] = str(sum(1 for x in best["VsRaw_standard"]["six"] if x > 0))
m["SvGenStdVsRawPos"] = str(sum(1 for x in best["VsRaw_standard"]["gen"] if x > 0))
m["TrkLSixOODVsRawGainL"] = sign(st.mean(last["TrkVsRaw_ood"]["six"]))
m["TrkLSixOODVsRawPosL"] = str(sum(1 for x in last["TrkVsRaw_ood"]["six"] if x > 0))
m["SceneBootZero"] = str(sum(1 for v in boot.values() if v["scene_ci_contains_zero"]))
m["SceneBootCells"] = str(len(boot))
m["SceneBootScenes"] = str(len(audit["scenes_per_split"]["ood"]))
with open(HERE.parent / "numbers.tex", "a") as f:
    f.write("\n% --- V16 extras (emit_v16_extras_v1.py) ---\n")
    for k, v in m.items(): f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
print(m)
