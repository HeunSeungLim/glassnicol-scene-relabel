#!/usr/bin/env python3
"""Last-epoch scores for the two external baselines, the check three reviewers asked for.

The paper reports its arms under a preregistered checkpoint rule and, post hoc, under the last epoch.
The comparison rows were added later and had only the preregistered rule, so a reader could ask whether
the ranking is an artefact of that rule.  This scores every cell of both baselines at the last epoch from
the saved weights and writes the contrast against the released labels.
"""
import json, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
# In the release layout the receipts sit beside this directory, not inside it.
if not (HERE / "SCENEVOTE_VERDICT_V1_LAST.json").exists() and (HERE.parent / "receipts").exists():
    HERE = HERE.parent / "receipts"
EV = HERE.parent / "scenes_v1" / "evals_official"
CELLS = [("scene", 1337), ("scene", 3407), ("scene", 4567), ("scene136", 1337), ("scene136", 5678)]


def ap(host, arm, seed):
    p = EV / f"ood_{host}_{arm}_s{seed}_last" / "RESULT.json"
    return 100 * json.loads(p.read_text())["six_class"]["ap50_95"] if p.exists() else None


def main():
    raw = [ap(h, "transparent", s) for h, s in CELLS]
    assert all(x is not None for x in raw), "released-label cells not scored at the last epoch"
    out, m = {}, {}
    for arm, key in (("lsmooth", "Ls"), ("confid", "Cf"), ("scenevote", "Sv")):
        v = [ap(h, arm, s) for h, s in CELLS]
        if any(x is None for x in v):
            continue
        d = [a - b for a, b in zip(v, raw)]
        out[arm] = {"cells": d, "mean": st.mean(d), "positive": sum(1 for x in d if x > 0)}
        m[f"Last{key}OODGain"] = ("+" if st.mean(d) >= 0 else "$-$") + f"{abs(st.mean(d)):.1f}"
        m[f"Last{key}OODPos"] = str(sum(1 for x in d if x > 0))
    (HERE / "LASTEPOCH_BASELINES_V1.json").write_text(
        json.dumps({"schema": "apsr-lastepoch-baselines-v1", "arms": out}, indent=1, sort_keys=True) + "\n")
    with open(HERE.parent / "numbers.tex", "a") as f:
        f.write("\n% --- V18c last-epoch baselines (emit_v18c_lastepoch.py) ---\n")
        for k, v in m.items():
            f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print(m)


if __name__ == "__main__":
    main()
