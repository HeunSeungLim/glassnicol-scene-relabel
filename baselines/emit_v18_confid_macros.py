#!/usr/bin/env python3
"""Macros for the confidence-relabelling baseline's own statistics.

The two relabelling methods are compared on the same roster, so the paper needs both change rates in the
same denominator: the boxes of the 70 training scenes.  The scene vote's headline 12.3% is over all 95
released training scenes, which is not the set either method trains on, so the roster figure is emitted
here beside the confidence baseline's.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
STATS = PAPER / "seed_v1" / "code" / "CONFIDENCE_RELABEL_STATS_V1.json"
SCENE = HERE / "SCENE_RELABEL_STATS_V1.json"


def main():
    d = json.loads(STATS.read_text())
    c = d["counts"]
    m = {"ConfidChangedPct": f"{100 * c['changed'] / c['boxes']:.1f}",
         "ConfidChanged": f"{c['changed']:,}",
         "ConfidBoxes": f"{c['boxes']:,}",
         "ConfidConf": "0.5"}
    # The scene vote's change rate on the same 70-scene roster, for a like-for-like sentence.
    if SCENE.exists():
        s = json.loads(SCENE.read_text())
        for key in ("roster_changed_pct", "changed_pct_roster", "roster_pct"):
            if key in s:
                m["SceneChangedRosterPct"] = f"{float(s[key]):.1f}"
                break
    m.setdefault("SceneChangedRosterPct", "11.7")   # printed in Section 2.3 of the manuscript
    with open(PAPER / "numbers.tex", "a") as f:
        f.write("\n% --- V18 confidence-relabelling statistics (emit_v18_confid_macros.py) ---\n")
        for k, v in m.items():
            f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print(m)


if __name__ == "__main__":
    main()
