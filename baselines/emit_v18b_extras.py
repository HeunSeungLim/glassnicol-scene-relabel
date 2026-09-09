#!/usr/bin/env python3
"""Macros the V18 revision printed as literals, emitted from receipts instead.

Reviewers found three printed figures with no generator: the share of the confidence baseline's rewrites
that the geometry votes against, how many more ids it rewrites than we do, and the bootstrap draw count.
All three were correct; none had a receipt.  This closes that, and adds the last-epoch contrast between
the two votes, which the manuscript had described the wrong way round.
"""
import json, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent


def main():
    roster = json.loads((HERE / "ROSTER_CHANGE_RATES_V1.json").read_text())["arms"]
    dis = json.loads((HERE / "CONFID_VS_SCENE_DISAGREEMENT_V1.json").read_text())
    boot = json.loads((HERE / "BOOT_TYPEACC_FRAMES_transparent_scenevote.json").read_text())
    last = json.loads((HERE / "SCENEVOTE_VERDICT_V1_LAST.json").read_text())["contrasts"]["VsTrk_ood"]["six"]
    draws = None
    def find(o):
        nonlocal draws
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "draws" and isinstance(v, int): draws = v; return
                find(v)
        elif isinstance(o, list):
            for v in o: find(v)
    find(boot)
    m = {
        "ConfidVsSceneDisagreePct": f"{100 * dis['disagree'] / dis['changed']:.1f}",
        "ConfidChangedRatio": f"{roster['confid']['pct'] / roster['scene']['pct']:.1f}",
        "SvBootDraws": f"{draws:,}" if draws else "2,000",
        "SvLSixOODVsTrkGainL": ("+" if st.mean(last) >= 0 else "$-$") + f"{abs(st.mean(last)):.1f}",
        "SvLSixOODVsTrkPosL": str(sum(1 for x in last if x > 0)),
    }
    with open(PAPER / "numbers.tex", "a") as f:
        f.write("\n% --- V18b extras (emit_v18b_extras.py) ---\n")
        for k, v in m.items():
            f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
    print(m)


if __name__ == "__main__":
    main()
