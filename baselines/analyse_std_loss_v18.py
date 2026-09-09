#!/usr/bin/env python3
"""Where the standard-split loss of the scene vote actually sits.

The paper concedes a 1.9-point six-class AP loss on the standard split and leaves it at that, which reads
as an unexplained cost.  Two questions are answerable from receipts already in the repository:

  1. is the loss about type at all?  Generic AP collapses the six types to one, so it moves only with
     localisation and confidence; comparing it with six-class AP on both splits separates the two effects.
  2. which direction does the vote push labels?  The labeller's estimated confusion says it reads glasses
     as shorter than they are, so a repair should move ids up the height ladder; the two arms' label sets
     are on disk and the edits can simply be counted.

Writes STD_LOSS_ANALYSIS_V18.json.  No GPU, no retraining, no re-inference.
"""
from __future__ import annotations
import collections, json, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
EV = PAPER / "scenes_v1" / "evals_official"
CELLS = [("scene", 1337), ("scene", 3407), ("scene", 4567), ("scene136", 1337), ("scene136", 5678)]
RAWDIR = PAPER / "seed_v1" / "datasets" / "scene136_transparent" / "labels" / "train"
VOTEDIR = PAPER / "seed_v1" / "datasets" / "scene136_scenevote" / "labels" / "train"
# height order of the six types, shortest first; the id order of the release is already this order
LADDER = ["shot", "whisky", "water", "beer", "wine", "high"]


def result(split, host, arm, seed):
    return json.loads((EV / f"{split}_{host}_{arm}_s{seed}" / "RESULT.json").read_text())


def metric(split, arm, path):
    out = []
    for h, s in CELLS:
        d = result(split, h, arm, s)
        for k in path:
            d = d[k]
        out.append(100 * d)
    return out


def gains(split, key):
    raw = metric(split, "transparent", key)
    ours = metric(split, "scenevote", key)
    d = [a - b for a, b in zip(ours, raw)]
    # keep the unrounded mean: rounding here and again at print time turned +7.055 into +7.0
    return {"raw": st.mean(raw), "ours": st.mean(ours), "gain": st.mean(d),
            "positive_cells": sum(x > 0 for x in d), "cells": len(d)}


def read_labels(d):
    out = {}
    for f in d.glob("*.txt"):
        rows = []
        for line in f.read_text().split("\n"):
            q = line.split()
            if len(q) >= 5:
                rows.append((int(q[0]), tuple(round(float(x), 6) for x in q[1:5])))
        out[f.name] = rows
    return out


def edits():
    raw, vote = read_labels(RAWDIR), read_labels(VOTEDIR)
    common = sorted(set(raw) & set(vote))
    m, unmatched, total = collections.Counter(), 0, 0
    for f in common:
        a, b = raw[f], vote[f]
        if len(a) != len(b):
            unmatched += 1
            continue
        by_box = {box: c for c, box in b}
        for c, box in a:
            total += 1
            if box in by_box:
                m[(c, by_box[box])] += 1
            else:
                unmatched += 1
    up = sum(n for (a, b), n in m.items() if b > a)
    down = sum(n for (a, b), n in m.items() if b < a)
    one_step = sum(n for (a, b), n in m.items() if abs(b - a) == 1)
    changed = up + down
    return {"files": len(common), "boxes": total, "unmatched": unmatched,
            "changed": changed, "changed_pct": round(100 * changed / total, 2),
            "up": up, "down": down, "up_pct_of_changed": round(100 * up / changed, 1),
            "one_step_pct_of_changed": round(100 * one_step / changed, 1),
            "top": [[LADDER[a], LADDER[b], n] for (a, b), n in
                    sorted(m.items(), key=lambda t: -t[1]) if a != b][:8],
            "class_shares": {LADDER[c]: [sum(n for (x, _), n in m.items() if x == c),
                                         sum(n for (_, y), n in m.items() if y == c)] for c in range(6)}}


def main():
    out = {"schema": "apsr-std-loss-v18", "cells": [f"{h}_s{s}" for h, s in CELLS], "splits": {}}
    for split in ("standard", "ood"):
        out["splits"][split] = {
            "six_class": gains(split, ["six_class", "ap50_95"]),
            "generic": gains(split, ["collapsed_general", "ap50_95"]),
            "per_class": {LADDER[c - 1]: gains(split, ["six_class", "per_class", str(c), "ap50_95"])
                          for c in range(1, 7)}}
    out["edits"] = edits()
    (HERE / "STD_LOSS_ANALYSIS_V18.json").write_text(json.dumps(out, indent=1) + "\n")

    for split in ("standard", "ood"):
        s = out["splits"][split]
        print(f"\n[{split}]  six {s['six_class']['gain']:+.1f} ({s['six_class']['positive_cells']}/5)"
              f"   generic {s['generic']['gain']:+.1f} ({s['generic']['positive_cells']}/5)")
        for k, v in s["per_class"].items():
            print(f"    {k:7s} raw {v['raw']:5.1f} ours {v['ours']:5.1f}  {v['gain']:+6.1f} "
                  f"({v['positive_cells']}/5)")
    e = out["edits"]
    print(f"\n[edits] {e['changed']} of {e['boxes']} ids ({e['changed_pct']}%), "
          f"{e['up_pct_of_changed']}% upward on the height ladder, "
          f"{e['one_step_pct_of_changed']}% by one step")
    for a, b, n in e["top"]:
        print(f"    {a:7s} -> {b:7s} {n}")


if __name__ == "__main__":
    main()
