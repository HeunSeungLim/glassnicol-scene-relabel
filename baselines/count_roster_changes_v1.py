#!/usr/bin/env python3
"""Count what each relabelling changed on the 70-scene training roster, and how the two disagree.

Three receipts the paper prints from had no generator in the release: the per-arm change rate, the share
of the confidence baseline's rewrites that the geometric vote contradicts, and the fold assignment.  This
writes all three from the label trees themselves, so a reader can reproduce 11.7%, 22.5% and 81.8%.

    count_roster_changes_v1.py RAW_LABELS SCENE_LABELS CONFID_LABELS SPLIT_JSON MANIFEST OUTDIR
"""
import json, re, sys
from pathlib import Path


def scene_of(path):
    m = re.search(r"_s(\d+)_f", Path(path).name)
    assert m is not None, path
    return m.group(1)


def main():
    raw, scene, confid, split_p, manifest_p, out = (Path(x) for x in sys.argv[1:7])
    split = json.loads(split_p.read_text())
    keep = set(split["train_scenes"])
    man = {json.loads(x)["stem"]: json.loads(x)
           for x in manifest_p.read_text().splitlines() if json.loads(x)["split"] == "train"}
    arms = {"scene": scene, "confid": confid}
    counts = {k: {"changed": 0, "boxes": 0} for k in arms}
    both = {"changed": 0, "agree": 0}
    for p in sorted(raw.glob("*.txt")):
        st = p.stem
        if st not in man or scene_of(man[st]["transparent_image"]) not in keep:
            continue
        R = [l.split()[0] for l in p.read_text().splitlines() if l.strip()]
        A = {k: [l.split()[0] for l in (d / f"{st}.txt").read_text().splitlines() if l.strip()]
             for k, d in arms.items()}
        for k in arms:
            counts[k]["boxes"] += len(R)
            counts[k]["changed"] += sum(1 for a, b in zip(R, A[k]) if a != b)
        for r, c, s in zip(R, A["confid"], A["scene"]):
            if c != r:
                both["changed"] += 1
                both["agree"] += int(c == s)
    for k in counts:
        counts[k]["pct"] = round(100 * counts[k]["changed"] / counts[k]["boxes"], 3)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ROSTER_CHANGE_RATES_V1.json").write_text(json.dumps(
        {"schema": "apsr-roster-change-rate-v1", "roster_scenes": len(keep), "arms": counts},
        indent=1, sort_keys=True) + "\n")
    (out / "CONFID_VS_SCENE_DISAGREEMENT_V1.json").write_text(json.dumps(
        {"schema": "apsr-confid-vs-scene-disagreement-v1", "changed": both["changed"],
         "agree_with_scene_vote": both["agree"], "disagree": both["changed"] - both["agree"],
         "disagree_pct": round(100 * (both["changed"] - both["agree"]) / both["changed"], 3),
         "population": "the 70-scene training roster"}, indent=1, sort_keys=True) + "\n")
    scenes = sorted(split["train_scenes"])
    (out / "CONFIDENCE_FOLD_ASSIGNMENT_V1.json").write_text(json.dumps(
        {"schema": "apsr-confidence-fold-assignment-v1",
         "rule": "training scenes sorted, then alternating; fixed by the roster alone",
         "folds": {"a": scenes[0::2], "b": scenes[1::2]},
         "counts": {"a": len(scenes[0::2]), "b": len(scenes[1::2])},
         "judge": "each box is judged by the model trained on the other fold"},
        indent=1, sort_keys=True) + "\n")
    print(json.dumps({"arms": counts, "disagree": both}, sort_keys=True))


if __name__ == "__main__":
    main()
