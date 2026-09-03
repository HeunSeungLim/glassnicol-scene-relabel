#!/usr/bin/env python3
"""Emit the raw-vs-relabelled results table for the paper.

One row per (host, seed, arm): six-class AP, generic AP, localisation recall and
box-level type accuracy on both official splits (manual labels). Cells whose
evaluation does not exist yet print as --, so the file is always compilable.
Also writes the macros the prose quotes, computed only from the three
preregistered H200 seeds.
"""
from __future__ import annotations
import json, statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
OFF = PAPER.parent / "official_test_v1"
EV = PAPER / "scenes_v1" / "evals_official"
ANN = {"standard": OFF / "standard_types_remap_1based.json", "ood": OFF / "ood_types_remap_1based.json"}
IOU, CONF = 0.5, 0.25
ROWS = [("scene", 1337), ("scene", 3407), ("scene", 4567), ("scene136", 1337), ("scene136", 5678)]
HOST = {"scene": "H200", "scene136": "RTX"}


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by); x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    i = max(0, x2 - x1) * max(0, y2 - y1); return i / (aw * ah + bw * bh - i + 1e-9)


def outcomes(tag, split):
    """per ground-truth box, in a fixed order: (localised, type correct)"""
    d = EV / f"{split}_{tag}"
    if not (d / "RESULT.json").exists(): return None, None
    res = json.loads((d / "RESULT.json").read_text())
    gt = json.loads(ANN[split].read_text()); by = defaultdict(list)
    for a in gt["annotations"]:
        if 1 <= a["category_id"] <= 6: by[a["image_id"]].append(a)
    pr = defaultdict(list)
    for p in json.loads((d / "predictions_six_class.json").read_text()):
        if p["score"] >= CONF: pr[p["image_id"]].append(p)
    out = []
    for img, anns in sorted(by.items()):
        used = set()
        for a in sorted(anns, key=lambda x: x["id"]):
            best, bi = IOU, None
            for i, p in enumerate(pr[img]):
                if i in used: continue
                v = iou(a["bbox"], p["bbox"])
                if v >= best: best, bi = v, i
            if bi is None: out.append((0, 0))
            else: used.add(bi); out.append((1, int(pr[img][bi]["category_id"] == a["category_id"])))
    return res, out


def pair_cells(host, seed, split):
    """Both arms of one (host, seed): recall per arm over all boxes; type accuracy over the
    boxes BOTH arms localised, which is the preregistered definition."""
    ra, oa = outcomes(f"{host}_transparent_s{seed}", split)
    rb, ob = outcomes(f"{host}_relabel_s{seed}", split)
    def one(res, o, both):
        if res is None: return None
        return {"six": 100 * res["six_class"]["ap50_95"], "gen": 100 * res["collapsed_general"]["ap50_95"],
                "recall": 100 * st.mean(x[0] for x in o),
                "type": 100 * st.mean(o[i][1] for i in both) if both else float("nan")}
    if oa is not None and ob is not None:
        both = [i for i in range(len(oa)) if oa[i][0] and ob[i][0]]
    else:
        both = [i for i in range(len(oa or ob or [])) if (oa or ob)[i][0]]
    return one(ra, oa, both), one(rb, ob, both)


def f(x): return "--" if x is None else f"{x:.1f}" if abs(x) < 1000 else str(x)


def main():
    lines = []; gains = defaultdict(list)
    for host, seed in ROWS:
        pc = {sp: pair_cells(host, seed, sp) for sp in ("standard", "ood")}
        for k, (arm, label) in enumerate((("transparent", "raw"), ("relabel", "relabelled"))):
            c = {sp: pc[sp][k] for sp in ("standard", "ood")}
            vals = []
            for sp in ("standard", "ood"):
                x = c[sp]; vals += [f(x and x["six"]), f(x and x["gen"]), f(x and x["recall"]), f(x and x["type"])]
            lines.append(f"{HOST[host]} & {seed} & {label} & " + " & ".join(vals) + r" \\")
            if host == "scene" and seed in (1337, 3407, 4567):
                for sp in ("standard", "ood"):
                    if c[sp]: gains[(sp, arm)].append(c[sp])
        lines.append(r"\cmidrule(lr){3-11}")
    lines[-1] = r"\bottomrule"
    (PAPER / "relabel_table_rows.tex").write_text("\n".join(lines) + "\n")
    m = {}
    for sp, key in (("standard", "Std"), ("ood", "OOD")):
        raw, rel = gains.get((sp, "transparent"), []), gains.get((sp, "relabel"), [])
        if len(raw) == 3 and len(rel) == 3:
            for stat, mk in (("type", "Type"), ("six", "Six"), ("recall", "Rec")):
                d = [b[stat] - a[stat] for a, b in zip(raw, rel)]
                m[f"Rel{mk}{key}Gain"] = f"{st.mean(d):.2f}"; m[f"Rel{mk}{key}Pos"] = str(sum(1 for x in d if x > 0))
            m[f"RelTypeSd{key}Raw"] = f"{st.stdev(a['type'] for a in raw):.2f}"
            m[f"RelTypeSd{key}Rel"] = f"{st.stdev(b['type'] for b in rel):.2f}"
    num = PAPER / "numbers.tex"
    keep = [l for l in num.read_text().splitlines() if not any(l.startswith(f"\\newcommand{{\\{k}}}") for k in m)]
    if m:
        keep += ["% relabelling experiment, three preregistered H200 seeds, official manual-label test"]
        keep += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    num.write_text("\n".join(keep) + "\n")
    print("\n".join(lines)); print("\nmacros:", json.dumps(m) if m else "PENDING (needs all three H200 seeds)")


if __name__ == "__main__":
    main()
