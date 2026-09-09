#!/usr/bin/env python3
"""One table for the whole comparison: five label variants, four metrics, both splits.

V18 carried two tables.  One pooled three arms per accelerator; the other compared five arms on six-class
AP alone, so the two external baselines had no recall and no type accuracy anywhere in the paper, and the
per-accelerator pooling repeated information the per-cell receipts already hold.  This merges them.

Six-class AP and generic AP come from the frozen RESULT.json of each cell.  Recall and type accuracy are
recomputed here from the frozen predictions with the paper's own matching rule (Section 3.1): manual boxes
in ascending annotation id greedily take the unused prediction of maximum IoU, at IoU >= 0.5 and confidence
>= 0.25, irrespective of type.  Recall is per arm, the fraction of manual boxes it localises.  Type accuracy
is scored on the boxes every arm localises in that cell, so all five columns share a denominator; scoring
each arm on what it happens to find would hand the arm that finds least the easiest test.

    emit_v18c_master_table.py [--rows OUT.tex] [--macros]
"""
from __future__ import annotations
import argparse, collections, json, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _resolve():
    """Workspace layout, or the copy of the fifty scored cells shipped beside this script."""
    local = HERE.parent / "receipts" / "table1_cells"
    if (local / "ood_types_remap_1based.json").is_file():
        return local, local
    paper = HERE.parent
    return paper / "scenes_v1" / "evals_official", paper.parent / "official_test_v1"


EV, OFF = _resolve()
PAPER = HERE.parent
CELLS = [("scene", 1337), ("scene", 3407), ("scene", 4567), ("scene136", 1337), ("scene136", 5678)]
ARMS = [("transparent", "Released labels"), ("lsmooth", "Label smoothing"),
        ("confid", "Detector confidence"), ("relabel", "Adjacent-frame vote (earlier)"),
        ("scenevote", "Scene vote (ours)")]
IOU_T, CONF_T = 0.5, 0.25


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    i = ix * iy
    u = aw * ah + bw * bh - i
    return i / u if u > 0 else 0.0


def match(gts, preds):
    free = sorted([p for p in preds if p["score"] >= CONF_T], key=lambda p: -p["score"])
    used, out = set(), {}
    for a in sorted(gts, key=lambda g: g["id"]):
        best, bi = None, IOU_T - 1e-12
        for k, p in enumerate(free):
            if k in used:
                continue
            v = iou(a["bbox"], p["bbox"])
            if v > bi:
                best, bi = k, v
        if best is None:
            out[a["id"]] = None
        else:
            used.add(best)
            out[a["id"]] = free[best]["category_id"]
    return out


def gt_of(split):
    d = json.loads((OFF / f"{split}_types_remap_1based.json").read_text())
    by = collections.defaultdict(list)
    for a in d["annotations"]:
        if 1 <= a["category_id"] <= 6:
            by[a["image_id"]].append(a)
    return by


def cell_metrics(split, host, seed, by_img):
    m = {}
    for arm, _ in ARMS:
        preds = json.loads((EV / f"{split}_{host}_{arm}_s{seed}" / "predictions_six_class.json").read_text())
        pim = collections.defaultdict(list)
        for p in preds:
            pim[p["image_id"]].append(p)
        m[arm] = {i: match(by_img[i], pim.get(i, [])) for i in by_img}
    total = sum(len(v) for v in by_img.values())
    common = [(i, a) for i, anns in by_img.items() for a in anns
              if all(m[arm][i][a["id"]] is not None for arm, _ in ARMS)]
    out = {}
    for arm, _ in ARMS:
        loc = sum(1 for i, anns in by_img.items() for a in anns if m[arm][i][a["id"]] is not None)
        cor = sum(1 for i, a in common if m[arm][i][a["id"]] == a["category_id"])
        r = json.loads((EV / f"{split}_{host}_{arm}_s{seed}" / "RESULT.json").read_text())
        out[arm] = {"six": 100 * r["six_class"]["ap50_95"], "gen": 100 * r["collapsed_general"]["ap50_95"],
                    "rec": 100 * loc / total, "type": 100 * cor / len(common)}
    out["_common"] = len(common)
    out["_boxes"] = total
    return out


def sign(x):
    return ("+" if x >= 0 else "$-$") + f"{abs(x):.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(PAPER / "master_v18c_rows.tex"))
    ap.add_argument("--macros", action="store_true")
    a = ap.parse_args()

    data, common = {}, {}
    for split in ("standard", "ood"):
        by = gt_of(split)
        cells = [cell_metrics(split, h, s, by) for h, s in CELLS]
        common[split] = [c["_common"] for c in cells]
        data[split] = {arm: [c[arm] for c in cells] for arm, _ in ARMS}

    lines, macros = [], {}
    key = {"transparent": "Raw", "lsmooth": "Ls", "confid": "Cf", "relabel": "Trk", "scenevote": "Sv"}
    for arm, label in ARMS:
        cells = []
        for split in ("standard", "ood"):
            d = data[split][arm]
            base = data[split]["transparent"]
            six = st.mean(x["six"] for x in d)
            dsix = [x["six"] - b["six"] for x, b in zip(d, base)]
            cells.append((six, st.mean(dsix), sum(1 for g in dsix if g > 0),
                          st.mean(x["gen"] for x in d), st.mean(x["rec"] for x in d),
                          st.mean(x["type"] for x in d)))
            k = key[arm]
            M = "Std" if split == "standard" else "OOD"
            macros[f"Tbl{k}{M}Six"] = f"{six:.1f}"
            macros[f"Tbl{k}{M}Gen"] = f"{cells[-1][3]:.1f}"
            macros[f"Tbl{k}{M}Rec"] = f"{cells[-1][4]:.1f}"
            macros[f"Tbl{k}{M}Type"] = f"{cells[-1][5]:.1f}"
            if arm != "transparent":
                macros[f"Tbl{k}{M}SixGain"] = sign(cells[-1][1])
                macros[f"Tbl{k}{M}SixPos"] = str(cells[-1][2])
        (s6, sg, sp, sgen, srec, styp), (o6, og, op, ogen, orec, otyp) = cells
        if arm == "transparent":
            lines.append(f"{label} & {s6:.1f} & --- & {sgen:.1f} & {srec:.1f} & {styp:.1f} & "
                         f"{o6:.1f} & --- & {ogen:.1f} & {orec:.1f} & {otyp:.1f} \\\\")
        else:
            lines.append(f"{label} & {s6:.1f} & {sign(sg)} ({sp}/5) & {sgen:.1f} & {srec:.1f} & {styp:.1f} & "
                         f"{o6:.1f} & {sign(og)} ({op}/5) & {ogen:.1f} & {orec:.1f} & {otyp:.1f} \\\\")
    macros["TblCommonStd"] = str(min(common["standard"]))
    macros["TblCommonStdMax"] = str(max(common["standard"]))
    macros["TblCommonOOD"] = str(min(common["ood"]))
    macros["TblCommonOODMax"] = str(max(common["ood"]))
    macros["TblBoxesStd"] = str(sum(len(v) for v in gt_of("standard").values()))
    macros["TblBoxesOOD"] = str(sum(len(v) for v in gt_of("ood").values()))

    # Rank on the unrounded values.  Marking the printed strings made two entries that both round to 84.2
    # share second place when one is 84.15744 and the other 84.15206.
    metric_cols = [(1, "s6"), (3, "sgen"), (4, "srec"), (5, "styp"),
                   (6, "o6"), (8, "ogen"), (9, "orec"), (10, "otyp")]
    exact = []
    for arm, _ in ARMS:
        row = {}
        for split, pre in (("standard", "s"), ("ood", "o")):
            d = data[split][arm]
            row[pre + "6"] = st.mean(x["six"] for x in d)
            row[pre + "gen"] = st.mean(x["gen"] for x in d)
            row[pre + "rec"] = st.mean(x["rec"] for x in d)
            row[pre + "typ"] = st.mean(x["type"] for x in d)
        exact.append(row)
    cells_tex = [l[:-3].split(" & ") for l in lines]
    for col, key in metric_cols:
        vals = [r[key] for r in exact]
        best = max(vals)
        second = max(v for v in vals if v < best)
        for i, v in enumerate(vals):
            if v == best:
                cells_tex[i][col] = "\\textbf{%s}" % cells_tex[i][col]
            elif v == second:
                cells_tex[i][col] = "\\second{%s}" % cells_tex[i][col]
    hl = [" & ".join(c) + " \\\\" for c in cells_tex]
    Path(str(a.rows).replace("_rows.tex", "_hl_rows.tex")).write_text("\n".join(hl) + "\n")
    Path(a.rows).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\ncommon boxes per cell  standard", common["standard"], " ood", common["ood"])
    json.dump({"schema": "apsr-master-table-v18c", "cells": [f"{h}_s{s}" for h, s in CELLS],
               "common_boxes": common,
               "arms": {arm: {sp: data[sp][arm] for sp in ("standard", "ood")} for arm, _ in ARMS}},
              open(HERE / "MASTER_TABLE_V18C.json", "w"), indent=1)
    if a.macros:
        with open(PAPER / "numbers.tex", "a") as f:
            f.write("\n% --- V18c merged results table (emit_v18c_master_table.py) ---\n")
            for k, v in macros.items():
                f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
        print("appended", len(macros), "macros")


if __name__ == "__main__":
    main()
