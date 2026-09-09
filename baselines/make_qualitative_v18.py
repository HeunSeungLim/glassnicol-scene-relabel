#!/usr/bin/env python3
"""Render the qualitative comparison of training-label arms on the OOD test split.

Six panels per frame: the manual annotation, then the detector trained on each label variant.  Boxes are
the frozen predictions that produced the paper's tables, matched to the manual boxes by the paper's rule,
so the picture and the numbers cannot drift apart.

Frames are chosen without looking at our arm's output, except for the row that is declared a failure:

  rows 1-2  the frame of each of two OOD scenes on which the released-label arm mistypes the most manual
            boxes (ties: more boxes, then file name);
  row 3     the frame on which our vote mistypes the most boxes that the released-label arm types
            correctly, i.e. our worst frame on this split.

    make_qualitative_v18.py [--out FIG.png]
"""
from __future__ import annotations
import argparse, collections, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_qualitative_v18 import ARMS, CELL, EV, GT, ROOT, SEED, match

IMAGES = ROOT / "official_test_v1" / "images"
TYPES = {1: "shot", 2: "whisky", 3: "water", 4: "beer", 5: "wine", 6: "high"}
COLOUR = {1: "#e41a1c", 2: "#ff9d00", 3: "#1f78b4", 4: "#33a02c", 5: "#6a3d9a", 6: "#00b4c8"}
PANEL_TITLES = ["Manual annotation"] + [lab for _, lab in ARMS]
PANEL_W_PT = 79.0


def scene_of(name):
    i = int(name[:3])
    return 501 if i < 91 else (502 if i < 116 else 503)


def load():
    gt = json.loads(GT.read_text())
    by_img = collections.defaultdict(list)
    for a in gt["annotations"]:
        if 1 <= a["category_id"] <= 6:
            by_img[a["image_id"]].append(a)
    names = {im["id"]: im["file_name"] for im in gt["images"]}
    m = {}
    for arm, _ in ARMS:
        preds = json.loads((EV / f"ood_{CELL}_{arm}_s{SEED}" / "predictions_six_class.json").read_text())
        pim = collections.defaultdict(list)
        for p in preds:
            pim[p["image_id"]].append(p)
        m[arm] = {i: match(by_img[i], pim.get(i, [])) for i in by_img}
    return by_img, names, m


def pick(by_img, names, m, n_neutral=2):
    """Two frames by the released-label arm's error, then our worst frame."""
    stat = []
    for i, anns in by_img.items():
        raw_err = sum(1 for a in anns
                      if not m["transparent"][i][a["id"]]
                      or m["transparent"][i][a["id"]][0] != a["category_id"])
        breaks = 0
        for a in anns:
            r, o = m["transparent"][i][a["id"]], m["scenevote"][i][a["id"]]
            breaks += bool(r) and r[0] == a["category_id"] and not (o and o[0] == a["category_id"])
        stat.append({"id": i, "name": names[i], "scene": scene_of(names[i]), "boxes": len(anns),
                     "raw_err": raw_err, "ours_breaks": breaks})
    hard = sorted(stat, key=lambda r: (-r["raw_err"], -r["boxes"], r["name"]))
    chosen, seen = [], set()
    for r in hard:
        if r["scene"] in seen:
            continue
        chosen.append(dict(r, why="released labels mistype the most here"))
        seen.add(r["scene"])
        if len(chosen) == n_neutral:
            break
    worst = sorted(stat, key=lambda r: (-r["ours_breaks"], -r["boxes"], r["name"]))[0]
    chosen.append(dict(worst, why="our worst frame on this split"))
    return chosen


def crop_box(anns, w, h, pad=0.12):
    xs = [a["bbox"][0] for a in anns] + [a["bbox"][0] + a["bbox"][2] for a in anns]
    ys = [a["bbox"][1] for a in anns] + [a["bbox"][1] + a["bbox"][3] for a in anns]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    mx, my = (x1 - x0) * pad, (y1 - y0) * pad
    x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my * 1.6, y1 + my
    # keep a constant aspect so panels align
    ar = 16 / 9.0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ww = max(x1 - x0, (y1 - y0) * ar); hh = ww / ar
    x0, x1, y0, y1 = cx - ww / 2, cx + ww / 2, cy - hh / 2, cy + hh / 2
    if x0 < 0: x1, x0 = x1 - x0, 0
    if y0 < 0: y1, y0 = y1 - y0, 0
    if x1 > w: x0, x1 = x0 - (x1 - w), w
    if y1 > h: y0, y1 = y0 - (y1 - h), h
    return max(0, x0), min(w, x1), max(0, y0), min(h, y1)


def draw(ax, img, box, anns, got, is_gt):
    x0, x1, y0, y1 = box
    ax.imshow(img, aspect="auto")
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color("0.35")
    span, FS = x1 - x0, 4.3
    unit = span / (PANEL_W_PT)          # data units per printed point, for label placement
    placed = []                          # (xc, y, half width, half height) of labels already drawn
    for a in sorted(anns, key=lambda a: a["bbox"][1]):
        t = a["category_id"] if is_gt else (got[a["id"]][0] if got[a["id"]] else None)
        bx, by, bw, bh = a["bbox"]
        if t is None:                    # the arm did not localise this glass
            ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec="0.5", lw=0.5, ls=(0, (1.2, 1.2))))
            continue
        ok = is_gt or t == a["category_id"]
        ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec=COLOUR[t], lw=0.8,
                               ls="-" if ok else (0, (2.0, 1.0))))
        lab = TYPES[t] if ok else TYPES[t] + "\u2717"
        hw = 0.5 * len(lab) * 0.58 * FS * unit
        hh = 0.62 * FS * unit
        # a box near the top of the crop keeps its label inside, so nothing lands on the column title
        inside = by - span * 0.008 < y0 + span * 0.035
        ly = by + hh * 1.4 if inside else by - span * 0.008 - hh
        lx = min(max(bx + bw / 2, x0 + hw + span * 0.004), x1 - hw - span * 0.004)
        for _ in range(6):               # nudge upward off a label already placed
            if not any(abs(lx - px) < hw + phw and abs(ly - py) < hh + phh
                       for px, py, phw, phh in placed):
                break
            ly -= hh * 2.15
        if ly - hh < y0:
            ly = by + hh * 1.4
        placed.append((lx, ly, hw, hh))
        ax.text(lx, ly, lab, color=COLOUR[t], fontsize=FS, ha="center", va="center",
                fontweight="bold" if not ok else "normal",
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.35))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neutral", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "paper_work_v4" / "build_v18_paper" / "figures" /
                                        "figure_qualitative.png"))
    a = ap.parse_args()
    by_img, names, m = load()
    rows = pick(by_img, names, m, a.neutral)

    # split-level type accuracy per arm, so three frames are read as examples of a measured whole
    split_acc = {}
    for arm, _ in ARMS:
        loc = cor = 0
        for i, anns in by_img.items():
            for x in anns:
                g = m[arm][i][x["id"]]
                if g:
                    loc += 1
                    cor += g[0] == x["category_id"]
        split_acc[arm] = 100.0 * cor / max(loc, 1)

    nc = 1 + len(ARMS)
    # Size the canvas so each axes slot has exactly the crop's aspect: any mismatch would come back as a
    # band of white between the rows, which no amount of hspace can remove.
    AR = 16 / 9.0
    W, L, R, TOP, BOT, WS, HS = 7.0, 0.040, 0.999, 0.872, 0.004, 0.030, 0.050
    pw = W * (R - L) / (nc + (nc - 1) * WS)
    globals()["PANEL_W_PT"] = pw * 72.0
    ph = pw / AR
    H = (len(rows) + (len(rows) - 1) * HS) * ph / (TOP - BOT)
    fig, axes = plt.subplots(len(rows), nc, figsize=(W, H), dpi=400)
    fig.subplots_adjust(left=L, right=R, top=TOP, bottom=BOT, wspace=WS, hspace=HS)
    for r, row in enumerate(rows):
        anns = sorted(by_img[row["id"]], key=lambda x: x["id"])
        img = Image.open(IMAGES / row["name"]).convert("RGB")
        box = crop_box(anns, img.width, img.height)
        for c in range(nc):
            ax = axes[r][c]
            arm = None if c == 0 else ARMS[c - 1][0]
            draw(ax, img, box, anns, None if c == 0 else m[arm][row["id"]], c == 0)
            if r == 0:
                sub = "manual types" if c == 0 else f"{split_acc[arm]:.1f}% types on this split"
                ax.set_title(PANEL_TITLES[c] + "\n" + sub, fontsize=4.9, pad=1.4, linespacing=1.12)
            if c == 0:
                ax.set_ylabel(f"scene {row['scene']}", fontsize=4.6, labelpad=1.3)
            n = len(anns)
            if c == 0:
                txt = f"{n} glasses"
            else:
                got = m[arm][row["id"]]
                ok = sum(1 for x in anns if got[x["id"]] and got[x["id"]][0] == x["category_id"])
                txt = f"{ok}/{n}"
            ax.text(0.986, 0.035, txt, transform=ax.transAxes, fontsize=4.3, ha="right", va="bottom",
                    color="black", bbox=dict(fc="white", ec="none", alpha=0.78, pad=0.6))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.004)
    print("wrote", out)
    receipt = {"schema": "apsr-qualitative-v18", "cell": f"A_s{SEED}", "split": "ood",
               "match": {"iou": 0.5, "conf": 0.25},
               "split_type_accuracy": {arm: round(v, 2) for arm, v in split_acc.items()},
               "rows": rows}
    (Path(__file__).parent / "QUALITATIVE_FIGURE_V18.json").write_text(
        json.dumps(receipt, indent=1) + "\n")
    for row in rows:
        print(f"  {row['name']} scene {row['scene']} boxes {row['boxes']} "
              f"raw_err {row['raw_err']} ours_breaks {row['ours_breaks']} :: {row['why']}")
    for arm, lab in ARMS:
        print(f"  {lab:22s} {split_acc[arm]:5.1f}%")


if __name__ == "__main__":
    main()
