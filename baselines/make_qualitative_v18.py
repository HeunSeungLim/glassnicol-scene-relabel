#!/usr/bin/env python3
"""Render the qualitative comparison of training-label arms on the OOD test split.

Six panels per frame: the manual annotation, then the detector trained on each label variant.  Boxes are
the frozen predictions that produced the paper's tables, matched to the manual boxes by the paper's rule,
so the picture and the numbers cannot drift apart.

Frames are chosen without looking at our arm's output, except for the row that is declared a failure.  The
published figure has two rows, which is what this script writes by default:

  row 1     the frame on which the released-label arm gets the most manual boxes wrong, whether mistyped or
            not localised at all (four frames tie at five; the first by file name is taken);
  row 2     the frame on which our vote mistypes the most boxes that the released-label arm types
            correctly, i.e. our worst frame on this split (a unique maximum).

--neutral N takes N frames by the first rule, one per scene, before the failure row.

    make_qualitative_v18.py [--out FIG.png]
"""
from __future__ import annotations
import argparse, collections, json, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from select_qualitative_v18 import ARMS, CELL, EV, GT, GTDIR, SEED, match

IMAGES = Path(os.environ.get("APSR_TEST_IMAGES", GTDIR / "images"))
TYPES = {1: "shot", 2: "whisky", 3: "water", 4: "beer", 5: "wine", 6: "high"}
COLOUR = {1: "#e41a1c", 2: "#ff9d00", 3: "#1f78b4", 4: "#33a02c", 5: "#6a3d9a", 6: "#00b4c8"}
# short column heads: at the size this figure prints, the full arm names collide (the caption spells them out)
PANEL_TITLES = ["Manual", "Released", "Smoothing", "Confidence", "Adjacent vote", "Ours"]
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
    span = x1 - x0
    # A panel that has to carry every glass -- the manual one -- runs out of room at the size the rest print
    # at, so its labels step down one notch rather than overprint each other.
    n_lab = sum(1 for a in anns if is_gt or (got[a["id"]] is not None))
    FS = 5.0 if n_lab < 5 else 4.1
    unit = span / (PANEL_W_PT)          # data units per printed point, for label placement
    # the per-panel count badge sits bottom right; reserve it so a label never prints on top of it
    placed = [(x1 - span * 0.11, y1 - (y1 - y0) * 0.07, span * 0.13, (y1 - y0) * 0.09)]
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
        # Place the label where it neither leaves the panel nor lands on a label already drawn.  Six boxes
        # in one crowded manual panel used to overprint each other ("shot" over "wine").
        cx = min(max(bx + bw / 2, x0 + hw + span * 0.004), x1 - hw - span * 0.004)
        top_ok = by - span * 0.008 - hh > y0
        cands = []
        SHIFT = (0.0, -hw * 0.9, hw * 0.9, -hw * 1.8, hw * 1.8)
        for dy in range(6):              # every upward slot first: above the box reads best
            for dx in SHIFT:
                if top_ok:
                    cands.append((cx + dx, by - span * 0.008 - hh - dy * hh * 2.1))
        for dy in range(6):
            for dx in SHIFT:
                cands.append((cx + dx, by + hh * 1.4 + dy * hh * 2.1))
        free = [(x, y) for x, y in cands
                if x0 + hw <= x <= x1 - hw and y0 + hh <= y <= y1 - hh
                and not any(abs(x - px) < hw + phw and abs(y - py) < hh + phh
                            for px, py, phw, phh in placed)]
        lx, ly = free[0] if free else (cx, min(max(by + hh * 1.4, y0 + hh), y1 - hh))
        placed.append((lx, ly, hw, hh))
        ax.text(lx, ly, lab, color=COLOUR[t], fontsize=FS, ha="center", va="center",
                fontweight="bold" if not ok else "normal",
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.35))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neutral", type=int, default=1,
                    help="frames chosen by the released-label arm's error; the last row is always "
                         "our worst frame. The published figure uses the default, 1.")
    ap.add_argument("--out", default=str(HERE.parents[1] / "build_v18_paper" / "figures" /
                                        "figure_qualitative.png"))
    a = ap.parse_args()
    by_img, names, m = load()
    rows = pick(by_img, names, m, a.neutral)

    # One denominator for every column: the manual boxes that all five arms localise.  Scoring each arm on
    # the boxes it happens to find would give the arm that finds fewest the easiest test, and ours finds
    # fewest, so the per-arm denominator would flatter us.
    common = [(i, x) for i, anns in by_img.items() for x in anns
              if all(m[arm][i][x["id"]] for arm, _ in ARMS)]
    split_acc = {arm: 100.0 * sum(1 for i, x in common
                                  if m[arm][i][x["id"]][0] == x["category_id"]) / len(common)
                 for arm, _ in ARMS}
    n_common = len(common)

    nc = 1 + len(ARMS)
    # Size the canvas so each axes slot has exactly the crop's aspect: any mismatch would come back as a
    # band of white between the rows, which no amount of hspace can remove.
    AR = 16 / 9.0
    # The page places this figure at \\textwidth, so drawing it on a narrower canvas makes LaTeX scale
    # everything up, text included: 4.6in of canvas becomes 7in of page and a 5.3pt label prints at 8.1pt.
    # Rendering at 7in and keeping the same point sizes is what left the labels at 4.4pt in the last draft.
    W, L, R, TOP, BOT, WS, HS = 4.6, 0.052, 0.999, 0.838, 0.004, 0.030, 0.050
    pw = W * (R - L) / (nc + (nc - 1) * WS)
    globals()["PANEL_W_PT"] = pw * 72.0
    ph = pw / AR
    H = (len(rows) + (len(rows) - 1) * HS) * ph / (TOP - BOT)
    fig, axes = plt.subplots(len(rows), nc, figsize=(W, H), dpi=620)
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
                sub = f"{n_common} shared" if c == 0 else f"{split_acc[arm]:.1f}% correct"
                ax.set_title(PANEL_TITLES[c] + "\n" + sub, fontsize=6.0, pad=1.6, linespacing=1.1)
            if c == 0:
                ax.set_ylabel(f"sc.{row['scene']}", fontsize=5.2, labelpad=1.2)
            n = len(anns)
            if c == 0:
                txt = f"{n} glasses"
            else:
                got = m[arm][row["id"]]
                ok = sum(1 for x in anns if got[x["id"]] and got[x["id"]][0] == x["category_id"])
                txt = f"{ok}/{n}"
            ax.text(0.986, 0.035, txt, transform=ax.transAxes, fontsize=5.2, ha="right", va="bottom",
                    color="black", bbox=dict(fc="white", ec="none", alpha=0.78, pad=0.6))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=620, bbox_inches="tight", pad_inches=0.004)
    print("wrote", out)
    receipt = {"schema": "apsr-qualitative-v18", "cell": f"A_s{SEED}", "split": "ood",
               "match": {"iou": 0.5, "conf": 0.25},
               "split_type_accuracy": {arm: round(v, 2) for arm, v in split_acc.items()},
               "denominator": {"boxes_localised_by_all_arms": n_common},
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
