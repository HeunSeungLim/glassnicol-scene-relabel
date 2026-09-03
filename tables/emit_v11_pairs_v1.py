#!/usr/bin/env python3
"""V12 confusable-pair table (arm pair from argv: ARM_A ARM_B PREFIX ROWSFILE): type-accuracy change, relabelled minus raw, on the manual boxes of each pair
(both arms localised), pooled per accelerator and over every finished cell.  Rows pair_v11_rows.tex, macros PairV*."""
from __future__ import annotations
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from emit_relabel_table_v1 import EV, ANN, IOU, CONF, iou, ROWS, HOST, PAPER
NAMES = ["shot", "whisky", "water", "beer", "wine", "high"]
PAIRS = [("shot", "whisky"), ("whisky", "water"), ("water", "beer"), ("wine", "high")]
ARM_A = sys.argv[1] if len(sys.argv) > 1 else "transparent"; ARM_B = sys.argv[2] if len(sys.argv) > 2 else "relabel"
PREFIX = sys.argv[3] if len(sys.argv) > 3 else "PairV"; ROWS_OUT = sys.argv[4] if len(sys.argv) > 4 else "pair_v11_rows.tex"

def per_box(tag, split):
    d = EV / f"{split}_{tag}"
    if not (d / "RESULT.json").exists(): return None
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
            if bi is None: out.append((a["category_id"] - 1, 0, 0))
            else: used.add(bi); out.append((a["category_id"] - 1, 1, int(pr[img][bi]["category_id"] == a["category_id"])))
    return out

def sg(v): return ("$-$" + f"{abs(v):.1f}") if v < 0 else f"+{v:.1f}"

def main():
    gains = defaultdict(dict)   # (split, pair) -> {(host, seed): gain}
    for host, seed in ROWS:
        for split in ("standard", "ood"):
            oa = per_box(f"{host}_{ARM_A}_s{seed}", split); ob = per_box(f"{host}_{ARM_B}_s{seed}", split)
            if oa is None or ob is None: continue
            for a, b in PAIRS:
                ia, ib = NAMES.index(a), NAMES.index(b)
                both = [(x[2], y[2]) for x, y in zip(oa, ob) if x[1] and y[1] and x[0] in (ia, ib)]
                if both: gains[(split, (a, b))][(host, seed)] = 100 * (st.mean(t for _, t in both) - st.mean(t for t, _ in both))
    lines, m = [], {}
    for split, S in (("standard", "Std."), ("ood", "OOD")):
        negs = 0
        for a, b in PAIRS:
            g = gains[(split, (a, b))]
            h = [v for (host, _), v in g.items() if host == "scene"]; r = [v for (host, _), v in g.items() if host == "scene136"]
            allv = list(g.values()); mean = st.mean(allv); pos = sum(v > 0 for v in allv)
            lines.append(f"{S} & {a}/{b} & {sg(st.mean(h))} & {sg(st.mean(r)) if r else '--'} & {sg(mean)} & {pos}/{len(allv)} \\\\")
            key0 = f"{PREFIX}{a.capitalize()}{'Std' if split == 'standard' else 'OOD'}"; m[key0 + "Max"] = sg(max(allv)); m[key0 + "Median"] = sg(st.median(allv))
            key = f"{PREFIX}{a.capitalize()}{'Std' if split == 'standard' else 'OOD'}"
            m[key + "Mean"] = sg(mean); m[key + "Pos"] = f"{pos}/{len(allv)}"; negs += mean < 0
        m[f"{PREFIX}{'Std' if split == 'standard' else 'OOD'}NegCount"] = str(negs)
    (PAPER / ROWS_OUT).write_text("\n".join(lines) + "\n")
    (Path(__file__).resolve().parent / f"PAIR_GAINS_{PREFIX}_V1.json").write_text(json.dumps({f"{sp}|{a}/{b}": {f"{h}_s{sd}": round(v, 2) for (h, sd), v in g.items()} for (sp, (a, b)), g in gains.items()}, indent=1))
    num = PAPER / "numbers.tex"; s = num.read_text()
    BEG, END = f"% ---- {PREFIX} pairs block (emit_v11_pairs_v1.py {ARM_A} {ARM_B}) ----", f"% ---- end {PREFIX} pairs block ----"
    if BEG in s: s = s[:s.index(BEG)] + s[s.index(END) + len(END):].lstrip("\n")
    num.write_text(s.rstrip("\n") + "\n" + "\n".join([BEG] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()] + [END]) + "\n")
    print("\n".join(lines)); print(m)

if __name__ == "__main__":
    main()
