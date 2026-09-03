#!/usr/bin/env python3
"""Three-arm comparison for the scene-vote experiment: raw (transparent), tracklet vote (relabel), scene vote
(scenevote).  Per host: mean +- SD over seeds of six-class AP, generic AP, recall, and type accuracy on the boxes
localised by ALL compared arms of that cell.  Pooled contrasts over every cell where all three arms finished:
scenevote - raw and scenevote - tracklet (type accuracy on boxes localised by both arms of the pair).
Writes arms_v12_rows.tex, cells_v12_rows.tex, macros Sv* (v12 block), SCENEVOTE_VERDICT_V1.json against
PREREGISTRATION_SCENEVOTE_V1.md P1-P4."""
from __future__ import annotations
import json, statistics as st, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from emit_relabel_table_v1 import outcomes, HOST, PAPER
HERE = Path(__file__).resolve().parent
ARMS = [("raw", "transparent"), ("tracklet", "relabel"), ("scenevote", "scenevote")]
ROWS = [("scene", 1337), ("scene", 3407), ("scene", 4567), ("scene136", 1337), ("scene136", 5678)]
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ""      # "_last" scores the last epoch instead of the selected checkpoint
MP = "SvL" if SUFFIX else "Sv"; FS = "_last" if SUFFIX else ""
sg = lambda v: ("$-$" + f"{abs(v):.1f}") if v < 0 else f"+{v:.1f}"

def cell(host, seed, split):
    res = {}
    for name, tag in ARMS:
        r, o = outcomes(f"{host}_{tag}_s{seed}{SUFFIX}", split)
        if r is None: return None
        res[name] = (r, o)
    return res

def metrics(r, o, both):
    return {"six": 100 * r["six_class"]["ap50_95"], "gen": 100 * r["collapsed_general"]["ap50_95"],
            "recall": 100 * st.mean(x[0] for x in o), "type": 100 * st.mean(o[i][1] for i in both)}

def main():
    cells = {}
    for host, seed in ROWS:
        c = {sp: cell(host, seed, sp) for sp in ("standard", "ood")}
        if all(c.values()): cells[(host, seed)] = c
    print("cells with all three arms:", sorted(cells))
    m = {f"{MP}Cells": str(len(cells))}
    if not cells: print("no complete cells yet"); return
    # per-host mean +- SD table, type accuracy on boxes localised by all three arms
    per = {}
    for (host, seed), c in cells.items():
        for sp in ("standard", "ood"):
            both = [i for i in range(len(c[sp]["raw"][1])) if all(c[sp][n][1][i][0] for n, _ in ARMS)]
            for n, _ in ARMS:
                mt = metrics(*c[sp][n], both)
                for k, v in mt.items(): per.setdefault((host, sp, n, k), []).append(v)
    lines = []
    for host in ("scene", "scene136"):
        seeds = sorted(s for h, s in cells if h == host)
        if not seeds: continue
        for n, _ in ARMS:
            vals = []
            for sp in ("standard", "ood"):
                for k in ("six", "gen", "recall", "type"):
                    v = per[(host, sp, n, k)]; mu = st.mean(v); sd = st.stdev(v) if len(v) > 1 else 0.0
                    vals.append(f"{mu:.1f}$\\pm${sd:.1f}" if len(v) > 1 else f"{mu:.1f}")
            lines.append(f"{HOST[host]} ({len(seeds)}) & {dict(ARMS)[n] if False else {'raw':'raw','tracklet':'tracklet vote','scenevote':'scene vote'}[n]} & " + " & ".join(vals) + " \\\\")
        if host == "scene": lines.append("\\addlinespace[2pt]")
    (PAPER / f"arms_v12{FS}_rows.tex").write_text("\n".join(lines) + "\n")
    # pooled pairwise contrasts (both-localised per pair) and per-cell rows
    cl = []; verdict = {}
    for a, b, tag in (("raw", "scenevote", "VsRaw"), ("tracklet", "scenevote", "VsTrk"), ("raw", "tracklet", "TrkVsRaw")):
        for sp, S in (("standard", "Std"), ("ood", "OOD")):
            d = {k: [] for k in ("six", "gen", "recall", "type")}
            for (host, seed), c in sorted(cells.items()):
                ra, oa = c[sp][a]; rb, ob = c[sp][b]
                both = [i for i in range(len(oa)) if oa[i][0] and ob[i][0]]
                ma, mb = metrics(ra, oa, both), metrics(rb, ob, both)
                for k in d: d[k].append(mb[k] - ma[k])
            for k, K in (("six", "Six"), ("gen", "Gen"), ("recall", "Rec"), ("type", "Type")):
                m[f"{MP}{K}{S}{tag}Gain"] = sg(st.mean(d[k])); m[f"{MP}{K}{S}{tag}Pos"] = str(sum(x > 0 for x in d[k]))
            verdict[f"{tag}_{sp}"] = {k: [round(x, 2) for x in v] for k, v in d.items()}
    for (host, seed), c in sorted(cells.items()):
        row = [f"{HOST[host]} & {seed}"]
        for sp in ("standard", "ood"):
            for a, b in (("raw", "scenevote"), ("tracklet", "scenevote")):
                ra, oa = c[sp][a]; rb, ob = c[sp][b]; both = [i for i in range(len(oa)) if oa[i][0] and ob[i][0]]
                row.append(sg(metrics(rb, ob, both)["type"] - metrics(ra, oa, both)["type"]))
        cl.append(" & ".join(row) + " \\\\")
    (PAPER / f"cells_v12{FS}_rows.tex").write_text("\n".join(cl) + "\n")
    # preregistered predictions
    vr = verdict; t = lambda tag, sp: vr[f"{tag}_{sp}"]["type"]
    P1 = st.mean(t("VsRaw", "standard")) > 0 and st.mean(t("VsRaw", "ood")) > 0
    P2 = st.mean(t("VsRaw", "ood")) > st.mean(t("TrkVsRaw", "ood"))
    P3 = all(abs(st.mean(vr[f"VsRaw_{sp}"][k])) < 2 for sp in ("standard", "ood") for k in ("recall", "gen"))
    P4 = sum(x > 0 for x in t("VsRaw", "ood")) >= 4
    out = {"schema": "apsr-scenevote-verdict-v1", "preregistration_sha256": (HERE / "PREREGISTRATION_SCENEVOTE_V1.sha256").read_text().split()[0],
           "cells": [f"{h}_s{s}" for h, s in sorted(cells)], "P1_mean_type_gain_vs_raw_positive_both_splits": P1,
           "P2_ood_type_gain_vs_raw_exceeds_tracklet": P2, "P3_recall_and_generic_within_2": P3, "P4_ood_positive_cells_ge4": P4,
           "contrasts": verdict}
    (HERE / f"SCENEVOTE_VERDICT_V1{FS.upper()}.json").write_text(json.dumps(out, indent=1))
    for k in ("P1_mean_type_gain_vs_raw_positive_both_splits", "P2_ood_type_gain_vs_raw_exceeds_tracklet", "P3_recall_and_generic_within_2", "P4_ood_positive_cells_ge4"): print(k, out[k])
    num = PAPER / "numbers.tex"; s = num.read_text()
    BEG, END = f"% ---- v12 scene-vote block {MP} (emit_v12_arms_v1.py {SUFFIX}) ----", f"% ---- end v12 block {MP} ----"
    if BEG in s: s = s[:s.index(BEG)] + s[s.index(END) + len(END):].lstrip("\n")
    num.write_text(s.rstrip("\n") + "\n" + "\n".join([BEG] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()] + [END]) + "\n")
    print("\n".join(lines)); print({k: v for k, v in m.items() if "Type" in k})

if __name__ == "__main__":
    main()
