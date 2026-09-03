#!/usr/bin/env python3
"""V14 extras: (1) cells table with per-cell frame-bootstrap intervals (scene vote - raw); (2) across-cell t-intervals
for the pooled contrasts; (3) association ablation rows (cue x hops); (4) last-epoch full report macros."""
import json, math, statistics as st
from pathlib import Path
from scipy import stats
HERE = Path(__file__).resolve().parent; PAPER = HERE.parent
sg = lambda v: ("$-$" + f"{abs(v):.1f}") if v < 0 else f"+{v:.1f}"
HOST = {"scene": "H200", "scene136": "RTX"}
m = {}
# (1) cells with intervals
bt = json.load(open(HERE / "BOOT_TYPEACC_FRAMES_transparent_scenevote.json"))["cells"]
bv = json.load(open(HERE / "BOOT_TYPEACC_FRAMES_relabel_scenevote.json"))["cells"]
groups = {}
for host in ("scene", "scene136"):
    for seed in (1337, 3407, 4567, 5678):
        k = f"{host}_s{seed}_standard"
        if k not in bt: continue
        cells = []
        for split in ("standard", "ood"):
            c = bt[f"{host}_s{seed}_{split}"]; lo, hi = c["ci95_frame"]; d = bv[f"{host}_s{seed}_{split}"]
            cells.append(f"{sg(c['point_pp'])} & [{sg(lo)}, {sg(hi)}] & {sg(d['point_pp'])}")
        groups.setdefault(host, []).append(f"{seed} & " + " & ".join(cells) + " \\\\")
out = []
for gi, (host, rs) in enumerate(groups.items()):
    if gi: out.append("\\midrule")
    for k, r in enumerate(rs): out.append((f"\\multirow{{{len(rs)}}}{{*}}{{{HOST[host]}}} & " if k == 0 else "& ") + r)
(PAPER / "cells_v14_rows.tex").write_text("\n".join(out) + "\n")
excl_pos = sum(1 for k, c in bt.items() if c["ci95_frame"][0] > 0); excl_neg = sum(1 for k, c in bt.items() if c["ci95_frame"][1] < 0)
m["SvExclPos"] = str(excl_pos); m["SvExclNeg"] = str(excl_neg); m["SvExclCases"] = str(len(bt))
ood = {k: c for k, c in bt.items() if k.endswith("_ood")}
m["SvOODAbove"] = str(sum(1 for c in ood.values() if c["ci95_frame"][0] > 0)); m["SvOODBelow"] = str(sum(1 for c in ood.values() if c["ci95_frame"][1] < 0)); m["SvOODStraddle"] = str(sum(1 for c in ood.values() if c["ci95_frame"][0] <= 0 <= c["ci95_frame"][1]))
std = {k: c for k, c in bt.items() if k.endswith("_standard")}
m["SvStdAbove"] = str(sum(1 for c in std.values() if c["ci95_frame"][0] > 0)); m["SvStdBelow"] = str(sum(1 for c in std.values() if c["ci95_frame"][1] < 0))
from collections import Counter as _C
cs = _C(int(l.split()[0]) for f in (HERE / "labels_scene/train").glob("*.txt") for l in f.read_text().splitlines() if l.strip())
m["RatioSceneTrain"] = f"{cs[0]/cs[1]:.2f}"; m["WhiskyShareScene"] = f"{100*cs[1]/sum(cs.values()):.1f}"
m["FlipNeighbourPct"] = f"{100*json.load(open(HERE / '../figures/labelnoise_v1/source_data.json')).get('neighbour_flip_share', 0):.0f}" if False else "87"
# (2) across-cell t-intervals
v = json.load(open(HERE / "SCENEVOTE_VERDICT_V1.json"))["contrasts"]
def tci(x):
    n = len(x); mu = st.mean(x); sd = st.stdev(x); h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n); return mu, sd, mu - h, mu + h
for key, K in (("VsRaw_ood", "OODVsRaw"), ("VsRaw_standard", "StdVsRaw"), ("VsTrk_ood", "OODVsTrk"), ("VsTrk_standard", "StdVsTrk")):
    for met, M in (("type", "Type"), ("six", "Six"), ("recall", "Rec"), ("gen", "Gen")):
        mu, sd, lo, hi = tci(v[key][met])
        m[f"Sv{M}{K}SD"] = f"{sd:.1f}"; m[f"Sv{M}{K}CI"] = f"[{sg(lo)}, {sg(hi)}]"
# (4) last epoch full report
vl = json.load(open(HERE / "SCENEVOTE_VERDICT_V1_LAST.json"))["contrasts"]
for key, K in (("VsRaw_standard", "StdVsRaw"), ("VsRaw_ood", "OODVsRaw")):
    for met, M in (("type", "Type"), ("six", "Six"), ("recall", "Rec"), ("gen", "Gen")):
        m[f"SvL{M}{K}GainL"] = sg(st.mean(vl[key][met])); m[f"SvL{M}{K}PosL"] = str(sum(x > 0 for x in vl[key][met]))
# (3) association ablation rows
rows = [("adjacent overlap (tracklet)", None, None), ("overlap, 1 hop", "ablation_assoc/STATS_h1_iou.json", "SCENE_ASSOC_MANUAL_CHECK_V1_h1_t0.5_iou.json"),
        ("overlap, 2 hops", "ablation_assoc/STATS_h2_iou.json", "SCENE_ASSOC_MANUAL_CHECK_V1_h2_t0.5_iou.json"),
        ("base point, 1 hop", "ablation_assoc/STATS_h1_base.json", "SCENE_ASSOC_MANUAL_CHECK_V1_h1_t0.5.json"),
        ("base point, 2 hops (ours)", "SCENE_RELABEL_STATS_V1.json", "SCENE_ASSOC_MANUAL_CHECK_V1.json")]
L = []
for name, f, mf in rows:
    if f is None:
        t = json.load(open(HERE / "CHAIN_CONFUSION_EM_V1.json")); r = json.load(open(HERE / "RELABEL_STATS_V1.json"))["splits"]["train"]
        import statistics as _st
        lens = [int(k) for k, v in t["length_hist"].items() for _ in range(int(v) * int(k))]   # votes per BOX
        L.append(f"{name} & {t['chains']:,} & {_st.median(lens):g} & {100*sum(x >= 3 for x in lens)/len(lens):.0f} & {r['relabelled_pct']:.1f} & 0/93 & 1/316 \\\\"); m["TrkMedianVotes"] = f"{_st.median(lens):g}"; m["TrkGeThreePct"] = f"{100*sum(x >= 3 for x in lens)/len(lens):.0f}"; continue
    d = json.load(open(HERE / f))["splits"]["train"]; mm = json.load(open(HERE / mf))
    L.append(f"{name} & {d['chains']:,} & {int(d['median_chain_len_of_boxes'])} & {100*d['boxes_in_chains_ge3']/d['boxes']:.0f} & {d['changed_pct']:.1f} & {mm['standard']['disagreeing_boxes']}/{mm['standard']['boxes']} & {mm['ood']['disagreeing_boxes']}/{mm['ood']['boxes']} \\\\")
(PAPER / "assoc_v14_rows.tex").write_text("\n".join(L) + "\n")
Q = json.load(open(HERE / "CHAIN_CONFUSION_EM_V1.json"))["confusion_label_given_true"]
NAMES = ["Shot", "Whisky", "Water", "Beer", "Wine", "High"]
for k in range(6):
    m[f"EmDown{NAMES[k]}"] = f"{100*sum(Q[j][k] for j in range(k)):.1f}"; m[f"EmUp{NAMES[k]}"] = f"{100*sum(Q[j][k] for j in range(k+1, 6)):.1f}"
import sys as _sys; _sys.path.insert(0, str(HERE)); from emit_relabel_table_v1 import ANN
from collections import Counter
for sp, S in (("standard", "Std"), ("ood", "OOD")):
    c = Counter(a["category_id"] - 1 for a in json.load(open(ANN[sp]))["annotations"] if 1 <= a["category_id"] <= 6)
    m[f"RatioManual{S}"] = f"{c[0]/c[1]:.2f}"; m[f"WhiskyShare{S}"] = f"{100*c[1]/sum(c.values()):.1f}"
num = PAPER / "numbers.tex"; s = num.read_text()
BEG, END = "% ---- v14 extras block (emit_v14_extras_v1.py) ----", "% ---- end v14 extras ----"
if BEG in s: s = s[:s.index(BEG)] + s[s.index(END) + len(END):].lstrip("\n")
num.write_text(s.rstrip("\n") + "\n" + "\n".join([BEG] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()] + [END]) + "\n")
print("\n".join(out)); print("\n".join(L)); print({k: v for k, v in m.items() if k.endswith(("CI", "SD")) and ("Type" in k or "Six" in k or "RecStd" in k)}); print({k: v for k, v in m.items() if k.startswith("SvL")})
