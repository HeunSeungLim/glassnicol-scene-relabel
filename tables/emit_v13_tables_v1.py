#!/usr/bin/env python3
"""V13 table rows: grouped by accelerator with \\multirow, rules between groups, pairs as one row per pair.
Reads the V12 receipts (SCENEVOTE_VERDICT_V1.json cells order, arms/cells/pair rows) and rewrites the layout only."""
import json, re
from pathlib import Path
HERE = Path(__file__).resolve().parent; PAPER = HERE.parent
# arms: group rows by accelerator
ACCEL = {"H200": "A", "RTX": "B"}   # the paper names the accelerators A and B (Section 3.1)
rows = [l for l in (PAPER / "arms_v12_rows.tex").read_text().splitlines() if l.strip() and not l.startswith("\\addlinespace")]
rows = [re.sub(r"^(H200|RTX)", lambda m: ACCEL[m.group(1)], l) for l in rows]   # print A/B, not machine names
groups = {}
for l in rows:
    acc, rest = l.split(" & ", 1); groups.setdefault(acc, []).append(rest)
out = []
for gi, (acc, rs) in enumerate(groups.items()):
    if gi: out.append("\\midrule")
    for k, r in enumerate(rs):
        out.append((f"\\multirow{{{len(rs)}}}{{*}}{{{acc}}} & " if k == 0 else "& ") + r)
(PAPER / "arms_v13_rows.tex").write_text("\n".join(out) + "\n")
# cells: group by accelerator
rows = [l for l in (PAPER / "cells_v12_rows.tex").read_text().splitlines() if l.strip()]
groups = {}
for l in rows:
    acc, rest = l.split(" & ", 1); groups.setdefault(acc, []).append(rest)
out = []
for gi, (acc, rs) in enumerate(groups.items()):
    if gi: out.append("\\midrule")
    for k, r in enumerate(rs):
        out.append((f"\\multirow{{{len(rs)}}}{{*}}{{{acc}}} & " if k == 0 else "& ") + r)
(PAPER / "cells_v13_rows.tex").write_text("\n".join(out) + "\n")
# pairs: one row per pair: Pair & Std mean & Std positive & OOD mean & OOD positive (pooled over all cells)
rows = [l for l in (PAPER / "pair_v12_rows.tex").read_text().splitlines() if l.strip()]
by = {}
for l in rows:
    split, pair, h, r, allv, pos = [x.strip() for x in l.rstrip(" \\").split("&")]
    by.setdefault(pair, {})[split] = (allv, pos)
out = [f"{pair} & {v['Std.'][0]} & {v['Std.'][1]} & {v['OOD'][0]} & {v['OOD'][1]} \\\\" for pair, v in by.items()]
(PAPER / "pairs_v13_rows.tex").write_text("\n".join(out) + "\n")
print("\n".join((PAPER / "arms_v13_rows.tex").read_text().splitlines()[:4])); print("\n".join(out))
