#!/usr/bin/env python3
"""Mark the best and second-best entry of every metric column in the two result tables.

The tables were readable but flat: a reader had to scan eight columns to see who wins what.  This rewrites
the generated row files with the usual convention, best in bold and second in blue, applied per column and,
in the per-accelerator table, per accelerator block.  Nothing is recomputed here: the values come from the
row files the receipts already produced, so a mark can never disagree with the number it decorates.

Only metric columns are marked.  The "vs raw" columns of the comparison table are differences against a row
that is itself in the table, and the released-label row has no difference to compare, so marking them would
rank a quantity against its own baseline.

    emit_v18_highlight_rows.py            # writes *_hl_rows.tex beside the originals
"""
from __future__ import annotations
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
BUILD = PAPER / "build_v18_paper"
NUM = re.compile(r"-?\d+\.\d+|-?\d+")


def value(cell: str):
    """The number a cell ranks by: the mean before any $\\pm$, or None if the cell is not a number."""
    head = cell.split("$\\pm$")[0].strip()
    m = NUM.fullmatch(head.replace(",", ""))
    return float(m.group()) if m else None


def mark(cells, cols):
    """Bold the best and colour the second best of each column index in cols."""
    out = [list(r) for r in cells]
    for c in cols:
        vals = [(i, value(r[c])) for i, r in enumerate(cells)]
        vals = [(i, v) for i, v in vals if v is not None]
        if len(vals) < 2:
            continue
        order = sorted(vals, key=lambda t: -t[1])
        best = order[0][1]
        winners = [i for i, v in vals if v == best]
        seconds = [i for i, v in vals if v < best and v == max(x for _, x in vals if x < best)]
        for i in winners:
            out[i][c] = "\\textbf{%s}" % out[i][c]
        for i in seconds:
            out[i][c] = "\\second{%s}" % out[i][c]
    return out


def split_row(line):
    body = line.rstrip()
    assert body.endswith("\\\\"), body
    return [c.strip() for c in body[:-2].split(" & ")]


def join_row(cells):
    return " & ".join(cells) + " \\\\"


def do_arms():
    """Per-accelerator table: rank inside each block, over the eight metric columns.

    Rows of a block do not have the same field count: the first carries the \\multirow cell and the others
    start with a bare ampersand, so the metrics are addressed from the end, never by absolute index.
    """
    src = (BUILD / "arms_v13_rows.tex").read_text().splitlines()
    blocks, cur = [], []
    for line in src:
        if line.strip() == "\\midrule":
            blocks.append(cur); cur = []
        elif line.strip():
            cur.append(line)
    blocks.append(cur)
    out = []
    for bi, block in enumerate(blocks):
        rows = [split_row(l) for l in block]
        heads = [r[:-8] for r in rows]
        marked = mark([r[-8:] for r in rows], range(8))
        if bi:
            out.append("\\midrule")
        out += [join_row(h + m) for h, m in zip(heads, marked)]
    (BUILD / "arms_v18_hl_rows.tex").write_text("\n".join(out) + "\n")
    return out


def do_master():
    """Merged results table: mark the four metric columns of each split, not the two delta columns."""
    rows = [split_row(l) for l in (BUILD / "master_v18c_rows.tex").read_text().splitlines() if l.strip()]
    marked = mark(rows, [1, 3, 4, 5, 6, 8, 9, 10])
    out = [join_row(r) for r in marked]
    (BUILD / "master_v18c_hl_rows.tex").write_text("\n".join(out) + "\n")
    return out


def do_baselines():
    """Comparison table: rank the two AP columns; leave the differences alone."""
    rows = [split_row(l) for l in (BUILD / "baselines_v18_rows.tex").read_text().splitlines() if l.strip()]
    marked = mark(rows, [1, 3])
    out = [join_row(r) for r in marked]
    (BUILD / "baselines_v18_hl_rows.tex").write_text("\n".join(out) + "\n")
    return out


if __name__ == "__main__":
    for l in do_master():
        print(l)
