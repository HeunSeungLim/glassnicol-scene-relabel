#!/usr/bin/env python3
"""Final overview figure: read the stage skeleton, crop the stray bottom caption, detect the pale
magenta windows, paste the real panels, add exact Times labels above each stage and one sentence below."""
import cv2, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def _place(im, width=2800):
    """Return the figure as the paper places it: 400 dpi at \\textwidth, with the blank bands removed.

    The full-resolution composite is 904 dpi, which put 8 MB of pixels no printer uses into the PDF.
    Doing the reduction here rather than by hand keeps the shipped figure reproducible from this file.
    """
    import numpy as _n
    a = _n.array(im.convert("L")) < 245
    rows = _n.where(a.any(1))[0]
    blank, s, drop = ~a.any(1), None, set()
    for i, b in enumerate(blank):
        if b and s is None:
            s = i
        elif not b and s is not None:
            if i - s > 12:
                drop |= set(range(s + 4, i - 4))
            s = None
    drop |= set(range(0, rows.min())) | set(range(rows.max() + 1, im.height))
    keep = [i for i in range(im.height) if i not in drop]
    im = Image.fromarray(_n.array(im.convert("RGB"))[keep])
    return im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
HERE = Path(__file__).resolve().parent
FONT = "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf"; FONTB = "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-bold.otf"; FONTI = "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-italic.otf"
im = cv2.imread(str(HERE / "skeleton.png")); Hh, W = im.shape[:2]
# find the stray text band: rows below the diagram with dark pixels
dark = (im.min(2) < 100)
rows = dark.sum(1); diag_bottom = max(y for y in range(Hh) if rows[y] > 50 and y < 0.9 * Hh)
crop_h = int(diag_bottom + 0.02 * Hh); im = im[:crop_h]
b, g, r = cv2.split(im); mask = ((r > 225) & (g > 170) & (g < 225) & (b > 225)).astype(np.uint8)
n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
comps = [(x, y, w, h) for i, (x, y, w, h, a) in enumerate(stats) if i > 0 and a > 20000]
left = sorted([c for c in comps if c[0] < 0.15 * W], key=lambda c: c[1]); right = [c for c in comps if c[0] > 0.78 * W]; mid = sorted([c for c in comps if 0.15 * W <= c[0] <= 0.78 * W], key=lambda c: c[0])
rx0 = min(c[0] for c in right); ry0 = min(c[1] for c in right); rx1 = max(c[0] + c[2] for c in right); ry1 = max(c[1] + c[3] for c in right); right = [(rx0, ry0, rx1 - rx0, ry1 - ry0)]
print("windows: left", len(left), "mid", len(mid), "right", len(right)); assert len(left) == 3 and len(mid) == 3
TOP = 260  # extra white band for labels
canvas = Image.new("RGB", (W, crop_h + TOP), "white"); canvas.paste(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), (0, TOP)); d = ImageDraw.Draw(canvas)
def put(win, name, border):
    x, y, w, h = win; y += TOP; p = Image.open(HERE / name).convert("RGB"); ar = p.width / p.height
    nw, nh = (w, int(round(w / ar))) if w / h <= ar else (int(round(h * ar)), h)
    d.rectangle([x - 3, y - 3, x + w + 2, y + h + 2], fill="white"); ox, oy = x + (w - nw) // 2, y + (h - nh) // 2
    canvas.paste(p.resize((nw, nh), Image.LANCZOS), (ox, oy)); d.rectangle([ox, oy, ox + nw - 1, oy + nh - 1], outline=border, width=6)
    print(f"  {name:12s} {w}x{h} -> {nw}x{nh}  print width {7*nw/W:.2f} in")
    return ox, oy, nw, nh
NAVY, TEAL, GREEN, ORANGE = (28, 43, 94), (23, 128, 128), (46, 125, 50), (214, 118, 20)
fV = ImageFont.truetype(FONT, 113)
for win, name, tag in zip(left, ["pA1.png", "pB_top.png", "pA2.png"], ["view 6", "view 8", "view 10"]):
    ox, oy, nw, nh = put(win, name, NAVY)
    tw = d.textlength(tag, font=fV); tx1 = ox + nw - 8
    d.rectangle([tx1 - tw - 18, oy + 8, tx1, oy + 128], fill="white")
    d.text((tx1 - 9, oy + 14), tag, font=fV, fill="black", anchor="ra")
for win, name, col in zip(mid, ["pB_flow.png", "pC.png", "pD1.png"], [TEAL, GREEN, ORANGE]): put(win, name, col)
put(right[0], "pE2.png", NAVY)
fT = ImageFont.truetype(FONTB, 120); fS = ImageFont.truetype(FONT, 113); fI = ImageFont.truetype(FONTI, 113)
# the stage subtitles print at 9pt, so they are kept short enough not to run into each other
labels = [("Released views", "automatic type ids"), ("Table-plane registration", "H (f → f+k),  k ≤ 2"), ("Base-point association", "‖H b − b'‖ < 0.5 w"), ("Scene chains", "union-find, plurality"), ("Detector", "relabelled ids")]
xs = [left[0][0] + left[0][2] // 2] + [c[0] + c[2] // 2 for c in mid] + [right[0][0] + right[0][2] // 2]
for (t, sub), cx in zip(labels, xs):
    d.text((cx, 20), t, font=fT, fill="black", anchor="ma"); d.text((cx, 150), sub, font=fS, fill=(40, 40, 40), anchor="ma")
bx0 = mid[0][0]; bx1 = right[0][0] + right[0][2]; by = TOP + mid[0][1] + mid[0][3] + 150
# the italic line the skeleton carried here repeated the caption; dropped to give the page its space back
# squeeze the empty band the skeleton leaves between the stage labels and the panels: it is dead print area
import numpy as _np
_a = _np.array(canvas.convert("L")) < 245; _empty = ~_a.any(1); _H = canvas.height
_runs, _st = [], None
for _y in range(_H):
    if _empty[_y] and _st is None: _st = _y
    if not _empty[_y] and _st is not None:
        _runs.append((_st, _y)); _st = None
if _st is not None: _runs.append((_st, _H))
KEEP = 60                                        # px of white left between the labels and the panels
for _y0, _y1 in _runs:
    if _y0 > 0 and _y1 < _H and _y1 - _y0 > KEEP:
        top = canvas.crop((0, 0, canvas.width, _y0 + KEEP)); bot = canvas.crop((0, _y1, canvas.width, _H))
        out = Image.new("RGB", (canvas.width, top.height + bot.height), "white")
        out.paste(top, (0, 0)); out.paste(bot, (0, top.height)); canvas = out
        print(f"  squeezed {_y1 - _y0 - KEEP} px of empty band at y={_y0}")
        break
_place(canvas).save(HERE / "figure_overview_v5.png"); _place(canvas).save(HERE.parent / "figure_overview.png"); print("saved", canvas.size, "height at 7in:", round(7 * canvas.height / canvas.width, 2), "in")
