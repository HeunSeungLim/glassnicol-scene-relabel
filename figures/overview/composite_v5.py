#!/usr/bin/env python3
"""Final overview from the layout skeleton (skeleton.png): crop the stray bottom caption, detect the pale
magenta windows, paste the real panels, add exact Times labels above each stage and one sentence below."""
import cv2, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
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
NAVY, TEAL, GREEN, ORANGE = (28, 43, 94), (23, 128, 128), (46, 125, 50), (214, 118, 20)
for win, name in zip(left, ["pA1.png", "pB_top.png", "pA2.png"]): put(win, name, NAVY)
for win, name, col in zip(mid, ["pB_flow.png", "pC.png", "pD1.png"], [TEAL, GREEN, ORANGE]): put(win, name, col)
put(right[0], "pE2.png", NAVY)
fT = ImageFont.truetype(FONTB, 120); fS = ImageFont.truetype(FONT, 96); fI = ImageFont.truetype(FONTI, 96)
labels = [("Released views", "automatic type ids"), ("Table-plane registration", "H (f → f+k),  k ≤ 2"), ("Base-point association", "‖H b − b'‖ < 0.5 · width"), ("Scene chains", "union-find · most frequent type"), ("Detector", "relabelled ids → manual test")]
xs = [left[0][0] + left[0][2] // 2] + [c[0] + c[2] // 2 for c in mid] + [right[0][0] + right[0][2] // 2]
for (t, sub), cx in zip(labels, xs):
    d.text((cx, 20), t, font=fT, fill="black", anchor="ma"); d.text((cx, 150), sub, font=fS, fill=(40, 40, 40), anchor="ma")
bx0 = mid[0][0]; bx1 = right[0][0] + right[0][2]; by = TOP + mid[0][1] + mid[0][3] + 150
d.multiline_text(((bx0 + bx1) // 2, by), "Views of a static scene are linked through the table plane by their base points;\neach chain votes one type per glass, and only type ids change.", font=fI, fill="black", anchor="ma", align="center", spacing=20)
canvas.save(HERE / "figure_overview_v5.png"); canvas.save(HERE.parent / "figure_overview.png"); print("saved", canvas.size, "height at 7in:", round(7 * canvas.height / canvas.width, 2), "in")
