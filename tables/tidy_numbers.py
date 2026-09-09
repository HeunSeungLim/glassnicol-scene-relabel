import sys
#!/usr/bin/env python3
"""Canonical last step after any macro emitter: de-duplicate numbers.tex (the last
definition of a macro wins), then place every macro main.tex prints above the
SUPERSEDED banner and everything else below it."""
import re
from pathlib import Path
P = Path(__file__).resolve().parents[1]
# In the workspace the manuscript sits beside numbers.tex; in this repository both live under paper/.
_default = "main.tex" if (P / "main.tex").is_file() else "paper/main_v18.tex"
n = (P / "numbers.tex") if (P / "numbers.tex").is_file() else (P / "paper" / "numbers.tex")
tex = (P / (sys.argv[1] if len(sys.argv) > 1 else _default)).read_text()
used = set(re.findall(r'\\([A-Za-z]+)(?=[^A-Za-z]|$)', tex))
last = {}
for l in n.read_text().splitlines():
    m = re.match(r'\\newcommand\{\\([A-Za-z]+)\}', l)
    if m: last[m.group(1)] = l          # later definitions override earlier ones
live = [last[k] for k in sorted(last) if k in used]
dead = [last[k] for k in sorted(last) if k not in used]
n.write_text("% ===== macros printed by main.tex. Each value is emitted by a script from a JSON receipt; "
             "see the emitters in relabel_v1/ and scenes_v1/. Re-run relabel_v1/tidy_numbers.py after any emitter. =====\n"
             + "\n".join(live) + "\n\n% SUPERSEDED. Nothing below is used by main.tex. Kept for provenance of earlier drafts only; "
             "recompute before citing any of it.\n" + "\n".join(dead) + "\n")
dups = sum(1 for _ in n.read_text().splitlines()) 
print(f"numbers.tex: {len(live)} live, {len(dead)} superseded, duplicates removed")
