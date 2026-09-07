#!/usr/bin/env python3
"""Independent check of what the paper attributes to the GlassNICOL release.

Re-derives, from the released artefacts rather than from our own notes:
  (a) whether the six glass heights and the 3 cm matching rule are in the dataset paper or only in its code,
  (b) how many base-point (key_point) annotations the release ships and in which annotation variant,
  (c) how many boxes each released annotation variant carries,
  (d) how large the hop/tolerance grid actually was, from the sealed preregistration and the stored logs.
Writes AUDIT_SOURCE_CLAIMS_V1.json.  Nothing here reuses the relabelling code."""
import json, re, subprocess
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
GLA = HERE.parents[3]                                    # .../gla
CODE = GLA / "<workspace>"
ANN_NO = GLA / "<workspace>"
ANN_WITH = GLA / "work/glassnicol_detection_v1/official_test_v1/annotations/coco_train_head_with_base_points.json"
PAPER_PDF = Path("WORKDIR")  # arXiv:2503.04308
out = {}

# (a) constants: released code vs paper text
src = (CODE / "process.py").read_text(); thr = (CODE / "utils/thresholding.py").read_text()
m = re.search(r"KNOWN_HEIGHTS\s*=\s*\[([^\]]*)\]", src)
out["code_known_heights"] = [float(x) for x in m.group(1).split(",")] if m else None
out["code_tolerance_cm"] = 3 if re.search(r"current\s*<\s*3\b", thr) else None
out["code_files"] = ["process.py", "utils/thresholding.py"]
if PAPER_PDF.exists():
    txt = subprocess.run(["pdftotext", str(PAPER_PDF), "-"], capture_output=True, text=True).stdout
    out["paper_cm_mentions"] = sorted(set(re.findall(r"\d+(?:\.\d+)?\s?cm", txt)))
    out["paper_states_heights"] = any(h in txt for h in ["17.5 cm", "17.5cm"])
    out["paper_provides_base_points"] = "We provide annotations for them in the dataset" in txt
else:
    out["paper_cm_mentions"] = "paper pdf not present at audit time"

# (b), (c) annotation variants
for tag, p in [("no_base_points", ANN_NO), ("with_base_points", ANN_WITH)]:
    d = json.load(open(p)); cats = {c["id"]: c["name"] for c in d["categories"]}
    cnt = Counter(cats[a["category_id"]] for a in d["annotations"])
    out[tag] = {"file": p.name, "images": len(d["images"]), "key_point": cnt.get("key_point", 0),
                "glass_boxes": sum(v for k, v in cnt.items() if k != "key_point"), "per_class": dict(sorted(cnt.items()))}
out["variant_box_difference"] = out["no_base_points"]["glass_boxes"] - out["with_base_points"]["glass_boxes"]

# (d) the grid actually searched
pre = (HERE / "PREREGISTRATION_SCENEVOTE_V1.md").read_text()
g = re.search(r"hops \{([^}]*)\} x tolerance \{([^}]*)\}", pre)
out["preregistered_grid"] = {"hops": g.group(1).split(","), "tolerances": g.group(2).split(",")} if g else None
logs = sorted(p.name for p in HERE.glob("SCENE_ASSOC_MANUAL_CHECK_V1_h*_t*.json") if "_iou" not in p.name)
out["grid_logs"] = {"count": len(logs), "files": logs}

json.dump(out, open(HERE / "AUDIT_SOURCE_CLAIMS_V1.json", "w"), indent=1)
for k in ["code_known_heights", "code_tolerance_cm", "paper_cm_mentions", "paper_states_heights",
          "paper_provides_base_points", "variant_box_difference", "preregistered_grid"]:
    print(f"{k}: {out.get(k)}")
print("key_point:", out["with_base_points"]["key_point"], "| boxes", out["no_base_points"]["glass_boxes"], "vs", out["with_base_points"]["glass_boxes"])
print("grid logs:", out["grid_logs"]["count"])
