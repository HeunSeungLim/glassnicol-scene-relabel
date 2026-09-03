#!/usr/bin/env python3
"""audit_scenevote_v1.py -- scene-vote 실험 측정 파이프라인 독립 감사.

파이프라인 스크립트(emit_v12_arms_v1.py, emit_relabel_table_v1.py, bootstrap_typeacc_frames_v1.py,
emit_v11_pairs_v1.py)의 코드를 가져오거나(import) 베끼지 않고, 그 안에 적힌 정의만 따라 처음부터 다시 구현한 뒤
파이프라인이 낸 숫자(SCENEVOTE_VERDICT_V1.json, arms_v12_rows.tex, cells_v12_rows.tex, numbers.tex의 Sv* 매크로,
BOOT_TYPEACC_FRAMES_*.json)와 대조한다.  기존 파일은 수정하지 않고 AUDIT_SCENEVOTE_V1.json만 새로 쓴다.

재구현한 정의
  1. six-class AP / generic AP : pycocotools COCO AP@[.50:.95], 1-based GT.  generic은 GT·예측의 모든 범주를 1로 합침.
  2. localisation recall       : 프레임별로 GT 박스를 annotation id 오름차순으로 돌며, 점수>=0.25이고 아직 쓰이지 않은
                                 예측 가운데 IoU가 가장 큰 것을 붙인다(탐욕). IoU>=0.5면 localised.  recall = localised / 전체 GT 박스.
  3. 두 arm (a,b)의 type accuracy: 두 arm 모두 localised한 GT 박스에서, 붙은 예측의 범주가 GT 범주와 같은 비율.  차이는 b-a (points).
                                 3-arm 표는 세 arm 모두 localised한 박스를 쓴다.
  4. 셀별 gain(scenevote-transparent, scenevote-relabel), 셀 평균, gain>0 셀 수.
  5. 프레임 단위 paired bootstrap: GT 이미지 N개를 복원추출(각 자리마다 random.Random(20260902).choice), 2,000회,
                                 뽑힌 프레임의 both-localised 박스로 gain 재계산, numpy.percentile 2.5 / 97.5.
"""
from __future__ import annotations
import contextlib, datetime, io, json, random, re, statistics
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HERE = Path(__file__).resolve().parent            # relabel_v1
PAPER = HERE.parent                                # paper_work_v4
OFF = PAPER.parent / "official_test_v1"
EV = PAPER / "scenes_v1" / "evals_official"
GT_FILES = {"standard": OFF / "standard_types_remap_1based.json", "ood": OFF / "ood_types_remap_1based.json"}
SPLITS = ("standard", "ood")
ARMS = ("transparent", "relabel", "scenevote")
ARM_LABEL = {"transparent": "raw", "relabel": "tracklet vote", "scenevote": "scene vote"}
HOST_LABEL = {"scene": "H200", "scene136": "RTX"}
CELLS = [("scene", 1337), ("scene", 3407), ("scene", 4567), ("scene136", 1337), ("scene136", 5678)]
PAIRS = [("transparent", "scenevote", "VsRaw"), ("relabel", "scenevote", "VsTrk"), ("transparent", "relabel", "TrkVsRaw")]
METRICS = ("six", "gen", "recall", "type")
SCORE_MIN, IOU_MIN = 0.25, 0.5
BOOT_DRAWS, BOOT_SEED = 2000, 20260902
TOL = 0.05 + 1e-9                                  # 0-100 스케일에서 0.05 point 초과만 불일치로 본다
OUT = HERE / "AUDIT_SCENEVOTE_V1.json"

cell_name = lambda host, seed: f"{host}_s{seed}"
run_name = lambda host, arm, seed: f"{host}_{arm}_s{seed}"


# ----------------------------------------------------------------------------- GT
def load_gt(split):
    g = json.loads(GT_FILES[split].read_text())
    boxes = [a for a in g["annotations"] if 1 <= a["category_id"] <= 6]
    return {"raw": g, "images": g["images"], "image_ids": [im["id"] for im in g["images"]], "boxes": boxes}


def check_gt(gt):
    """점검 (b): annotation id 0 없음, 범주 id는 1..6만.  덤으로 id 고유성과 박스 없는 이미지 수."""
    anns = gt["raw"]["annotations"]
    ann_ids = [a["id"] for a in anns]; cats = sorted({a["category_id"] for a in anns})
    img_ids = gt["image_ids"]; with_box = {a["image_id"] for a in gt["boxes"]}
    return {"n_images": len(img_ids), "n_annotations": len(anns), "n_boxes_cat_1_to_6": len(gt["boxes"]),
            "annotation_id_zero_present": 0 in ann_ids, "annotation_id_min": min(ann_ids), "annotation_id_max": max(ann_ids),
            "annotation_ids_unique": len(set(ann_ids)) == len(ann_ids),
            "category_ids_present": cats, "category_ids_only_1_to_6": set(cats) <= set(range(1, 7)),
            "image_ids_unique": len(set(img_ids)) == len(img_ids), "image_id_min": min(img_ids), "image_id_max": max(img_ids),
            "images_without_boxes": sorted(set(img_ids) - with_box),
            "ok": (0 not in ann_ids) and set(cats) <= set(range(1, 7))}


# ----------------------------------------------------------------------------- AP (정의 1)
def coco_ap(gt, preds, generic):
    remap = (lambda c: 1) if generic else (lambda c: c)
    ds = {"images": [dict(im) for im in gt["images"]],
          "annotations": [dict(a, category_id=remap(a["category_id"])) for a in gt["boxes"]],
          "categories": [{"id": 1, "name": "glass"}] if generic else [{"id": c, "name": f"type{c}"} for c in range(1, 7)]}
    dets = [{"image_id": p["image_id"], "category_id": remap(p["category_id"]), "bbox": list(p["bbox"]), "score": float(p["score"])}
            for p in preds]
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO(); cg.dataset = ds; cg.createIndex()
        cd = cg.loadRes(dets)
        ev = COCOeval(cg, cd, "bbox")
        ev.params.imgIds = list(gt["image_ids"])       # GT 이미지 id 그대로 (pycocotools가 내부에서 unique 처리)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return 100.0 * float(ev.stats[0]), 100.0 * float(ev.stats[1])


# ----------------------------------------------------------------------------- localisation (정의 2)
def box_iou_xywh(a, b):
    ax0, ay0, aw, ah = a; bx0, by0, bw, bh = b
    iw = min(ax0 + aw, bx0 + bw) - max(ax0, bx0); ih = min(ay0 + ah, by0 + bh) - max(ay0, by0)
    if iw <= 0 or ih <= 0: return 0.0
    inter = iw * ih; union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def localise(gt, preds):
    """GT 박스마다 (image_id, ann_id, gt_cat, localised, pred_cat, iou) 를 고정 순서(이미지 id, ann id 오름차순)로 돌려준다."""
    pred_by_img, gt_by_img = {}, {}
    for p in preds:
        if p["score"] >= SCORE_MIN: pred_by_img.setdefault(p["image_id"], []).append(p)
    for a in gt["boxes"]: gt_by_img.setdefault(a["image_id"], []).append(a)
    rows = []; diag = {"ties_at_best_iou": 0, "ties_with_different_category": 0, "best_iou_within_1e-7_of_0.5": 0}
    for img in sorted(gt_by_img):
        cand = pred_by_img.get(img, []); free = [True] * len(cand)
        for a in sorted(gt_by_img[img], key=lambda z: z["id"]):
            ious = [box_iou_xywh(a["bbox"], p["bbox"]) if free[i] else -1.0 for i, p in enumerate(cand)]
            best = max(ious) if ious else -1.0
            if best >= IOU_MIN:
                j = ious.index(best); free[j] = False
                tied = [i for i, v in enumerate(ious) if v == best]
                if len(tied) > 1:
                    diag["ties_at_best_iou"] += 1
                    if len({cand[i]["category_id"] for i in tied}) > 1: diag["ties_with_different_category"] += 1
                rows.append((img, a["id"], a["category_id"], 1, cand[j]["category_id"], best))
            else:
                rows.append((img, a["id"], a["category_id"], 0, None, max(best, 0.0)))
            if abs(best - IOU_MIN) < 1e-7: diag["best_iou_within_1e-7_of_0.5"] += 1
    return rows, diag


def type_acc(rows, idx):
    return 100.0 * statistics.mean(int(rows[i][4] == rows[i][2]) for i in idx) if idx else float("nan")


def recall(rows):
    return 100.0 * statistics.mean(r[3] for r in rows)


# ----------------------------------------------------------------------------- bootstrap (정의 5)
def frame_bootstrap(rows_a, rows_b):
    """both-localised 박스의 (a맞음, b맞음) 를 프레임별로 모아 프레임 복원추출.  반환: point, ci95, p(gain<=0), n."""
    per_frame = {}; frames = sorted({r[0] for r in rows_a})
    for ra, rb in zip(rows_a, rows_b):
        if ra[3] and rb[3]: per_frame.setdefault(ra[0], []).append((int(ra[4] == ra[2]), int(rb[4] == rb[2])))
    allb = [d for f in frames for d in per_frame.get(f, [])]
    point = 100.0 * (statistics.mean(b for _, b in allb) - statistics.mean(a for a, _ in allb))
    rng = random.Random(BOOT_SEED); draws = []
    for _ in range(BOOT_DRAWS):
        picked = [rng.choice(frames) for _ in frames]
        s = [d for f in picked for d in per_frame.get(f, [])]
        if s: draws.append(100.0 * (sum(b for _, b in s) - sum(a for a, _ in s)) / len(s))
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return {"point_pp": point, "ci95_frame": [lo, hi], "p_le_zero_frame": sum(x <= 0 for x in draws) / len(draws),
            "n_frames": len(frames), "n_frames_with_both": len(per_frame), "n_boxes_both": len(allb), "draws": len(draws), "seed": BOOT_SEED}


# ----------------------------------------------------------------------------- 파이프라인 산출물 읽기
def parse_signed(s):
    s = s.strip().replace("$-$", "-").replace("$+$", "+")
    return float(s)


def read_arms_tex():
    p = PAPER / "arms_v12_rows.tex"
    if not p.exists(): return None
    out = {}
    for line in p.read_text().splitlines():
        if "&" not in line: continue
        f = [x.strip() for x in line.replace("\\\\", "").split("&")]
        host_lab, n = re.match(r"(\S+)\s*\((\d+)\)", f[0]).groups(); arm_lab = f[1]
        vals = []
        for x in f[2:]:
            if "$\\pm$" in x: mu, sd = x.split("$\\pm$"); vals.append((float(mu), float(sd)))
            else: vals.append((float(x), None))
        out[(host_lab, arm_lab)] = {"n_seeds": int(n), "values": vals}
    return out


def read_cells_tex():
    p = PAPER / "cells_v12_rows.tex"
    if not p.exists(): return None
    out = {}
    for line in p.read_text().splitlines():
        if "&" not in line: continue
        f = [x.strip() for x in line.replace("\\\\", "").split("&")]
        out[(f[0], int(f[1]))] = [parse_signed(x) for x in f[2:]]
    return out


def read_sv_macros():
    p = PAPER / "numbers.tex"
    if not p.exists(): return None
    s = p.read_text()
    beg, end = "% ---- v12 scene-vote block", "% ---- end v12 block ----"
    # tidy_numbers.py 가 돌면 블록 표식이 사라지고 매크로가 파일 전체에 흩어진다(마지막 정의가 유효). 둘 다 지원.
    block = s[s.index(beg):s.index(end)] if beg in s and end in s else s
    out = {}
    for k, v in re.findall(r"\\newcommand\{\\(Sv\w+)\}\{([^}]*)\}", block):
        out[k] = int(v) if re.fullmatch(r"-?\d+", v) else parse_signed(v)
    return out or None


def read_json(p):
    return json.loads(p.read_text()) if p.exists() else None


# ----------------------------------------------------------------------------- main
def main():
    gt = {sp: load_gt(sp) for sp in SPLITS}
    gt_checks = {sp: check_gt(gt[sp]) for sp in SPLITS}
    discrepancies, matches = [], {}

    def note(group, where, pipeline, audit, tol=TOL, exact=False):
        """비교 하나를 기록. 불일치면 discrepancies에 추가."""
        m = matches.setdefault(group, {"compared": 0, "matched": 0, "max_abs_diff": 0.0})
        m["compared"] += 1
        if pipeline is None or audit is None:
            discrepancies.append({"group": group, "where": where, "pipeline": pipeline, "audit": audit, "diff": None, "reason": "값 없음"}); return
        d = abs(float(pipeline) - float(audit)); m["max_abs_diff"] = max(m["max_abs_diff"], d)
        ok = (pipeline == audit) if exact else (d <= tol)
        if ok: m["matched"] += 1
        else: discrepancies.append({"group": group, "where": where, "pipeline": pipeline, "audit": audit, "diff": round(float(audit) - float(pipeline), 4)})

    # ---- run 단위: AP 재계산, localisation, 이미지 id 점검 (a)
    runs, run_report = {}, {}
    for host, seed in CELLS:
        for arm in ARMS:
            for sp in SPLITS:
                d = EV / f"{sp}_{run_name(host, arm, seed)}"
                rf, pf = d / "RESULT.json", d / "predictions_six_class.json"
                if not (rf.exists() and pf.exists()):
                    run_report[f"{sp}_{run_name(host, arm, seed)}"] = {"status": "missing", "dir": str(d)}; continue
                res = json.loads(rf.read_text()); preds = json.loads(pf.read_text())
                six_mine, six50 = coco_ap(gt[sp], preds, False); gen_mine, gen50 = coco_ap(gt[sp], preds, True)
                six_pipe = 100.0 * res["six_class"]["ap50_95"]; gen_pipe = 100.0 * res["collapsed_general"]["ap50_95"]
                rows, diag = localise(gt[sp], preds)
                pimg = {p["image_id"] for p in preds}; gimg = set(gt[sp]["image_ids"])
                pimg25 = {p["image_id"] for p in preds if p["score"] >= SCORE_MIN}
                runs[(host, seed, sp, arm)] = {"six": six_mine, "gen": gen_mine, "rows": rows, "recall": recall(rows)}
                run_report[f"{sp}_{run_name(host, arm, seed)}"] = {
                    "status": "ok", "n_predictions": len(preds), "n_predictions_score_ge_0.25": sum(p["score"] >= SCORE_MIN for p in preds),
                    "six_ap50_95_audit": round(six_mine, 4), "six_ap50_95_result_json": round(six_pipe, 4),
                    "generic_ap50_95_audit": round(gen_mine, 4), "generic_ap50_95_result_json": round(gen_pipe, 4),
                    "six_ap50_audit": round(six50, 4), "generic_ap50_audit": round(gen50, 4),
                    "recall_audit": round(recall(rows), 4), "n_gt_boxes": len(rows), "n_localised": sum(r[3] for r in rows),
                    "image_ids_missing_from_predictions": sorted(gimg - pimg), "image_ids_not_in_gt": sorted(pimg - gimg),
                    "image_ids_without_prediction_ge_0.25": sorted(gimg - pimg25), "matching_diagnostics": diag}
                note("AP_six_vs_RESULT.json", f"{sp}_{run_name(host, arm, seed)}", six_pipe, six_mine)
                note("AP_generic_vs_RESULT.json", f"{sp}_{run_name(host, arm, seed)}", gen_pipe, gen_mine)
    # 같은 셀/split의 arm들이 같은 GT 박스 순서를 갖는지, 같은 이미지 집합을 보는지
    for host, seed in CELLS:
        for sp in SPLITS:
            have = [arm for arm in ARMS if (host, seed, sp, arm) in runs]
            for arm in have[1:]:
                assert [r[:3] for r in runs[(host, seed, sp, arm)]["rows"]] == [r[:3] for r in runs[(host, seed, sp, have[0])]["rows"]]

    # ---- 셀 구성
    cells_avail = {cell_name(h, s): {sp: [arm for arm in ARMS if (h, s, sp, arm) in runs] for sp in SPLITS} for h, s in CELLS}
    cells_all3 = [(h, s) for h, s in CELLS if all(len(cells_avail[cell_name(h, s)][sp]) == 3 for sp in SPLITS)]

    # ---- 정의 3/4: 쌍별 both-localised type accuracy, gain
    pair_cells = {}      # (tag, sp) -> {cell -> {metrics diff, type_a, type_b, n_both}}
    for a, b, tag in PAIRS:
        for sp in SPLITS:
            for h, s in CELLS:
                if (h, s, sp, a) not in runs or (h, s, sp, b) not in runs: continue
                ra, rb = runs[(h, s, sp, a)], runs[(h, s, sp, b)]
                both = [i for i in range(len(ra["rows"])) if ra["rows"][i][3] and rb["rows"][i][3]]
                ta, tb = type_acc(ra["rows"], both), type_acc(rb["rows"], both)
                pair_cells.setdefault((tag, sp), {})[cell_name(h, s)] = {
                    "six": rb["six"] - ra["six"], "gen": rb["gen"] - ra["gen"], "recall": rb["recall"] - ra["recall"], "type": tb - ta,
                    "type_a": ta, "type_b": tb, "n_both": len(both), "n_gt": len(ra["rows"])}

    # ---- 3-arm 표: 세 arm 모두 localised한 박스의 type accuracy, host별 mean +- SD (표본 SD)
    three_arm = {}
    for h, s in cells_all3:
        for sp in SPLITS:
            R = {arm: runs[(h, s, sp, arm)] for arm in ARMS}
            all3 = [i for i in range(len(R["transparent"]["rows"])) if all(R[arm]["rows"][i][3] for arm in ARMS)]
            three_arm[(cell_name(h, s), sp)] = {"n_all3": len(all3), **{arm: {
                "six": R[arm]["six"], "gen": R[arm]["gen"], "recall": R[arm]["recall"], "type": type_acc(R[arm]["rows"], all3)} for arm in ARMS}}
    host_table = {}
    for host in ("scene", "scene136"):
        seeds = [s for h, s in cells_all3 if h == host]
        if not seeds: continue
        for arm in ARMS:
            vals = {}
            for sp in SPLITS:
                for k in METRICS:
                    v = [three_arm[(cell_name(host, s), sp)][arm][k] for s in seeds]
                    vals[f"{sp}_{k}"] = {"mean": statistics.mean(v), "sd": statistics.stdev(v) if len(v) > 1 else None, "n": len(v)}
            host_table[f"{host}/{arm}"] = {"seeds": seeds, **vals}

    # ---- pooled (쌍이 있는 셀 전체, 그리고 파이프라인처럼 세 arm이 다 있는 셀만)
    pooled = {}
    for a, b, tag in PAIRS:
        for sp in SPLITS:
            pc = pair_cells.get((tag, sp), {})
            for scope, names in (("cells_with_pair", list(pc)), ("cells_all_three_arms", [cell_name(h, s) for h, s in cells_all3 if cell_name(h, s) in pc])):
                for k in METRICS:
                    g = [pc[c][k] for c in names]
                    pooled[f"{tag}_{sp}_{scope}_{k}"] = {"cells": names, "values": [round(x, 4) for x in g],
                                                          "mean": statistics.mean(g) if g else None, "n_positive": sum(x > 0 for x in g), "n_cells": len(g)}

    # ---- 정의 5: 프레임 bootstrap
    boot = {}
    for a, b, tag in PAIRS:
        for h, s in CELLS:
            for sp in SPLITS:
                if (h, s, sp, a) in runs and (h, s, sp, b) in runs:
                    boot[f"{a}_{b}"] = boot.get(f"{a}_{b}", {})
                    boot[f"{a}_{b}"][f"{cell_name(h, s)}_{sp}"] = frame_bootstrap(runs[(h, s, sp, a)]["rows"], runs[(h, s, sp, b)]["rows"])

    # ---- 파이프라인 산출물과 대조
    verdict = read_json(HERE / "SCENEVOTE_VERDICT_V1.json")
    verdict_check = None
    if verdict:
        vcells = verdict["cells"]
        for a, b, tag in PAIRS:
            for sp in SPLITS:
                for k in METRICS:
                    plist = verdict["contrasts"].get(f"{tag}_{sp}", {}).get(k)
                    if plist is None: continue
                    for c, pv in zip(vcells, plist):
                        mine = pair_cells.get((tag, sp), {}).get(c, {}).get(k)
                        note(f"VERDICT_contrasts_{k}", f"{tag}_{sp}[{c}]", pv, mine)
        # 예측 P1-P4 를 같은 셀 집합으로 다시 판정
        def mean_of(tag, sp, k): return statistics.mean(pair_cells[(tag, sp)][c][k] for c in vcells)
        P1 = mean_of("VsRaw", "standard", "type") > 0 and mean_of("VsRaw", "ood", "type") > 0
        P2 = mean_of("VsRaw", "ood", "type") > mean_of("TrkVsRaw", "ood", "type")
        P3 = all(abs(mean_of("VsRaw", sp, k)) < 2 for sp in SPLITS for k in ("recall", "gen"))
        P4 = sum(pair_cells[("VsRaw", "ood")][c]["type"] > 0 for c in vcells) >= 4
        verdict_check = {"cells_used": vcells, "P1_audit": P1, "P2_audit": P2, "P3_audit": P3, "P4_audit": P4,
                         "P1_pipeline": verdict["P1_mean_type_gain_vs_raw_positive_both_splits"], "P2_pipeline": verdict["P2_ood_type_gain_vs_raw_exceeds_tracklet"],
                         "P3_pipeline": verdict["P3_recall_and_generic_within_2"], "P4_pipeline": verdict["P4_ood_positive_cells_ge4"]}
        for i in range(1, 5):
            note("VERDICT_P1-P4", f"P{i}", int(verdict_check[f"P{i}_pipeline"]), int(verdict_check[f"P{i}_audit"]), exact=True)

    cells_tex = read_cells_tex()
    if cells_tex:
        for (host_lab, seed), vals in cells_tex.items():
            host = {v: k for k, v in HOST_LABEL.items()}[host_lab]; c = cell_name(host, seed)
            for j, (tag, sp) in enumerate((("VsRaw", "standard"), ("VsTrk", "standard"), ("VsRaw", "ood"), ("VsTrk", "ood"))):
                note("cells_v12_rows.tex_type_gain", f"{c} {sp} {tag}", vals[j], pair_cells.get((tag, sp), {}).get(c, {}).get("type"))

    arms_tex = read_arms_tex()
    if arms_tex:
        lab2arm = {v: k for k, v in ARM_LABEL.items()}; lab2host = {v: k for k, v in HOST_LABEL.items()}
        for (host_lab, arm_lab), row in arms_tex.items():
            key = f"{lab2host[host_lab]}/{lab2arm[arm_lab]}"; ht = host_table.get(key)
            note("arms_v12_rows.tex_n_seeds", key, row["n_seeds"], len(ht["seeds"]) if ht else None, exact=True)
            for j, (sp, k) in enumerate([(sp, k) for sp in SPLITS for k in METRICS]):
                mu, sd = row["values"][j]; mine = ht[f"{sp}_{k}"] if ht else None
                note("arms_v12_rows.tex_mean", f"{key} {sp} {k}", mu, mine["mean"] if mine else None)
                if sd is not None: note("arms_v12_rows.tex_sd", f"{key} {sp} {k}", sd, mine["sd"] if mine else None)

    macros = read_sv_macros()
    if macros:
        note("numbers.tex_SvCells", "SvCells", macros.get("SvCells"), len(cells_all3), exact=True)
        for a, b, tag in PAIRS:
            for sp, S in (("standard", "Std"), ("ood", "OOD")):
                for k, K in (("six", "Six"), ("gen", "Gen"), ("recall", "Rec"), ("type", "Type")):
                    p = pooled[f"{tag}_{sp}_cells_all_three_arms_{k}"]
                    note("numbers.tex_Sv_Gain", f"Sv{K}{S}{tag}Gain", macros.get(f"Sv{K}{S}{tag}Gain"), p["mean"])
                    note("numbers.tex_Sv_Pos", f"Sv{K}{S}{tag}Pos", macros.get(f"Sv{K}{S}{tag}Pos"), p["n_positive"], exact=True)

    boot_files = {"transparent_scenevote": HERE / "BOOT_TYPEACC_FRAMES_transparent_scenevote.json",
                  "relabel_scenevote": HERE / "BOOT_TYPEACC_FRAMES_relabel_scenevote.json",
                  "transparent_relabel": HERE / "BOOT_TYPEACC_FRAMES_V1.json"}
    boot_compared = {}
    for pair, p in boot_files.items():
        j = read_json(p); boot_compared[pair] = {"file": str(p), "present": j is not None}
        if j is None: continue
        for c, v in j.get("cells", {}).items():
            mine = boot.get(pair, {}).get(c)
            note(f"BOOT_{pair}_point", c, v["point_pp"], mine["point_pp"] if mine else None)
            note(f"BOOT_{pair}_ci_lo", c, v["ci95_frame"][0], mine["ci95_frame"][0] if mine else None)
            note(f"BOOT_{pair}_ci_hi", c, v["ci95_frame"][1], mine["ci95_frame"][1] if mine else None)
            note(f"BOOT_{pair}_n_boxes_both", c, v["n_boxes_both"], mine["n_boxes_both"] if mine else None, exact=True)
            note(f"BOOT_{pair}_p_le_zero", c, v["p_le_zero_frame"], mine["p_le_zero_frame"] if mine else None, tol=0.001)

    # ---- seed 5678 scenevote 상태
    s5678 = {"FINAL_RESULT.json": (PAPER / "seed_v1/runs/scene136_scenevote_s5678/FINAL_RESULT.json").exists(),
             "ood_RESULT.json": (EV / "ood_scene136_scenevote_s5678/RESULT.json").exists(),
             "standard_RESULT.json": (EV / "standard_scene136_scenevote_s5678/RESULT.json").exists()}

    # ---- 출력
    r4 = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)
    out = {"schema": "apsr-scenevote-audit-v1", "generated": datetime.datetime.now().isoformat(timespec="seconds"),
           "definitions": {"score_min": SCORE_MIN, "iou_min": IOU_MIN, "bootstrap": {"draws": BOOT_DRAWS, "seed": BOOT_SEED, "unit": "frame"}, "tolerance_points": 0.05},
           "gt_checks_b": gt_checks,
           "seed5678_scenevote_status": s5678,
           "cells_available": cells_avail, "cells_all_three_arms": [cell_name(h, s) for h, s in cells_all3],
           "runs": run_report,
           "pair_cells": {f"{tag}_{sp}": {c: {k: r4(v) for k, v in d.items()} for c, d in pc.items()} for (tag, sp), pc in pair_cells.items()},
           "three_arm_cells": {f"{c}_{sp}": {"n_all3": d["n_all3"], **{arm: {k: r4(v) for k, v in d[arm].items()} for arm in ARMS}} for (c, sp), d in three_arm.items()},
           "host_table": {k: {kk: ({m: r4(x) if m != "n" else x for m, x in vv.items()} if isinstance(vv, dict) else vv) for kk, vv in v.items()} for k, v in host_table.items()},
           "pooled": {k: {**v, "mean": r4(v["mean"])} for k, v in pooled.items()},
           "bootstrap": {pair: {c: {**v, "point_pp": r4(v["point_pp"]), "ci95_frame": [r4(v["ci95_frame"][0]), r4(v["ci95_frame"][1])], "p_le_zero_frame": r4(v["p_le_zero_frame"])}
                                for c, v in d.items()} for pair, d in boot.items()},
           "verdict_check": verdict_check, "bootstrap_files_compared": boot_compared,
           "comparison_summary": {k: {**v, "max_abs_diff": round(v["max_abs_diff"], 6)} for k, v in matches.items()},
           "discrepancies": discrepancies}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    # ---- 콘솔 보고
    print(f"감사 시각 {out['generated']}   출력 {OUT}")
    print("GT 점검(b):", {sp: {"ann_id_0": g["annotation_id_zero_present"], "cats": g["category_ids_present"], "ok": g["ok"]} for sp, g in gt_checks.items()})
    print("seed 5678 scenevote:", s5678)
    print("세 arm 모두 있는 셀:", [cell_name(h, s) for h, s in cells_all3])
    print("\n[셀별 type-acc gain, both-localised, points]  (scenevote-raw / scenevote-tracklet)")
    for h, s in CELLS:
        c = cell_name(h, s); row = []
        for sp in SPLITS:
            for tag in ("VsRaw", "VsTrk"):
                v = pair_cells.get((tag, sp), {}).get(c); row.append(f"{v['type']:+6.2f}(n={v['n_both']})" if v else "   --   ")
        print(f"  {c:16s} std {row[0]} {row[1]}   ood {row[2]} {row[3]}")
    print("\n[pooled]")
    for k, v in pooled.items():
        if k.endswith("_type") or "_recall" in k or "_gen" in k or "_six" in k:
            if v["n_cells"]: print(f"  {k:48s} mean {v['mean']:+6.2f}  pos {v['n_positive']}/{v['n_cells']}  {v['values']}")
    print("\n[bootstrap, frame unit, 95% CI]")
    for pair, d in boot.items():
        for c, v in d.items():
            print(f"  {pair:22s} {c:26s} point {v['point_pp']:+6.2f}  CI [{v['ci95_frame'][0]:+6.2f}, {v['ci95_frame'][1]:+6.2f}]  p<=0 {v['p_le_zero_frame']:.4f}  nB {v['n_boxes_both']}")
    print("\n[대조 요약]")
    for k, v in matches.items(): print(f"  {k:40s} {v['matched']}/{v['compared']} 일치, 최대 |차이| {v['max_abs_diff']:.6f}")
    print(f"\n불일치 {len(discrepancies)}건")
    for d in discrepancies: print("  ", d)


if __name__ == "__main__":
    main()
