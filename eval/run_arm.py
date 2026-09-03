#!/usr/bin/env python3
"""Train one APSR arm at one seed, using the recipe frozen on the 136 runs.

Usage: run_arm.py ARM SEED EPOCHS EXPECTED_UPDATES DATASET RUN_ROOT
Every hyperparameter below is copied from run_exposure_matched_control.py so the
H200 runs stay comparable to the original single-seed results.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from copy import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO as UltralyticsYOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "pool" / "pretrained" / "yolo11n.pt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CollapsedGenericValidator(DetectionValidator):
    """Class-aware NMS first, then collapse subtype IDs for generic COCO matching."""

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            pbatch["cls"] = torch.zeros_like(pbatch["cls"])
            predn = self._prepare_pred(pred)
            predn = {k: v.clone() if k == "cls" else v for k, v in predn.items()}
            predn["cls"] = torch.zeros_like(predn["cls"])
            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0
            self.metrics.update_stats({
                **self._process_batch(predn, pbatch),
                "target_cls": cls,
                "target_img": np.unique(cls),
                "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                "im_name": Path(pbatch["im_file"]).name,
            })


class CountingTrainer(DetectionTrainer):
    optimizer_step_attempts = 0

    def optimizer_step(self):
        self.optimizer_step_attempts += 1
        return super().optimizer_step()

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return CollapsedGenericValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


class CollapsedYOLO(UltralyticsYOLO):
    def val(self, validator=None, **kwargs):
        return super().val(validator=CollapsedGenericValidator, **kwargs)


def main() -> None:
    arm, seed, epochs, expected_updates = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    dataset, run_root = Path(sys.argv[5]), Path(sys.argv[6])
    if run_root.exists():
        raise SystemExit(f"no-clobber: {run_root}")
    audit_path, yaml_path = dataset / "DATASET_AUDIT.json", dataset / "data_full.yaml"
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "PASS" or audit["arm"] != arm:
        raise SystemExit("dataset admission failed")

    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip().splitlines()[0]
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    run_root.mkdir(parents=True)
    contract = {
        "schema": "apsr-h200-arm-contract-v1", "status": "FROZEN_BEFORE_RESULT",
        "arm": arm, "seed": seed, "epochs": epochs,
        "expected_optimizer_updates": expected_updates,
        "host": "h200", "gpu_uuid": gpu_uuid,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dataset_audit_sha256": sha256(audit_path),
        "dataset_yaml_sha256": sha256(yaml_path),
        "roster_sha256": audit["roster_sha256"],
        "pretrained_weights_sha256": sha256(WEIGHTS),
        "runner_sha256": sha256(Path(__file__)),
        "official_test_member_payload_reads": 0,
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "cv2_version": cv2.__version__,
    }
    contract_path = run_root / "CONTRACT.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")

    model = CollapsedYOLO(str(WEIGHTS))
    started = time.time()
    model.train(
        trainer=CountingTrainer,
        data=str(yaml_path), imgsz=640, batch=64, nbs=64, epochs=epochs,
        device=0, seed=seed, deterministic=True, workers=8, cache="ram",
        optimizer="AdamW", lr0=1e-3, lrf=1e-2, weight_decay=5e-4,
        cos_lr=True, amp=False, patience=0, close_mosaic=30,
        val=True, plots=False, save=True, save_period=-1,
        project=str(run_root), name="train", exist_ok=False, verbose=False,
    )
    updates = int(model.trainer.optimizer_step_attempts)
    if updates != expected_updates:
        raise SystemExit(f"optimizer update mismatch: {updates} != {expected_updates}")
    best, last = run_root / "train" / "weights" / "best.pt", run_root / "train" / "weights" / "last.pt"
    if not best.is_file() or not last.is_file():
        raise SystemExit("missing checkpoints")
    evaluated = CollapsedYOLO(str(best)).val(
        data=str(yaml_path), split="val", imgsz=640, batch=64, device=0,
        workers=0, plots=False, project=str(run_root), name="best_eval",
        exist_ok=False, verbose=False,
    )
    result = {
        "schema": "apsr-h200-arm-result-v1", "status": "COMPLETE_UNADJUDICATED",
        "arm": arm, "seed": seed, "optimizer_updates": updates,
        "generic_val": {"ap50_95": float(evaluated.box.map),
                        "ap50": float(evaluated.box.map50),
                        "ap75": float(evaluated.box.map75)},
        "best_sha256": sha256(best), "last_sha256": sha256(last),
        "contract_sha256": sha256(contract_path),
        "elapsed_s": time.time() - started,
        "official_test_member_payload_reads": 0,
    }
    (run_root / "FINAL_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
