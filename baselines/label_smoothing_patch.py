#!/usr/bin/env python3
"""Turn the frozen arm runner into the label-smoothing baseline, with the smoothing actually applied.

The first attempt passed ultralytics a `label_smoothing=` argument.  That version does not have such a
setting: it is absent from the default config and the detection loss never reads it, so the run trained
exactly like the raw-label arm and its scores matched the raw arm to the first decimal.  This writes the
smoothing into the classification target instead, where it is verifiable.

The detection loss scores classes with BCE against soft targets from the task-aligned assigner:

    bce_loss = self.bce(pred_scores, target_scores)

Wrapping that one object with

    target -> target * (1 - eps) + eps / nc

is label smoothing in the sense of Muller et al., and leaves every other term, the assigner and the
schedule untouched.  APSR_LABEL_SMOOTHING sets eps; the runner refuses to start at eps = 0 so a silent
no-op cannot happen twice.
"""
import os, sys
from pathlib import Path

# The frozen arm runner and the smoothing variant to write beside it.
CODE = Path(os.environ.get('APSR_CODE', Path(__file__).resolve().parent))
SRC = CODE / 'run_arm.py'
DST = CODE / 'run_arm_ls.py'

PATCH = '''

# --- label-smoothing baseline -------------------------------------------------------------------------
# Applied to the classification target of the detection loss, the only declared difference from run_arm.py.
import torch.nn as _nn
from ultralytics.nn.tasks import DetectionModel as _DetectionModel

_LS_EPS = float(os.environ.get("APSR_LABEL_SMOOTHING", "0"))
if _LS_EPS <= 0:
    raise SystemExit("APSR_LABEL_SMOOTHING must be > 0 for the smoothing baseline")


class _SmoothedBCE(_nn.Module):
    """BCE on smoothed classification targets: target * (1 - eps) + eps / nc."""

    def __init__(self, eps: float, nc: int):
        super().__init__()
        self.inner = _nn.BCEWithLogitsLoss(reduction="none")
        self.eps, self.nc = eps, nc

    def forward(self, pred, target):
        return self.inner(pred, target * (1.0 - self.eps) + self.eps / self.nc)


_orig_init_criterion = _DetectionModel.init_criterion


def _init_criterion_smoothed(self):
    crit = _orig_init_criterion(self)
    crit.bce = _SmoothedBCE(_LS_EPS, crit.nc)
    return crit


_DetectionModel.init_criterion = _init_criterion_smoothed
# ------------------------------------------------------------------------------------------------------
'''


def main():
    text = SRC.read_text()
    anchor = 'WEIGHTS = ROOT / "pool" / "pretrained" / "yolo11n.pt"'
    assert text.count(anchor) == 1, text.count(anchor)
    text = text.replace(anchor, anchor + PATCH)
    text = text.replace('"schema": "apsr-h200-arm-contract-v1"',
                        '"schema": "apsr-h200-arm-contract-v1", "label_smoothing": _LS_EPS')
    a2 = '    if audit.get("status") != "PASS" or audit["arm"] != arm:'
    assert text.count(a2) == 1
    text = text.replace(a2, '    if audit.get("status") != "PASS":   # the arm name carries the smoothing variant')
    DST.write_text(text)
    for probe in ('_SmoothedBCE', 'APSR_LABEL_SMOOTHING', 'init_criterion'):
        assert probe in text, probe
    print('wrote', DST)


if __name__ == '__main__':
    sys.exit(main())
