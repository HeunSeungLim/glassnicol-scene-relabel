# Preregistration: scene-level type-consistent relabelling (scene vote) — v1

Written 2026-09-02 21:20 KST, before any detector was trained on the scene-vote labels.

## Method under test
relabel_scene_v1.py (hops K=4, base-point tolerance 0.5 of box width): frames of a scene are registered
pairwise up to four frames apart by a table-plane homography from background ORB features; each box is
carried by its base point (bottom-centre); links are one-to-one per frame pair; union-find merges links with
the constraint of at most one box per frame per chain; each chain takes its majority type; ties unchanged;
no box moves. Labels: labels_scene/{train,val}. Training split statistics at sealing time:
741 chains for 4,336 boxes, median chain length 10 (by box), 4,007 boxes in chains of three or more,
546 boxes (12.6%) change type, 164 tied. The adjacent-frame tracklet vote (labels_relabelled) has
1,561 chains, median length 1, 364 boxes (8.4%) changed.

## Precondition (association validity), evaluated before training
On the hand-labelled official splits the same association must produce fewer than 2% within-chain type
disagreements (manual types are constant per object, so disagreement measures merged objects). Result
recorded in SCENE_ASSOC_MANUAL_CHECK_V1.json. If the precondition fails the experiment is not run.

## Arms, seeds, hosts
- raw: existing scene_transparent runs (H200 seeds 1337, 3407, 4567; RTX seeds 1337, 5678).
- tracklet: existing scene_relabel runs (same seeds and hosts).
- scenevote (new): dataset scene_scenevote / scene136_scenevote, identical roster, repeated slots, budget
  (300 epochs, 15,600 updates), hyperparameters and checkpoint rule (best validation generic AP);
  H200 seeds 1337, 3407, 4567 on GPUs 1-3; RTX seeds 1337, 5678.

## Metrics (as PREREGISTRATION_RELABEL_V1)
Official standard and OOD test splits, manual labels, 1-based renumbered annotations. Six-class AP, generic AP,
localisation recall (IoU >= 0.5, confidence >= 0.25), type accuracy on boxes localised by BOTH compared arms.
Per-cell paired frame bootstrap (2,000 draws, seed 20260902) for the type-accuracy gain.

## Predictions (fixed now)
P1. Over the five (host, seed) cells the mean type-accuracy gain of scenevote over raw is positive on both splits.
P2. On the OOD split the mean type-accuracy gain of scenevote over raw exceeds that of tracklet over raw
    (+5.0 points over the same five cells).
P3. Localisation recall and generic AP move by less than 2 points on average on both splits (the vote touches
    type ids only).
P4. The number of cells with a positive OOD type-accuracy gain is at least 4 of 5.

## Refutation
If P1 fails on both splits the scene vote does not help and is reported as such. If P1 holds and P2 fails,
the scene vote is reported as equivalent to the tracklet vote with more coverage, not as an improvement.
Nothing else is added post hoc without being labelled post hoc.

## Addendum v1.1 (2026-09-02 21:45 KST, still before any scene-vote detector was trained)
The v1 setting (hops 4, tolerance 0.5) failed the precondition on the standard split: 4 of 167 manual boxes
(2.4%) disagreed within their chain (two merged pairs, water/beer and wine/high); OOD 0 of 351. A grid over
hops {2,3,4} x tolerance {0.25,0.3,0.4,0.5} was run on the manual splits (SCENE_ASSOC_MANUAL_CHECK_V1_h*_t*.json).
The loosest setting with zero disagreement on both splits is hops 2, tolerance 0.5 (coverage 163/167 and
346/351 boxes in chains of three or more; median chain length 10 and 15). That setting is adopted for the
experiment. The manual splits are used here only to validate the association; they never enter training,
and no detector had been trained on any scene-vote labels when this addendum was written. Predictions P1-P4
and the refutation rule are unchanged.
