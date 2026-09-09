# Scene-level type-consistent relabelling for transparent glass detection

Code, receipts and paper source for the ICASSP 2027 submission
*Scene-Level Type-Consistent Relabelling for Transparent Glass Detection*.

GlassNICOL assigns each glass a type automatically from a depth-derived height, in every view.
The released training labels disagree on the same glass in 18.1% of matched adjacent-view pairs,
while the hand-labelled test splits almost never do. This repository contains the association that links
the views of each glass through the table plane, the vote that gives each chain one type, the
evaluation on the official manual test splits, and every number printed in the paper.

## Layout

| directory | contents |
|---|---|
| `relabel/` | `relabel_scene_v1.py` (table-plane registration, base-point linking, union-find chains, vote), `check_scene_assoc_manual_v1.py` (validity check on the hand-labelled splits), `relabel_multiview_v1.py` (adjacent-frame tracklet baseline), `chain_confusion_em_v1.py` (Dawid–Skene estimate of the labeller's confusion) |
| `eval/` | `build_scene_arm_labels_v1.py` (dataset build with the shared roster), `run_arm.py` (YOLO11n training, one arm), `eval_official_v1.py` (scoring on the manual splits: six-class AP, generic AP, recall, type accuracy) |
| `tables/` | emitters that turn the receipts into the paper's tables and macros, the paired frame bootstrap, an independent re-implementation of the scoring (`audit_scenevote_v1.py`) |
| `receipts/` | every JSON the tables and sentences are built from: verdicts, bootstrap intervals, association statistics, ablation grid, manual-split checks, confusion matrix |
| `preregistration/` | the sealed protocol, its SHA-256, and an errata note on two clock strings typed inside it |
| `labels/` | `scene_vote_train.tar.gz`: the relabelled training type ids (YOLO format; boxes identical to the release) |
| `figures/` | scripts and source data of Fig. 1 and Fig. 2 |
| `paper/` | LaTeX source of the submitted manuscript (`main_v18.tex`, `numbers.tex`, table rows, references) |

Not included: the GlassNICOL images and released labels (from the dataset authors), trained weights.

## Reproduce

1. Get GlassNICOL (head-camera split) and export the released boxes to YOLO format under `yolo_head_v1/`.
2. `python relabel/relabel_scene_v1.py --hops 2 --tol 0.5` writes the relabelled type ids and `SCENE_RELABEL_STATS_V1.json`.
3. `python relabel/check_scene_assoc_manual_v1.py 2 0.5` runs the same association on the manual splits (expects no within-chain disagreement).
4. `python eval/build_scene_arm_labels_v1.py` builds the training set; `python eval/run_arm.py scenevote <seed> 300 15600 <dataset> <run_dir>` trains one arm; repeat for `transparent` (raw) and `relabel` (tracklet vote).
5. `python eval/eval_official_v1.py <run_dir> <tag>` scores a run on the manual splits.
6. `tables/emit_v12_arms_v1.py`, `tables/bootstrap_typeacc_frames_v1.py transparent scenevote`, `tables/emit_v11_pairs_v1.py transparent scenevote PairS pair_v12_rows.tex`, `tables/emit_v13_tables_v1.py`, `tables/emit_v14_extras_v1.py` regenerate the table rows and `numbers.tex`; `tables/audit_scenevote_v1.py` recomputes everything independently.

Seeds, budgets and the checkpoint rule are those of `preregistration/PREREGISTRATION_SCENEVOTE_V1.md`.
