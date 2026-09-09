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
| `figures/` | scripts and source data of Fig. 1 and Fig. 2. `figures/overview/composite_v5.py` redraws Fig. 1 from `skeleton.png` and the seven panel crops beside it; `figures/script.py` redraws Fig. 2 from `source_data.json` |
| `baselines/` | the arms compared in Table 2 (label smoothing, out-of-fold detector-confidence relabelling, roster change counts) and the scripts behind Fig. 3 and the Section 4 loss analysis |
| `paper/` | LaTeX source of the submitted manuscript (`main_v18.tex`, `numbers.tex`, table rows, references) |

Not included: the GlassNICOL images and released labels (from the dataset authors), trained weights.

## Reproduce

1. Get GlassNICOL (head-camera split) and export the released boxes to YOLO format under `yolo_head_v1/`.
2. `python relabel/relabel_scene_v1.py --hops 2 --tol 0.5` writes the relabelled type ids and `SCENE_RELABEL_STATS_V1.json`.
3. `python relabel/check_scene_assoc_manual_v1.py 2 0.5` runs the same association on the manual splits (expects no within-chain disagreement).
4. `python eval/build_scene_arm_labels_v1.py` builds the training set; `python eval/run_arm.py scenevote <seed> 300 15600 <dataset> <run_dir>` trains one arm; repeat for `transparent` (raw) and `relabel` (tracklet vote).
5. `python eval/eval_official_v1.py <run_dir> <tag>` scores a run on the manual splits.
6. `tables/emit_v12_arms_v1.py`, `tables/bootstrap_typeacc_frames_v1.py transparent scenevote`, `tables/emit_v11_pairs_v1.py transparent scenevote PairS pair_v12_rows.tex`, `tables/emit_v13_tables_v1.py`, `tables/emit_v14_extras_v1.py` regenerate the table rows and `numbers.tex`; `tables/audit_scenevote_v1.py` recomputes everything independently.

7. `python baselines/audit_qualitative_v18.py` recounts Fig. 3 from the frozen predictions in
   `receipts/qualitative_v18/` with a matcher written independently of the figure's own: it checks the
   shared denominator (273 boxes), the five column accuracies, the ten panel counts and both frame
   selection rules, including the tie counts the caption states. It needs nothing but this repository.
8. `python baselines/make_qualitative_v18.py` redraws Fig. 3. This one also needs the GlassNICOL test
   images; point `APSR_TEST_IMAGES` at the directory holding `072_000026.png` and `091_000020.png`.
   The default arguments produce the published two-row figure.
9. `python baselines/analyse_std_loss_v18.py` reproduces the per-class breakdown behind Section 4 and
   counts which way the vote moves a type id. This one needs the full evaluation tree and both label
   sets, so it runs where steps 2-5 ran, not from this repository alone; its output is
   `receipts/STD_LOSS_ANALYSIS_V18.json` and `receipts/VOTE_DIRECTION_V18.json`.

10. `python baselines/emit_v18c_master_table.py` regenerates Table 1. The fifty scored cells it needs
    (five arms x five cells x two splits, predictions and COCO results) are in `receipts/table1_cells/`
    with both annotation files, so this reproduces every entry of the table from this repository alone.

11. `python relabel/relabel_scene_gated_v1.py --hops 2 --tol 0.5` writes the gated label set of the
    Section 4 ablation: the same association, but a chain votes only with three views and a plurality
    leading by two. Like step 2 it reads the exported dataset of step 1 (`yolo_head_v1/`) and the pair
    manifest that lists each frame's transparent and chalk-coated capture, so it runs where step 2 ran; the
    label set it writes is in `labels/scene_vote_gated_train.tar.gz`. `python baselines/emit_v18f_gate.py` turns the three scored cells in
    `receipts/gate_ablation_cells/` into the macros the sentence prints.

`APSR_QUAL_DATA` overrides where steps 7 and 8 look for the predictions and the OOD annotation file.

Seeds, budgets and the checkpoint rule are those of `preregistration/PREREGISTRATION_SCENEVOTE_V1.md`.
