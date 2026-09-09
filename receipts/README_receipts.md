# Receipts

Every number the manuscript prints is emitted from a file here. The table below names the receipts the
V18 emitters read directly; the other files in this directory are inputs to those emitters or receipts of
earlier drafts, and each one records the script that wrote it in its own `schema` field.

| receipt | written by |
|---|---|
| `ROSTER_CHANGE_RATES_V1.json` | `baselines/count_roster_changes_v1.py` |
| `CONFID_VS_SCENE_DISAGREEMENT_V1.json` | `baselines/count_roster_changes_v1.py` |
| `CONFIDENCE_FOLD_ASSIGNMENT_V1.json` | `baselines/count_roster_changes_v1.py` |
| `CONFIDENCE_RELABEL_STATS_V1.json` | `baselines/relabel_confidence_v1.py` |
| `CONFIDENCE_RELABEL_STATS_TAU08.json` | `baselines/relabel_confidence_tau.py` |
| `CONFID_THRESHOLD_SENSITIVITY_V1.json` | `baselines/emit_v18b_extras.py` reads it; produced by scoring the tau-0.8 cells |
| `LASTEPOCH_BASELINES_V1.json` | `baselines/emit_v18c_lastepoch.py` |
| `BASELINE_CELLS_V18.json`, `BASELINE_COMPARISON_V18.json` | `baselines/emit_v18_baselines.py` |
| `MASTER_TABLE_V18C.json` | `baselines/emit_v18c_master_table.py` (Table 1: five arms, four metrics, both splits) |
| `STD_LOSS_ANALYSIS_V18.json` | `baselines/analyse_std_loss_v18.py` (per-class AP changes behind Section 4) |
| `VOTE_DIRECTION_V18.json`, `VOTE_DIRECTION_FLOOR_V18.json` | `baselines/emit_v18b_fixes.py` (which way the vote moves a type id, and the share forced by the two extreme types) |
| `QUALITATIVE_FIGURE_V18.json`, `QUALITATIVE_SELECTION_V18.json` | `baselines/make_qualitative_v18.py`, `baselines/select_qualitative_v18.py` (Fig. 3) |
| `AUDIT_QUALITATIVE_V18.json` | `baselines/audit_qualitative_v18.py` (independent recount of Fig. 3) |
| `HIGHLIGHT_IDENTITY_V18.json` | `baselines/emit_v18_highlight_rows.py` (the table's bold/blue marks change no value) |
| `qualitative_v18/` | the ten frozen prediction files and the OOD annotations Fig. 3 is drawn from |
| `table1_cells/` | the fifty scored cells behind Table 1: predictions and COCO results for five arms x five cells x two splits, with both annotation files |

The macros the manuscript prints are emitted from these by `baselines/emit_v18*.py`; run
`tables/tidy_numbers.py` afterwards so every printed macro sits in the live block of `numbers.tex`.
