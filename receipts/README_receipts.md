# Receipts

Every printed figure in the paper comes from a file here, and each file names the script that wrote it.

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

The macros the manuscript prints are emitted from these by `baselines/emit_v18*.py`; run
`tables/tidy_numbers.py` afterwards so every printed macro sits in the live block of `numbers.tex`.
