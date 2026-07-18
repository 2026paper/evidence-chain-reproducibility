
# Supplementary materials and reproducibility index

## Evidence-to-file map

| Evidence component | Analysis-ready data | Code | Main frozen outputs |
|---|---|---|---|
| Fixed nine-judge panel | `data/40_GitHub/rebuttal_update_20260714/api_test_scores_7290.csv` | `scripts/reliability_analysis.py` | `analysis/reliability_*` |
| Broad and selected human alignment | `analysis/cleaned_human_ratings_long.csv`, `analysis/human_api_crosswalk.csv` | `scripts/alignment_analysis.py` | `analysis/alignment_*` |
| Aggregation baselines against humans | `experiments/api_aggregation_human_baselines/aggregation_human_common_data.csv` plus the fixed panel and released human tables | `experiments/api_aggregation_human_baselines/analyze_aggregation_human_baselines.py` | `experiments/api_aggregation_human_baselines/aggregation_human_results.*` |
| Three-fresh-call repeat stability | Frozen 90/270 design inputs and nine provider score snapshots in `experiments/api_repeat_stability/` | `experiments/api_repeat_stability/analyze_repeat_stability.py` | `repeat_call_status_2430.csv`, provider/panel/variance tables and JSON report |
| Reader experiment | `analysis/public_abc_cleaned_long.csv` | `scripts/recompute_public_from_release.py` | `analysis/public_abc_analysis_results.json` |
| Exact 18-text API--reader bridge | Inputs, mapping, nine provider score snapshots and merged 162-row table in `experiments/api_reader_bridge_18/` | `qa_inputs.py`, `analyze_reader_bridge.py` | bridge cells, associations, version summary and JSON report |
| Controlled A/B failure probes | `analysis/controlled_ab_*`, `analysis/difference_survey_cleaned_long.csv` | `scripts/recompute_public_from_release.py` | controlled-case summary and results JSON |
| Figures 1--4 | `output/figures/source_data/` | four current `scripts/make_*` files named in the root README | PDF/PNG (and Fig. 1 SVG) under `output/figures/` |

## Scope of reproducibility

The package supports re-running all analyses from de-identified or frozen
analysis-ready inputs. It does not support replaying the original platform
exports or live API transactions because those would require restricted raw
records, credentials, and provider payloads. Provider score CSVs are frozen
snapshots sufficient for the reported downstream statistics.

The 90 stimuli in the repeat-stability experiment are treated as a frozen,
outcome-blind design input. The selection constraints, seed, hashes, and QA are
documented, but the upstream selection program is not represented as an
end-to-end reconstruction. All downstream 90 x 3 x 9 = 2,430 planned-score
analyses are reproducible from released files.

## Withheld by design

- Direct/platform identifiers, IP/location/device metadata and timestamps.
- Exact completion durations and sparse participant background cells.
- Participant open text and reversible re-identification maps.
- API keys, environment files, raw HTTP responses/full provider payloads.
- Call logs, stdout/stderr/resume logs, failed attempts and pilot runs.
- Raw survey workbooks and word-processing files.

These omissions do not remove the analysis variables used in the reported
models. Instrument text is released independently of participant responses.
