# API repeat-stability experiment

This directory freezes an outcome-blind, balanced 90-stimulus sample and expands it to three genuinely fresh repetitions per stimulus.

## Execution inputs

- `runner_inputs_270.csv` is the only stimulus file needed by the API runner. Its exact columns are `item_id,domain,concept,task_type,generated_text`. The opaque `item_id` contains no source or repetition label.
- `mapping_270.csv` is an internal-only crosswalk from opaque IDs to the 90 source stimuli and repetition indices. Do not send it in prompts.
- `judge_repetition_grid.csv` specifies nine judges and three repetitions. The Cartesian product of 270 runner rows and the nine judges is 2,430 fresh API requests.
- `repeat_sample_90.csv` is the canonical sampled-stimulus table. `repeat_sample_90.json` is its machine-readable item index.

Every call must be a new provider request. Disable response caching where controllable. The historical 7,290-row matrix is provenance/QA evidence only and must not be counted as repetition 1. A retry after transport or parse failure remains a retry of the same planned call, not an additional repetition.

## Frozen design

Sampling seed: `20260717`. The sample contains one generator output from every concept-by-task stratum: 30 concepts x 3 tasks = 90 stimuli. Each of the nine generators contributes exactly 10 stimuli; each domain contributes 18; each task contributes 30; every generator contributes exactly two stimuli per domain and three or four per task. All 90 visible-text hashes are distinct.

Runner order seed namespace: `20260717|runner-order-v1`. The first SHA256 ordering with no adjacent rows from the same source was attempt 3. QA confirms 270 unique opaque IDs, 90 stimuli repeated exactly three times, and zero adjacent same-source pairs.

No credential is stored in this directory.
