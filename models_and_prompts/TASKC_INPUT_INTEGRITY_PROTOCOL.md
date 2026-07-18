# Corrected Task-C input-integrity protocol

Each Task-C item must expose the following semantic fields to the judge before any API call:

1. complete question stem;
2. options A, B, C, and D;
3. the correct answer label together with its option text;
4. the generated explanation;
5. task/domain/concept context used by the judging prompt;
6. no generator identity in the judged content block.

The offline audit rejects a payload when any required field is absent, the rendered block is below the minimum length, the block collapses to an answer label, a source-item mapping is not one-to-one, or the source/rendered hashes do not match the retained manifest. Output validation separately checks parseability, required score keys, numeric values, and the 1–5 range. Attempt rows and retained final rows are stored separately; a retry never becomes an additional authoritative row.

`data/taskc_test_inputs_270.csv` contains the exact task-aware content blocks. `data/taskc_payload_audit_270.csv` contains the 270 item-level decisions. The final prompt hash for every corrected Task-C item–judge pair is in `data/api_test_scores_7290.csv`.

This protocol establishes semantic input completeness and output-schema validity only. It does not establish construct validity or the scientific correctness of any individual score.

