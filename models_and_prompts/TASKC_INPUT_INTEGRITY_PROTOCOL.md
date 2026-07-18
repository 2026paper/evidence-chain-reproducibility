# One-shot matrix input-integrity provenance

The released historical scoring matrix is a fixed one-shot snapshot with one authoritative retained score row for each of 810 items evaluated by nine judges (7,290 item-judge rows). It is evidence about agreement among judges under the recorded single-call conditions; it is not repeat-call stability evidence.

The reviewer-facing provenance chain is:

- `materials/generated_items_810.csv`: normalized scientific item and stimulus content;
- `data/40_GitHub/rebuttal_update_20260714/api_test_scores_7290.csv`: the frozen one-shot score matrix and its retained provenance and hash fields;
- `analysis/api_stimulus_equivalence_810.csv`: task-aware rendered-stimulus signatures and equivalence audit;
- `models_and_prompts/judge_system_prompt.txt` and `models_and_prompts/judge_user_prompt_template.txt`: the released judging instructions;
- `models_and_prompts/taskc_run_inventory.csv`: credential-free run-level provenance extracted from the locked run manifests.

The `api_source_regime` values in the released matrix are frozen internal provenance codes that distinguish the recorded input regimes used for Tasks A/B and Task C. They are not experimental conditions, quality labels, or scientific conclusions. All 7,290 retained rows passed the declared input-integrity checks and the six-score output-schema checks before inclusion in the matrix.

The normalized item archive is not a raw provider HTTP payload. Credentials, raw attempt logs, provider payloads, and unsuccessful retries are withheld. This note establishes traceable input provenance and score-schema validity only; it does not establish construct validity or the scientific correctness of any individual score.
