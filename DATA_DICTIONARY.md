# Release data dictionary

## Individual-level minimized files

- `analysis/cleaned_human_ratings_long.csv`: human-review rating long table. `participant_id` is a new non-reversible random key; source filename was removed. Scores, item hashes, wave/domain/task, and clustering are unchanged.
- `analysis/public_abc_cleaned_long.csv`: randomized A/B/C reader-response long table. `response_id` and `participant_id` are new study-local random keys. Exact duration and education/field/reading-frequency fields were removed. `duration_qc_status` distinguishes retained missing/nonpositive time from retained-at-or-above-floor without publishing seconds.
- `analysis/difference_survey_cleaned_long.csv`: eight-case forced-choice validation table. `respondent_id` is a new study-local random key. Exact duration, role, field, and familiarity fields were removed. `reason_category` is a closed derived category; no open text is present.

Keys are randomized independently across studies. They cannot be used to link the same person across instruments. The rekey map is not retained.

## Crosswalks

- `human_api_crosswalk.csv` and `api_stimulus_equivalence_810.csv` contain item/provenance IDs and hashes, not participant data. The paper's 796 distinct stimuli are the 796 unique task-aware `stimulus_signature_sha256` values in the latter file; deduplicating only one text/output column from the 810-record archive is not the same operation.
- `public_semantic_crosswalk_hash_release.csv` and `questionnaire_item_dictionary_hash_release.csv` are compact hash-only analysis crosswalks.
- `controlled_ab_pair_text_crosswalk.csv` contains pair hashes and exact-match flags only.

## Complete instrument-only materials

- `materials/generated_items_810.csv` contains the full normalized 810-item stimulus archive; it is not an HTTP/API response dump.
- `materials/human_review_questionnaire_items.csv` contains the 210 human-review item texts; `materials/human_review_wave1_*_instrument.txt` contains the complete five broad-review form wrappers. The exact published wrapper for the selected 30-item second wave was not independently recovered and is not reconstructed.
- `materials/public_form_{A,B,C}_instrument.txt` contains all visible question and option text from each separately published 见数 form. `materials/public_abc_answer_key.json` provides the attention and comprehension keys.
- `materials/public_semantic_crosswalk_full.csv` releases the complete meaning-based mapping used for the six purposively selected, popularized cases.
- `materials/difference_survey_instrument.txt`, `materials/difference_survey_answer_key.json`, and `materials/controlled_ab_pairs_full.json` provide the eight controlled pairs, response prompts, keys, and expected directions.

These files contain scientific content only. They contain no response, participant key, timestamp, duration, background value, device/location field, or participant open text.

## Controlled A/B judge dimensions

- `analysis/controlled_ab_judge_dimension_scores.csv` contains the 144 validated judge-item records used to construct the controlled A/B quality contrasts (eight cases, two versions, and nine judge providers). The released columns are limited to the join keys `case_id`, `version`, and `judge_provider`; the five ordinal quality dimensions `fa`, `cc`, `lc`, `tf`, and `mq`; and the derived `quality_mean`.
- `quality_mean` is the unweighted arithmetic mean `(fa + cc + lc + tf + mq) / 5`. Pivoting by `version` and joining on `case_id` plus `judge_provider` reproduces `quality_A` and `quality_B` in `analysis/controlled_ab_machine_judge_case_long.csv` exactly.
- Provider-response payloads, backend identifiers, token counts, retry metadata, timestamps, local paths, and free-text responses are not included in the dimension-score release.

## Score direction

`fa`, `cc`, `lc`, `tf`, and `mq` are positive 1–5 dimensions. `risk` is a 1–5 misleading-risk dimension, where larger is worse. Files explicitly named `score_quality_aligned` reverse risk only when stated by the formal script.

## Small cells

Background variables are absent from all public individual-level files. Sparse background summaries were removed from the released JSON results. Study-design cells (for example human-review wave/domain panels) are scientific sampling strata rather than demographic/background disclosures and are retained.
