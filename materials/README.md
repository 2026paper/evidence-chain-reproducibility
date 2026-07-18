# Scientific materials and instruments

This directory contains instrument-only materials, never participant responses.

- `generated_items_810.csv`: all 810 generated experimental items and normalized content.
- `human_review_questionnaire_items.csv`: the 210 item texts shown in the two human-review waves.
- `human_review_wave1_*_instrument.txt`: the complete visible wrappers, prompts, options, calibration/repeat items, and rating matrices for the five broad-review domain forms underlying the 180-row analyzable item dictionary.
- `public_form_A_instrument.txt`, `public_form_B_instrument.txt`, and `public_form_C_instrument.txt`: complete visible question and option text extracted from the three separately published 见数 forms.
- `public_semantic_crosswalk_full.csv`: the full meaning-based crosswalk for the six public-facing concepts and the controlled cases.
- `public_abc_answer_key.json`: attention key, pre/post comprehension keys, scale coding, and the fixed 72-second duration floor.
- `difference_survey_instrument.txt`, `difference_survey_answer_key.json`, and `controlled_ab_pairs_full.json`: the eight A/B materials, questions, coding keys, and expected quality/risk directions.

Design boundary: platform entrants were completely randomly allocated among the separately published A/B/C forms. The six scientific materials were purposively selected disagreement cases and then popularized for general readers; they were not randomly sampled or copied verbatim from the 810-item corpus. The difference survey is a controlled expected-direction probe, not a population-accuracy sample.

The legacy `.doc` sources are Word containers whose visible questionnaire HTML is stored in `word/afchunk.mht`. The released `.txt` files are deterministic visible-text extractions. Source and release hashes are recorded in `manifests/source_to_release_map.csv`.

The selected second-wave item dictionary contains all 30 scientific items, but an independently source-bound copy of the exact published second-wave wrapper was not recoverable. Candidate reconstruction files were not substituted; see `MISSING.md`.
