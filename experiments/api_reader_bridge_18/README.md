# API–reader bridge: 18 exact reader-visible texts

This directory freezes the 18 texts actually shown in public forms A, B, and C: six concept slots in each of three independently fielded reader-facing versions.

## Files

- `inputs.csv`: runner-friendly RFC 4180 CSV; `visible_text` is a quoted multiline field.
- `inputs.jsonl`: lossless JSON Lines equivalent with full per-record provenance.
- `runner_inputs.csv`: blinded runner schema (`item_id`, `domain`, `concept`, `task_type`, `generated_text`), hash-sorted with opaque IDs.
- `mapping.csv`: held-out mapping from opaque runner IDs to A/B/C source identity and provenance.
- `manifest.json`: frozen design, source hashes, input-file hashes, and the 18 text hashes.
- `qa_inputs.py`: read-only QA against the three instruments, answer key, semantic crosswalk, and manifest.

## Interpretation boundary

The A/B/C labels are public-form versions, not corpus task labels. These are reader-facing rewrites linked one-to-one to six upstream items; they are not exact subsets of the 810-item corpus, and existing upstream API scores therefore cannot be reused as direct scores for these strings.

Only `visible_text`/`generated_text` is intended as model stimulus. `comprehension_question` is provenance metadata. Correct answers are checked against the frozen answer key but deliberately excluded from model-input files to prevent leakage. Do not join `mapping.csv` into runner prompts.

## Canonical text and hashing

The canonical text uses Unicode NFC, LF line endings, and no trailing newline. The instrument transcription has one LF between visible lines. The crosswalk uses paragraph spacing, so QA collapses consecutive crosswalk line breaks to one LF before requiring exact equality. `visible_text_sha256` is SHA-256 over the canonical UTF-8 bytes.

Run QA from the repository root:

```powershell
python -X utf8 experiments/api_reader_bridge_18/qa_inputs.py
```

Expected result: `PASS: 18 records; 6 concepts x 3 versions; all source, text, and file hashes verified.`

No participant responses, identifiers, timing, device/location fields, API credentials, or correct-answer strings are present in the model-input files.
