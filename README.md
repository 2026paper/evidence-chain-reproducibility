
# Anonymous blind-review reproducibility repository

This repository contains the de-identified analysis data, frozen model scores,
released scientific item/stimulus texts, available judge prompts and survey
wrappers, analysis scripts, figure source data, and verification files used by
the ADMA 2026 Full Paper submission. It is designed for anonymous peer review:
no author identity, local filesystem path, API credential, raw provider payload,
or direct platform identifier is distributed. Historical materials that could
not be recovered exactly, or are intentionally withheld, are listed in
`MISSING.md`; no substitute material is presented as original.

## Start here

Use Python 3.12.7. Install the analysis and figure dependencies, then run all
three read-only package checks:

```bash
python -m pip install -r environment/requirements-analysis.txt
python -m pip install -r environment/requirements-figures.txt
python scripts/audit_anonymity.py
python scripts/verify_release.py
python scripts/verify_nested_manifests.py
```

The SHA-256 inventory at `manifests/sha256_manifest.csv` covers every tracked
file except the inventory itself.

Run the three checks before any analysis. The analysis and figure commands below
write their documented outputs in place; use a disposable copy or fresh checkout
if the frozen distributed outputs must remain byte-for-byte unchanged. Some
preprocessing/provenance scripts require privacy-restricted raw platform exports
and are retained for auditability rather than advertised as clean-clone entry
points. The commands explicitly listed in this README are the reviewer-facing
runnable entry points.

## Reproduce the core human and panel analyses

The following commands rebuild the human reliability and human--API alignment
outputs from the released, non-reversibly rekeyed response table and the frozen
7,290-score API matrix. The alignment command is the full locked run (5,000
bootstrap draws, 10,000 broad-review permutations, and all 32,768 selected-set
swaps), so it is substantially slower than the other checks.

```bash
python scripts/reliability_analysis.py --api-scores data/40_GitHub/rebuttal_update_20260714/api_test_scores_7290.csv
python scripts/alignment_analysis.py --api-scores data/40_GitHub/rebuttal_update_20260714/api_test_scores_7290.csv
python scripts/recompute_public_from_release.py
```

The 810 source records correspond to 796 distinct task-aware rendered stimuli,
as defined by `stimulus_signature_sha256` in
`analysis/api_stimulus_equivalence_810.csv`. A one-column text/output deduplication
is not equivalent to this complete rendering signature and yields a different
count.

## Reproduce the added analyses

These are three added analyses: two fresh API-call experiments and one
aggregation reanalysis.

The repository ships analysis-ready score tables, not API credentials or raw
HTTP/provider responses. Live calls may drift after provider model updates.

```bash
python experiments/api_repeat_stability/analyze_repeat_stability.py --require-complete
python experiments/api_reader_bridge_18/qa_inputs.py
python experiments/api_reader_bridge_18/analyze_reader_bridge.py
python experiments/api_aggregation_human_baselines/analyze_aggregation_human_baselines.py
python experiments/api_aggregation_human_baselines/recompute_aggregation_from_common.py
```

The full aggregation analyzer reconstructs the comparison from the released
human criterion and fixed API matrix. The final command independently verifies
the frozen common comparison table; the two commands therefore overlap by
design and need not both be run for a quick inspection.

For repeat stability, the balanced 90-stimulus table is a **frozen design
input**. Its upstream selection is documented in
`experiments/api_repeat_stability/sampling_manifest.json` and
`experiments/api_repeat_stability/STATISTICAL_ANALYSIS_PLAN.md`, but the
repository does not claim an end-to-end rebuild of that 90-item selection from
pre-selection code. Starting from the frozen 90 items, the 2,430 planned
fresh-call score analysis is reproducible: all nine provider score snapshots,
the blinded 270-row runner input, mapping, manifests, and analysis outputs are
included. Failed retries and raw call logs are not.

## Reproduce figures

```bash
python scripts/make_five_layer_evidence_figure.py
python scripts/make_reliability_figure_v3.py
python scripts/make_alignment_figure.py
python scripts/make_reader_bridge_figure_v2.py
```

Current vector/raster outputs and their source CSV files are under
`output/figures/`.

## Repository map

- `analysis/`: privacy-minimized human/public tables and frozen formal results.
- `data/`: the 7,290-row fixed API score matrix used by the formal analyses.
- `experiments/`: repeat stability, exact reader-visible-text bridge, and
  human-criterion aggregation baselines.
- `materials/`: complete scientific instruments without participant responses.
- `models_and_prompts/`: available credential-free judge prompts, schemas, and
  model metadata.
- `scripts/`: formal analysis/verification scripts and current Fig. 1--4 code.
- `output/figures/`: current figures, manifests, and source data.
- `manifests/`: package integrity, privacy, provenance, and anonymity checks.

See `SUPPLEMENTARY_MATERIALS.md`, `DATA_DICTIONARY.md`, `DESIGN_BOUNDARY.md`,
and `MISSING.md` for evidence-to-file mapping and explicit boundaries.

## Privacy and access boundary

The repository excludes raw survey workbooks, platform IDs, IP/location/device
fields, timestamps, exact response durations, sparse background cells,
participant open text, credentials, raw API call logs, raw provider payloads,
failed-attempt logs, and reversible re-identification maps. Released response
tables use non-reversible random keys and only fields required for the reported
analyses. Released scientific wording and the available response-free form
wrappers are in `materials/`; the unrecovered historical wrapper boundary is
stated in `MISSING.md`.
