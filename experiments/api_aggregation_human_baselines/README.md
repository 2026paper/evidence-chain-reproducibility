
# API aggregation baselines against the broad human criterion

This experiment compares the nine individual judges with four fixed panel
aggregations on the same 178 visible texts and 1,068 text-by-dimension rows:
arithmetic mean, median, one-score-per-tail trimmed mean, and the generator-
family-matched no-self mean. Every method uses the same controls, frozen
standardization contract, 5,000 domain-stratified concept-cluster bootstrap
draws, and 10,000 shared Freedman--Lane permutations.

Reproduce the frozen tables with:

```bash
python recompute_aggregation_from_common.py
```

The public recomputation starts from the frozen, de-identified
`aggregation_human_common_data.csv`, re-runs every point/raw-scale metric,
5,000-draw cluster bootstrap, 10,000-permutation Freedman--Lane test, paired
beta difference and standardization constant, and checks the frozen results
CSV and JSON. The original construction script is retained for provenance,
but the public entry point does not depend on restricted raw participant
exports or the superseded broad-review release. It does not read API
credentials or provider payloads. See
`AGGREGATION_HUMAN_BASELINES_REPORT.md` for estimates and interpretation
boundaries.
