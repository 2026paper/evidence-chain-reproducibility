# Statistical analysis plan: fresh-call repeat stability

## Frozen analysis set

The analysis uses 90 sampled stimuli, nine fixed judges, three fresh calls per stimulus-judge cell, and six rating dimensions (`fa`, `cc`, `lc`, `tf`, `mq`, `risk`). The expected long table therefore contains 2,430 valid call records and 14,580 dimension scores. Historical scores are not part of the three-repetition estimand.

The 90 stimuli are fixed by the committed hashes before fresh responses are inspected. Inference to new content resamples the 30 concept clusters; the nine-judge panel is treated as fixed unless an explicitly labelled leave-one-judge-out sensitivity analysis is reported.

## Primary estimands

1. **Within-judge repeatability.** For each judge and dimension, report absolute-agreement single-measure ICC across the three fresh calls, exact agreement, within-one-category agreement, and mean absolute pairwise difference. Report bootstrap 95% intervals by resampling concepts and retaining all three tasks and repetitions within a sampled concept.
2. **Fixed-panel repeatability.** Within each repetition, aggregate the nine judges by the arithmetic mean. Across the three independently obtained panel means, report absolute-agreement ICC, pairwise Lin concordance, Spearman correlation, MAE, and RMSE. Repeat for median, one-high/one-low trimmed mean, and no-self mean as labelled secondary analyses.
3. **Variance decomposition.** Fit a cross-classified mixed model per dimension with task and domain fixed effects and random components for stimulus, judge, stimulus-by-judge, repetition occasion, and residual call variation. Use it to report the proportions attributable to judge disagreement versus fresh-call instability. Treat a cumulative-link mixed model as the ordinal sensitivity model; if it fails to converge, retain the linear mixed model and disclose the failure.

Holm correction is applied across the six dimensions within each primary family. Effect sizes and confidence intervals remain primary; corrected p-values are supporting evidence. A pooled six-dimension mean may be reported only if declared before outcome inspection and accompanied by dimension-specific results.

## Missingness and failures

Do not impute failed calls. Report transport failure, provider refusal, schema/parse failure, and out-of-range scores by provider and repetition. A parse retry is linked to the same planned opaque ID and judge; the first valid response is used, while all attempts remain in the run log. Primary complete-cell estimates use cells with three valid fresh responses; available-case sensitivity estimates must be labelled.

## Execution metadata and leakage controls

Retain opaque item ID, judge provider, requested and returned model identifiers, UTC timestamp, runner order, repetition index from the internal mapping, retry count, latency, prompt hash, response hash, and parse status. Do not include source or repetition labels in the model prompt. The runner file is randomly ordered with zero adjacent occurrences of the same source stimulus.

## Interpretation boundary

This experiment estimates repeat-call stability for the frozen model snapshots, prompt, 90-stimulus sample, and fixed nine-judge panel. It does not by itself establish human validity, temporal stability after model updates, or calibration to reader outcomes.
