# Three-fresh-call API repeat-stability report

**Analysis status: FINAL COMPLETE.**

Only the three calls in this experiment are analyzed. No score from the historical 7,290-row matrix is used as a repetition. Each provider-by-dimension estimate uses only stimuli with all three fresh responses; missing calls are not imputed.

## Completion audit

| Provider | Valid planned calls | Missing | Complete 3-call stimuli | Complete |
|---|---:|---:|---:|:---:|
| anthropic | 270/270 | 0 | 90/90 | yes |
| deepseek | 270/270 | 0 | 90/90 | yes |
| doubao | 270/270 | 0 | 90/90 | yes |
| gemini | 270/270 | 0 | 90/90 | yes |
| glm | 270/270 | 0 | 90/90 | yes |
| kimi | 270/270 | 0 | 90/90 | yes |
| mimo | 270/270 | 0 | 90/90 | yes |
| openai | 270/270 | 0 | 90/90 | yes |
| qwen | 270/270 | 0 | 90/90 | yes |

The provider files contained 2431 physical rows. Exactly 2,430 unique planned calls were selected; 1 invalid audit row(s), including 1 duplicate planned-ID row(s), were counted in the audit but were not analyzed as repetitions.

Provider-specific retry deviations (failed attempts remain retries, not repetitions):

- kimi: one of 270 planned Kimi calls: the planned call had one SSL EOF attempt and one response that ended at the 8192-token provider cap before forming complete schema-valid JSON; retry cap 8192->16384.

## Repeatability results

Across the 54 provider-by-dimension cells, single-call absolute-agreement ICC(A,1) ranged from 0.187 to 0.869 (median 0.773). Mean pairwise exact agreement ranged from 0.407 to 0.922.

For the fixed nine-judge arithmetic mean, replicate ICC(A,1) across the six dimensions ranged from 0.885 to 0.939; the median was 0.931. Dimension-specific mean, median, trimmed-mean, and no-self estimates are all retained in the tables.

The variance table separates fixed-snapshot judge disagreement from fresh-call instability using the balanced 90x9x3 design after domain/task residualization.

## Interpretation boundary

These statistics estimate repeat-call stability for the frozen prompt, model identifiers, and 90-stimulus sample. They do not establish human validity, stability after provider model updates, or calibration to reader outcomes.
