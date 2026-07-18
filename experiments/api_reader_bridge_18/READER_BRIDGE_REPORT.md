# Reader bridge: exact displayed texts scored by the nine-judge API panel

## Design and estimand

All nine API judges scored the 18 texts that readers actually saw (six concepts by three randomized versions). Reader answers were first reduced to concept-by-version cell estimates. The association analysis therefore has 18 text cells in six concept blocks; it does not treat 1,068 participant-concept observations as independent texts.

The analysis reports a fixed two-endpoint primary set: (i) panel mean quality versus paired reader accuracy change and (ii) panel mean misleading-risk versus paired reader accuracy change. Two-sided exact tests enumerate all 6^6 = 46,656 independent within-concept version-label permutations. Confidence intervals use a hierarchical bootstrap that resamples both participant clusters and concept blocks.

## Primary results

Panel quality was positively associated with reader accuracy change after within-concept centering (r=0.670, hierarchical 95% CI [-0.140, 0.893], exact two-sided p=0.0277, Holm p=0.0554).

Panel misleading-risk was negatively associated with reader accuracy change (r=-0.580, hierarchical 95% CI [-0.831, 0.206], exact two-sided p=0.0552, Holm p=0.0554).

## Fixed sensitivity and descriptive version pattern

The already-documented content-parallelism sensitivity excludes the adaptation pre/post pair because its two questions are scientifically related but not equivalent. The corresponding quality association was r=0.702 (exact p=0.0419), and the risk association was r=-0.575 (exact p=0.0947).

Version-level means are descriptive because there are only three version bundles:

| Version | API quality | API risk | Pre accuracy | Post accuracy | Accuracy change |
|---|---:|---:|---:|---:|---:|
| A | 4.081 | 1.741 | 0.900 | 0.956 | 0.056 |
| B | 4.563 | 1.241 | 0.811 | 0.919 | 0.108 |
| C | 4.696 | 1.222 | 0.856 | 0.980 | 0.124 |

## Boundary of the claim

This closes the previous text-identity gap: the API panel and readers are now linked through the same 18 visible texts. It is still a small, purposively selected six-concept bridge, not a held-out calibration set or evidence of generalization to new concepts. The absence of a no-exposure retest arm also means accuracy change cannot be interpreted as a pure causal learning effect.

## Quality assurance

- API rows: 162 (9 providers x 18 texts); parse success: 162/162.
- Reader data: 178 responses, 167 participant clusters, 1068 paired participant-concept rows.
- Analysis cells: 18 texts in 6 concept blocks.
