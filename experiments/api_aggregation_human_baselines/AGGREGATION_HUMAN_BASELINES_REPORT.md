# Aggregation baselines against the broad human criterion

## Common analysis contract

All 13 methods use the same 178 visible texts (1,068 text-by-dimension rows), 30 domain-concept clusters, controls for domain, task, and dimension, 5,000 domain-stratified concept-cluster bootstrap draws, and 10,000 shared domain-by-task Freedman--Lane permutations. No method-specific row deletion occurs.

Risk is quality-aligned for both humans and APIs as 6 - raw risk. Consequently, higher scores are better on all six dimensions. CCC and MAE are calculated on the original aligned 1--5 scale; beta uses method-by-dimension z scores whose constants are frozen before resampling.

## Complete results

| Method | Beta | 95% cluster CI | FL p | Holm p | Lin CCC | MAE | Spearman | Beta on old mean scale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 4.6 | 0.123 | [0.021, 0.217] | 0.0069 | 0.0276 | 0.047 | 1.398 | 0.123 | 0.074 |
| DeepSeek-V4 Expert Mode | 0.178 | [0.081, 0.290] | 0.0034 | 0.0198 | 0.157 | 0.854 | 0.179 | 0.124 |
| Doubao 2.0 Pro | 0.162 | [0.085, 0.248] | 0.0004 | 0.0028 | 0.119 | 0.982 | 0.134 | 0.108 |
| GLM-5.1 | 0.098 | [-0.002, 0.199] | 0.0242 | 0.0570 | 0.140 | 0.921 | 0.147 | 0.056 |
| GPT-5.5 Thinking | 0.190 | [0.093, 0.290] | 0.0001 | 0.0013 | 0.133 | 0.774 | 0.122 | 0.147 |
| Gemini 3.1 Pro | 0.213 | [0.126, 0.303] | 0.0001 | 0.0013 | 0.172 | 0.726 | 0.185 | 0.184 |
| Kimi K2.6 Thinking | 0.120 | [0.064, 0.184] | 0.0033 | 0.0198 | 0.095 | 0.877 | 0.117 | 0.080 |
| Mimo v2.5 Pro | 0.103 | [0.030, 0.180] | 0.0503 | 0.0570 | 0.085 | 1.011 | 0.134 | 0.053 |
| Qwen3.5 Thinking | 0.140 | [0.022, 0.263] | 0.0190 | 0.0570 | 0.099 | 0.851 | 0.105 | 0.099 |
| Nine-judge mean | 0.232 | [0.133, 0.335] | 0.0002 | 0.0022 | 0.172 | 0.696 | 0.178 | 0.232 |
| Nine-judge median | 0.197 | [0.101, 0.301] | 0.0003 | 0.0027 | 0.152 | 0.802 | 0.157 | 0.154 |
| Nine-judge trimmed mean | 0.233 | [0.135, 0.338] | 0.0002 | 0.0022 | 0.173 | 0.719 | 0.176 | 0.221 |
| No-self mean | 0.210 | [0.113, 0.312] | 0.0003 | 0.0027 | 0.156 | 0.708 | 0.155 | 0.205 |

## Audit of the earlier exploratory values

The unified method-standardized analysis reproduces the single-judge range (0.098--0.213), mean (0.232), trimmed mean (0.233), and no-self mean (0.210).

The earlier median value near .154 is also reproducible, but it is a different scale: beta=0.154 when the median uses the mean panel's frozen API SD. With the same method-specific standardization used for every formal baseline, the median beta is 0.197. The latter is the comparable primary value; the former is retained only for provenance.

## Exploratory paired beta differences

Each interval below is calculated from the draw-by-draw difference between the mean panel beta and the comparator beta under the same 5,000 concept-cluster resamples. These comparisons are exploratory and were not multiplicity-adjusted.

| Comparator | Mean - comparator beta | Paired 95% CI | Crosses zero |
|---|---:|---:|:---:|
| Claude Sonnet 4.6 | 0.109 | [0.043, 0.187] | no |
| DeepSeek-V4 Expert Mode | 0.053 | [-0.020, 0.123] | yes |
| Doubao 2.0 Pro | 0.069 | [0.004, 0.130] | no |
| GLM-5.1 | 0.133 | [0.070, 0.201] | no |
| GPT-5.5 Thinking | 0.041 | [-0.020, 0.098] | yes |
| Gemini 3.1 Pro | 0.019 | [-0.048, 0.093] | yes |
| Kimi K2.6 Thinking | 0.111 | [0.025, 0.198] | no |
| Mimo v2.5 Pro | 0.128 | [0.050, 0.207] | no |
| Qwen3.5 Thinking | 0.092 | [0.025, 0.169] | no |
| Nine-judge median | 0.035 | [-0.003, 0.064] | yes |
| Nine-judge trimmed mean | -0.002 | [-0.016, 0.010] | yes |
| No-self mean | 0.021 | [0.005, 0.038] | no |

## No-self family mapping

| Generator family | Excluded judge provider |
|---|---|
| Alibaba Qwen | qwen |
| Anthropic Claude | anthropic |
| ByteDance Doubao | doubao |
| DeepSeek | deepseek |
| Google Gemini | gemini |
| Mimo | mimo |
| Moonshot Kimi | kimi |
| OpenAI GPT | openai |
| Zhipu GLM | glm |

## Interpretation boundary

These are correlated baseline estimates on the same broad-review criterion. A larger point beta does not by itself prove one aggregation is superior to another; the paired intervals above are exploratory, unadjusted for multiplicity, and do not define a confirmatory superiority test. CCC remains modest because association and raw-score agreement answer different questions.
