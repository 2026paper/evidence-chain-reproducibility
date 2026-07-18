# Provider configuration boundary

The public `judge_configuration_public.json` records the common target/default settings. It must not be read as proof that every API accepted identical literal parameters. `provider_configuration_deviations.csv` is the credential-free recovery of the authoritative provider overrides.

In particular, OpenAI did not accept/send temperature or seed in the recorded adapter; Kimi required temperature 1, an 8,192-token budget and streaming; GLM used `do_sample=false` with a 2,000-token limit; and Qwen used streaming thinking mode. Therefore any methods wording should describe temperature 0 and seed 20260514 as target settings, followed by provider-specific deviations, not as a perfectly uniform intervention.

The private configuration itself is excluded because it contains obsolete absolute paths, endpoints, and credential-environment metadata. No credential value is copied here.
