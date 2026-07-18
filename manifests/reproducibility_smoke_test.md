# Reproducibility smoke test

**Status: PASS**

An isolated temporary copy of this package successfully ran the byte-exact formal alignment and reliability scripts. The alignment smoke test read 5,364 human rows, 210 crosswalk rows, 810 equivalence rows, and the complete 7,290-row nine-judge API matrix; its internal QA passed and `final_primary` was byte-identical to the released cleaned input. The reliability smoke test also passed and produced the expected 54 human-result, 540 panel-curve, 24 API-result, and 216 ensemble-curve rows.

The test deliberately used 100 bootstrap/permutation repetitions and quick mode to validate portability and input contracts. It does not replace the frozen formal 5,000/10,000/full-enumeration results in `analysis/`. The test ran only in the system temporary directory, did not overwrite this package, and the temporary clone was deleted after success.

The public cleaned-only entry point was then run against the released tables. It reproduced the A/B/C primary omnibus p-value (`0.009668141427595464`), the fixed Q11/Q17 exclusion p-value (`0.003266657160939916`), all eight controlled-case aggregates, and the seven exact-text-case summary within `1e-10` of the frozen outputs. It read no raw workbook or call log.
