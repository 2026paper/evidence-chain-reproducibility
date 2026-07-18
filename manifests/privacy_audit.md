# Privacy audit

**Overall status: PASS**

- Public individual files: 3
- Non-reversible random rekey: PASS
- Row-order shuffle: PASS
- Exact duration removed/coarsened: PASS
- Background quasi-identifiers removed: PASS
- Background cells with n<5 released: none
- Raw workbooks/documents: none
- Raw API responses, call logs, complete payloads, `.env`, or keys: none
- Complete scientific instrument/stimulus text: released separately in `materials/`; no participant responses are present there
- Hash-only analysis crosswalks: no complete text fields; SHA-256 and character counts only
- Absolute local path scan: 0 hit(s)
- Literal credential-signature scan: 0 hit(s)

The same person cannot be linked across the human-review, A/B/C, and difference-survey releases because each study uses an independent random namespace and no mapping is retained. The human-review domain/wave cells are design strata, not participant background fields; they remain necessary for the scientific analysis.
