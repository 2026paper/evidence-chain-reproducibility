# Independent release audit

**Status: PASS**

The privacy-transformed files were compared read-only with their formal sources. After excluding only the declared removed fields and new random identifiers, every within-participant human-review rating signature, every A/B/C response signature, and every difference-survey respondent signature was preserved. Source and release identifiers were disjoint, and all release identifiers matched their cryptographically randomized namespace.

All 34 public crosswalk rows were independently rehashed: every removed text cell's SHA-256 and UTF-8 character count matched the withheld formal source. Exact duration and participant-background columns were absent. Sparse background result sections were also absent, so no background cell with n<5 is reconstructible from the package.

Observed release counts were 5,364 human-review rows/104 clusters, 2,136 A/B/C rows/178 response instances/167 clusters, and 480 difference-survey rows/60 respondents.
