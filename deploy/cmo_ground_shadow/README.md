# CMO receipt-time candidate — 2026-09-19

This is a frozen research candidate, not an enabled forecast service.
`decision.json` records its decision and digests; `metrics.csv` records all
four historical periods and three predeclared delays.

The candidate beats the strongest matched control in native and log RMSE in
all twelve cases. Its May 2024 ten-minute-delay upper coverage is 0.879595,
below the unchanged 0.88 floor. It therefore does not start prospective
collection under the [receipt-time protocol](../../validation/operational/ground_shadow_protocol.md).
The original A3 and FRD decisions are unaffected.

To reproduce with the preserved local USGS receipts and training data:

```sh
julia --project=. validation/operational/ground_delay_check.jl REMEDIATION_ARCHIVE NEW_OUTPUT_DIR
julia --project=. validation/operational/check_ground_delay_output.jl REMEDIATION_ARCHIVE NEW_OUTPUT_DIR
```

The checker independently reconstructs targets from the earlier checked
zero-delay panels, delayed feedback, quantile ranks, baseline statistics,
and the resulting decision. Raw inputs and large prediction tables remain
outside git; their hashes are retained with the local receipts.
