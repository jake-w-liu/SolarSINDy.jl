# Complete-hour served-stack V2.1 chronological holdout

The locked point calibration and conformal sidecar are evaluated after the complete-hour implementation of the V2.1 tail and safeguards. Fractional subhourly live windows are not reconstructed. Static conformal widths are shifted to the served center; no holdout residual updates the interval.

Rows: 90400. Served RMSE: 6.638 nT. Pooled served coverage: 0.871 (78766/90400), against the declared 0.85 promotion floor and 0.90 nominal target.

| cohort | lead [h] | n | served RMSE [nT] | served coverage | frozen-tail coverage |
|---|---:|---:|---:|---:|---:|
| overall | 0 | 90400 | 6.638 | 0.871 | 0.866 |
| lead_1 | 1 | 22602 | 3.418 | 0.886 | 0.872 |
| lead_2 | 2 | 22601 | 5.386 | 0.879 | 0.871 |
| lead_3 | 3 | 22600 | 6.792 | 0.871 | 0.865 |
| lead_6 | 6 | 22597 | 9.457 | 0.849 | 0.854 |
| quiet | 0 | 89516 | 6.256 | 0.875 | 0.868 |
| quiet_lead_1 | 1 | 22381 | 3.334 | 0.887 | 0.873 |
| quiet_lead_2 | 2 | 22380 | 5.185 | 0.882 | 0.873 |
| quiet_lead_3 | 3 | 22379 | 6.443 | 0.875 | 0.868 |
| quiet_lead_6 | 6 | 22376 | 8.778 | 0.856 | 0.859 |
| storm | 0 | 884 | 23.293 | 0.498 | 0.589 |
| storm_lead_1 | 1 | 221 | 8.341 | 0.733 | 0.769 |
| storm_lead_2 | 2 | 221 | 15.621 | 0.643 | 0.697 |
| storm_lead_3 | 3 | 221 | 22.661 | 0.462 | 0.552 |
| storm_lead_6 | 6 | 221 | 36.650 | 0.154 | 0.339 |

The pooled floor is the deployment gate. Lead- and activity-stratified coverage is reported as a limitation diagnostic rather than silently promoted to a separate pass criterion.

Audit identity: `v2.1`, 20 candidates / 11 active terms.
