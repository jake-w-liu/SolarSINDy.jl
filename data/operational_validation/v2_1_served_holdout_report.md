# Complete-hour served-stack V2.1 chronological holdout

The locked point calibration and conformal sidecar are evaluated after the complete-hour implementation of the V2.1 tail and safeguards. Fractional subhourly live windows are not reconstructed. Static conformal widths are shifted to the served center; no holdout residual updates the interval.

Rows: 135817. Served RMSE: 7.534 nT. Pooled served coverage: 0.866 (117575/135817), against the declared 0.85 promotion floor and 0.90 nominal target.

| cohort | lead [h] | n | served RMSE [nT] | served coverage | frozen-tail coverage |
|---|---:|---:|---:|---:|---:|
| overall | 0 | 135817 | 7.534 | 0.866 | 0.861 |
| lead_1 | 1 | 22639 | 3.512 | 0.890 | 0.856 |
| lead_2 | 2 | 22638 | 5.396 | 0.881 | 0.870 |
| lead_3 | 3 | 22637 | 6.777 | 0.874 | 0.867 |
| lead_4 | 4 | 22636 | 7.826 | 0.865 | 0.864 |
| lead_6 | 6 | 22634 | 9.408 | 0.847 | 0.854 |
| lead_7 | 7 | 22633 | 10.170 | 0.838 | 0.852 |
| quiet | 0 | 134491 | 7.060 | 0.870 | 0.864 |
| quiet_lead_1 | 1 | 22418 | 3.433 | 0.891 | 0.857 |
| quiet_lead_2 | 2 | 22417 | 5.197 | 0.883 | 0.872 |
| quiet_lead_3 | 3 | 22416 | 6.438 | 0.878 | 0.870 |
| quiet_lead_4 | 4 | 22415 | 7.365 | 0.870 | 0.869 |
| quiet_lead_6 | 6 | 22413 | 8.741 | 0.854 | 0.858 |
| quiet_lead_7 | 7 | 22412 | 9.390 | 0.844 | 0.858 |
| storm | 0 | 1326 | 27.542 | 0.416 | 0.516 |
| storm_lead_1 | 1 | 221 | 8.273 | 0.751 | 0.769 |
| storm_lead_2 | 2 | 221 | 15.577 | 0.638 | 0.697 |
| storm_lead_3 | 3 | 221 | 22.362 | 0.471 | 0.548 |
| storm_lead_4 | 4 | 221 | 27.798 | 0.330 | 0.421 |
| storm_lead_6 | 6 | 221 | 36.268 | 0.163 | 0.357 |
| storm_lead_7 | 7 | 221 | 40.645 | 0.140 | 0.303 |

The pooled floor is the deployment gate. Lead- and activity-stratified coverage is reported as a limitation diagnostic rather than silently promoted to a separate pass criterion.

Every issuable model step (1;2;3;4;6;7) is present in the chronological holdout; the minimum lead-specific coverage is 0.838.

Audit identity: `v2.1`, 20 candidates / 11 active terms.
