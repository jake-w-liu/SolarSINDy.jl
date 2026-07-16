# Storm-time replay — V2 vs baselines (1/2/3/6 h ahead)

Causal replay of the strongest 2023–2024 storms through the locked `replay_recent_table` engine, scored against persistence / Burton / Burton-full / O'Brien-McPherron. Built from the OMNI archive; the locked live record is untouched. Rows filtered to each storm's issue-time window.

## May 2024 (Gannon, G5)

**Horizon 1 h** (v2 90% CI coverage 0.80):

| model | RMSE [nT] | MAE [nT] | n |
|---|---|---|---|
| persistence | 27.93 | 15.13 | 85 |
| O'Brien-McPherron | 28.88 | 15.63 | 85 |
| V2 | 30.48 | 16.55 | 85 |
| SINDy v1 | 30.53 | 16.60 | 85 |
| Burton-full | 37.69 | 22.25 | 85 |
| Burton | 37.72 | 22.40 | 85 |

**Horizon 2 h** (v2 90% CI coverage 0.86):

| model | RMSE [nT] | MAE [nT] | n |
|---|---|---|---|
| persistence | 44.36 | 24.16 | 85 |
| O'Brien-McPherron | 45.68 | 24.17 | 85 |
| V2 | 50.14 | 29.51 | 85 |
| SINDy v1 | 50.31 | 29.48 | 85 |
| Burton-full | 61.73 | 39.40 | 85 |
| Burton | 61.77 | 39.69 | 85 |

**Horizon 3 h** (v2 90% CI coverage 0.92):

| model | RMSE [nT] | MAE [nT] | n |
|---|---|---|---|
| persistence | 56.87 | 30.69 | 85 |
| O'Brien-McPherron | 57.20 | 30.81 | 85 |
| V2 | 63.76 | 36.83 | 85 |
| SINDy v1 | 64.13 | 36.94 | 85 |
| Burton | 77.25 | 51.30 | 85 |
| Burton-full | 77.31 | 51.06 | 85 |

**Horizon 6 h** (v2 90% CI coverage 0.94):

| model | RMSE [nT] | MAE [nT] | n |
|---|---|---|---|
| O'Brien-McPherron | 90.57 | 47.03 | 85 |
| persistence | 94.24 | 51.81 | 85 |
| V2 | 109.74 | 63.05 | 85 |
| SINDy v1 | 110.31 | 63.25 | 85 |
| Burton | 125.84 | 83.31 | 85 |
| Burton-full | 126.19 | 82.93 | 85 |

## Oct 2024 (G4)

**Horizon 1 h** (v2 90% CI coverage 0.95):

| model | RMSE [nT] | MAE [nT] | n |
|---|---|---|---|
| V2 | 18.05 | 10.15 | 73 |
| SINDy v1 | 18.21 | 10.47 | 73 |
| persistence | 19.89 | 11.95 | 73 |
| O'Brien-McPherron | 20.49 | 11.05 | 73 |
| Burton | 29.52 | 16.91 | 73 |
| Burton-full | 29.53 | 16.93 | 73 |

**Horizon 2 h** (v2 90% CI coverage 0.96):

| model | RMSE [nT] | MAE [nT] | n |
|---|---|---|---|
| V2 | 31.02 | 18.43 | 73 |
| SINDy v1 | 31.22 | 18.87 | 73 |
| persistence | 34.13 | 20.40 | 73 |
| O'Brien-McPherron | 35.81 | 20.20 | 73 |
| Burton | 54.23 | 31.16 | 73 |
| Burton-full | 54.23 | 31.22 | 73 |

**Horizon 3 h** (v2 90% CI coverage 0.95):

| model | RMSE [nT] | MAE [nT] | n |
|---|---|---|---|
| V2 | 43.89 | 26.14 | 73 |
| SINDy v1 | 44.13 | 26.67 | 73 |
| persistence | 47.03 | 27.89 | 73 |
| O'Brien-McPherron | 48.52 | 26.78 | 73 |
| Burton | 76.40 | 44.38 | 73 |
| Burton-full | 76.41 | 44.51 | 73 |

**Horizon 6 h** (v2 90% CI coverage 0.95):

| model | RMSE [nT] | MAE [nT] | n |
|---|---|---|---|
| O'Brien-McPherron | 76.37 | 45.11 | 73 |
| V2 | 77.04 | 48.05 | 73 |
| SINDy v1 | 77.41 | 48.58 | 73 |
| persistence | 79.01 | 49.99 | 73 |
| Burton | 128.66 | 76.51 | 73 |
| Burton-full | 128.67 | 76.60 | 73 |

## Mar 2023 (G4)

**Horizon 1 h** (v2 90% CI coverage 1.00):

| model | RMSE [nT] | MAE [nT] | n |
|---|---|---|---|
| O'Brien-McPherron | 6.22 | 4.76 | 61 |
| Burton | 7.17 | 5.32 | 61 |
| V2 | 7.19 | 5.30 | 61 |
| Burton-full | 7.19 | 5.37 | 61 |
| SINDy v1 | 7.57 | 5.65 | 61 |
| persistence | 9.39 | 7.03 | 61 |

**Horizon 2 h** (v2 90% CI coverage 1.00):

| model | RMSE [nT] | MAE [nT] | n |
|---|---|---|---|
| O'Brien-McPherron | 9.43 | 7.26 | 61 |
| Burton | 11.95 | 8.92 | 61 |
| Burton-full | 11.97 | 8.99 | 61 |
| V2 | 12.29 | 9.34 | 61 |
| SINDy v1 | 12.73 | 9.68 | 61 |
| persistence | 16.17 | 11.93 | 61 |

**Horizon 3 h** (v2 90% CI coverage 1.00):

| model | RMSE [nT] | MAE [nT] | n |
|---|---|---|---|
| O'Brien-McPherron | 12.83 | 9.73 | 61 |
| Burton | 17.25 | 13.10 | 61 |
| Burton-full | 17.29 | 13.23 | 61 |
| V2 | 17.41 | 13.31 | 61 |
| SINDy v1 | 17.81 | 13.68 | 61 |
| persistence | 22.19 | 16.08 | 61 |

**Horizon 6 h** (v2 90% CI coverage 0.98):

| model | RMSE [nT] | MAE [nT] | n |
|---|---|---|---|
| O'Brien-McPherron | 28.37 | 19.81 | 61 |
| V2 | 34.44 | 25.06 | 61 |
| SINDy v1 | 34.67 | 25.40 | 61 |
| Burton | 38.24 | 27.80 | 61 |
| Burton-full | 38.33 | 28.13 | 61 |
| persistence | 38.38 | 29.00 | 61 |

## POOLED (all storm windows)

**Horizon 1 h** (v2 90% CI coverage 0.90):

| model | RMSE [nT] | MAE [nT] | n |
|---|---|---|---|
| persistence | 21.43 | 11.81 | 219 |
| O'Brien-McPherron | 21.78 | 11.07 | 219 |
| V2 | 21.99 | 11.28 | 219 |
| SINDy v1 | 22.10 | 11.50 | 219 |
| Burton-full | 29.26 | 15.78 | 219 |
| Burton | 29.28 | 15.81 | 219 |

**Horizon 2 h** (v2 90% CI coverage 0.93):

| model | RMSE [nT] | MAE [nT] | n |
|---|---|---|---|
| persistence | 35.00 | 19.50 | 219 |
| O'Brien-McPherron | 35.52 | 18.14 | 219 |
| V2 | 36.59 | 20.20 | 219 |
| SINDy v1 | 36.78 | 20.43 | 219 |
| Burton-full | 49.99 | 28.20 | 219 |
| Burton | 50.01 | 28.27 | 219 |

**Horizon 3 h** (v2 90% CI coverage 0.95):

| model | RMSE [nT] | MAE [nT] | n |
|---|---|---|---|
| O'Brien-McPherron | 45.83 | 23.60 | 219 |
| persistence | 46.15 | 25.69 | 219 |
| V2 | 48.00 | 26.72 | 219 |
| SINDy v1 | 48.31 | 27.03 | 219 |
| Burton | 65.91 | 38.35 | 219 |
| Burton-full | 65.95 | 38.34 | 219 |

**Horizon 6 h** (v2 90% CI coverage 0.95):

| model | RMSE [nT] | MAE [nT] | n |
|---|---|---|---|
| O'Brien-McPherron | 73.16 | 38.81 | 219 |
| persistence | 77.06 | 44.85 | 219 |
| V2 | 83.56 | 47.47 | 219 |
| SINDy v1 | 83.99 | 47.82 | 219 |
| Burton | 109.87 | 65.58 | 219 |
| Burton-full | 110.04 | 65.56 | 219 |

## Rapid main phase (Dst dropping > 15 nT/h)

The pooled windows are recovery/quiet-dominated, where persistence is nearly unbeatable. This subset isolates the rapid main phase — the hours a Dst forecaster exists to serve.

**Horizon 1 h** (n=25): SINDy v1 46.2 · V2 47.0 · persistence 52.6 · O'Brien-McPherron 55.7 · Burton 58.6 · Burton-full 58.6 ·   → v2 vs persistence: v2 better
**Horizon 2 h** (n=22): SINDy v1 82.8 · V2 83.4 · persistence 95.3 · Burton 96.3 · Burton-full 96.4 · O'Brien-McPherron 98.9 ·   → v2 vs persistence: v2 better
**Horizon 3 h** (n=23): SINDy v1 105.4 · V2 105.6 · Burton 119.0 · Burton-full 119.3 · persistence 125.5 · O'Brien-McPherron 126.9 ·   → v2 vs persistence: v2 better
**Horizon 6 h** (n=20): V2 196.1 · SINDy v1 196.2 · Burton 201.1 · Burton-full 202.2 · O'Brien-McPherron 215.8 · persistence 222.9 ·   → v2 vs persistence: v2 better
