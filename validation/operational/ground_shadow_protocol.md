# Ground forecast receipt-time protocol — 2026-09-19

This is a separately identified experimental CMO shadow. It does not change
the served Dst model, A3 record, FRD forecast decision, or operational alerts.
The user approved this experiment on 2026-09-19. No retrospective row counts
as a prospective issuance.

## Frozen model and target

Use the CMO `log_features_ridge` coefficients and 1,440 calibration seeds from
the already evaluated artifact with SHA-256
`a7386f50ea3e5fe920e224fb56b5040b9a4e7c188b9e54ec68559d0cabf2c47d`.
No coefficient, feature, quantile level, or history length is selected again.
The four features are current, mean, maximum, and sample standard deviation
of 30 consecutive one-minute horizontal derivatives, in nT/min. Apply log1p,
the frozen standardization and linear coefficients, then max(0, expm1(z)).

Live input is CMO adjusted X/Y only. A missing, malformed, non-finite, wrongly
identified, future-dated, or older-than-ten-minute input is unavailable, not
zero; neither another station nor the variation product replaces it. Preserve
the exact response and receipt completion time before issuing a forecast.

Let s be the first minute boundary strictly after the completed forecast
calculation and let e=s+30 minutes. The target is the maximum of the 30
one-minute derivatives in (s,e]. Thus both endpoints of every target derivative
are future at issuance, despite input publication delay. Record the actual
anchor lag and target lead. This differs from pretending issuance occurred at
the latest observation timestamp.
The target-start minus observation-anchor delay must also be at most ten
minutes; the next-minute rounding does not extend that limit.

The upper estimate uses the finite-sample rank ceil(0.90*(n+1)), capped at n,
of the last 1,440 eligible normalized residuals. A residual enters only after
all target observations have actually been received. Seed only from the frozen
2017 calibration sample; reset after a collection gap longer than one day.
The normalization is 1+the forecast's own past-30-minute maximum. Missing
outcomes never become calibration feedback. Scores and forecasts are immutable.

## Delay check before launch

Before inspecting results, fix observation-to-target-start delays to 0, 5,
and 10 minutes on each of the archived 2018, 2021, 2024, and September 2026
CMO adjusted series. Keep complete windows only and record missing support.
Compare the frozen candidate against the same-row past-30-minute maximum and
both frozen training climatologies in native and log RMSE. Simulate delayed
feedback conservatively using the same observation publication delay.

Every period and delay must retain the unchanged native/log accuracy and
0.88–0.92 upper-coverage screens. These already inspected periods are
development stress evidence, not new independent confirmation. If a delay
fails, preserve the result and do not declare the receipt-time model qualified
or change the admissible delay in response to the outcome.

## Prospective assessment

Freeze the deployed artifact digest and start receipt. Issue at most once per
minute, persist raw receipts and unavailable attempts, and score the first
complete received outcome window. No backfill, duplicate retry selection,
product mixing, model retuning, or automatic promotion is permitted.

Daily reports separate process health, data availability, score maturity,
point accuracy, and upper coverage. The initial review requires 30 consecutive
collection days and at least 2,000 matured forecasts, pooled coverage
0.88–0.92, each complete seven-day period at least 0.85, and native/log RMSE no
worse than the strongest of the three matched controls. These are diagnostic
advancement screens, not storm certification or a theorem. Overlapping targets
require UTC-day-block uncertainty and a separately approved event protocol
before any storm claim or operational promotion.

## Independent checks

Hand-computed X/Y differences and sample variance; high-precision frozen-model
arithmetic; exact sorted quantiles; future-outcome mutation; delayed receipt,
missing minute, duplicate timestamp, false metadata, non-finite value, malformed
artifact, restart, conflicting duplicate, concurrent writer, and interrupted
write tests. Include an independent target-window and metric reconstruction.
Require package tests, experiments, the development harness, and the final
GitHub Julia 1.12 jobs before completion.
