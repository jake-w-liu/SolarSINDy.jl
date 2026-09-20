# CMO delayed adaptive calibration experiment — 2026-09-20

This offline experiment follows the authorized freshness and calibration
diagnosis. It changes neither the served Dst product nor the A3 record, original
CMO candidate, live dB/dt availability, or manuscript. No live shadow is
authorized by a retrospective result.

## Question and fixed candidate

Test whether feedback about previous upper-bound misses improves the transfer
of the CMO ground-only upper estimate under receipt delay. The diagnosed May
2024 ten-minute-delay shortfall persists after all 1,440 initial calibration
scores have been replaced. This motivates testing adaptation during collection,
not replacing the initial scores or increasing the nominal coverage target.

Keep the original CMO point-model artifact, four features, calibration seed,
normalization by one plus the past-30-minute maximum, 1,440-score window,
missing-window exclusions, and more-than-one-day reset. Its model SHA-256 is
`c1af43235dfdea8525100227a7649535bfc977b43672be5e994310a14cae6cea`.

There is one new candidate and no parameter search. Start alpha at 0.10.
After receiving `m` outcomes, of which `M` exceeded their own immutable issued
upper bounds, set alpha to `0.10 + (0.10*m - M)/6000`. The rate 1/6000 is
the existing ground-study pooled controller's 2017-selected rate, not selected
from the four receipt-delay periods. This is a new combination with rolling
scores and a ground-only point model; the earlier study does not establish
its performance or transfer.

Before each issue, process every newly available outcome in order, at most
once. Outcome receipt is target-end plus the specified publication delay and
must be strictly earlier than the new issue. Use the updated alpha to select
rank ceil((1-alpha)*(n+1)), capped to [1,n], from the retained sorted normalized
scores. Convert with the original native-scale formula. Preserve alpha<=0 as
an infinite upper bound and alpha>=1 as an empty set; do not clip the controller
state or silently replace these cases. Either case fails advancement. Reset
both score history and the controller after a collection gap longer than one
day. Report overlapping outcomes descriptively; no independent-minute claim
or distribution-free conditional guarantee is inferred.

## Fixed evaluation sequence

1. Exercise hand-worked quantiles, strict receipt boundaries, gaps, future-label
   mutations, once-only feedback, and finite/full/empty-set handling.
2. Evaluate once on the twelve preserved period/delay panels: 2018, 2021,
   2024 and September 2026, each at 0, 5 and 10 minutes. These panels have
   already been inspected and are development stress evidence. Do not tune
   after reading this result. Keep the original candidate on identical rows.
3. If and only if every original screen passes, fetch the complete May months
   of 2019, 2020, 2022 and 2025 as new retrospective transfer panels, always
   CMO adjusted X/Y at one-minute cadence. These years are absent from the
   original fitting dataset. Freeze raw receipts before scoring, report
   missing support, and use the same three delays and comparisons. A missing
   or incomplete source is not a pass. No alternate month is selected in
   response to data or outcomes.

Every period/delay must retain native and log RMSE no worse than the strongest
of the original three same-row controls, upper coverage 0.88–0.92, and finite
nonempty upper estimates. Also report per-day coverage, mean upper estimate,
and 0.90 upper pinball loss against the original candidate. Reject if pooled
same-row pinball loss worsens; do not buy a coverage pass by unbounded intervals.
The original 30-day, 2,000-matured-forecast, seven-day, uncertainty and comparison
requirements in `ground_shadow_protocol.md` remain unchanged for any later
prospective proposal. A new shadow identity/start receipt and a separate
deployment decision are still required.

## Verification and stopping rule

Independently reconstruct all issued alpha values, quantile ranks, received
history counts, bounds, metrics and the decision. Hash inputs, model, protocol
and executable source. Run full package tests, experiments, the development
harness and exact-revision CI. Preserve failed checks and the original model.
If any development screen fails, end this candidate; do not fetch the new
transfer panels, alter its rate, change the window or relax a threshold.

## A3 boundary

The interval-only, point-offset and causal expert-weight studies have already
closed without a qualifying successor. The new A3 diagnosis updates the
descriptive horizon/day/anchor-lag evidence only. It does not reopen those
candidate grids, reset the prospective clock, erase any of the 48 historical
fallback findings, or authorize a serving change. A different Dst model needs
a separate model-development proposal, not another width or offset retuning.
