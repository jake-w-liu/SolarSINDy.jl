# Matched-information GBM initial role: material feature proposal

Prepared for root technical review before production code or fits. This selects
one concrete ordered342-column design. It uses the maintained29-column direct
GBM design plus the maintained136-column seventeen-channel raw-lag design,
their165 missing flags, and12 checked age/run fields. It retains originalGBM29
as its own diagnostic. The mandatory validation-selected strongest full-channel
physics/recipe/raw-lag comparator and every other frozen baseline remain required.

## Ordered feature blocks

Columns1:29 are the exact source `direct_gbm_feature_names()` order, prefixed
`gbm29/`. This includes the existing south_run capped12; it is not a new field.
Columns30:165 use SanduFullChannelFeatures.RAW_NAMES in exact17-channel order,
then lags(0,1,2,3,4,8,12,24) in that nested order, prefixed`raw17/`.
The complete raw lag name is the source name followed by`_raw_lag_h`.
Columns166:330 are the corresponding165 pre-imputation nonfinite/missing flags
in the same order, named`missing/<base name>`. NaN encodes missing; infinity
and invalid physical/source inputs are rejected before design, not repaired.
Columns331:342 are the12 names in ORDERED_FEATURE_SCHEMA_PROPOSED_V1.json.

All raw cells are first censored by the maintained source fill thresholds and
masked unless both completion and declared receipt are<=issue. The raw-lag
block uses those real tags. Raw Dst is not forward-filled or shifted inside
that block. Full-channel engineered173-column physics designs remain separate
mandatory model families, evaluated and selected on their own frozen validation
panels. Their original causal imputation/recipe controls remain intact.

The29-column block uses the qualified source13 `gbm29_design` convention:
drivers retain their raw tags; Dst at virtual frame label t is the observed
raw mean tagged t-1-L. At issue I it therefore uses raw anchor I-1-L, while
its Dst lag j uses I-1-L-j. Materialize the virtual issue row even on a
truncated prefix; virtual driver cells remain missing and are not queried.
L1 is a separately fitted changed-prefix model, not a relabelledL0 artifact.
Retain the source's complete-history29-block behavior; an unavailable29 block
gets missing flags and finite encoding without dropping its scoreable issue.

## Observation/run fields and clocks

Latest observation age is (I-anchor_complete)/Hour=L in the assumed receipt
L0/L1 archives. Required anchor must be the exact raw tag I-1-L, finite and
received by I; never choose an older finite anchor. Target tag is I+h-1,
completion I+h, label receipt I+h+L. No target/required anchor is imputed.

Run fields use the exact B archival wind trace on completed/received rows,
including its Bz-sign classification and q-finite requirement. Bz<0 is South
even when thresholded q==0. A missing/gapped row resets the run. An initial
run's true age is unknown until a confirmed sign flip or49 same-state hours.
Known age caps49; North duration is retained too. `trace_left_censored` says
the current trace run began without a contiguous known opposite state; it can
remain1 when the49+ bin becomes known. Observed run is separately capped49.

The latest eligible trace is retained when the expected latest driver is late;
the unavailable flag and latest-wind age distinguish it from a fresh current
driver. A latest eligible missing row has traceN=S=age=run=0. No eligible row
uses finite sentinel wind age0 with latest-driver-unavailable1 and all trace
fields0. Unknown-age is1 only for a knownN/S trace with age0. Known-age is1
only for a knownN/S trace with positive age. Age49plus is1 iff known age49.
These are clock/status representations of B's source information, not a new
lookahead, new hazard feature or inferred state from q>0.

## Finite encoding and fitting

For each of165 base columns, fit an arithmetic mean from its finite values on
that cell's eligible training rows only; an all-missing column uses0. Replace
only missing design entries by that stored mean. Missing flags retain their
pre-imputation values. Store count/mean for every column. Fit the maintained
explicit training mean/sample-standard-deviation scaler after this encoding,
with constant/one-row column scale1. Persist and independently replay every
parameter; require finite encoded/scaled values and exact ordered schema.
No validation/test row participates in either imputation or scaling.

GBM learns the target increment y-anchor. Prediction is anchor+increment;
the common persistence fallback and per-lead scoreability rules are explicit.
All four cells are L0h4,L0h6,L1h4,L1h6, with separate per-cell fit/scalers.
Use the exact original1976-start source, pre2020 training and2020-22 validation,
with168h target completion AND label receipt embargo. Each lead retains its
own full scoreable cohort. The union is used for shared execution/joining;
neither h4 evaluation nor h4 baseline fit is truncated to an h6 intersection.
Final pre2023 refits and exposed development need a later permit.

Tune exactly the eight depth(4,6) × rounds(200,400) × bins(64,255) settings,
depth outermost, bins innermost, seed22022026. Reuse pinned EvoTrees0.18.7 CPU
MSE wrapper settings eta0.05,min_weight64,unit sampling and existing guards.
This is32 initial fits. Choose strict first exact minimum validation pooled
RMSE per cell. Save all8 candidates, masks, scales, native receipts, predictions
and fallback rows. Never tune from test/development, prune failed rows or use
an epsilon tie promotion. The primary-role reader must independently verify
this new schema/grid/source proof; changing a role string cannot admit old29.

## Independent oracles before code

Freeze complete342-coordinate literals at L0/L1 for a rational/constant-driver
fixture, then missing current and delayed-receipt variants; verify every column
and mask, not just a digest. Keep rawzero/missing-zero distinct. Freeze initial
unknown48,49+, knownNorth, zero-qSouth, missing-gap and stale-driver traces.
Check train-only imputer/scaler literals, all-missing and constant columns,
strict first ties, separate lead cohorts including the two h4-only tail keys,
and handwritten trees/bin/Float32 fixtures splitting every new/high coordinate.
Mutation cases must change an actual decision/output: swapped feature order,
omitted missing flags, q>0 state, stale anchor substitution, wrong receipt boundary,
L0 relabelledL1, full-archive scaling, omitted grid member and rehashed model
payload corruption. Genuine tiny synthetic fits/save-reload and independent
reader replay follow, before any actual-data permit.

Execution should reuse bounded26-hour raw windows and a serial B-compatible
trace, not clean/hash the full prefix per issue. Profile native memory and CPU
at the frozen source bound before proposing actual32-fit resources. No model
implementation, native fit, actual score or scientific verdict is in this packet.
