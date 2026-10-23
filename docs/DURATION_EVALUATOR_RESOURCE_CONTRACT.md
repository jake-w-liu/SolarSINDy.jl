# Duration evaluator resource contract

The original pre-code mean contract is retained unchanged. A separately reviewed
resource correction supersedes only its requirement that the exhaustive evaluator
body remain byte-identical. The exhaustive affine rollout and every numerical
expression/order/precision/variance formula remain unchanged. The pre-correction
module is preserved verbatim in
`test/duration_forcing/frozen/evaluator_before_barrier_v1.jl`, SHA-256
`f2dfca48267b5feaeb15174f6d1921fb1580d91a9ecc3d2d09896441fbecb558`.

The evaluator copies the original paths into a local Vector for iteration, captures
only its schedule/horizon in internal closures, and attaches evaluation fields
through an ordered NamedTuple assembly. Original fields stay in order;
existing gains/summaries/components slots are overwritten, otherwise appended in
that order. Extra caller fields and repeated evaluation remain supported. Public
paths and components remain immutable tuples with identical values/order. The
evaluated NamedTuple's field types deliberately widen to Any; the affine return
does not change. Unavailable passthrough, validation and source identity checks
remain unchanged.

Direct cold tuple-path public h1:6,N/S calls and exact frozen-reference comparisons
passed on Julia1.12.6/1.13.1 before adaptation. The measured original all-corner
focused runs reached600s watchdogs. Cold timings depend strongly on caller
specialization; no steady-state speed claim follows from compilation ratios.
The correction's warmed synthetic median is approximately1% above the original
full evaluator. The separate mean-only computation reduces point-design work;
its benchmark results are reported independently in OPTIMIZATION_REPORT.md.

`test/duration_forcing/cold_public_api.jl` is a standalone regression for a fresh,
bounded Julia process. It passes ordinary public affine output directly to the
public evaluator for every horizon and both starts, with independent exact
Brownian target expectations. Unit compatibility checks retain a labelled Vector
reference of every frozen original branch, exact component/metadata values,
extra-field preservation, overwrites and repeated evaluation. Oracle literals,
tolerances, forcing probabilities and A dependencies are unchanged.

This is a computation/resource correction. It changes no actual-data protocol,
fit, forecasting outcome, model selection budget, manuscript claim or serving
interface. Full software/harness/clean-checkout gates remain separate from these
focused controls.
