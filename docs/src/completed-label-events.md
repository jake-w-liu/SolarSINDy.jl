# Shared completed-label research events

`SolarSINDy.CompletedSourceDBDT.CompletedLabelEvents` sends one ordered source
and completion stream to fixed static, phase, FIFO, early and maturity consumers.
It is a synthetic research adapter. Every output context identifies the explicit
stream/protocol, point SHA, policy and layout; it also states
`source_bound_provenance_verified=false`. Labels and a declared SHA do not prove
actual source-bound refitting or receipt vintage.

```@example completed_labels
using Dates, SolarSINDy
using SolarSINDy.CompletedSourceDBDT.CompletedLabelEvents
D = SolarSINDy.CompletedSourceDBDT
issue = DateTime(2024,1,1)
ground = Dict(issue+Minute(k)=>(0.0,0.0) for k in -40:31)
prepared = D.prepare_schedule(ground,[issue])
cal = D.CompletedCalibration(zeros(9),fill(issue-Millisecond(1),9);
    model_schema="dbdt_completed_source_point_v1",
    point_model_sha256=repeat("1",64),purpose=:synthetic)
receipts = D.materialize_receipts(ground,issue+Minute(1):Minute(1):issue+Minute(30);
    asof=issue+Minute(31))
result = replay_shared(prepared,[0.0],[1.0],receipts.batches,cal;
    subscribers=(SubscriberSpec(:phase,:phase),SubscriberSpec(:early,:early)),
    stream_id="synthetic-zero-ground",protocol_id="manual-shared-v1",
    asof=issue+Minute(31))
@assert result.source.attempted == result.source.completed_targets == 1
@assert all(c.binary_resolved == c.completed_scores == 1 for c in result.subscribers)
@assert result.study_pass && !result.context.source_bound_provenance_verified
nothing # hide
```

Supply complete same-time batches to `prepare_time`, then call `commit_time!`.
`replay_shared` requires one-based prepared, point and scale arrays; offset axes
are rejected before source-batch access to preserve positional alignment.
Preparation validates the whole batch and every projected finite residual
without changing the owner. Commit binds the exact owner, context and revision
and replaces one immutable snapshot. Receipts and completed targets precede an
issue at the same clock. Repeating the identical last transaction emits nothing;
stale or cross-owner/context plans reject atomically. Single-owner use is
required; arbitrary callbacks and concurrent mutation are unsupported.

A feature-eligible attempt always registers its source window. Its30 future
rate centers require31 ground vectors. With default policy the last center is
ready after30 minutes45 seconds and feedback is scheduled at31 minutes. Receipts
at scheduled feedback plus grace are accepted; censoring occurs strictly after
that clock. A censored window never reopens. Complete scores require all30 rates
and readiness; only early mode may apply an already-proved binary miss before
the maximum is known. A completed label never reapplies that early binary miss.
Source zeros canonicalize to positive zero after scalar validation, while source
events retain converted raw bits and supplied identities separately.

Static, phase and FIFO retain fixed calibration scores. Early and maturity
retain their declared rolling-score behavior. FIFO updates follow completion
order; simultaneous completions use issue/id order. These shared events do not
expand a controller's mathematical assumptions or coverage scope.

The default phase selector `:policy` uses the policy-derived period (31 minutes
by default). If that phase's prior target remains pending, the attempted issue
records an unavailable bound and retains its source window. A mandatory
comparator's unavailability makes `study_pass=false` permanently. The alternative
`:safe` fixes a terminal-safe period from declared grace before replay:
policy period + floor(grace/60000) +1. It is never selected from outcomes.
Terminalization can mean censoring and does not establish coverage for unknowns.

Certificates distinguish attempted source windows, bounds actually issued,
unavailable attempts, completed targets and classifiable binary outcomes. A
known target for an unavailable bound cannot become a hit/miss retrospectively.
The all-attempted enclosure retains unavailable rows as unknown. The separately
labeled issued-bound enclosure cannot rescue a failed study or establish a
matched skill verdict. `scientific_coverage_qualified=false` is explicit.

Live windows, source retention and rolling scores are bounded by the declared
policy/grace/capacity. Returned full event logs grow with stream length outside
the live state. Warmup membership requires both readiness and completion
strictly before start and matching point/policy/layout/phase identity. This
implements membership only; publisher-UP numerical state is a separate stage.

The standalone `examples/completed_label_events_synthetic.jl` exercises a44-issue
ground step and1000-issue state bound and is registered in the package experiment
entry. Actual data, refits, fit/export provenance, observational skill, full
package/experiment/harness checks and native platform CI remain separate gates.

```@docs
SolarSINDy.CompletedSourceDBDT.CompletedLabelEvents
SolarSINDy.CompletedSourceDBDT.CompletedLabelEvents.SubscriberSpec
SolarSINDy.CompletedSourceDBDT.CompletedLabelEvents.IssueRequest
SolarSINDy.CompletedSourceDBDT.CompletedLabelEvents.EventReplay
SolarSINDy.CompletedSourceDBDT.CompletedLabelEvents.prepare_time
SolarSINDy.CompletedSourceDBDT.CompletedLabelEvents.commit_time!
SolarSINDy.CompletedSourceDBDT.CompletedLabelEvents.source_certificate
SolarSINDy.CompletedSourceDBDT.CompletedLabelEvents.subscriber_certificate
SolarSINDy.CompletedSourceDBDT.CompletedLabelEvents.safe_terminal_period
SolarSINDy.CompletedSourceDBDT.CompletedLabelEvents.warmup_membership
SolarSINDy.CompletedSourceDBDT.CompletedLabelEvents.replay_shared
```
