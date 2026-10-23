# Archival current-state bridge

`SolarSINDy.DurationCurrentStateStream` provides a bounded, one-pass current-state
consumer for explicitly assumed archival hourly receipts. It also works with the
optional `issue_sink` keyword on `IntervalMeanForecaster.stream_basis`, allowing
the existing A historical filter to supply detached issue states. These are
research primitives; they supply no gain/hazard fitting, calibration, complete B
executor, prediction artifact or operational service.

## Receipt admission and exact state identity

Pass an ordinary one-based `AbstractVector` of at most500000 immutable
`DurationConditionedForcing.WindHour` values, with strictly increasing unique
tags and nondecreasing receipts in tag order. Gaps and missing wind values are
allowed. Equal-receipt rows enter together in tag order. Completion/receipt
eligibility uses inclusive equality. Finite North wind requires q0; South is
defined by Bz<0 even when its already thresholded q is0.

This restriction is explicit. General public `current_state` also accepts
nonmonotone receipts; a newly received old tag can then insert into an earlier
position of its tag-ordered digest. The archival tracker rejects such a source
before its callback and does not claim live receipt/revision support. Public
`current_state`, forecast boundary checks and old model/artifact guards retain
their definitions.

On admitted inputs, each `advance(issue)` returns exactly the public
`current_state(rows,issue)` NamedTuple, including field order, state, capped age,
q, eligible-prefix digest and state ID. Issues are nondecreasing; equal issues
are idempotent. Missing rows reset the trace, and a gap gives initially unknown
age. A witnessed state flip establishes age1. An initially unknown same-state
run becomes known capped49+ at its49th contiguous hour. The latest eligible tag
must equal issue minus1hour; an older available driver cannot substitute for it.

Prefix identity remains SHA256 of the exact public ASCII row serialization:
tag/completion/receipt integer milliseconds, unsigned decimal Float64 bit
tokens, separators and newline. NaN is `missing`; signed-zero bits remain
distinct. It is unrelated to A's separate consumed-input hash chain.

## Scoped ownership and checkpoints

```julia
using SolarSINDy, Dates, SHA
D = SolarSINDy.DurationConditionedForcing
C = SolarSINDy.DurationCurrentStateStream
t = DateTime(2000)
rows = [D.WindHour(t, t+Hour(1), t+Hour(1), 1.0, 0.0),
        D.WindHour(t+Hour(1), t+Hour(2), t+Hour(2), -1.0, 0.0)]
source = bytes2hex(sha256("declared-generated-source"))
result = C.with_current_state_tracker(rows; source_sha256=source) do (advance, snapshot)
    first_state = advance(t+Hour(1))
    current = advance(t+Hour(2))
    @assert first_state.reason == :unknown_initial_age
    @assert current.state == :S && current.age == 1 && current.q == 0.0
    current
end
```

The source vector is privately copied once. It is held as a concrete vector,
never a source-length-specialized tuple. The two closures belong to the serial
calling task and expire when the scope ends. Cross-task, reentrant, expired,
backward-clock and overflowing-clock calls reject before tracker mutation.
Each advance uses a private candidate hash/trace and commits after constructing
the complete result. The mutable source vector and SHA context are not returned
in checkpoint data.

`snapshot()` returns a distinct immutable `CurrentStateCheckpoint`. Resume with
the `checkpoint` keyword validates source/configuration/code identity and
reconstructs its consumed prefix once. Full wind-sequence identity is pinned:
appending or changing even a future row rejects that checkpoint. This version
provides in-memory checkpoints and no portable disk codec, Julia Serialization
or executable artifact. Checkpoint provenance has a caller-declared source SHA,
an independently computed wind-sequence SHA and implementation hashes. The
declared SHA does not verify raw-source attribution.

Loaded dependencies are checked at scope entry and before successful return,
including after a callback exception. Results observed inside the callback are
provisional until that final check. Caller side effects and previously published
data cannot be rolled back by this primitive. A changed dependency prevents a
successful wrapper result. Byte-identical relocation does not enter IDs.

## Detached A issue states

`IntervalMeanForecaster.stream_basis(...; issue_sink=callback)` calls the optional
callback once per processed hour after historical filtering and before future
continuation. The default `nothing` preserves ordinary A behavior. The payload
includes issue/configuration clocks, current driver, availability/reset metadata,
source/prefix identity, a zero-gain issue checkpoint, a fresh2x2 historical gain
basis and a separate resume checkpoint. It includes no targets or scoreability.

L0's issue state is its sealed assimilated state. L1's issue state advances the
current driver without its unrevealed Dst observation; its resume checkpoint
retains the earlier sealed state and pending driver. Warmup/anchor-missing hours
can supply a valid historical issue state while keeping A's model gate false.
An invalid current driver supplies no issue state/basis and retains A's existing
reset precedence. Mutating a supplied basis or collected checkpoint arrays does
not affect ordinary A rows, later payloads or returned checkpoints. A callback
exception aborts before that hour's row is published; earlier chunks remain.

The two gates remain separate: A model availability and B current-state
availability do not themselves authorize a fitted forecast. Historical
source-bound A/B artifacts keep strict source checks. Descendant implementation
bytes have new identities; old artifacts are never relabeled to match them.

The synthetic example `examples/b_stream_bridge_synthetic.jl` connects the two
primitives without fitting. Independent literal tests cover prefix/state hashes
and integrated-OU issue states, while a separate A187 source fixture checks old
A numeric outputs by bits. Generated resource observations are software checks,
not forecasting skill, publication timing or calibrated probability evidence.

```@docs
SolarSINDy.DurationCurrentStateStream
SolarSINDy.DurationCurrentStateStream.CurrentStateCheckpoint
SolarSINDy.DurationCurrentStateStream.with_current_state_tracker
```
