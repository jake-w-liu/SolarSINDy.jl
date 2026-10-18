using Test, LinearAlgebra, Dates

const REPLAY = DstIntervalReplay
state_matrix(state) = [state.covariance[i][j] for i in eachindex(state.mean), j in eachindex(state.mean)]
state_mean(state) = collect(state.mean)

function peer_inputs(f; gains=f.gains, observation_records=f.observations)
    initial = REPLAY.Checkpoint(f.origin, f.mean, f.covariance)
    intervals = [REPLAY.ForcingInterval("bin$j", f.origin+Hour(j-1), f.origin+Hour(j),
                                       -gains .* q, f.origin+Hour(j)) for (j,q) in enumerate(f.q)]
    observations = [REPLAY.MeanObservation(o.revision_id,
        "bin$(Int(Dates.value(o.stop-f.origin) ÷ 3_600_000))", o.start, o.stop,
        o.value, o.R, o.receipt, o.verified) for o in observation_records]
    return (; initial, intervals, observations)
end

function peer_replay(f, boundary; gains=f.gains, observation_records=f.observations,
                     asof=f.origin+Hour(boundary)+Millisecond(1), receipt_boundary=:strict)
    inputs = peer_inputs(f; gains, observation_records)
    return REPLAY.replay_asof(inputs.initial, f.lambdas, f.sigmas, inputs.intervals,
                             inputs.observations, f.origin+Hour(boundary); asof, receipt_boundary)
end

@testset "Independent full as-of batch Gaussian replay" begin
    f = fixture()
    for boundary in 1:8
        actual = peer_replay(f, boundary)
        expected = expected_posterior(f, boundary)
        @test compare(state_mean(actual.state), expected.mean)
        @test compare(state_matrix(actual.state), expected.covariance)
        @test actual.state.boundary == f.origin+Hour(boundary)
        @test length(actual.intervals) == boundary
        @test length(actual.selected) == boundary
        selected_ids = [o.id for o in actual.selected if o !== nothing]
        @test sort(selected_ids) == sort(f.selected[boundary])
        @test length(actual.history) == boundary+1
        @test actual.history[1].mean == Tuple(f.mean)
        @test compare(state_matrix(actual.history[1]), f.covariance)
        for bin in 1:boundary
            historical_ids = [id for id in f.selected[boundary] if
                only(filter(o -> o.revision_id==id, f.observations)).stop <= f.origin+Hour(bin)]
            historical = expected_posterior(f, bin; selected_ids=historical_ids)
            @test actual.history[bin+1].boundary == f.origin+Hour(bin)
            @test compare(state_mean(actual.history[bin+1]), historical.mean)
            @test compare(state_matrix(actual.history[bin+1]), historical.covariance)
        end
        inputs = peer_inputs(f)
        permuted = REPLAY.replay_asof(inputs.initial, f.lambdas, f.sigmas,
            reverse(inputs.intervals), reverse(inputs.observations), f.origin+Hour(boundary);
            asof=f.origin+Hour(boundary)+Millisecond(1))
        @test permuted.state.mean == actual.state.mean
        @test permuted.state.covariance == actual.state.covariance
    end
end

@testset "One likelihood per bin and capture-order revisions" begin
    f = fixture()
    original = peer_replay(f, 6)
    duplicated = peer_replay(f, 6; observation_records=[f.observations; f.observations; reverse(f.observations)])
    @test duplicated.state.mean == original.state.mean
    @test duplicated.state.covariance == original.state.covariance
    @test count(!isnothing, duplicated.selected) == 4
    revision = only(filter(o -> o.revision_id=="revision-bin1", f.observations))
    equivalent_later = merge(revision, (revision_id="same-value-later-receipt",
        receipt=f.origin+Hour(5)+Minute(50), verified=f.origin+Hour(5)+Minute(50)+Millisecond(1)))
    repeated = peer_replay(f, 6; observation_records=[f.observations; equivalent_later])
    @test repeated.state.mean == original.state.mean
    @test repeated.state.covariance == original.state.covariance
    @test count(!isnothing, repeated.selected) == 4
    same_capture_equivalent = merge(revision, (revision_id="equivalent-same-capture",))
    equivalent = peer_replay(f, 6; observation_records=[f.observations; same_capture_equivalent])
    @test equivalent.state.mean == original.state.mean
    @test equivalent.state.covariance == original.state.covariance
    conflicting = merge(revision, (revision_id="conflicting-same-capture", value=revision.value+1))
    @test_throws ArgumentError peer_replay(f, 6; observation_records=[f.observations; conflicting])
    @test_throws ArgumentError peer_replay(f, 6; observation_records=reverse([f.observations; conflicting]))
    reused_id = merge(revision, (value=revision.value+1, receipt=revision.receipt+Millisecond(1), verified=revision.verified+Millisecond(1)))
    @test_throws ArgumentError peer_replay(f, 6; observation_records=[f.observations; reused_id])
    # Older capture -99 is verified later than the newer -17 revision. Correct
    # selection was fixed manually and strongly distinguishes capture ordering.
    @test original.selected[1].id == "revision-bin1"
    wrong_ids = ["older-capture-late-verification", "late-bin2", "initial-bin4", "boundary-bin6"]
    wrong = expected_posterior(f, 6; selected_ids=wrong_ids)
    @test norm(state_mean(original.state)-wrong.mean) > 0.1
end

@testset "Strict receipt and verification millisecond boundaries" begin
    f = fixture()
    boundary, asof = f.origin+Hour(6), f.origin+Hour(6)+Millisecond(1)
    records = deepcopy(f.observations)
    index = only(findall(o -> o.revision_id=="boundary-bin6", records))
    no6 = ["revision-bin1", "late-bin2", "initial-bin4"]
    excluded = expected_posterior(f, 6; selected_ids=no6)
    included = expected_posterior(f, 6)
    for replacement in (
            merge(records[index], (verified=asof,)),
            merge(records[index], (receipt=asof, verified=asof)))
        changed = copy(records)
        changed[index] = replacement
        strict = peer_replay(f, 6; observation_records=changed, asof)
        inclusive = peer_replay(f, 6; observation_records=changed, asof, receipt_boundary=:hypothetical_inclusive)
        @test compare(state_mean(strict.state), excluded.mean)
        @test compare(state_matrix(strict.state), excluded.covariance)
        @test compare(state_mean(inclusive.state), included.mean)
        @test compare(state_matrix(inclusive.state), included.covariance)
        @test norm(state_mean(strict.state)-state_mean(inclusive.state)) > 0.1
    end
    inputs = peer_inputs(f)
    @test_throws ArgumentError REPLAY.replay_asof(inputs.initial, f.lambdas, f.sigmas,
        inputs.intervals, inputs.observations, boundary; asof=boundary)
    inclusive = peer_replay(f, 6; asof=boundary, receipt_boundary=:hypothetical_inclusive)
    @test compare(state_mean(inclusive.state), included.mean)
    @test compare(state_matrix(inclusive.state), included.covariance)
    premature = merge(records[3], (revision_id="pre-completion", receipt=f.origin+Hour(1)-Millisecond(1), verified=f.origin+Hour(1)))
    rollback = merge(records[3], (revision_id="clock-rollback", receipt=f.origin+Hour(1), verified=f.origin+Hour(1)-Millisecond(1)))
    @test_throws ArgumentError peer_replay(f, 6; observation_records=[records; premature])
    @test_throws ArgumentError peer_replay(f, 6; observation_records=[records; rollback])
end

@testset "Future arrays and receipts cannot change current state" begin
    f = fixture()
    original = peer_replay(f, 6)
    changed = [if o.revision_id in ("future-bin7", "future-receipt-bin5")
        merge(o, (value=o.value+1e6, R=1e5, revision_id="revision-bin1"))
        else o end for o in f.observations]
    future_changed = peer_replay(f, 6; observation_records=changed)
    @test future_changed.state.mean == original.state.mean
    @test future_changed.state.covariance == original.state.covariance
    inputs = peer_inputs(f)
    mutated_intervals = [j<=6 ? interval : REPLAY.ForcingInterval(interval.id, interval.start,
        interval.stop, [1e6, -1e6], interval.received+Hour(5)) for (j,interval) in enumerate(inputs.intervals)]
    future_forcing_changed = REPLAY.replay_asof(inputs.initial, f.lambdas, f.sigmas,
        mutated_intervals, inputs.observations, f.origin+Hour(6); asof=f.origin+Hour(6)+Millisecond(1))
    @test future_forcing_changed.state.mean == original.state.mean
    @test future_forcing_changed.state.covariance == original.state.covariance
    # Even with a later actual decision, intervals ending after model boundary
    # cannot enter that model state. Legitimately arrived bin5 may now enter.
    late_ids = ["revision-bin1", "late-bin2", "initial-bin4", "future-receipt-bin5", "boundary-bin6"]
    late_expected = expected_posterior(f, 6; selected_ids=late_ids)
    late = peer_replay(f, 6; asof=f.origin+Hour(8))
    @test compare(state_mean(late.state), late_expected.mean)
    @test compare(state_matrix(late.state), late_expected.covariance)
    @test length(late.intervals)==6 && count(!isnothing, late.selected)==5
end

@testset "Implementation replay is affine in the two gains" begin
    f = fixture()
    for boundary in (2, 5, 6, 8)
        zero = peer_replay(f, boundary; gains=[0.0, 0.0])
        first = peer_replay(f, boundary; gains=[1.0, 0.0])
        second = peer_replay(f, boundary; gains=[0.0, 1.0])
        for gains in ([0.25, 7.0], [2.5, 4.0], [10.0, 0.01])
            actual = peer_replay(f, boundary; gains)
            affine = state_mean(zero.state) + gains[1]*(state_mean(first.state)-state_mean(zero.state)) +
                     gains[2]*(state_mean(second.state)-state_mean(zero.state))
            expected = expected_posterior(f, boundary; gains)
            @test compare(state_mean(actual.state), affine)
            @test compare(state_mean(actual.state), expected.mean)
            @test actual.state.covariance == zero.state.covariance == first.state.covariance == second.state.covariance
        end
    end
end
