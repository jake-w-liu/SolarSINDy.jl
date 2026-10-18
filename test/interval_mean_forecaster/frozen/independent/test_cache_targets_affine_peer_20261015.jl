using Test, Dates, LinearAlgebra
include("ZeroPrefixCase.jl")

@testset "Independent same first-bin revision case after fix" begin
    case = zero_prefix_case(REPLAY)
    @test case.failure === nothing
    if case.failure === nothing
        @test compare(state_mean(case.result.state), case.expected.mean)
        @test compare(state_matrix(case.result.state), case.expected.covariance)
        @test case.result.reused_steps == 0
        @test case.result.replayed_steps == 6
        @test case.cache.snapshot === case.result
    else
        println(case.trace)
    end
end

@testset "Exact future interval-mean targets and immutable issuance" begin
    f = fixture()
    scenarios(boundary,gains=f.gains) = [REPLAY.ScenarioInterval(f.origin+Hour(boundary+k-1),
        f.origin+Hour(boundary+k), -gains.*f.q[boundary]) for k in 1:6]
    for boundary in (2, 5, 6, 8)
        replay = peer_replay(f, boundary)
        targets = REPLAY.rollout_targets(replay.state, f.lambdas, f.sigmas, scenarios(boundary))
        @test length(targets) == 6
        for horizon in 1:6
            expected = expected_target(f, boundary, horizon)
            @test targets[horizon].start == f.origin+Hour(boundary+horizon-1)
            @test targets[horizon].stop == f.origin+Hour(boundary+horizon)
            @test compare(targets[horizon].mean, expected.mean)
            @test compare(targets[horizon].variance, expected.variance)
        end
    end
    # Prespecified exact deterministic one-mode example, with a known
    # intervening forcing interval before the model boundary.
    rate = log(2.0)
    initial = REPLAY.Checkpoint(f.origin+Hour(2), [-20rate-50], zeros(1,1))
    target = only(REPLAY.rollout_targets(initial,[rate],[0.0],
        [REPLAY.ScenarioInterval(f.origin+Hour(2),f.origin+Hour(3),[-100rate])]))
    @test target.mean ≈ 25/rate-110 atol=1e-12 rtol=1e-12
    @test abs(target.mean-(15/rate-100)) > 1.0
    @test target.variance == 0

    inputs = peer_inputs(f)
    cache = REPLAY.ReplayCache(inputs.initial, f.lambdas, f.sigmas)
    boundary, asof = f.origin+Hour(6), f.origin+Hour(6)+Millisecond(1)
    first = REPLAY.replay!(cache,inputs.intervals,inputs.observations,boundary;asof)
    issued = REPLAY.record_forecast!(cache,"independent-issued",first,scenarios(6))
    issued_copy = deepcopy(issued)
    @test issued == REPLAY.record_forecast!(cache,"independent-issued",first,scenarios(6))
    @test !ismutabletype(typeof(issued)) && !ismutabletype(typeof(issued.targets))
    revision = merge(f.observations[2],(revision_id="bin2-later",value=-40.0,
        receipt=boundary+Millisecond(10),verified=boundary+Millisecond(11)))
    revised_records = [f.observations;revision]
    revised_inputs = peer_inputs(f;observation_records=revised_records)
    later = REPLAY.replay!(cache,revised_inputs.intervals,revised_inputs.observations,boundary;
        asof=boundary+Millisecond(20))
    revised_fixture = merge(f,(observations=revised_records,))
    ids = ["revision-bin1","bin2-later","initial-bin4","future-receipt-bin5","boundary-bin6"]
    expected = expected_posterior(revised_fixture,6;selected_ids=ids)
    @test compare(state_mean(later.state),expected.mean)
    @test compare(state_matrix(later.state),expected.covariance)
    @test later.reused_steps==1 && later.replayed_steps==5
    @test issued == issued_copy == cache.issued["independent-issued"]
    @test_throws ArgumentError REPLAY.record_forecast!(cache,"independent-issued",later,scenarios(6))
    @test_throws ArgumentError REPLAY.record_forecast!(cache,"stale-result",first,scenarios(6))
    @test !haskey(cache.issued,"stale-result")
    snapshot = cache.snapshot
    conflict = merge(revision,(value=revision.value+1,))
    invalid = peer_inputs(f;observation_records=[revised_records;conflict])
    @test_throws ArgumentError REPLAY.replay!(cache,invalid.intervals,invalid.observations,boundary;
        asof=boundary+Millisecond(20))
    @test cache.snapshot===snapshot && cache.issued["independent-issued"]==issued_copy
    # Explicitly supported retrospective backward-asof replay must rebuild
    # correctly and retain already issued immutable values.
    backward = REPLAY.replay!(cache,revised_inputs.intervals,revised_inputs.observations,f.origin+Hour(4);
        asof=f.origin+Hour(4)+Millisecond(1))
    expected_back = expected_posterior(f,4)
    @test compare(state_mean(backward.state),expected_back.mean)
    @test compare(state_matrix(backward.state),expected_back.covariance)
    @test cache.issued["independent-issued"]==issued_copy

    mutable_mean,mutable_P = copy(f.mean),copy(f.covariance)
    copied = REPLAY.Checkpoint(f.origin,mutable_mean,mutable_P)
    mutable_mean .= 1e6
    mutable_P .= 0
    @test copied.mean==Tuple(f.mean)
    @test state_matrix(copied)==f.covariance
end

@testset "Public affine gain basis versus independent batch law" begin
    f = fixture()
    for boundary in (2,6,8), initial_basis in (zeros(2,2),[1.0 0.2;-0.3 2.0])
        zero_inputs = peer_inputs(f;gains=[0.0,0.0])
        bases = Dict("bin$j" => -q.*Matrix{Float64}(I,2,2) for (j,q) in enumerate(f.q))
        affine = REPLAY.affine_gain_replay(zero_inputs.initial,f.lambdas,f.sigmas,
            zero_inputs.intervals,zero_inputs.observations,f.origin+Hour(boundary),bases,initial_basis;
            asof=f.origin+Hour(boundary)+Millisecond(1))
        @test affine.parameter_count==2
        @test compare(state_mean(affine.base),expected_posterior(f,boundary;gains=[0.0,0.0]).mean)
        for gains in ([0.25,7.0],[2.5,4.0],[10.0,0.01])
            value = REPLAY.evaluate_affine(affine,gains)
            changed_initial = merge(f,(mean=f.mean+initial_basis*gains,))
            independent = expected_posterior(changed_initial,boundary;gains)
            @test compare(state_mean(value),independent.mean)
            @test compare(state_matrix(value),independent.covariance)
            direct = peer_replay(changed_initial,boundary;gains)
            @test compare(state_mean(value),state_mean(direct.state))
            @test value.covariance==direct.state.covariance
        end
        @test_throws DimensionMismatch REPLAY.evaluate_affine(affine,[1.0,2.0,3.0])
    end
end
