module V21StateInertiaTests

using Test
using DataFrames

const STATE_SELECTION_SCRIPT = normpath(joinpath(
    @__DIR__, "..", "validation", "operational",
    "v2_1_state_inertia_selection.jl",
))
include(STATE_SELECTION_SCRIPT)

@testset "V2.1 state-inertia operator" begin
    @test _state_inertia_blend(-120.0, -100.0, 1, -10.0) == -100.0
    @test _state_inertia_blend(10.0, 20.0, 1, 0.0) == 12.5
    @test _state_inertia_blend(-120.0, -100.0, 1, -16.0) == -120.0
    @test _state_inertia_blend(10.0, 20.0, 2, 0.0) == 13.75
    @test _state_inertia_blend(10.0, 20.0, 3, 0.0) == 11.25
    @test _state_inertia_blend(-40.0, 20.0, 2, 0.0) == -40.0
    @test _state_inertia_blend(10.0, -40.0, 2, 0.0) == 10.0
    @test _state_inertia_blend(10.0, 20.0, 6, 0.0) == 10.0
    @test_throws ArgumentError _state_inertia_blend(
        10.0, 20.0, 2, 0.0; h2_quiet_weight=1.1,
    )
    @test_throws ArgumentError _state_inertia_blend(
        10.0, 20.0, 2, 0.0; deepening_lo=-5.0, deepening_hi=-15.0,
    )
end

@testset "V2.1 state-inertia selection candidate is causal" begin
    rows = DataFrame(
        lead=[1, 1, 2, 3, 6, 1],
        obs=[-999.0, 999.0, -999.0, 999.0, -999.0, -300.0],
        persistence=[-100.0, 20.0, 20.0, 20.0, 20.0, -250.0],
        rate=[-10.0, 0.0, 0.0, 0.0, 0.0, -10.0],
        v2_0=fill(0.0, 6),
        v2_1_pre_state_inertia=[-120.0, 10.0, 10.0, 10.0, 10.0, -300.0],
    )
    weights = (
        V2_STATE_INERTIA_H1_QUIET_WEIGHT,
        V2_STATE_INERTIA_H1_DEEPENING_WEIGHT,
        V2_STATE_INERTIA_H2_QUIET_WEIGHT,
        V2_STATE_INERTIA_H3_QUIET_WEIGHT,
    )
    pred = _state_candidate(rows, weights...)
    @test pred == [-100.0, 12.5, 13.75, 11.25, 10.0, -250.0]

    changed_targets = copy(rows)
    changed_targets.obs .= reverse(changed_targets.obs)
    @test _state_candidate(changed_targets, weights...) == pred

    @test_throws ErrorException _state_candidate(
        select(rows, Not(:rate)), 0.75, 0.0, 0.75, 0.875,
    )
    @test_throws ErrorException _state_candidate(rows, 1.1, 0.0, 0.75, 0.875)
end

@testset "V2.1 state-inertia observational tie ordering" begin
    candidates = DataFrame(
        h1_quiet_weight=[0.625, 0.75, 0.75],
        h1_deepening_weight=[0.0, 0.0, 0.0],
        h2_quiet_weight=[0.875, 0.75, 0.625],
        h3_quiet_weight=[0.875, 0.875, 0.875],
        validation_rmse_sum_nt=[53.6715, 53.6779, 53.6798],
    )
    tied = collect(1:nrow(candidates))
    key(i) = (
        -candidates.h1_quiet_weight[i],
        -(candidates.h1_quiet_weight[i] + candidates.h1_deepening_weight[i] +
          candidates.h2_quiet_weight[i] + candidates.h3_quiet_weight[i]),
        -candidates.h2_quiet_weight[i], -candidates.h3_quiet_weight[i],
        -candidates.h1_deepening_weight[i], candidates.validation_rmse_sum_nt[i],
    )
    @test first(sort(tied; by=key)) == 2
end

end # module V21StateInertiaTests
