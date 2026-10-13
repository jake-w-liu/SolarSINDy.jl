module SuperlearnerLearnTests

# Unit oracles for the Operational super-learner learning and scoring stage
# (`validation/operational/superlearner_learn.jl`).
#
# The stage turns Task A's out-of-fold expert tables into the served candidate
# and the preregistered verdict, so its failure modes are the study's failure
# modes: a stack that is not the constrained optimum, a residual that escapes its
# cap, a guard that lifts a deepening forecast, intervals that do not cover, a
# gate that passes on the wrong arithmetic, a selection rule that ignores the
# storm guards, and — worst of all — a fit that sees the year it is about to
# score. Every test below pins one of those against an expectation written
# independently of the code under test: an exact convex mixture, an
# independently implemented optimiser, the package's own conformal quantile, the
# deployed static stack simplex projection, a hand-built gate table, and a mutation of
# the scored year's observations.

using Test
using CSV
using DataFrames
using Dates
using LinearAlgebra
using Random
using Statistics
using SolarSINDy

include(normpath(joinpath(@__DIR__, "..", "validation", "operational", "superlearner_learn.jl")))
include(joinpath(@__DIR__, "superlearner_fixture.jl"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"""
Minimal fold object for the metric and pairing tests: only the columns those
functions read are populated, and the centers are supplied directly so the
expected root mean square error can be written down by hand.
"""
function tiny_year(year::Int, issues::Vector{DateTime}, steps::Vector{Int},
                   obs::Vector{Float64}; latest=nothing, rate=nothing, coupling=nothing,
                   centers=Dict{Symbol,Vector{Float64}}(),
                   comparators=Dict{Symbol,Vector{Float64}}(),
                   half_widths=Dict{Symbol,Vector{Float64}}())
    n = length(issues)
    latest_v = latest === nothing ? fill(-10.0, n) : latest
    rate_v = rate === nothing ? zeros(n) : rate
    coupling_v = coupling === nothing ? zeros(n) : coupling
    regime = [SolarSINDy.static_stack_regime(latest_v[i], rate_v[i], coupling_v[i])
              for i in 1:n]
    return SuperlearnerYearData(
        year, issues, steps, obs, latest_v, rate_v, coupling_v, fill(false, n), regime,
        trues(n), zeros(n, SUPERLEARNER_EXPERT_COUNT), zeros(n, SUPERLEARNER_FEATURE_COUNT),
        comparators, "analog_shadow", fill(NaN, n, SUPERLEARNER_INNOVATION_LAGS), falses(n),
        fill(NaN, n), fill(NaN, n), fill(NaN, n), fill(SUPERLEARNER_POOLED_REGIME, n),
        fill(SUPERLEARNER_POOLED_DEPTH, n), falses(n),
        zeros(n), zeros(n), falses(n), centers, half_widths,
    )
end

"Summary row in the schema `superlearner_gate_rows` reads."
summary_row(scope, model, step, n, rmse; bias=0.0, mae=0.0) =
    (scope=scope, model=model, model_step_hours=step, n=n, rmse_nt=rmse, bias_nt=bias,
     mae_nt=mae)

"Cell row in the schema `superlearner_storm_guard` reads."
cell_row(scope, cell, model, step, n, rmse; bias=0.0) =
    (scope=scope, cell=cell, model=model, model_step_hours=step, n=n, rmse_nt=rmse,
     bias_nt=bias)

"Bootstrap row in the schema `superlearner_gate_rows` reads."
bootstrap_row(scope, comparator, step, n, gain, lower, holm; candidate="superlearner_c",
              matched=true, rmse_candidate=1.0, rmse_comparator=2.0) =
    (scope=scope, candidate=candidate, comparator=comparator, model_step_hours=step, n=n,
     n_candidate_scored=n, n_comparator_scored=n, rows_matched=matched,
     rmse_candidate_nt=rmse_candidate, rmse_comparator_nt=rmse_comparator, gain_nt=gain,
     lower_nt=lower, p_one_sided=1e-4, n_blocks=20, holm_p=holm, family_size=66)

"Interval row in the schema `superlearner_gate_rows` reads."
interval_row(scope, variant, subset, step, n, coverage, width, score) =
    (scope=scope, variant=variant, subset=subset, model_step_hours=step, n=n,
     coverage=coverage, mean_width_nt=width, mean_interval_score_nt=score)

"""
A gate table for one era in which every G1, G2 and G3 requirement is satisfied,
so that each test below can break exactly one requirement and watch that gate —
and only that gate — turn false.
"""
function passing_inputs(scope::String; candidate::String="superlearner_c", oracle::Float64=3.0)
    summary = NamedTuple[]
    for step in SUPERLEARNER_STEPS
        push!(summary, summary_row(scope, candidate, step, 1000, 5.0))
        for comparator in SUPERLEARNER_GATED_COMPARATORS
            push!(summary, summary_row(scope, String(comparator), step, 1000, 6.0))
        end
        # The realized-driver ceiling: its distance to the best comparator is the
        # headroom the Amendment A2 G1 clause switches on.
        push!(summary, summary_row(scope, String(SUPERLEARNER_ORACLE_COLUMN), step, 1000, oracle))
    end
    cells = NamedTuple[]
    for cell in (SUPERLEARNER_G2_CELLS..., SUPERLEARNER_G2_INTENSE_CELL), step in SUPERLEARNER_STEPS
        push!(cells, cell_row(scope, String(cell), candidate, step, 200, 5.0))
        for comparator in SUPERLEARNER_GATED_COMPARATORS
            push!(cells, cell_row(scope, String(cell), String(comparator), step, 200, 6.0))
        end
    end
    boot = NamedTuple[]
    for comparator in SUPERLEARNER_GATED_COMPARATORS, step in SUPERLEARNER_STEPS
        push!(boot, bootstrap_row(scope, String(comparator), step, 1000, 1.0, 0.5, 0.001;
                                  candidate=candidate))
    end
    intervals = NamedTuple[]
    for subset in ("pooled", "storm_le_m50", "storm_le_m100")
        push!(intervals, interval_row(scope, candidate, subset, 0, 1000, 0.90, 10.0, 12.0))
        push!(intervals, interval_row(scope, "served_inertial_sindy", subset, 0, 1000, 0.90, 11.0,
                                      13.0))
        for step in SUPERLEARNER_STEPS
            push!(intervals, interval_row(scope, candidate, subset, step, 160, 0.90, 10.0,
                                          12.0))
            push!(intervals, interval_row(scope, "served_inertial_sindy", subset, step, 160, 0.90,
                                          11.0, 13.0))
        end
    end
    return (summary=summary, cells=cells, bootstrap=boot, intervals=intervals)
end

"Run the gate evaluation for one era and return its verdict dictionary."
function verdicts_for(inputs, scope::String; candidate::Symbol=:superlearner_c)
    eras = NamedTuple{(Symbol(scope),)}((2014:2019,))
    gates = superlearner_gate_rows(inputs.summary, inputs.cells, inputs.bootstrap,
                          inputs.intervals, eras, candidate)
    return gates.verdicts
end

"Replace the first row matching `predicate` with `replacement`."
function replace_row(rows, predicate, replacement)
    out = collect(rows)
    index = findfirst(predicate, out)
    index === nothing && error("test helper found no row to replace")
    out[index] = replacement
    return out
end

# ---------------------------------------------------------------------------
# L1: exact non-negative simplex least squares
# ---------------------------------------------------------------------------

@testset "the simplex projection restates the deployed static stack projection" begin
    rng = MersenneTwister(4041)
    for _ in 1:300
        v = randn(rng, 6) .* 8.0
        mine = superlearner_project_simplex(v, 1.0)
        theirs = SolarSINDy._static_stack_project_simplex(v, 1.0)
        @test maximum(abs, mine .- theirs) <= 1e-14
        @test all(>=(0.0), mine)
        @test abs(sum(mine) - 1.0) <= 1e-14
    end
    # A feasible point is its own projection, and a positive mass is honoured.
    feasible = [0.2, 0.3, 0.5]
    @test maximum(abs, superlearner_project_simplex(feasible, 1.0) .- feasible) <= 1e-15
    @test abs(sum(superlearner_project_simplex(randn(rng, 4), 0.6)) - 0.6) <= 1e-14
    @test superlearner_project_simplex(randn(rng, 4), 0.0) == zeros(4)
    @test_throws ArgumentError superlearner_project_simplex([1.0, NaN], 1.0)
    @test_throws ArgumentError superlearner_project_simplex([1.0, 2.0], -1.0)
    @test_throws ArgumentError superlearner_project_simplex(Float64[], 1.0)
end

@testset "the floor projection is the nearest point of its constraint set" begin
    rng = MersenneTwister(5052)
    sindy = collect(SUPERLEARNER_SINDY_FAMILY)
    for _ in 1:120
        v = randn(rng, SUPERLEARNER_EXPERT_COUNT) .* 3.0
        projected = superlearner_project_floor(v, SUPERLEARNER_SINDY_FLOOR)
        @test all(>=(-1e-15), projected)
        @test abs(sum(projected) - 1.0) <= 1e-12
        @test sum(projected[sindy]) >= SUPERLEARNER_SINDY_FLOOR - 1e-12
        # No randomly drawn feasible point is closer than the projection, which is
        # the defining property of a Euclidean projection.
        best = sum(abs2, projected .- v)
        for _ in 1:200
            head = superlearner_project_simplex(randn(rng, 3), SUPERLEARNER_SINDY_FLOOR + 0.4 * rand(rng))
            mass = sum(head)
            tail = superlearner_project_simplex(randn(rng, 6), 1.0 - mass)
            candidate = vcat(head, tail)
            @test best <= sum(abs2, candidate .- v) + 1e-10
        end
    end
    # A slack floor leaves the plain projection untouched.
    slack = [4.0, 4.0, 4.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0]
    @test superlearner_project_floor(slack, SUPERLEARNER_SINDY_FLOOR) == superlearner_project_simplex(slack, 1.0)
    @test superlearner_project_floor(slack, 0.0) == superlearner_project_simplex(slack, 1.0)
    # The projection now serves both expert sets, so its shape check is on the floor
    # group rather than on a fixed length: a single coordinate has no free mass, and
    # a group outside the index range or covering everything is malformed.
    @test_throws DimensionMismatch superlearner_project_floor(zeros(1), 0.5)
    @test_throws ArgumentError superlearner_project_floor(zeros(SUPERLEARNER_EXPERT_COUNT), 1.5)
    @test_throws ArgumentError superlearner_project_floor(zeros(4), 0.5; family=(1, 2, 3, 4))
    @test_throws ArgumentError superlearner_project_floor(zeros(4), 0.5; family=(1, 9))
    # On the ten-expert set the floor binds on the four-column family.
    ten = vcat(fill(4.0, 3), fill(-2.0, 6), 4.0)
    projected = superlearner_project_floor(ten, SUPERLEARNER_SINDY_FLOOR; family=SUPERLEARNER_SINDY_FAMILY_TEN)
    @test length(projected) == SUPERLEARNER_EXPERT_TEN_COUNT
    @test isapprox(sum(projected), 1.0; atol=1e-12)
    @test sum(projected[collect(SUPERLEARNER_SINDY_FAMILY_TEN)]) >= SUPERLEARNER_SINDY_FLOOR - 1e-12
end

@testset "the stack recovers a known convex mixture exactly" begin
    rng = MersenneTwister(6063)
    A = randn(rng, 800, SUPERLEARNER_EXPERT_COUNT) .* 25.0 .- 45.0
    truth = zeros(SUPERLEARNER_EXPERT_COUNT)
    truth[[1, 3, 5, 8]] = [0.40, 0.25, 0.20, 0.15]
    fit = superlearner_fit_nnls(A, A * truth)
    @test maximum(abs, fit.weights .- truth) <= 1e-6
    @test fit.objective_mse <= 1e-18
    @test all(>=(0.0), fit.weights)
    @test abs(sum(fit.weights) - 1.0) <= 1e-12
    @test Set(fit.support) == Set([1, 3, 5, 8])
    @test fit.stationarity <= 1e-9
    @test fit.dual_min >= -1e-9
    @test !fit.floor_active
    # A second mixture, so the recovery is not an artefact of one support.
    other = zeros(SUPERLEARNER_EXPERT_COUNT)
    other[[2, 9]] = [0.7, 0.3]
    @test maximum(abs, superlearner_fit_nnls(A, A * other).weights .- other) <= 1e-6
end

@testset "the stack agrees with an independent projected-gradient optimiser" begin
    rng = MersenneTwister(7074)
    for trial in 1:12
        A = randn(rng, 400, SUPERLEARNER_EXPERT_COUNT) .* 15.0 .- 30.0
        # Correlate the experts, which is the regime the real study is in and the
        # regime where a badly conditioned optimiser drifts.
        A[:, 2] = 0.9 .* A[:, 1] .+ 0.1 .* A[:, 2]
        A[:, 3] = 0.8 .* A[:, 1] .+ 0.2 .* A[:, 3]
        y = A * superlearner_project_simplex(rand(rng, SUPERLEARNER_EXPERT_COUNT), 1.0) .+
            2.0 .* randn(rng, 400)
        exact = superlearner_fit_nnls(A, y)
        reference = superlearner_pgd_nnls(A, y)
        @test exact.objective_mse <= reference.objective_mse + 1e-9
        @test abs(exact.objective_mse - reference.objective_mse) <= 1e-7
        @test maximum(abs, exact.weights .- reference.weights) <= 1e-5
        floored = superlearner_fit_nnls(A, y; floor_mass=SUPERLEARNER_SINDY_FLOOR)
        floored_reference = superlearner_pgd_nnls(A, y; floor_mass=SUPERLEARNER_SINDY_FLOOR)
        @test sum(floored.weights[collect(SUPERLEARNER_SINDY_FAMILY)]) >= SUPERLEARNER_SINDY_FLOOR - 1e-10
        @test floored.objective_mse <= floored_reference.objective_mse + 1e-9
        @test abs(floored.objective_mse - floored_reference.objective_mse) <= 1e-7
        # The floor is a constraint, so it can never improve the objective.
        @test floored.objective_mse >= exact.objective_mse - 1e-12
    end
end

@testset "the SINDy-family floor binds when the free optimum violates it" begin
    rng = MersenneTwister(8085)
    n = 600
    # The physical experts carry the signal and the SINDy family carries noise, so
    # the free optimum starves the family and the floor has to intervene.
    signal = randn(rng, n) .* 30.0
    A = Matrix{Float64}(undef, n, SUPERLEARNER_EXPERT_COUNT)
    for j in 1:SUPERLEARNER_EXPERT_COUNT
        A[:, j] = j in SUPERLEARNER_SINDY_FAMILY ? signal .+ 25.0 .* randn(rng, n) :
            signal .+ 1.0 .* randn(rng, n)
    end
    y = signal
    free = superlearner_fit_nnls(A, y)
    @test sum(free.weights[collect(SUPERLEARNER_SINDY_FAMILY)]) < SUPERLEARNER_SINDY_FLOOR
    @test !free.floor_active
    floored = superlearner_fit_nnls(A, y; floor_mass=SUPERLEARNER_SINDY_FLOOR)
    @test floored.floor_active
    @test abs(sum(floored.weights[collect(SUPERLEARNER_SINDY_FAMILY)]) - SUPERLEARNER_SINDY_FLOOR) <= 1e-10
    @test abs(sum(floored.weights) - 1.0) <= 1e-12
    @test all(>=(0.0), floored.weights)
    @test floored.objective_mse > free.objective_mse
    reference = superlearner_pgd_nnls(A, y; floor_mass=SUPERLEARNER_SINDY_FLOOR)
    @test abs(floored.objective_mse - reference.objective_mse) <= 1e-6
end

@testset "the stack survives collinear experts and rejects malformed input" begin
    rng = MersenneTwister(9096)
    A = randn(rng, 300, SUPERLEARNER_EXPERT_COUNT) .* 10.0
    A[:, 4] = A[:, 1]
    A[:, 5] = A[:, 2]
    y = A[:, 1] .* 0.5 .+ A[:, 2] .* 0.5 .+ 0.5 .* randn(rng, 300)
    fit = superlearner_fit_nnls(A, y)
    @test all(>=(0.0), fit.weights)
    @test abs(sum(fit.weights) - 1.0) <= 1e-12
    @test fit.stationarity <= 1e-7
    @test fit.dual_min >= -1e-7
    # A single column has no free mass beside the floor group, and the target must
    # have one entry per row.
    @test_throws DimensionMismatch superlearner_fit_nnls(randn(rng, 10, 1), randn(rng, 10))
    @test_throws DimensionMismatch superlearner_fit_nnls(randn(rng, 10, SUPERLEARNER_EXPERT_COUNT),
                                                randn(rng, 9))
    bad = randn(rng, 10, SUPERLEARNER_EXPERT_COUNT)
    bad[3, 2] = NaN
    @test_throws ArgumentError superlearner_fit_nnls(bad, randn(rng, 10))
    @test_throws ArgumentError superlearner_fit_nnls(randn(rng, 10, SUPERLEARNER_EXPERT_COUNT),
                                           randn(rng, 10); floor_mass=-0.1)
    # The floor group must be a proper subset of the expert indices.
    @test_throws ArgumentError superlearner_fit_nnls(randn(rng, 10, SUPERLEARNER_EXPERT_COUNT),
                                            randn(rng, 10); family=(1, 2, 99))
    @test_throws ArgumentError superlearner_fit_nnls(randn(rng, 10, SUPERLEARNER_EXPERT_COUNT),
                                            randn(rng, 10),
                                            ; family=Tuple(1:SUPERLEARNER_EXPERT_COUNT))
end

@testset "the ten-expert stack of Amendment A3 floors the static stack with the family" begin
    # The static stack is expert ten and belongs to the floor group, so a fit
    # whose free optimum starves the SINDy family must move mass back onto exactly
    # those four columns and no others.
    rng = MersenneTwister(3141)
    n = 900
    signal = randn(rng, n) .* 20.0 .- 40.0
    A = Matrix{Float64}(undef, n, SUPERLEARNER_EXPERT_TEN_COUNT)
    for j in 1:SUPERLEARNER_EXPERT_TEN_COUNT
        A[:, j] = j in SUPERLEARNER_SINDY_FAMILY_TEN ? signal .+ 25.0 .* randn(rng, n) :
                  signal .+ 0.4 .* randn(rng, n)
    end
    y = signal .+ 0.1 .* randn(rng, n)
    free = superlearner_fit_nnls(A, y; family=SUPERLEARNER_SINDY_FAMILY_TEN)
    @test length(free.weights) == SUPERLEARNER_EXPERT_TEN_COUNT
    @test sum(free.weights[collect(SUPERLEARNER_SINDY_FAMILY_TEN)]) < SUPERLEARNER_SINDY_FLOOR
    @test !free.floor_active
    floored = superlearner_fit_nnls(A, y; floor_mass=SUPERLEARNER_SINDY_FLOOR, family=SUPERLEARNER_SINDY_FAMILY_TEN)
    @test floored.floor_active
    @test abs(sum(floored.weights[collect(SUPERLEARNER_SINDY_FAMILY_TEN)]) - SUPERLEARNER_SINDY_FLOOR) <= 1e-10
    @test all(>=(-1e-12), floored.weights)
    @test abs(sum(floored.weights) - 1.0) <= 1e-10
    @test floored.stationarity <= 1e-7
    @test floored.dual_min >= -1e-7
    @test floored.objective_mse >= free.objective_mse - 1e-12
    # An independent optimiser over the same constraint set agrees.
    reference = superlearner_pgd_nnls(A, y; floor_mass=SUPERLEARNER_SINDY_FLOOR, family=SUPERLEARNER_SINDY_FAMILY_TEN)
    @test abs(floored.objective_mse - reference.objective_mse) <= 1e-6
    # The nine-expert floor group is unchanged, and the tenth expert is the static
    # stack in the contracted column order.
    @test SUPERLEARNER_SINDY_FAMILY_TEN == (1, 2, 3, 10)
    @test SUPERLEARNER_EXPERTS_TEN[SUPERLEARNER_EXPERT_TEN_COUNT] === :static_stack
    @test SUPERLEARNER_EXPERTS_TEN[1:SUPERLEARNER_EXPERT_COUNT] == SUPERLEARNER_EXPERTS
    # The written order, pinned literally: a silent reorder would relabel every
    # persisted weight column and the deployed expert list.
    @test [String(e) for e in SUPERLEARNER_EXPERTS_TEN] ==
          ["served_inertial_sindy", "frozen_inertial_sindy", "t1r_analog", "persistence", "burton",
           "burton_full", "obrien", "direct_gbm", "climatology", "static_stack"]
end

@testset "the depth bins are the written Amendment A1 intervals" begin
    @test superlearner_depth_bin(0.0) === :shallow
    @test superlearner_depth_bin(-29.999) === :shallow
    # The edges close from below, so a row exactly on one belongs to the deeper bin.
    @test superlearner_depth_bin(SUPERLEARNER_DEPTH_MODERATE_NT) === :moderate
    @test superlearner_depth_bin(-69.999) === :moderate
    @test superlearner_depth_bin(SUPERLEARNER_DEPTH_DEEP_NT) === :deep
    @test superlearner_depth_bin(-400.0) === :deep
    @test SUPERLEARNER_DEPTH_BINS == (:shallow, :moderate, :deep)
    @test (SUPERLEARNER_DEPTH_MODERATE_NT, SUPERLEARNER_DEPTH_DEEP_NT) == (-30.0, -70.0)
    # The fallback chain is specific, then regime-pooled, then fully pooled.
    @test superlearner_cell_chain(3, :recovery, :deep) ==
          [(3, :recovery, :deep), (3, :recovery, :pooled), (3, :pooled, :pooled)]
    grid = superlearner_cell_grid(6)
    @test length(grid) == 1 + length(SUPERLEARNER_REGIMES) * (1 + length(SUPERLEARNER_DEPTH_BINS))
    @test allunique(grid)
    @test all(key -> key[1] == 6, grid)
    @test first(grid) == (6, SUPERLEARNER_POOLED_REGIME, SUPERLEARNER_POOLED_DEPTH)
end

@testset "a pool row whose target falls inside the embargo is excluded" begin
    # Hourly issues straddling the fold-2016 cutoff, each carrying every model step,
    # so the admissible set differs per step exactly as `issue + step <= cutoff` says.
    # Expert one reproduces the observation on the admissible rows and expert four on
    # the embargoed ones, so a fit that keeps the late rows cannot produce the
    # embargoed answer and the test cannot pass by accident.
    cutoff = superlearner_pool_cutoff(2016)
    @test cutoff == DateTime(2016, 1, 1) - Hour(168)
    origin = cutoff - Hour(300)
    issues = DateTime[]
    steps = Int[]
    for k in 0:419, step in SUPERLEARNER_STEPS
        push!(issues, origin + Hour(k))
        push!(steps, step)
    end
    n = length(issues)
    inside = [issues[i] + Hour(steps[i]) > cutoff for i in 1:n]
    @test 0 < count(inside) < n
    obs = collect(range(-120.0, -20.0; length=n))
    comparators = Dict{Symbol,Vector{Float64}}()
    for (j, name) in enumerate(SUPERLEARNER_EXPERTS)
        comparators[name] = j == 1 ? [inside[i] ? obs[i] + 60.0 : obs[i] for i in 1:n] :
                            (j == 4 ? [inside[i] ? obs[i] : obs[i] - 60.0 for i in 1:n] :
                             fill(-500.0, n))
    end
    comparators[:static_stack] = fill(-400.0, n)
    year = tiny_year(2015, issues, steps, obs; comparators=comparators)
    @test all(i -> superlearner_in_pool(year, i, cutoff) == !inside[i], 1:n)
    # The exact boundary is admissible, one hour past it is not.
    boundary = findfirst(i -> issues[i] + Hour(steps[i]) == cutoff, 1:n)
    @test boundary !== nothing
    @test superlearner_in_pool(year, boundary, cutoff)
    past = findfirst(i -> issues[i] + Hour(steps[i]) == cutoff + Hour(1), 1:n)
    @test !superlearner_in_pool(year, past, cutoff)
    @test superlearner_in_pool(year, past, nothing)

    embargoed_fit = superlearner_fit_l1([year]; cutoff=cutoff)
    full_fit = superlearner_fit_l1([year])
    for step in SUPERLEARNER_STEPS
        key = (step, SUPERLEARNER_POOLED_REGIME, SUPERLEARNER_POOLED_DEPTH)
        admissible = count(i -> steps[i] == step && !inside[i], 1:n)
        @test embargoed_fit[key].n_rows == admissible
        @test full_fit[key].n_rows == count(==(step), steps)
        @test embargoed_fit[key].n_rows < full_fit[key].n_rows
        # With the embargo the first expert is exact and takes all the mass; without
        # it the fit is pulled toward the fourth expert.
        @test embargoed_fit[key].weights[1] > 0.999
        @test embargoed_fit[key].objective_mse < 1e-18
        @test full_fit[key].weights[1] < 0.999
        @test full_fit[key].objective_mse > embargoed_fit[key].objective_mse
    end

    # The conformal calibration follows the same rule: residuals from embargoed rows
    # must not widen the interval.
    centers = [inside[i] ? obs[i] + 60.0 : obs[i] for i in 1:n]
    tight = superlearner_fit_conformal(Tuple{SuperlearnerYearData,Vector{Float64}}[(year, centers)];
                              cutoff=cutoff)
    loose = superlearner_fit_conformal(Tuple{SuperlearnerYearData,Vector{Float64}}[(year, centers)])
    for step in SUPERLEARNER_STEPS
        key = (step, SUPERLEARNER_POOLED_DEPTH)
        @test tight[key].n == count(i -> steps[i] == step && !inside[i], 1:n)
        @test loose[key].n == count(==(step), steps)
        @test tight[key].half_width < loose[key].half_width
    end
    # The L2 pool eligibility check composes with the embargo the same way.
    year.l1 .= centers
    year.innovations .= 0.0
    year.innovation_ok .= true
    eligible = [i for i in 1:n if superlearner_l2_eligible(year, i) && superlearner_in_pool(year, i, cutoff)]
    @test length(eligible) == count(!, inside)
end

@testset "L1 cells are keyed by regime and depth and fall back in order" begin
    dir = mktempdir()
    superlearner_synthesize_fixture(dir; years=2013:2013, hours_per_year=1200)
    pool = [superlearner_read_year(2013; dir=dir)]
    cells = superlearner_fit_l1(pool)
    for step in SUPERLEARNER_STEPS
        @test haskey(cells, (step, SUPERLEARNER_POOLED_REGIME, SUPERLEARNER_POOLED_DEPTH))
        for key in superlearner_cell_grid(step)
            cell = get(cells, key, nothing)
            cell === nothing && continue
            @test (cell.model_step_hours, cell.regime, cell.depth) == key
            @test cell.n_rows >= SUPERLEARNER_MIN_CELL_ROWS
            @test abs(sum(cell.weights) - 1.0) <= 1e-10
            @test all(>=(0.0), cell.weights)
        end
    end
    # The fixture reaches depth-resolved cells, so the amendment is exercised and
    # not merely available.
    @test any(key -> key[3] in SUPERLEARNER_DEPTH_BINS, keys(cells))
    # Every row's cell is the first fitted key of its own chain, so a row can
    # never be scored by a coarser cell than the one that exists for it.
    fold = pool[1]
    applied = superlearner_l1_centers(fold, cells)
    for i in 1:length(fold)
        fold.usable[i] || continue
        chain = superlearner_cell_chain(fold.step[i], fold.regime[i],
                               superlearner_depth_bin(fold.latest[i]))
        expected = first([key for key in chain if haskey(cells, key)])
        @test (fold.step[i], applied.cell_regime[i], applied.cell_depth[i]) == expected
        @test applied.used_pooled[i] == (expected != first(chain))
        weights = cells[expected].weights
        @test applied.centers[i] ≈ sum(weights[j] * fold.experts[i, j]
                                       for j in 1:SUPERLEARNER_EXPERT_COUNT)
    end
    # A minimum so large that no resolved cell qualifies must leave only the
    # pooled cells, and every row must then fall back to them.
    coarse = superlearner_fit_l1(pool; minimum_cell_rows=10_000_000)
    @test Set(keys(coarse)) ==
          Set((step, SUPERLEARNER_POOLED_REGIME, SUPERLEARNER_POOLED_DEPTH) for step in SUPERLEARNER_STEPS)
    pooled_applied = superlearner_l1_centers(fold, coarse)
    scored = [i for i in 1:length(fold) if fold.usable[i]]
    @test all(pooled_applied.used_pooled[scored])
    @test all(i -> pooled_applied.cell_regime[i] === SUPERLEARNER_POOLED_REGIME, scored)
    @test all(i -> pooled_applied.cell_depth[i] === SUPERLEARNER_POOLED_DEPTH, scored)
    # Fallback rows keep the served product, which is the plan section 3 rule.
    served = fold.comparators[:served_inertial_sindy]
    for i in 1:length(fold)
        fold.usable[i] && continue
        @test pooled_applied.centers[i] == served[i]
        @test pooled_applied.cell_regime[i] === :served
        @test pooled_applied.cell_depth[i] === :served
    end
    @test_throws ErrorException superlearner_fit_l1(SuperlearnerYearData[])
end

# ---------------------------------------------------------------------------
# L2: residual cap and inner split
# ---------------------------------------------------------------------------

@testset "the residual cap is exactly +/-(10 + 5h) nT" begin
    for step in SUPERLEARNER_STEPS
        cap = 10.0 + 5.0 * step
        @test superlearner_residual_cap(step) == cap
        @test superlearner_cap_residual(cap + 1.0, step) == cap
        @test superlearner_cap_residual(-cap - 1.0, step) == -cap
        @test superlearner_cap_residual(cap, step) == cap
        @test superlearner_cap_residual(cap - 1e-9, step) == cap - 1e-9
        @test superlearner_cap_residual(0.25, step) == 0.25
        @test superlearner_cap_residual(NaN, step) == 0.0
        @test superlearner_cap_residual(Inf, step) == 0.0
    end
    # The cap widens with lead time; a flat cap would fail this.
    @test superlearner_residual_cap(7) > superlearner_residual_cap(1)
    @test superlearner_residual_cap(7) - superlearner_residual_cap(6) == 5.0
end

@testset "the inner split follows the plan rule and degrades on a short pool" begin
    origin = DateTime(2016, 1, 1)
    long = tiny_year(2016, [origin + Hour(k) for k in 0:(60 * 730)],
                     fill(1, 60 * 730 + 1), zeros(60 * 730 + 1))
    long_rows = Tuple{SuperlearnerYearData,Int}[(long, i) for i in 1:length(long)]
    split = superlearner_inner_split(long_rows)
    @test split.rule == "last_$(SUPERLEARNER_INNER_VALIDATION_MONTHS)_months"
    @test split.boundary == split.last_issue - Month(SUPERLEARNER_INNER_VALIDATION_MONTHS)
    # The halves no longer partition the pool: the embargo of the testset below
    # keeps a gap between them, and every row is in exactly one of the three.
    @test length(split.train) + length(split.validate) + split.n_embargoed ==
          length(long_rows)
    @test all(i -> long.issue[long_rows[i][2]] < split.boundary, split.train)
    @test all(i -> long.issue[long_rows[i][2]] >= split.boundary, split.validate)
    @test length(split.train) >= SUPERLEARNER_INNER_MIN_ROWS
    @test length(split.validate) >= SUPERLEARNER_INNER_MIN_ROWS

    short = tiny_year(2016, [origin + Hour(k) for k in 0:9_000], fill(1, 9_001),
                      zeros(9_001))
    short_rows = Tuple{SuperlearnerYearData,Int}[(short, i) for i in 1:length(short)]
    short_split = superlearner_inner_split(short_rows)
    @test short_split.rule == "chronological_two_thirds"
    @test length(short_split.train) > length(short_split.validate)
    @test length(short_split.validate) >= SUPERLEARNER_INNER_MIN_ROWS
    @test maximum(short.issue[[short_rows[i][2] for i in short_split.train]]) <
          minimum(short.issue[[short_rows[i][2] for i in short_split.validate]])
    @test_throws ErrorException superlearner_inner_split(Tuple{SuperlearnerYearData,Int}[])
end

@testset "the inner split embargoes its training block by 168 h, per row's own step" begin
    # Amendment A3 puts a 168 h target embargo between a fold's out-of-fold pool
    # and the year it scores. The same statement has to hold one level down: the
    # inner split decides the residual's grid point, its joint-versus-per-step
    # form and its per-step acceptance, so a contiguous split lets targets that
    # mature after the validation window opens sit inside the fitting rows. The
    # pool carries all six model steps at every issue hour, so the rule is a
    # statement about `issue + step` evaluated per row, not about the issue hour.
    #
    # A contiguous split fails every assertion of the exact-count, kept/dropped
    # and per-step blocks below: it reports no embargoed row, keeps the row whose
    # target matures one hour into the window, and puts all six steps of one issue
    # hour on the same side.
    origin = DateTime(2016, 1, 1)
    gap = sum(SUPERLEARNER_EMBARGO_HOURS - 1 + step for step in SUPERLEARNER_STEPS)

    """
    Hourly pool carrying every model step at every issue hour, with a lookup from
    `(issue, step)` to the row index so single rows can be named by their target.
    """
    function stepped_pool(hours::Int)
        issues = DateTime[]
        steps = Int[]
        index = Dict{Tuple{DateTime,Int},Int}()
        for k in 0:(hours - 1), step in SUPERLEARNER_STEPS
            push!(issues, origin + Hour(k))
            push!(steps, step)
            index[(origin + Hour(k), step)] = length(issues)
        end
        year = tiny_year(2016, issues, steps, zeros(length(issues)))
        rows = Tuple{SuperlearnerYearData,Int}[(year, i) for i in 1:length(year)]
        return (year=year, rows=rows, index=index)
    end

    "The three assertions that separate an embargoed split from a contiguous one."
    function check_gap(pool, split)
        year = pool.year
        train = Set(split.train)
        validate = Set(split.validate)
        @test isempty(intersect(train, validate))
        @test split.cutoff == split.boundary - Hour(SUPERLEARNER_EMBARGO_HOURS)
        @test all(k -> year.issue[k] + Hour(year.step[k]) <= split.cutoff, split.train)
        @test all(k -> year.issue[k] >= split.boundary, split.validate)
        dropped = setdiff(Set(eachindex(pool.rows)), union(train, validate))
        @test length(dropped) == split.n_embargoed
        @test split.n_embargoed == gap
        @test all(k -> year.issue[k] < split.boundary &&
                       year.issue[k] + Hour(year.step[k]) > split.cutoff, dropped)
        # The pair a contiguous split gets wrong: the last row whose target lands
        # exactly on the cutoff is kept, and the next issue hour at the same step
        # is dropped although it is still issued long before the window opens.
        for step in SUPERLEARNER_STEPS
            kept = pool.index[(split.cutoff - Hour(step), step)]
            embargoed = pool.index[(split.cutoff - Hour(step) + Hour(1), step)]
            @test kept in train
            @test !(embargoed in train)
            @test !(embargoed in validate)
            @test year.issue[embargoed] < split.boundary
        end
        # One issue hour, six steps, two verdicts: the rule reads each row's own
        # step, so a 1 h row three hours before the cutoff still trains while the
        # 7 h row issued beside it does not.
        probe = split.cutoff - Hour(3)
        for step in SUPERLEARNER_STEPS
            row = pool.index[(probe, step)]
            @test (row in train) == (step <= 3)
            @test !(row in validate)
        end
        @test length(split.train) >= SUPERLEARNER_INNER_MIN_ROWS
        @test length(split.validate) >= SUPERLEARNER_INNER_MIN_ROWS
    end

    # Plan rule: 20,000 hours leave the last 24 months as validation and still a
    # training block above the row and span floors.
    long = stepped_pool(20_000)
    long_split = superlearner_inner_split(long.rows)
    @test long_split.rule == "last_$(SUPERLEARNER_INNER_VALIDATION_MONTHS)_months"
    check_gap(long, long_split)

    # Fallback rule: 9,001 hours span 9,000 h, so the two-thirds boundary lands on
    # an exact hour and the same named rows can be checked there.
    short = stepped_pool(9_001)
    short_split = superlearner_inner_split(short.rows)
    @test short_split.rule == "chronological_two_thirds"
    @test short_split.boundary == origin + Hour(6_000)
    check_gap(short, short_split)
end

@testset "the residual is accepted only at steps whose inner gain is positive" begin
    # A synthetic pool in which the residual target is learnable at some steps and
    # deliberately harmful at the others. At a learnable step the residual is a
    # multiple of one feature, so a shallow tree recovers it. At a harmful step the
    # residual is a constant that flips sign across the inner split, so whatever
    # the tree learns on the training half is exactly wrong on the validation half
    # and the gain is negative by construction rather than by luck.
    rng = MersenneTwister(4242)
    origin = DateTime(2016, 1, 1)
    hours = 4_000
    learnable = (2, 4, 7)
    issues = DateTime[]
    steps = Int[]
    drive = Float64[]
    for k in 0:(hours - 1), step in SUPERLEARNER_STEPS
        push!(issues, origin + Hour(k))
        push!(steps, step)
        push!(drive, 4.0 * randn(rng))
    end
    m = length(issues)
    center = fill(-20.0, m)

    function build(target_steps)
        # `static_stack` is the guard reference of the A2 variant, so the center
        # builder needs it even where the test only exercises the residual.
        year = tiny_year(2016, copy(issues), copy(steps), copy(center);
                         comparators=Dict(:static_stack => fill(-25.0, m)))
        for i in 1:m
            year.features[i, 1] = drive[i]
        end
        year.l1 .= center
        year.innovations .= 0.0
        year.innovation_ok .= true
        rows = Tuple{SuperlearnerYearData,Int}[(year, i) for i in 1:length(year)]
        split = superlearner_inner_split(rows)
        in_train = falses(m)
        in_train[split.train] .= true
        for i in 1:m
            residual = steps[i] in target_steps ? 3.0 * drive[i] + 0.20 * randn(rng) :
                       (in_train[i] ? 8.0 : -8.0)
            year.obs[i] = center[i] + residual
        end
        return (year=year, rows=rows, in_train=in_train)
    end

    built = build(learnable)
    year = built.year
    layer = superlearner_fit_l2(built.rows; grid=((3, 60),))
    # The layer carries its inner split's embargo forward, which is what the
    # persisted selection table reports per fold.
    @test layer.split.cutoff == layer.split.boundary - Hour(SUPERLEARNER_EMBARGO_HOURS)
    @test layer.n_inner_embargoed ==
          sum(SUPERLEARNER_EMBARGO_HOURS - 1 + step for step in SUPERLEARNER_STEPS)
    @test layer.n_inner_train + layer.n_inner_validate + layer.n_inner_embargoed ==
          layer.n_pool_rows
    @test Set(layer.accepted_steps) == Set(learnable)
    for row in layer.acceptance
        @test row.accepted == (row.gain_nt > 0.0)
        @test row.accepted == (row.model_step_hours in learnable)
        @test row.rmse_identity_nt > 0.0
        @test row.n_inner_validate > 0
        @test row.reason ==
              (row.accepted ? "inner_gain_positive" : "inner_gain_not_positive")
        if row.accepted
            @test row.rmse_residual_nt < row.rmse_identity_nt
        else
            @test row.rmse_residual_nt > row.rmse_identity_nt
        end
    end
    # Applying the layer must leave the rejected steps exactly at their L1 center.
    superlearner_apply_l2!(year, layer)
    for i in 1:length(year)
        if year.step[i] in learnable
            @test year.l2_applied[i]
        else
            @test !year.l2_applied[i]
            @test year.residual[i] == 0.0
            @test year.residual_raw[i] == 0.0
        end
    end
    year.l1_floor .= year.l1
    year.l1_ten .= year.l1
    superlearner_build_centers!(year)
    for i in 1:length(year)
        year.step[i] in learnable && continue
        @test year.centers[:superlearner_b][i] == year.centers[:superlearner_a][i]
        @test year.centers[:superlearner_c][i] == year.centers[:superlearner_a][i]
    end
    # A layer that helps nowhere is fitted but never applied, and no model is even
    # refitted for it.
    hostile = build(())
    hostile_layer = superlearner_fit_l2(hostile.rows; grid=((3, 60),))
    @test isempty(hostile_layer.accepted_steps)
    @test isempty(hostile_layer.models)
    @test all(row -> !row.accepted, hostile_layer.acceptance)
    superlearner_apply_l2!(hostile.year, hostile_layer)
    @test !any(hostile.year.l2_applied)
    @test all(iszero, hostile.year.residual)
    @test all(iszero, hostile.year.residual_raw)
end

# ---------------------------------------------------------------------------
# L3: depth-safe guard
# ---------------------------------------------------------------------------

@testset "the deepening cell matches the written thresholds" begin
    @test superlearner_deepening(-15.001, 0.0, 0.0)
    @test !superlearner_deepening(-15.0, 0.0, 0.0)           # strict inequality
    @test !superlearner_deepening(-14.999, 0.0, 0.0)
    @test superlearner_deepening(0.0, 0.1, -50.0)            # coupling active at the depth edge
    @test !superlearner_deepening(0.0, 0.0, -50.0)           # coupling must be positive
    @test !superlearner_deepening(0.0, 0.1, -49.999)         # depth threshold is inclusive
    @test superlearner_deepening(0.0, 3.0, -200.0)
    @test !superlearner_deepening(2.0, 0.0, 10.0)
end

@testset "the A2 variant guards the floor stack against the served static stack" begin
    # superlearner_d is the floor-constrained stack outside a deepening cell and the minimum
    # of that stack and the static stack inside one.
    rng = MersenneTwister(2426)
    n = 600
    origin = DateTime(2017, 4, 1)
    issues = [origin + Hour(k) for k in 0:(n - 1)]
    steps = [SUPERLEARNER_STEPS[1 + (k % length(SUPERLEARNER_STEPS))] for k in 0:(n - 1)]
    obs = [-40.0 - 60.0 * rand(rng) for _ in 1:n]
    latest = [-160.0 * rand(rng) for _ in 1:n]
    rate = [-30.0 * rand(rng) for _ in 1:n]
    coupling = [4.0 * (rand(rng) < 0.5) for _ in 1:n]
    static = [-100.0 * rand(rng) for _ in 1:n]
    year = tiny_year(2017, issues, steps, obs; latest=latest, rate=rate,
                     coupling=coupling,
                     comparators=Dict(:served_inertial_sindy => copy(static),
                                      :static_stack => static))
    year.l1 .= [-90.0 * rand(rng) for _ in 1:n]
    year.l1_floor .= [-90.0 * rand(rng) for _ in 1:n]
    year.l1_ten .= [-90.0 * rand(rng) for _ in 1:n]
    year.residual .= [10.0 * (rand(rng) - 0.5) for _ in 1:n]
    superlearner_build_centers!(year)
    deepening = 0
    for i in 1:n
        if superlearner_deepening(rate[i], coupling[i], latest[i])
            deepening += 1
            @test year.centers[:superlearner_d][i] == min(year.centers[:superlearner_a_floor][i], static[i])
            @test year.centers[:superlearner_d][i] <= static[i]
            @test year.centers[:superlearner_d][i] <= year.centers[:superlearner_a_floor][i]
        else
            @test year.centers[:superlearner_d][i] == year.centers[:superlearner_a_floor][i]
        end
    end
    # Both branches must be exercised, or the identity above proves nothing.
    @test 0 < deepening < n
    # The guard reference is the static stack, not the candidate's own center: a
    # variant that guarded against L1 would differ wherever the two disagree.
    @test any(i -> year.centers[:superlearner_d][i] != year.centers[:superlearner_c][i], 1:n)
    @test :superlearner_d in SUPERLEARNER_VARIANTS && :superlearner_d in SUPERLEARNER_SELECTABLE_VARIANTS
    # Amendment A3: superlearner_e is the ten-expert floor stack and superlearner_f is that stack under
    # the same static guard; the guarded one leads the tie-break order.
    for i in 1:n
        if superlearner_deepening(rate[i], coupling[i], latest[i])
            @test year.centers[:superlearner_f][i] == min(year.centers[:superlearner_e][i], static[i])
        else
            @test year.centers[:superlearner_f][i] == year.centers[:superlearner_e][i]
        end
    end
    @test year.centers[:superlearner_e] == year.l1_ten
    @test Set(SUPERLEARNER_SELECTABLE_VARIANTS) ==
          Set((:superlearner_a_floor, :superlearner_d, :superlearner_e, :superlearner_f))
    @test first(SUPERLEARNER_SELECTABLE_VARIANTS) === :superlearner_f
    @test :superlearner_b ∉ SUPERLEARNER_SELECTABLE_VARIANTS && :superlearner_c ∉ SUPERLEARNER_SELECTABLE_VARIANTS
    @test :superlearner_b in SUPERLEARNER_VARIANTS && :superlearner_c in SUPERLEARNER_VARIANTS
end

@testset "the guard lets the residual deepen but never lift" begin
    # Inside a deepening cell a lifting residual is discarded and a deepening one
    # is kept; outside, the residual passes through untouched.
    @test superlearner_guard(-100.0, -90.0, -20.0, 0.0, -100.0) == -100.0
    @test superlearner_guard(-100.0, -110.0, -20.0, 0.0, -100.0) == -110.0
    @test superlearner_guard(-100.0, -90.0, -1.0, 0.0, -10.0) == -90.0
    @test superlearner_guard(-100.0, -110.0, -1.0, 0.0, -10.0) == -110.0
    rng = MersenneTwister(1213)
    for _ in 1:500
        l1 = -200.0 * rand(rng)
        residual = 40.0 * (rand(rng) - 0.5)
        l2 = l1 + residual
        rate = -30.0 * rand(rng)
        coupling = 4.0 * rand(rng)
        latest = -150.0 * rand(rng)
        guarded = superlearner_guard(l1, l2, rate, coupling, latest)
        if superlearner_deepening(rate, coupling, latest)
            @test guarded <= l1 + 1e-12
            @test guarded == min(l1, l2)
        else
            @test guarded == l2
        end
    end
end

# ---------------------------------------------------------------------------
# L4: split conformal and the interval score
# ---------------------------------------------------------------------------

@testset "the conformal half-width restates the package quantile" begin
    rng = MersenneTwister(1415)
    for n in (1, 5, 19, 20, 137, 1000)
        residuals = randn(rng, n) .* 12.0
        mine = superlearner_conformal_halfwidth(residuals, SUPERLEARNER_COVERAGE)
        theirs = SolarSINDy._conformal_quantile(residuals, SUPERLEARNER_COVERAGE)
        @test mine.half_width == theirs[1]
        @test mine.coverage_floor == theirs[2]
        @test mine.n == n
        # The half-width is the k-th smallest absolute residual by construction.
        k = clamp(ceil(Int, (n + 1) * SUPERLEARNER_COVERAGE), 1, n)
        @test mine.half_width == sort(abs.(residuals))[k]
    end
    @test_throws ArgumentError superlearner_conformal_halfwidth(Float64[], SUPERLEARNER_COVERAGE)
    @test_throws ArgumentError superlearner_conformal_halfwidth([1.0], 1.0)
    @test_throws ArgumentError superlearner_conformal_halfwidth([1.0], 0.0)
end

@testset "split conformal covers at its nominal rate on exchangeable data" begin
    rng = MersenneTwister(1617)
    origin = DateTime(2015, 1, 1)
    n = 4 * 6_000
    issues = DateTime[]
    steps = Int[]
    for k in 0:(div(n, length(SUPERLEARNER_STEPS)) - 1), step in SUPERLEARNER_STEPS
        push!(issues, origin + Hour(k))
        push!(steps, step)
    end
    m = length(issues)
    # Heteroscedastic by depth bin and by step, which is exactly what the
    # (step, depth) strata of Amendment A1 exist to absorb.
    latest = [rand(rng) < 0.4 ? -80.0 - 40.0 * rand(rng) :
              (rand(rng) < 0.5 ? -40.0 - 20.0 * rand(rng) : -5.0 - 10.0 * rand(rng))
              for _ in 1:m]
    depth_scale = Dict(:shallow => 3.0, :moderate => 6.0, :deep => 12.0)
    scale = [depth_scale[superlearner_depth_bin(latest[i])] * (1.0 + 0.25 * steps[i])
             for i in 1:m]
    centers = zeros(m)
    calibration_obs = [scale[i] * randn(rng) for i in 1:m]
    evaluation_obs = [scale[i] * randn(rng) for i in 1:m]
    calibration = tiny_year(2015, issues, steps, calibration_obs; latest=latest)
    evaluation = tiny_year(2016, issues, steps, evaluation_obs; latest=latest)
    strata = superlearner_fit_conformal(Tuple{SuperlearnerYearData,Vector{Float64}}[(calibration, centers)])
    half = superlearner_apply_conformal(evaluation, strata)
    covered = count(i -> abs(evaluation_obs[i]) <= half[i], 1:m)
    coverage = covered / m
    # Five binomial standard errors around the nominal level: a stratum mix-up or
    # an off-by-one in the quantile index moves coverage far outside this band.
    band = 5.0 * sqrt(SUPERLEARNER_COVERAGE * (1 - SUPERLEARNER_COVERAGE) / m)
    @test abs(coverage - SUPERLEARNER_COVERAGE) <= band
    # The bins must be ordered in width at every step, which a single "disturbed"
    # stratum could not represent: that is what drove Amendment A1.
    for step in SUPERLEARNER_STEPS
        @test strata[(step, :deep)].half_width > strata[(step, :moderate)].half_width
        @test strata[(step, :moderate)].half_width > strata[(step, :shallow)].half_width
        @test strata[(step, :deep)].half_width > strata[(step, SUPERLEARNER_POOLED_DEPTH)].half_width
        for depth in (SUPERLEARNER_DEPTH_BINS..., SUPERLEARNER_POOLED_DEPTH)
            @test strata[(step, depth)].source == "own"
        end
    end
    # The half-width must grow with lead time under this generator.
    @test strata[(7, :deep)].half_width > strata[(1, :deep)].half_width
    # Per-bin coverage, not only the pooled rate: the deep rows are the ones the
    # single-stratum calibration under-covered.
    for depth in SUPERLEARNER_DEPTH_BINS
        rows = [i for i in 1:m if superlearner_depth_bin(latest[i]) === depth]
        @test length(rows) > 1_000
        bin_coverage = count(i -> abs(evaluation_obs[i]) <= half[i], rows) / length(rows)
        @test abs(bin_coverage - SUPERLEARNER_COVERAGE) <=
              5.0 * sqrt(SUPERLEARNER_COVERAGE * (1 - SUPERLEARNER_COVERAGE) / length(rows))
    end
end

@testset "a thin conformal depth stratum inherits its step's pooled stratum" begin
    origin = DateTime(2015, 6, 1)
    issues = DateTime[]
    steps = Int[]
    latest = Float64[]
    obs = Float64[]
    # Plenty of shallow rows, and only three deep rows, at every step.
    for step in SUPERLEARNER_STEPS
        for k in 1:400
            push!(issues, origin + Hour(k))
            push!(steps, step)
            push!(latest, -5.0)
            push!(obs, 2.0)
        end
        for k in 401:403
            push!(issues, origin + Hour(k))
            push!(steps, step)
            push!(latest, -80.0)
            push!(obs, 30.0)
        end
    end
    year = tiny_year(2015, issues, steps, obs; latest=latest)
    strata = superlearner_fit_conformal(Tuple{SuperlearnerYearData,Vector{Float64}}[(year, zeros(length(obs)))])
    for step in SUPERLEARNER_STEPS
        @test strata[(step, :shallow)].source == "own"
        @test strata[(step, SUPERLEARNER_POOLED_DEPTH)].source == "own"
        # Three rows cannot carry a 0.90 quantile, so the deep bin takes the
        # step's pooled stratum, and the moderate bin — empty here — takes it too.
        @test startswith(strata[(step, :deep)].source, "pooled_fallback_thin_3")
        @test strata[(step, :moderate)].source == "pooled_fallback_absent"
        @test strata[(step, :deep)].half_width ==
              strata[(step, SUPERLEARNER_POOLED_DEPTH)].half_width
        @test strata[(step, :moderate)].half_width ==
              strata[(step, SUPERLEARNER_POOLED_DEPTH)].half_width
        @test strata[(step, :deep)].n == strata[(step, SUPERLEARNER_POOLED_DEPTH)].n
    end
    # A row in a bin whose stratum fell back is still given a half-width.
    half = superlearner_apply_conformal(year, strata)
    @test all(isfinite, half)
    @test all(>(0.0), half)
    @test_throws ErrorException superlearner_fit_conformal(
        Tuple{SuperlearnerYearData,Vector{Float64}}[],
    )
end

@testset "the interval score is the Winkler score at the nominal level" begin
    alpha = 1.0 - SUPERLEARNER_COVERAGE
    # Covering: the score is the width.
    @test superlearner_interval_score(-50.0, -60.0, -40.0, alpha) == 20.0
    @test superlearner_interval_score(-60.0, -60.0, -40.0, alpha) == 20.0
    # Below: width plus 2/alpha times the shortfall.
    @test superlearner_interval_score(-70.0, -60.0, -40.0, alpha) ≈ 20.0 + 2 * 10.0 / alpha
    # Above: the symmetric penalty.
    @test superlearner_interval_score(-30.0, -60.0, -40.0, alpha) ≈ 20.0 + 2 * 10.0 / alpha
    # A degenerate interval scores its miss alone.
    @test superlearner_interval_score(1.0, 0.0, 0.0, alpha) ≈ 2 * 1.0 / alpha
    @test_throws ArgumentError superlearner_interval_score(0.0, 1.0, 0.0, alpha)
    @test_throws ArgumentError superlearner_interval_score(0.0, 0.0, 1.0, 0.0)
end

# ---------------------------------------------------------------------------
# Bootstrap and pairing
# ---------------------------------------------------------------------------

@testset "the block bootstrap reproduces a constant-difference oracle" begin
    origin = DateTime(2014, 1, 1)
    issues = [origin + Hour(k) for k in 0:(24 * 70 - 1)]
    # Constant squared errors make every resampled multiset give the same two
    # root mean square errors, so the point estimate, the lower bound and the
    # p-value are all known in closed form.
    candidate = fill(4.0, length(issues))
    comparator = fill(9.0, length(issues))
    result = SolarSINDy.analog_block_bootstrap(comparator, candidate, issues;
                                            replicates=500)
    @test result.point ≈ 3.0 - 2.0
    @test result.lower ≈ 1.0
    @test result.p_one_sided ≈ 1 / 501
    # The blocks are the fixed calendar grid anchored at 2010-01-01, not windows
    # anchored at the first scored row: 2014-01-01 lies five days into a block
    # (1461 days after the epoch is 208 weeks plus five days), so a 70-day span
    # covers a two-day stub, nine whole blocks and a five-day stub.
    @test result.n_blocks == 11
    aligned_origin = SolarSINDy.ANALOG_BOOTSTRAP_EPOCH +
                     Hour(208 * SolarSINDy.ANALOG_BOOTSTRAP_BLOCK_HOURS)
    aligned_issues = [aligned_origin + Hour(k) for k in 0:(24 * 70 - 1)]
    @test SolarSINDy.analog_block_bootstrap(
        fill(9.0, length(aligned_issues)), fill(4.0, length(aligned_issues)),
        aligned_issues; replicates=50,
    ).n_blocks == 10
    # Reversing the roles flips the sign and exhausts the one-sided tail.
    reversed = SolarSINDy.analog_block_bootstrap(candidate, comparator, issues;
                                              replicates=500)
    @test reversed.point ≈ -1.0
    @test reversed.p_one_sided ≈ 1.0
end

@testset "pairing keeps identical rows and a reproducible order" begin
    origin = DateTime(2014, 3, 1)
    issues = [origin + Hour(k) for k in 0:99]
    steps = fill(3, 100)
    obs = collect(range(-100.0, -1.0; length=100))
    candidate = obs .+ 1.0
    comparator = obs .+ 2.0
    comparator[7] = NaN
    year = tiny_year(2014, issues, steps, obs;
                     centers=Dict(:superlearner_c => candidate),
                     comparators=Dict(:direct_gbm => comparator))
    paired = superlearner_paired_rows([year], 2014:2014, :superlearner_c, :direct_gbm, 3)
    @test length(paired.issues) == 99
    @test paired.n_candidate == 100
    @test paired.n_comparator == 99
    @test issorted(paired.issues)
    @test all(x -> x ≈ 1.0, paired.candidate_se)
    @test all(x -> x ≈ 4.0, paired.comparator_se)
    # Shuffling the fold order cannot move the paired arrays.
    shuffled = superlearner_paired_rows([year], 2014:2014, :superlearner_c, :direct_gbm, 3)
    @test shuffled.issues == paired.issues
    @test superlearner_paired_rows([year], 2014:2014, :superlearner_c, :direct_gbm, 1).issues == DateTime[]
end

@testset "pooled metrics and interval metrics match hand arithmetic" begin
    origin = DateTime(2014, 5, 1)
    issues = DateTime[]
    steps = Int[]
    for k in 0:9, step in (1, 3)
        push!(issues, origin + Hour(k))
        push!(steps, step)
    end
    n = length(issues)
    obs = fill(-40.0, n)
    centers = Dict(variant => fill(-43.0, n) for variant in SUPERLEARNER_VARIANTS)
    comparators = Dict(Symbol(c) => fill(-36.0, n)
                       for c in (SUPERLEARNER_GATED_COMPARATORS..., SUPERLEARNER_REPORTED_COMPARATORS...))
    half = Dict(:superlearner_c => fill(2.5, n), :served_inertial_sindy => fill(5.0, n))
    year = tiny_year(2014, issues, steps, obs; centers=centers,
                     comparators=comparators, half_widths=half)
    summary = superlearner_summary_rows([year], (ALL=2014:2014,))
    hit = only([r for r in summary if r.model == "superlearner_c" && r.model_step_hours == 3])
    @test hit.n == 10
    @test hit.rmse_nt ≈ 3.0
    @test hit.bias_nt ≈ 3.0
    @test hit.mae_nt ≈ 3.0
    served = only([r for r in summary if r.model == "served_inertial_sindy" && r.model_step_hours == 1])
    @test served.rmse_nt ≈ 4.0
    @test served.bias_nt ≈ -4.0
    # Every scored model appears at every step, with a NaN where nothing is scored.
    @test length(summary) == length(superlearner_scored_models()) * length(SUPERLEARNER_STEPS)
    @test all(r -> r.n == 0 && isnan(r.rmse_nt),
              [r for r in summary if r.model_step_hours == 7])

    # The candidate interval (+/-2.5 nT) never covers a 3 nT error, so coverage is
    # zero and the interval score is the width plus the symmetric penalty.
    intervals = superlearner_interval_rows([year], (ALL=2014:2014,), (:superlearner_c, :served_inertial_sindy))
    candidate = only([r for r in intervals if r.variant == "superlearner_c" &&
                      r.subset == "pooled" && r.model_step_hours == 0])
    @test candidate.n == 20
    @test candidate.coverage == 0.0
    @test candidate.mean_width_nt ≈ 5.0
    @test candidate.mean_interval_score_nt ≈ 5.0 + 2 * 0.5 / (1 - SUPERLEARNER_COVERAGE)
    # The served interval (+/-5 nT) covers a 4 nT error every time.
    reference = only([r for r in intervals if r.variant == "served_inertial_sindy" &&
                      r.subset == "pooled" && r.model_step_hours == 0])
    @test reference.coverage == 1.0
    @test reference.mean_interval_score_nt ≈ 10.0
end

# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

@testset "G1 needs a strict win over every comparator with the written margin" begin
    inputs = passing_inputs("ALL")
    @test verdicts_for(inputs, "ALL")[("ALL", "G1")]

    # Tying the best comparator is not "lower RMSE than every comparator".
    tied = replace_row(inputs.summary,
                       r -> r.model == "direct_gbm" && r.model_step_hours == 3,
                       summary_row("ALL", "direct_gbm", 3, 1000, 5.0))
    @test !verdicts_for(merge(inputs, (summary=tied,)), "ALL")[("ALL", "G1")]

    # A comparator that beats the candidate anywhere fails the gate.
    beaten = replace_row(inputs.summary,
                         r -> r.model == "obrien" && r.model_step_hours == 6,
                         summary_row("ALL", "obrien", 6, 1000, 4.0))
    @test !verdicts_for(merge(inputs, (summary=beaten,)), "ALL")[("ALL", "G1")]

    # Amendment A2 reports the max(0.10 nT, 1 %) margin and gates on it nowhere:
    # a 0.09 nT gain with a positive lower bound now passes, and the margin flag in
    # the persisted row says the margin was not met.
    small = [bootstrap_row("ALL", String(c), s, 1000, 0.09, 0.05, 0.001)
             for c in SUPERLEARNER_GATED_COMPARATORS for s in SUPERLEARNER_STEPS]
    small_gates = superlearner_gate_rows(inputs.summary, inputs.cells, small, inputs.intervals,
                                (ALL=2014:2019,), :superlearner_c)
    @test small_gates.verdicts[("ALL", "G1")]
    for row in [r for r in small_gates.rows if r.gate == "G1"]
        @test row.rule == "superiority"
        @test row.margin_required_nt ≈ 0.10
        @test !row.margin_pass
        @test row.headroom_nt ≈ 3.0
        @test row.beat_all
    end
    # With a 20 nT best comparator the reported margin follows the 1 % branch.
    big = NamedTuple[]
    for step in SUPERLEARNER_STEPS
        push!(big, summary_row("ALL", "superlearner_c", step, 1000, 19.0))
        for comparator in SUPERLEARNER_GATED_COMPARATORS
            push!(big, summary_row("ALL", String(comparator), step, 1000, 20.0))
        end
        push!(big, summary_row("ALL", String(SUPERLEARNER_ORACLE_COLUMN), step, 1000, 3.0))
    end
    big_gates = superlearner_gate_rows(big, inputs.cells,
                              [bootstrap_row("ALL", String(c), s, 1000, 0.19, 0.05, 0.001)
                               for c in SUPERLEARNER_GATED_COMPARATORS for s in SUPERLEARNER_STEPS],
                              inputs.intervals, (ALL=2014:2019,), :superlearner_c)
    @test big_gates.verdicts[("ALL", "G1")]
    for row in [r for r in big_gates.rows if r.gate == "G1"]
        @test row.margin_required_nt ≈ 0.20
        @test !row.margin_pass
    end

    # A non-positive bootstrap lower bound, a Holm-adjusted p at or above the
    # level, and a row-count mismatch each fail on their own.
    for rows in (
        [bootstrap_row("ALL", String(c), s, 1000, 1.0, 0.0, 0.001)
         for c in SUPERLEARNER_GATED_COMPARATORS for s in SUPERLEARNER_STEPS],
        [bootstrap_row("ALL", String(c), s, 1000, 1.0, 0.5, SUPERLEARNER_ALPHA)
         for c in SUPERLEARNER_GATED_COMPARATORS for s in SUPERLEARNER_STEPS],
        [bootstrap_row("ALL", String(c), s, 1000, 1.0, 0.5, 0.001; matched=false)
         for c in SUPERLEARNER_GATED_COMPARATORS for s in SUPERLEARNER_STEPS],
    )
        @test !verdicts_for(merge(inputs, (bootstrap=rows,)), "ALL")[("ALL", "G1")]
    end
    # Unequal scored-row counts in the summary also break "identical rows".
    uneven = replace_row(inputs.summary,
                         r -> r.model == "analog_shadow" && r.model_step_hours == 2,
                         summary_row("ALL", "analog_shadow", 2, 999, 6.0))
    @test !verdicts_for(merge(inputs, (summary=uneven,)), "ALL")[("ALL", "G1")]
end

@testset "the Amendment A2 headroom rule replaces superiority near the ceiling" begin
    # Headroom is best comparator minus the realized-driver oracle. Above the
    # threshold the superiority clause applies; below it, only non-inferiority.
    wide = passing_inputs("ALL"; oracle=3.0)
    narrow = passing_inputs("ALL"; oracle=6.0 - 0.24)
    edge = passing_inputs("ALL"; oracle=6.0 - SUPERLEARNER_G1_HEADROOM_NT)
    rule_of(inputs, boot) = begin
        gates = superlearner_gate_rows(inputs.summary, inputs.cells, boot, inputs.intervals,
                              (ALL=2014:2019,), :superlearner_c)
        (verdict=gates.verdicts[("ALL", "G1")],
         rows=[r for r in gates.rows if r.gate == "G1"])
    end
    passing_boot = [bootstrap_row("ALL", String(c), s, 1000, 1.0, 0.5, 0.001)
                    for c in SUPERLEARNER_GATED_COMPARATORS for s in SUPERLEARNER_STEPS]
    @test all(r -> r.rule == "superiority", rule_of(wide, passing_boot).rows)
    @test all(r -> r.rule == "non_inferiority", rule_of(narrow, passing_boot).rows)
    # The threshold is strict, so a headroom exactly at 0.25 nT keeps superiority.
    @test all(r -> r.rule == "superiority", rule_of(edge, passing_boot).rows)
    for row in rule_of(narrow, passing_boot).rows
        @test row.headroom_nt ≈ 0.24
        @test row.oracle_rmse_nt ≈ 5.76
        @test row.best_comparator_rmse_nt ≈ 6.0
    end
    # Under the non-inferiority clause a small negative lower bound still passes,
    # and one below -0.05 nT does not; the Holm p-value stops being decisive.
    for (lower, expected) in ((0.5, true), (-0.049, true), (-0.05, false),
                              (-0.30, false))
        boot = [bootstrap_row("ALL", String(c), s, 1000, -0.01, lower, 0.9)
                for c in SUPERLEARNER_GATED_COMPARATORS for s in SUPERLEARNER_STEPS]
        @test rule_of(narrow, boot).verdict == expected
    end
    # A step that loses outright still passes the headroom-limited clause as long as
    # the loss is inside the non-inferiority bound: that is what the amendment says.
    tied = replace_row(narrow.summary,
                       r -> r.model == "direct_gbm" && r.model_step_hours == 1,
                       summary_row("ALL", "direct_gbm", 1, 1000, 4.99))
    boot = [bootstrap_row("ALL", String(c), s, 1000, -0.01, -0.02, 0.9)
            for c in SUPERLEARNER_GATED_COMPARATORS for s in SUPERLEARNER_STEPS]
    result = rule_of(merge(narrow, (summary=tied,)), boot)
    @test result.verdict
    @test !first([r for r in result.rows if r.model_step_hours == 1]).beat_all
    # The same loss under the superiority clause fails.
    @test !rule_of(merge(wide, (summary=replace_row(
        wide.summary, r -> r.model == "direct_gbm" && r.model_step_hours == 1,
        summary_row("ALL", "direct_gbm", 1, 1000, 4.99)),)), boot).verdict
    # Without an oracle row the headroom is unknown and the strict clause applies.
    without = [r for r in wide.summary if r.model != String(SUPERLEARNER_ORACLE_COLUMN)]
    rows = rule_of(merge(wide, (summary=without,)), passing_boot).rows
    @test all(r -> r.rule == "superiority" && isnan(r.headroom_nt), rows)
end

@testset "G2 enforces the storm-cell losses exactly as written" begin
    inputs = passing_inputs("ALL")
    @test verdicts_for(inputs, "ALL")[("ALL", "G2")]

    # A guarded cell may lose at most 0.50 nT to the best comparator: 6.50 passes,
    # 6.51 fails. Served inertial-SINDy sits at 6.0, so the candidate must also stay at or
    # below 6.0 there; the loss-to-served clause is tested separately below.
    for (rmse, expected) in ((6.5, false), (6.0, true))
        rows = replace_row(inputs.cells,
                           r -> r.cell == "recovery" && r.model == "superlearner_c" &&
                                r.model_step_hours == 4,
                           cell_row("ALL", "recovery", "superlearner_c", 4, 200, rmse))
        @test verdicts_for(merge(inputs, (cells=rows,)), "ALL")[("ALL", "G2")] == expected
    end
    # Loss to served inertial-SINDy is not tolerated at all, even well inside 0.50 nT.
    rows = replace_row(inputs.cells,
                       r -> r.cell == "latest_le_m50" && r.model == "superlearner_c" &&
                            r.model_step_hours == 2,
                       cell_row("ALL", "latest_le_m50", "superlearner_c", 2, 200, 6.01))
    @test !verdicts_for(merge(inputs, (cells=rows,)), "ALL")[("ALL", "G2")]
    # Beyond 0.50 nT against the best comparator fails as well.
    rows = replace_row(inputs.cells,
                       r -> r.cell == "latest_le_m100" && r.model == "superlearner_c" &&
                            r.model_step_hours == 6,
                       cell_row("ALL", "latest_le_m100", "superlearner_c", 6, 200, 6.75))
    @test !verdicts_for(merge(inputs, (cells=rows,)), "ALL")[("ALL", "G2")]

    # A cell thinner than 40 rows is not evaluated, so a large loss there passes.
    thin = collect(inputs.cells)
    for index in eachindex(thin)
        row = thin[index]
        (row.cell == "active_deepening" && row.model_step_hours == 7) || continue
        thin[index] = cell_row("ALL", "active_deepening", row.model, 7, 39,
                               row.model == "superlearner_c" ? 99.0 : 6.0)
    end
    @test verdicts_for(merge(inputs, (cells=thin,)), "ALL")[("ALL", "G2")]

    # Intense deepening at six hours: any RMSE loss fails, and the mean signed
    # error must stay inside +/-10 nT.
    rows = replace_row(inputs.cells,
                       r -> r.cell == String(SUPERLEARNER_G2_INTENSE_CELL) && r.model == "superlearner_c" &&
                            r.model_step_hours == SUPERLEARNER_G2_INTENSE_STEP,
                       cell_row("ALL", String(SUPERLEARNER_G2_INTENSE_CELL), "superlearner_c",
                                SUPERLEARNER_G2_INTENSE_STEP, 200, 6.05))
    @test !verdicts_for(merge(inputs, (cells=rows,)), "ALL")[("ALL", "G2")]
    for (bias, expected) in ((10.0, true), (10.01, false), (-10.01, false))
        rows = replace_row(inputs.cells,
                           r -> r.cell == String(SUPERLEARNER_G2_INTENSE_CELL) &&
                                r.model == "superlearner_c" &&
                                r.model_step_hours == SUPERLEARNER_G2_INTENSE_STEP,
                           cell_row("ALL", String(SUPERLEARNER_G2_INTENSE_CELL), "superlearner_c",
                                    SUPERLEARNER_G2_INTENSE_STEP, 200, 5.0; bias=bias))
        @test verdicts_for(merge(inputs, (cells=rows,)), "ALL")[("ALL", "G2")] == expected
    end
end

@testset "Amendment A2 counts a G2 cell loss only when the bootstrap supports it" begin
    # One deep cell, hourly for sixty days, so the fixed 168 h block grid has
    # several blocks to resample. The candidate loses the same pooled amount in two
    # different ways: concentrated in a single block, and spread over every row.
    origin = DateTime(2015, 3, 1)
    hours = 24 * 60
    issues = [origin + Hour(k) for k in 0:(hours - 1)]
    steps = fill(4, hours)
    obs = zeros(hours)
    latest = fill(-60.0, hours)
    function fixture(candidate_error, comparator_error)
        centers = Dict(variant => copy(candidate_error) for variant in SUPERLEARNER_VARIANTS)
        comparators = Dict{Symbol,Vector{Float64}}()
        for name in (SUPERLEARNER_GATED_COMPARATORS..., SUPERLEARNER_REPORTED_COMPARATORS...)
            comparators[name] = comparator_error .+ 50.0
        end
        # The served product is the best comparator in the cell, so it carries both
        # the best-comparator clause and the never-below-served clause.
        comparators[:served_inertial_sindy] = copy(comparator_error)
        year = tiny_year(2015, issues, steps, obs; latest=latest, centers=centers,
                         comparators=comparators)
        eras = (ALL=2015:2015,)
        cells = superlearner_cell_rows([year], eras)
        return (years=[year], cells=cells)
    end
    # (a) The whole loss sits in the first 168 h block: a resample that misses that
    # block sees no loss at all, so the lower bound cannot exceed zero.
    spike_candidate = fill(1.0, hours)
    spike_candidate[1:168] .= 120.0
    concentrated = fixture(spike_candidate, fill(1.0, hours))
    guard = superlearner_storm_guard(concentrated.cells, :ALL, :superlearner_c;
                            years=concentrated.years, scope_years=2015:2015,
                            replicates=500)
    detail = [row for row in guard.detail if row.cell == "latest_le_m50"]
    @test !isempty(detail)
    @test all(row -> row.loss_nt > SUPERLEARNER_G2_MAX_LOSS_NT, detail)
    @test all(row -> row.lower_nt <= 0.0, detail)
    @test all(row -> !row.counted, detail)
    @test guard.pass
    @test isempty(guard.failures)
    @test occursin("latest_le_m50", guard.within_noise)
    # Without the bootstrap the same cell table is a hard breach, which is what the
    # amendment changed and what a unit caller still sees.
    @test !superlearner_storm_guard(concentrated.cells, :ALL, :superlearner_c).pass

    # (b) The same pooled loss spread over every row: every resample sees it, so the
    # lower bound is positive and the cell counts.
    spread = fixture(fill(2.0, hours), fill(1.0, hours))
    supported = superlearner_storm_guard(spread.cells, :ALL, :superlearner_c; years=spread.years,
                                scope_years=2015:2015, replicates=500)
    spread_detail = [row for row in supported.detail if row.cell == "latest_le_m50"]
    @test all(row -> row.lower_nt > 0.0, spread_detail)
    @test all(row -> row.counted, spread_detail)
    @test !supported.pass
    @test occursin("latest_le_m50", supported.failures)
    @test isempty(supported.within_noise)
    # The bootstrap is run on the cell's own rows, and the reported loss agrees with
    # the cell table to floating-point noise.
    for row in spread_detail
        @test row.n == hours
        @test row.n_blocks >= 2
        @test isapprox(row.bootstrap_loss_nt, row.loss_nt; atol=1e-9)
    end
    # A candidate that wins in the cell produces no record at all.
    winning = fixture(fill(1.0, hours), fill(2.0, hours))
    clean = superlearner_storm_guard(winning.cells, :ALL, :superlearner_c; years=winning.years,
                            scope_years=2015:2015, replicates=500)
    @test clean.pass
    @test isempty(clean.detail)
end

@testset "G3 enforces coverage, width and interval score" begin
    inputs = passing_inputs("ALL")
    @test verdicts_for(inputs, "ALL")[("ALL", "G3")]

    for (coverage, expected) in ((0.85, true), (0.95, true), (0.8499, false),
                                 (0.9501, false))
        rows = replace_row(inputs.intervals,
                           r -> r.variant == "superlearner_c" && r.subset == "pooled" &&
                                r.model_step_hours == 0,
                           interval_row("ALL", "superlearner_c", "pooled", 0, 1000, coverage, 10.0,
                                        12.0))
        @test verdicts_for(merge(inputs, (intervals=rows,)), "ALL")[("ALL", "G3")] ==
              expected
    end
    # Mean width may reach 1.10 x inertial-SINDy (11.0 nT) but no further.
    for (width, expected) in ((12.1, true), (12.11, false))
        rows = replace_row(inputs.intervals,
                           r -> r.variant == "superlearner_c" && r.subset == "pooled" &&
                                r.model_step_hours == 0,
                           interval_row("ALL", "superlearner_c", "pooled", 0, 1000, 0.90, width,
                                        12.0))
        @test verdicts_for(merge(inputs, (intervals=rows,)), "ALL")[("ALL", "G3")] ==
              expected
    end
    # The interval score may tie inertial-SINDy's but not exceed it.
    for (score, expected) in ((13.0, true), (13.01, false))
        rows = replace_row(inputs.intervals,
                           r -> r.variant == "superlearner_c" && r.subset == "pooled" &&
                                r.model_step_hours == 0,
                           interval_row("ALL", "superlearner_c", "pooled", 0, 1000, 0.90, 10.0,
                                        score))
        @test verdicts_for(merge(inputs, (intervals=rows,)), "ALL")[("ALL", "G3")] ==
              expected
    end
    # Per-step and storm coverage each carry their own 0.80 floor.
    for (subset, step) in (("pooled", 6), ("storm_le_m50", 0))
        for (coverage, expected) in ((0.80, true), (0.7999, false))
            rows = replace_row(inputs.intervals,
                               r -> r.variant == "superlearner_c" && r.subset == subset &&
                                    r.model_step_hours == step,
                               interval_row("ALL", "superlearner_c", subset, step, 160, coverage,
                                            10.0, 12.0))
            @test verdicts_for(merge(inputs, (intervals=rows,)), "ALL")[("ALL", "G3")] ==
                  expected
        end
    end
    # Without a inertial-SINDy reference row the gate cannot be evaluated and must fail.
    stripped = [r for r in inputs.intervals
                if !(r.variant == "served_inertial_sindy" && r.subset == "pooled" &&
                     r.model_step_hours == 0)]
    @test !verdicts_for(merge(inputs, (intervals=stripped,)), "ALL")[("ALL", "G3")]
end

@testset "the A3 serve rule compares against the served product only" begin
    # Sixty days of hourly rows at every step, so the fixed block grid has enough
    # blocks for the bootstrap, and one deep cell that can be manipulated.
    origin = DateTime(2021, 3, 1)
    hours = 24 * 60
    issues = DateTime[]
    steps = Int[]
    latest = Float64[]
    for k in 0:(hours - 1), step in SUPERLEARNER_STEPS
        push!(issues, origin + Hour(k))
        push!(steps, step)
        # A fifth of the rows sit below -50 nT, which is the storm cell the rule
        # checks; the rest are quiet.
        push!(latest, k % 5 == 0 ? -60.0 : -10.0)
    end
    m = length(issues)
    obs = zeros(m)
    function fixture(candidate_error, served_error)
        centers = Dict(variant => copy(candidate_error) for variant in SUPERLEARNER_VARIANTS)
        comparators = Dict{Symbol,Vector{Float64}}()
        for name in (SUPERLEARNER_GATED_COMPARATORS..., SUPERLEARNER_REPORTED_COMPARATORS...)
            comparators[name] = served_error .+ 30.0
        end
        comparators[SUPERLEARNER_SERVE_REFERENCE] = copy(served_error)
        half = Dict(variant => fill(20.0, m) for variant in SUPERLEARNER_VARIANTS)
        half[:served_inertial_sindy] = fill(20.0, m)
        year = tiny_year(2021, issues, steps, obs; latest=latest, centers=centers,
                         comparators=comparators, half_widths=half)
        eras = (ALL=2021:2021, E2=2021:2021)
        intervals = superlearner_interval_rows([year], eras, (SUPERLEARNER_VARIANTS..., :served_inertial_sindy))
        g3 = Dict(("ALL", "G3") => true, ("E2", "G3") => true)
        return superlearner_serve_rule_rows([year], eras, :superlearner_f, intervals, g3; replicates=400)
    end
    # (a) The candidate beats the served product everywhere by 1 nT.
    better = fixture(fill(1.0, m), fill(2.0, m))
    @test better.serve
    @test Set(better.deciding) == Set(["ALL", "E2"])
    pooled = [r for r in better.rows if r.check == "pooled_gain"]
    @test length(pooled) == 2 * length(SUPERLEARNER_STEPS)
    @test all(r -> r.gain_nt > 0.0 && r.lower_nt > 0.0 && r.pass, pooled)
    @test all(r -> r.comparator == String(SUPERLEARNER_SERVE_REFERENCE), better.rows)
    cells = [r for r in better.rows if r.check == "storm_cell"]
    @test !isempty(cells)
    @test all(r -> r.pass, cells)
    @test all(r -> r.n >= SUPERLEARNER_G2_MIN_CELL_ROWS, cells)
    @test only([r for r in better.rows if r.check == "serve_rule"]).pass
    # (b) A uniform loss fails clause one on both eras.
    worse = fixture(fill(2.0, m), fill(1.0, m))
    @test !worse.serve
    @test all(r -> !r.pass, [r for r in worse.rows if r.check == "pooled_gain"])
    # (c) Pooled superiority with a supported loss inside the storm cell fails
    #     clause two: the candidate is better overall and worse where it matters.
    # Pooled: sqrt(0.2*4 + 0.8*0.04) = 0.91 nT against the served 1.0 nT, so the
    # candidate wins overall while losing 2:1 inside the storm cell.
    candidate = [latest[i] <= -50.0 ? 2.0 : 0.2 for i in 1:m]
    mixed = fixture(candidate, fill(1.0, m))
    @test !mixed.serve
    storm = [r for r in mixed.rows if r.check == "storm_cell" && r.cell == "latest_le_m50"]
    @test !isempty(storm)
    @test all(r -> !r.pass && r.lower_nt > 0.0, storm)
    @test all(r -> r.pass, [r for r in mixed.rows if r.check == "pooled_gain"])
    # (d) G3 is a clause of the rule, not decoration.
    origin_rows = fixture(fill(1.0, m), fill(2.0, m))
    @test origin_rows.serve
    eras = (ALL=2021:2021, E2=2021:2021)
    centers = Dict(variant => fill(1.0, m) for variant in SUPERLEARNER_VARIANTS)
    comparators = Dict{Symbol,Vector{Float64}}()
    for name in (SUPERLEARNER_GATED_COMPARATORS..., SUPERLEARNER_REPORTED_COMPARATORS...)
        comparators[name] = fill(32.0, m)
    end
    comparators[SUPERLEARNER_SERVE_REFERENCE] = fill(2.0, m)
    year = tiny_year(2021, issues, steps, obs; latest=latest, centers=centers,
                     comparators=comparators,
                     half_widths=Dict(variant => fill(20.0, m) for variant in SUPERLEARNER_VARIANTS))
    failed_g3 = superlearner_serve_rule_rows([year], eras, :superlearner_f,
                                    superlearner_interval_rows([year], eras, SUPERLEARNER_VARIANTS),
                                    Dict(("ALL", "G3") => true, ("E2", "G3") => false);
                                    replicates=200)
    @test !failed_g3.serve
    @test failed_g3.verdicts["ALL"]
    @test !failed_g3.verdicts["E2"]
    # E1 is scored but never decides, because the served product is partly in-sample
    # there.
    with_e1 = superlearner_serve_rule_rows([year], (ALL=2021:2021, E1=2021:2021), :superlearner_f,
                                  superlearner_interval_rows([year], (ALL=2021:2021, E1=2021:2021),
                                                    SUPERLEARNER_VARIANTS),
                                  Dict(("ALL", "G3") => true, ("E1", "G3") => false);
                                  replicates=200)
    @test with_e1.deciding == ["ALL"]
    @test with_e1.serve
    @test !with_e1.verdicts["E1"]
    @test all(r -> !r.decides, [r for r in with_e1.rows if r.scope == "E1"])
    @test SUPERLEARNER_SERVE_RULE_ERAS == (:ALL, :E2)
    @test SUPERLEARNER_SERVE_REFERENCE === :static_stack
end

@testset "the decision arithmetic follows plan section 6" begin
    eras = (ALL=2014:2025, E1=2014:2019, E2=2020:2025)
    all_pass = Dict{Tuple{String,String},Bool}()
    for scope in ("ALL", "E1", "E2"), gate in ("G1", "G2", "G3")
        all_pass[(scope, gate)] = true
    end
    @test superlearner_decision(all_pass, eras).state == "SERVE_PENDING_G4"
    @test isempty(superlearner_decision(all_pass, eras).failing)

    one_era = copy(all_pass)
    one_era[("E2", "G1")] = false
    one_era[("ALL", "G1")] = false
    decision = superlearner_decision(one_era, eras)
    @test decision.state == "SHADOW"
    @test "G1@E2" in decision.failing
    @test "G1@ALL" in decision.failing

    both_eras = copy(all_pass)
    both_eras[("E1", "G1")] = false
    both_eras[("E2", "G1")] = false
    both_eras[("ALL", "G1")] = false
    @test superlearner_decision(both_eras, eras).state == "NO_GO"

    guard_failure = copy(all_pass)
    guard_failure[("E1", "G2")] = false
    @test superlearner_decision(guard_failure, eras).state == "SHADOW"

    interval_failure = copy(all_pass)
    interval_failure[("ALL", "G3")] = false
    @test superlearner_decision(interval_failure, eras).state == "NO_GO"

    @test_throws ErrorException superlearner_decision(Dict{Tuple{String,String},Bool}(), eras)
end

@testset "the selection rule reads Amendment A1 literally" begin
    origin = DateTime(2015, 2, 1)
    eras = (E1=2014:2019,)
    # `errors` gives each variant its constant absolute error, so the mean over the
    # selection steps of the per-step pooled RMSE is that constant; `guards` says
    # which variants clear the E1 storm cells.
    function selection_fixture(errors::Dict{Symbol,Float64};
                               guards=Dict(v => true for v in SUPERLEARNER_SELECTABLE_VARIANTS))
        issues = DateTime[]
        steps = Int[]
        for k in 0:199, step in SUPERLEARNER_SELECTION_STEPS
            push!(issues, origin + Hour(k))
            push!(steps, step)
        end
        n = length(issues)
        obs = zeros(n)
        centers = Dict(variant => fill(errors[variant], n)
                       for variant in SUPERLEARNER_SELECTABLE_VARIANTS)
        year = tiny_year(2015, issues, steps, obs; centers=centers)
        cells = NamedTuple[]
        for cell in SUPERLEARNER_G2_CELLS, step in SUPERLEARNER_STEPS
            for variant in SUPERLEARNER_SELECTABLE_VARIANTS
                push!(cells, cell_row("E1", String(cell), String(variant), step, 200,
                                      guards[variant] ? 5.0 : 99.0))
            end
            for comparator in SUPERLEARNER_GATED_COMPARATORS
                push!(cells, cell_row("E1", String(cell), String(comparator), step, 200,
                                      6.0))
            end
        end
        # The cell table here is hand-built rather than derived from the fold, so
        # the A2 bootstrap support has no rows to resample; the strict deterministic
        # reading of the guards is what this test is about, and the bootstrap
        # variant has its own test above.
        return superlearner_select_variant([year], cells, eras; bootstrap_guards=false)
    end
    constant(value) = Dict(variant => value for variant in SUPERLEARNER_SELECTABLE_VARIANTS)
    # Every variant is selectable now, which is the point of the amendment: the
    # stack-only center wins when it is the most accurate.
    for winner in SUPERLEARNER_SELECTABLE_VARIANTS
        errors = constant(5.0)
        errors[winner] = 1.0
        result = selection_fixture(errors)
        @test result.selected === winner
        @test result.scores[winner] ≈ 1.0
        row = only([r for r in result.trace if r.variant == String(winner)])
        @test row.mean_step_rmse_nt ≈ 1.0
        @test row.pooled_rmse_nt ≈ 1.0
        @test row.n == 200 * length(SUPERLEARNER_SELECTION_STEPS)
    end
    # An exact tie across all four goes to the first, safest entry of the order.
    @test selection_fixture(constant(1.5)).selected === first(SUPERLEARNER_SELECTABLE_VARIANTS)
    # A variant that breaches the E1 storm guards is not eligible even when it is
    # the most accurate.
    guards = Dict(v => true for v in SUPERLEARNER_SELECTABLE_VARIANTS)
    guards[:superlearner_e] = false
    errors = constant(5.0)
    errors[:superlearner_e] = 1.0
    errors[:superlearner_a_floor] = 2.0
    breached = selection_fixture(errors; guards=guards)
    @test breached.selected === :superlearner_a_floor
    @test :superlearner_e ∉ breached.eligible
    # When nothing clears the guards the accuracy rule still decides and the breach
    # is recorded rather than hidden.
    both = selection_fixture(errors; guards=Dict(v => false
                                                 for v in SUPERLEARNER_SELECTABLE_VARIANTS))
    @test both.selected === :superlearner_e
    @test both.guards_all_failed
    @test all(row -> !row.guards_pass, both.trace)
    @test !selection_fixture(constant(1.0)).guards_all_failed
    @test_throws ErrorException superlearner_select_variant(
        SuperlearnerYearData[], NamedTuple[], (ALL=2014:2014,),
    )
end

# ---------------------------------------------------------------------------
# The Task A file contract
# ---------------------------------------------------------------------------

@testset "the fold reader accepts both contracted spellings" begin
    prefixed = mktempdir()
    superlearner_synthesize_fixture(prefixed; years=2013:2013, hours_per_year=200)
    bare = mktempdir()
    superlearner_synthesize_fixture(bare; years=2013:2013, hours_per_year=200,
                           shadow_column=:analog_lat, feature_prefix="")
    a = superlearner_read_year(2013; dir=prefixed)
    b = superlearner_read_year(2013; dir=bare)
    @test a.shadow_source == "analog_shadow"
    @test b.shadow_source == "analog_lat"
    @test length(a) == length(b) == 200 * length(SUPERLEARNER_STEPS)
    @test size(a.features, 2) == SUPERLEARNER_FEATURE_COUNT
    @test size(a.experts, 2) == SUPERLEARNER_EXPERT_COUNT
    # Fallback rows are excluded from fitting; every other row is usable.
    @test all(i -> a.usable[i] == !a.fallback[i], 1:length(a))
    @test haskey(a.comparators, :analog_shadow) && haskey(b.comparators, :analog_shadow)
    # Only one composition column exists in either fixture, so the lead-aware
    # comparator is that same column and the two are equal by construction.
    @test a.comparators[:analog_lat] == a.comparators[:analog_shadow]
    @test b.comparators[:analog_lat] == b.comparators[:analog_shadow]
    # A fold whose error layers were refitted persists two different compositions,
    # and both are then read as separate comparators.
    both = mktempdir()
    superlearner_synthesize_fixture(both; years=2013:2013, hours_per_year=200,
                           separate_lat=true)
    c = superlearner_read_year(2013; dir=both)
    @test c.shadow_source == "analog_shadow"
    @test c.comparators[:analog_lat] != c.comparators[:analog_shadow]
    @test all(isfinite, c.comparators[:analog_lat])
    @test :analog_lat in SUPERLEARNER_GATED_COMPARATORS && :analog_shadow in SUPERLEARNER_GATED_COMPARATORS
    @test all(isfinite, a.comparators[:served_inertial_sindy])
    @test superlearner_available_years(2013:2016; dir=prefixed) == [2013]
    @test superlearner_available_years(2014:2016; dir=prefixed) == Int[]
end

@testset "the fold reader fails closed on a broken contract" begin
    source = mktempdir()
    superlearner_synthesize_fixture(source; years=2013:2013, hours_per_year=120)
    template = CSV.read(joinpath(source, "oof_year_2013.csv"), DataFrame;
                        types=Dict("issue_time_utc" => DateTime))
    function rejected(mutate!)
        dir = mktempdir()
        frame = copy(template)
        mutate!(frame)
        CSV.write(joinpath(dir, "oof_year_2013.csv"), frame)
        return dir
    end
    @test_throws ErrorException superlearner_read_year(2013;
        dir=rejected(frame -> select!(frame, Not(:direct_gbm))))
    @test_throws ErrorException superlearner_read_year(2013;
        dir=rejected(frame -> select!(frame, Not(:analog_shadow))))
    @test_throws ErrorException superlearner_read_year(2013;
        dir=rejected(frame -> select!(frame, Not(:f_dst_lag24))))
    # Rows are written one per (issue, step) sorted by issue then step, so the
    # first six rows share an issue; repeating a step inside that block is what
    # repeats a key.
    @test_throws ErrorException superlearner_read_year(2013;
        dir=rejected(frame -> (frame.model_step_hours[3] = frame.model_step_hours[2])))
    @test_throws ErrorException superlearner_read_year(2013;
        dir=rejected(frame -> (frame.issue_time_utc[3] += Year(1))))
    @test_throws ErrorException superlearner_read_year(2013;
        dir=rejected(frame -> (frame.model_step_hours[4] = 5)))
    @test_throws ErrorException superlearner_read_year(2013;
        dir=rejected(frame -> (frame.observation_dst_nt[5] = NaN)))
    @test_throws ErrorException superlearner_read_year(2013;
        dir=rejected(frame -> (frame.coupling_active_mvm[6] = -1.0)))
    # Task A writes the served product into every forecast column of a fallback
    # row, so an empty center is a contract breach wherever it appears; the
    # fallback flag exempts the feature block, not the forecasts.
    @test_throws ErrorException superlearner_read_year(2013;
        dir=rejected(frame -> (frame.fallback[7] = false; frame.t1r_analog[7] = NaN)))
    @test_throws ErrorException superlearner_read_year(2013;
        dir=rejected(frame -> (frame.fallback[7] = true; frame.t1r_analog[7] = NaN)))
    # An empty feature block is what a fallback row means; the row is then scored
    # from its served centers but never fitted on.
    ok = rejected(frame -> (frame.fallback[7] = true; frame.f_dst_lag24[7] = NaN))
    year = superlearner_read_year(2013; dir=ok)
    @test !year.usable[7]
    @test isfinite(year.comparators[:t1r_analog][7])
    @test_throws ErrorException superlearner_read_year(2014; dir=source)
end

@testset "the deployment bundle ships only the residual models the layer accepted" begin
    # V05-01: superlearner_fit_l2 builds a per-step model only for accepted steps and a
    # joint model only when some step is accepted; the writer must not index a
    # model that was never fitted.
    cell = SuperlearnerStackCell(1, :pooled, :pooled, 100, collect(SUPERLEARNER_EXPERTS),
                        fill(1 / SUPERLEARNER_EXPERT_COUNT, SUPERLEARNER_EXPERT_COUNT), 1.0,
                        collect(1:SUPERLEARNER_EXPERT_COUNT), 0.0, 0.0, false)
    cells = Dict{SuperlearnerCellKey,SuperlearnerStackCell}((1, :pooled, :pooled) => cell)
    conformal = Dict{Tuple{Int,Symbol},NamedTuple}(
        (step, depth) => (half_width=10.0, n=50, coverage_floor=0.9, source="pooled")
        for step in SUPERLEARNER_STEPS for depth in (SUPERLEARNER_POOLED_DEPTH, SUPERLEARNER_DEPTH_BINS...))
    rng = MersenneTwister(11)
    X = randn(rng, 200, SUPERLEARNER_L2_FEATURE_COUNT)
    model = SolarSINDy.gbm_fit(X, X[:, 1]; max_depth=2, nrounds=5, min_weight=8,
                                   feature_names=superlearner_l2_feature_names())
    bundle_with(layer) = (fold_year=2025, l1_cells=cells, l1_floor_cells=cells,
        l1_ten_cells=cells, pool_cutoff=DateTime(2024, 12, 25), l2_layer=layer,
        conformal=conformal, selected_variant=:superlearner_e, seed=1, l1_pool_years=[2013],
        l2_pool_years=[2014], conformal_pool_years=[2014])
    partial_step = last(SUPERLEARNER_STEPS)
    for (layer, expected) in (
            ((form="per_step", models=Dict{Int,Any}(partial_step => model), max_depth=2,
              nrounds=5, accepted_steps=[partial_step]),
             ["residual_model_step$(partial_step).bson"]),
            ((form="joint", models=Dict{Int,Any}(), max_depth=2, nrounds=5,
              accepted_steps=Int[]), String[]),
            ((form="joint", models=Dict{Int,Any}(0 => model), max_depth=2, nrounds=5,
              accepted_steps=[partial_step]), ["residual_model_joint.bson"]),
        )
        outdir = mktempdir()
        deployment = superlearner_write_deployment(outdir, bundle_with(layer))
        directory = joinpath(outdir, SUPERLEARNER_DEPLOY_SUBDIR)
        guard_json = JSON3.read(read(joinpath(directory, "guard.json"), String))
        @test collect(String, guard_json.residual_model_files) == expected
        @test sort(filter(name -> endswith(name, ".bson"), readdir(directory))) ==
              sort(expected)
        @test all(path -> isfile(path), deployment.files)
    end
end

# ---------------------------------------------------------------------------
# End to end, and the property the whole study rests on
# ---------------------------------------------------------------------------

@testset "the whole stage runs, and no fit sees the year it scores" begin
    root = mktempdir()
    base_in = joinpath(root, "base")
    superlearner_synthesize_fixture(base_in; years=2013:2016, hours_per_year=720)
    eras = (ALL=2014:2016, E1=2014:2015, E2=2016:2016)
    settings = (years=2014:2016, eras=eras, l2_grid=((3, 60),),
                bootstrap_replicates=200)
    base_out = joinpath(root, "base_out")
    base = run_superlearner_learn(; indir=base_in, outdir=base_out, settings...)

    # --- artifacts ---
    for name in ("superlearner_summary.csv", "superlearner_cells.csv", "superlearner_bootstrap.csv",
                 "superlearner_gates.csv", "superlearner_decision.csv", "superlearner_intervals.csv",
                 "superlearner_selection.csv", "superlearner_l1_weights.csv", "superlearner_l2_selection.csv",
                 "superlearner_l2_acceptance.csv", "superlearner_gates_by_variant.csv",
                 "superlearner_serve_rule.csv", "superlearner_bootstrap_by_variant.csv",
                 "superlearner_conformal.csv", "superlearner_folds.csv", "superlearner_report.md",
                 "superlearner_learn_manifest.csv", "learn_year_2014.csv", "learn_year_2015.csv",
                 "learn_year_2016.csv")
        path = joinpath(base_out, name)
        @test isfile(path)
        @test filesize(path) > 0
    end
    @test base.years == [2014, 2015, 2016]
    @test base.selected in SUPERLEARNER_SELECTABLE_VARIANTS
    @test base.decision.state in ("SERVE_PENDING_G4", "SHADOW", "NO_GO")
    manifest = CSV.read(joinpath(base_out, "superlearner_learn_manifest.csv"), DataFrame)
    @test any(row -> row.entry_type == "output_sha256" && row.name == "superlearner_gates.csv",
              eachrow(manifest))
    @test any(row -> row.entry_type == "input_sha256" &&
                     row.name == "oof_year_2013.csv", eachrow(manifest))
    @test any(row -> row.entry_type == "protocol" && row.name == "folds_scored" &&
                     row.value == "2014|2015|2016", eachrow(manifest))

    # --- the deployable bundle is the last fold's fitted state, and only that ---
    deploy = joinpath(base_out, SUPERLEARNER_DEPLOY_SUBDIR)
    for name in ("stack_weights.csv", "residual_features.csv", "residual_cap.csv",
                 "conformal.csv", "guard.json", "deploy_manifest.csv")
        @test isfile(joinpath(deploy, name))
        @test filesize(joinpath(deploy, name)) > 0
    end
    deployed_stack = CSV.read(joinpath(deploy, "stack_weights.csv"), DataFrame)
    @test unique(deployed_stack.fold_year) == [2016]
    @test Set(deployed_stack.variant) == Set(["L1", "L1a", "L1e"])
    # Exactly the stack the selected variant is built from is flagged served.
    served_stack = base.selected in (:superlearner_e, :superlearner_f) ? "L1e" :
                   (base.selected in (:superlearner_a_floor, :superlearner_d) ? "L1a" : "L1")
    @test Set(deployed_stack[deployed_stack.served, :variant]) == Set([served_stack])
    @test all(row -> row.n_experts == (row.variant == "L1e" ? SUPERLEARNER_EXPERT_TEN_COUNT :
                                       SUPERLEARNER_EXPERT_COUNT), eachrow(deployed_stack))
    @test all(row -> row.variant == "L1e" || row.w_static_stack == 0.0,
              eachrow(deployed_stack))
    deployed_features = CSV.read(joinpath(deploy, "residual_features.csv"), DataFrame)
    @test deployed_features.feature_name == superlearner_l2_feature_names()
    @test deployed_features.column_index == collect(1:SUPERLEARNER_L2_FEATURE_COUNT)
    deployed_caps = CSV.read(joinpath(deploy, "residual_cap.csv"), DataFrame)
    @test deployed_caps.residual_cap_nt == [superlearner_residual_cap(s) for s in SUPERLEARNER_STEPS]
    guard_json = JSON3.read(read(joinpath(deploy, "guard.json"), String))
    @test guard_json.selected_variant == String(base.selected)
    @test guard_json.guard_applied == (base.selected in (:superlearner_c, :superlearner_d, :superlearner_f))
    @test guard_json.guard_reference ==
          (base.selected in (:superlearner_d, :superlearner_f) ? String(SUPERLEARNER_D_GUARD_REFERENCE) :
           (base.selected === :superlearner_c ? "l1_center" : "none"))
    @test guard_json.served_stack == served_stack
    @test collect(guard_json.served_expert_order) ==
          [String(e) for e in (served_stack == "L1e" ? SUPERLEARNER_EXPERTS_TEN : SUPERLEARNER_EXPERTS)]
    @test collect(guard_json.expert_order_ten) == [String(e) for e in SUPERLEARNER_EXPERTS_TEN]
    @test guard_json.pool_target_embargo_hours == SUPERLEARNER_EMBARGO_HOURS
    @test guard_json.pool_target_cutoff_utc == string(superlearner_pool_cutoff(2016))
    @test guard_json.residual_used_by_selected_variant ==
          (base.selected in (:superlearner_b, :superlearner_c))
    @test guard_json.fold_year == 2016
    @test collect(guard_json.expert_order) == [String(e) for e in SUPERLEARNER_EXPERTS]
    @test collect(guard_json.model_steps) == collect(SUPERLEARNER_STEPS)
    @test guard_json.deepening_rate_nt_per_h_strict_below == SUPERLEARNER_GUARD_RATE_NT_PER_H
    @test guard_json.conformal_coverage == SUPERLEARNER_COVERAGE
    @test collect(guard_json.l1_pool_years) == [2013, 2014, 2015]
    @test collect(guard_json.l2_pool_years) == [2014, 2015]
    # The residual models carry the design they were fitted on, so a feature-order
    # drift between the study and a serving path would fail here.
    residual_files = collect(guard_json.residual_model_files)
    @test !isempty(residual_files)
    for name in residual_files
        model = SolarSINDy.gbm_load(joinpath(deploy, name))
        @test String.(model.info[:feature_names]) == superlearner_l2_feature_names()
    end
    # The bundled half-widths are exactly the ones the last fold was scored with.
    deployed_conformal = CSV.read(joinpath(deploy, "conformal.csv"), DataFrame)
    bundle_half = Dict((row.model_step_hours, row.depth_bin) => row.half_width_nt
                       for row in eachrow(deployed_conformal))
    @test Set(deployed_conformal.depth_bin) ==
          Set(String.((SUPERLEARNER_POOLED_DEPTH, SUPERLEARNER_DEPTH_BINS...)))
    last_fold = CSV.read(joinpath(base_out, "learn_year_2016.csv"), DataFrame)
    selected_half = Symbol(String(base.selected) * "_half_width_nt")
    for row in eachrow(last_fold)
        depth = String(superlearner_depth_bin(row.latest_dst_nt))
        @test row.depth_bin == depth
        @test bundle_half[(row.model_step_hours, depth)] ≈ row[selected_half]
    end
    deployed_manifest = CSV.read(joinpath(deploy, "deploy_manifest.csv"), DataFrame)
    @test Set(deployed_manifest.file) ==
          Set(vcat(["stack_weights.csv", "residual_features.csv", "residual_cap.csv",
                    "conformal.csv", "guard.json"], residual_files))
    for row in eachrow(deployed_manifest)
        @test row.sha256 == bytes2hex(SHA.sha256(read(joinpath(deploy, row.file))))
    end

    # --- the embargo is recorded per fold, over the earlier years only ---
    folds_table = CSV.read(joinpath(base_out, "superlearner_folds.csv"), DataFrame)
    @test DateTime.(folds_table.pool_target_cutoff_utc) ==
          [superlearner_pool_cutoff(y) for y in (2014, 2015, 2016)]
    # Fold 2014's pool is the seed year alone, and the fixture's 720 h of hourly
    # issues all sit far inside the year, so nothing of it falls in the embargo;
    # the count is over prior years only and can never reach a fold's own size.
    @test all(folds_table.n_prior_pool_rows_embargoed .< folds_table.n_rows)

    # --- the A3 serve rule and the per-variant gate table ---
    serve = CSV.read(joinpath(base_out, "superlearner_serve_rule.csv"), DataFrame)
    @test all(serve.comparator .== String(SUPERLEARNER_SERVE_REFERENCE))
    @test Set(serve.check) ⊆ Set(["pooled_gain", "storm_cell", "intervals_G3",
                                  "era_verdict", "serve_rule"])
    @test count(r -> r.check == "serve_rule", eachrow(serve)) == 1
    # Only ALL and E2 decide; every E1 row is disclosure.
    @test all(r -> r.decides == (r.scope in ("ALL", "E2", "-")), eachrow(serve))
    for row in eachrow(serve[serve.check .== "pooled_gain", :])
        @test row.n > 0
        @test row.pass == (row.lower_nt > 0.0)
    end
    for row in eachrow(serve[serve.check .== "storm_cell", :])
        @test row.n >= SUPERLEARNER_G2_MIN_CELL_ROWS
        @test row.pass == !(row.lower_nt > 0.0)
    end
    variant_gates = CSV.read(joinpath(base_out, "superlearner_gates_by_variant.csv"), DataFrame)
    @test Set(variant_gates.variant) == Set(String.(SUPERLEARNER_VARIANTS))
    @test Set(variant_gates[variant_gates.served_candidate, :variant]) ==
          Set([String(base.selected)])
    # The served candidate's rows in the per-variant table are the gate table itself.
    gate_table = CSV.read(joinpath(base_out, "superlearner_gates.csv"), DataFrame)
    served_rows = variant_gates[variant_gates.variant .== String(base.selected), :]
    @test nrow(served_rows) == nrow(gate_table)
    @test served_rows.pass == gate_table.pass
    @test served_rows.observed == gate_table.observed

    # --- L1 weights are feasible everywhere and the floor variant honours its floor ---
    weights = CSV.read(joinpath(base_out, "superlearner_l1_weights.csv"), DataFrame)
    @test nrow(weights) > 0
    @test all(w -> abs(w - 1.0) <= 1e-9, weights.weight_sum)
    @test all(<=(1e-7), weights.kkt_stationarity)
    @test all(>=(-1e-7), weights.kkt_dual_min)
    @test Set(weights.variant) == Set(["L1", "L1a", "L1e"])
    for row in eachrow(weights)
        expert_weights = [row[Symbol("w_", e)] for e in SUPERLEARNER_EXPERTS_TEN]
        @test all(>=(-1e-12), expert_weights)
        @test isapprox(sum(expert_weights), 1.0; atol=1e-9)
        # The nine-expert stacks never place mass on the tenth expert, and the
        # ten-expert stack is the only one that may.
        row.variant == "L1e" || @test row.w_static_stack == 0.0
        @test row.n_experts == (row.variant == "L1e" ? SUPERLEARNER_EXPERT_TEN_COUNT :
                                SUPERLEARNER_EXPERT_COUNT)
        row.variant in ("L1a", "L1e") || continue
        @test row.sindy_family_mass >= SUPERLEARNER_SINDY_FLOOR - 1e-9
    end
    @test any(row -> row.variant == "L1" &&
                     row.sindy_family_mass < SUPERLEARNER_SINDY_FLOOR - 1e-9, eachrow(weights))
    @test all(row -> !row.floor_active, [r for r in eachrow(weights) if r.variant == "L1"])
    # The ten-expert stack does use its extra input somewhere, or E10 is decoration.
    @test any(row -> row.variant == "L1e" && row.w_static_stack > 0.0, eachrow(weights))

    # --- the residual layer, its cap, and the guard as persisted ---
    l2 = CSV.read(joinpath(base_out, "superlearner_l2_selection.csv"), DataFrame)
    @test Set(l2.fold_year) == Set([2014, 2015, 2016])
    @test !any(l2[l2.fold_year .== 2014, :available])
    @test all(l2[l2.fold_year .== 2016, :available])
    @test only(unique(l2[(l2.fold_year .== 2016) .& l2.selected, :inner_rule])) ==
          "chronological_two_thirds"
    # Amendment A3 one level down: the inner split's own 168 h target cutoff and
    # the row count it dropped are persisted, so the embargo is auditable per fold
    # and not only a property of the code that produced the fold.
    l2_2016 = l2[(l2.fold_year .== 2016) .& l2.selected, :]
    boundary_2016 = DateTime(only(unique(string.(l2_2016.inner_boundary))))
    cutoff_2016 = DateTime(only(unique(string.(l2_2016.inner_target_cutoff_utc))))
    @test cutoff_2016 == boundary_2016 - Hour(SUPERLEARNER_EMBARGO_HOURS)
    # The fixture's pool years are 30-day blocks eleven months apart, so the
    # two-thirds boundary falls in the empty gap between them and the embargo has
    # nothing to drop. What the whole stage pins here is that the cutoff is
    # persisted, that it sits 168 h before the boundary, and that the three row
    # counts still partition the pool; the gap arithmetic itself is pinned on
    # hourly pools in the inner-split testsets above.
    @test only(unique(l2_2016.n_inner_embargoed)) == 0
    @test only(unique(l2_2016.n_inner_train)) +
          only(unique(l2_2016.n_inner_validate)) +
          only(unique(l2_2016.n_inner_embargoed)) == only(unique(l2_2016.n_pool_rows))
    @test all(l2[l2.fold_year .== 2014, :n_inner_embargoed] .== 0)
    # Amendment A1: acceptance is decided per step on the inner validation, and the
    # table says so for every fold and step.
    acceptance = CSV.read(joinpath(base_out, "superlearner_l2_acceptance.csv"), DataFrame)
    @test Set(acceptance.fold_year) == Set([2014, 2015, 2016])
    @test nrow(acceptance) == 3 * length(SUPERLEARNER_STEPS)
    @test !any(acceptance[acceptance.fold_year .== 2014, :accepted])
    @test all(acceptance[acceptance.fold_year .== 2014, :reason] .== "no_residual_layer")
    for row in eachrow(acceptance[acceptance.fold_year .== 2016, :])
        @test row.accepted == (row.gain_nt > 0.0)
        @test row.n_inner_validate > 0
        @test isfinite(row.rmse_identity_nt) && isfinite(row.rmse_residual_nt)
    end
    accepted_2016 = Set(acceptance[(acceptance.fold_year .== 2016) .& acceptance.accepted,
                                   :model_step_hours])
    # The fixture carries a systematic bias the residual can remove, so at least
    # one step must accept — otherwise the residual path is never exercised here.
    @test !isempty(accepted_2016)
    @test Set(parse.(Int, split(only(unique(
        l2[(l2.fold_year .== 2016) .& l2.selected, :accepted_steps])), "|"))) ==
        accepted_2016
    for year in (2014, 2015, 2016)
        rows = CSV.read(joinpath(base_out, "learn_year_$(year).csv"), DataFrame)
        @test all(isfinite, rows.superlearner_a)
        @test all(isfinite, rows.superlearner_b)
        @test all(isfinite, rows.superlearner_c)
        @test all(isfinite, rows.superlearner_a_floor)
        @test all(isfinite, rows.superlearner_d)
        @test all(isfinite, rows.superlearner_e)
        @test all(isfinite, rows.superlearner_f)
        for row in eachrow(rows)
            cap = superlearner_residual_cap(row.model_step_hours)
            @test abs(row.l2_residual_capped_nt) <= cap + 1e-12
            @test row.superlearner_b ≈ row.superlearner_a + row.l2_residual_capped_nt
            @test row.superlearner_c == (row.deepening_cell ? min(row.superlearner_b, row.superlearner_a) : row.superlearner_b)
            @test row.superlearner_d == (row.deepening_cell ?
                                min(row.superlearner_a_floor, row.static_stack) : row.superlearner_a_floor)
            @test row.superlearner_f == (row.deepening_cell ?
                                min(row.superlearner_e, row.static_stack) : row.superlearner_e)
            @test row.l2_applied || row.l2_residual_capped_nt == 0.0
            @test row.superlearner_c_lo_nt ≈ row.superlearner_c - row.superlearner_c_half_width_nt
            @test row.superlearner_c_hi_nt ≈ row.superlearner_c + row.superlearner_c_half_width_nt
            @test row.usable == (!row.fallback)
            row.usable || @test row.superlearner_a == row.served_inertial_sindy
            # Only an accepted step of this fold may carry a correction.
            year == 2016 || @test row.l2_residual_capped_nt == 0.0
            (year == 2016 && !(row.model_step_hours in accepted_2016)) &&
                @test row.l2_residual_capped_nt == 0.0
        end
        # A residual outside the cap would have been clipped, so at least one row
        # must carry a nonzero correction — otherwise the cap test above is vacuous.
        @test any(abs.(rows.l2_residual_capped_nt) .> 0.0) == (year == 2016)
        @test all(rows.depth_bin .==
                  String.(superlearner_depth_bin.(rows.latest_dst_nt)))
        @test issubset(Set(rows.l1_cell_regime),
                       Set(String.((SUPERLEARNER_REGIMES..., SUPERLEARNER_POOLED_REGIME, :served))))
        @test issubset(Set(rows.l1_cell_depth),
                       Set(String.((SUPERLEARNER_DEPTH_BINS..., SUPERLEARNER_POOLED_DEPTH, :served))))
    end
    guarded = CSV.read(joinpath(base_out, "learn_year_2016.csv"), DataFrame)
    @test any(guarded.deepening_cell)
    @test all(row -> !row.deepening_cell || row.superlearner_c <= row.superlearner_a + 1e-12,
              eachrow(guarded))

    # --- mutation A: the scored year's pure targets cannot move anything fitted ---
    target_in = joinpath(root, "target")
    cp(base_in, target_in)
    frame = CSV.read(joinpath(target_in, "oof_year_2016.csv"), DataFrame;
                     types=Dict("issue_time_utc" => DateTime))
    later = frame.model_step_hours .>= 2
    @test count(later) > 0
    frame.observation_dst_nt[later] .+= 1_000.0
    CSV.write(joinpath(target_in, "oof_year_2016.csv"), frame)
    target_out = joinpath(root, "target_out")
    run_superlearner_learn(; indir=target_in, outdir=target_out, settings...)
    unchanged(name, columns) = begin
        left = CSV.read(joinpath(base_out, name), DataFrame)
        right = CSV.read(joinpath(target_out, name), DataFrame)
        all(column -> isequal(left[!, column], right[!, column]), columns)
    end
    @test unchanged("superlearner_l1_weights.csv", names(
        CSV.read(joinpath(base_out, "superlearner_l1_weights.csv"), DataFrame)))
    @test unchanged("superlearner_conformal.csv", names(
        CSV.read(joinpath(base_out, "superlearner_conformal.csv"), DataFrame)))
    @test unchanged("superlearner_l2_selection.csv", names(
        CSV.read(joinpath(base_out, "superlearner_l2_selection.csv"), DataFrame)))
    for year in (2014, 2015, 2016)
        @test unchanged("learn_year_$(year).csv",
                        ["superlearner_a", "superlearner_b", "superlearner_c", "superlearner_a_floor",
                         "l2_residual_raw_nt", "l2_residual_capped_nt",
                         "superlearner_c_half_width_nt", "served_inertial_sindy_half_width_nt"])
    end
    # The mutation must still have been felt somewhere, or the test proves nothing.
    mutated_summary = CSV.read(joinpath(target_out, "superlearner_summary.csv"), DataFrame)
    base_summary = CSV.read(joinpath(base_out, "superlearner_summary.csv"), DataFrame)
    @test !isequal(mutated_summary.rmse_nt, base_summary.rmse_nt)

    # --- mutation B: every observation of the scored year, against the fits only ---
    every_in = joinpath(root, "every")
    cp(base_in, every_in)
    frame = CSV.read(joinpath(every_in, "oof_year_2016.csv"), DataFrame;
                     types=Dict("issue_time_utc" => DateTime))
    frame.observation_dst_nt .+= 1_000.0
    CSV.write(joinpath(every_in, "oof_year_2016.csv"), frame)
    every_out = joinpath(root, "every_out")
    run_superlearner_learn(; indir=every_in, outdir=every_out, settings...)
    for (name, columns) in (
        ("superlearner_l1_weights.csv",
         names(CSV.read(joinpath(base_out, "superlearner_l1_weights.csv"), DataFrame))),
        ("superlearner_conformal.csv",
         names(CSV.read(joinpath(base_out, "superlearner_conformal.csv"), DataFrame))),
        ("superlearner_l2_selection.csv",
         names(CSV.read(joinpath(base_out, "superlearner_l2_selection.csv"), DataFrame))),
    )
        left = CSV.read(joinpath(base_out, name), DataFrame)
        right = CSV.read(joinpath(every_out, name), DataFrame)
        @test all(column -> isequal(left[!, column], right[!, column]), columns)
    end
    # Folds before the mutated year keep their centers as well.
    for year in (2014, 2015)
        left = CSV.read(joinpath(base_out, "learn_year_$(year).csv"), DataFrame)
        right = CSV.read(joinpath(every_out, "learn_year_$(year).csv"), DataFrame)
        @test isequal(left.superlearner_c, right.superlearner_c)
    end
end

@testset "an incomplete Task A run is scored on its contiguous prefix" begin
    root = mktempdir()
    indir = joinpath(root, "partial")
    superlearner_synthesize_fixture(indir; years=2013:2015, hours_per_year=400)
    outdir = joinpath(root, "out")
    result = run_superlearner_learn(; years=2014:2016, eras=(ALL=2014:2016, E1=2014:2015),
                            indir=indir, outdir=outdir, l2_grid=((3, 20),),
                            bootstrap_replicates=100)
    @test result.years == [2014, 2015]
    @test !isfile(joinpath(outdir, "learn_year_2016.csv"))
    manifest = CSV.read(joinpath(outdir, "superlearner_learn_manifest.csv"), DataFrame)
    @test any(row -> row.name == "folds_scored" && row.value == "2014|2015",
              eachrow(manifest))
    @test any(row -> row.name == "folds_requested" && row.value == "2014|2015|2016",
              eachrow(manifest))
    # A missing seed fold is a hard stop, not a silent start.
    bare = mktempdir()
    superlearner_synthesize_fixture(bare; years=2014:2015, hours_per_year=200)
    @test_throws ErrorException run_superlearner_learn(; years=2014:2015,
                                               eras=(ALL=2014:2015,), indir=bare,
                                               outdir=mktempdir())
end

end # module
