module OperationalV23StatsTests

using Test
using Dates
using Random
using Statistics
using SolarSINDy

"""
Issue times placed one per 168 h block: block `b` receives its issues three hours after the block
boundary, so the mapping from issue to block is unambiguous and independent of the block grid's
rounding at the boundaries.
"""
_block_issue(block::Int; offset_h::Int=3) =
    V23_BOOTSTRAP_EPOCH + Hour(168 * (block - 1) + offset_h)

"""
Independent whole-issue resampling support. Enumerates every multiplicity vector over `n_blocks`
blocks that sums to `n_blocks` and returns the RMSE gain each one produces. Written directly from
the definition of the block bootstrap (draw blocks, pool their rows, take RMSEs of the pooled
multiset), not from the implementation under test.
"""
function _reachable_statistics(counts::Vector{Int},
                               comparator_sse::Vector{Float64},
                               candidate_sse::Vector{Float64})
    n_blocks = length(counts)
    values = Float64[]
    multiplicities = zeros(Int, n_blocks)
    function recurse(position::Int, remaining::Int)
        if position > n_blocks
            remaining == 0 || return nothing
            n = sum(multiplicities .* counts)
            push!(values,
                  sqrt(sum(multiplicities .* comparator_sse) / n) -
                  sqrt(sum(multiplicities .* candidate_sse) / n))
            return nothing
        end
        for m in 0:remaining
            multiplicities[position] = m
            recurse(position + 1, remaining - m)
        end
        multiplicities[position] = 0
        return nothing
    end
    recurse(1, n_blocks)
    return values
end

@testset "V2.3 block bootstrap on a constant difference returns that difference" begin
    # Plan §7 oracle (v): with every row carrying the same comparator error (10 nT) and the same
    # candidate error (8 nT), every resampled multiset has RMSEs 10 and 8 regardless of which
    # blocks are drawn, so the replicate distribution is degenerate at the point gain 10 - 8 = 2.
    issues = [_block_issue(b) for b in 1:12 for _ in 1:3]
    comparator = fill(100.0, length(issues))
    candidate = fill(64.0, length(issues))

    result = v23_block_bootstrap(comparator, candidate, issues; replicates=1_000)
    @test result.n_rows == 36
    @test result.n_blocks == 12
    @test result.replicates == 1_000
    @test result.point == 2.0
    @test result.lower == result.point
    # Continuity-corrected p-value with no replicate at or below zero.
    @test result.p_one_sided == 1 / 1_001

    # Constant difference c = 36 nT² subtracted from a constant comparator squared error.
    @test all(candidate .== comparator .- 36.0)

    # Orientation check: swapping the arguments negates the gain and sends every replicate to a
    # non-positive value, so the p-value saturates at (R + 1) / (R + 1).
    reversed = v23_block_bootstrap(candidate, comparator, issues; replicates=1_000)
    @test reversed.point == -2.0
    @test reversed.lower == -2.0
    @test reversed.p_one_sided == 1.0

    # A different constant pair reproduces its own exact difference (guards against a hard-coded
    # or accidentally cancelled statistic).
    other = v23_block_bootstrap(fill(225.0, length(issues)), fill(81.0, length(issues)), issues;
                                replicates=200)
    @test other.point == 6.0
    @test other.lower == 6.0
end

@testset "V2.3 block bootstrap on an exactly null difference straddles zero" begin
    # Sixty block pairs; within a pair the two blocks carry mirrored squared errors, so the pooled
    # comparator and candidate sums are equal integer sums and the point gain is exactly zero.
    # Swapping the two blocks of every pair is a measure-preserving involution on the draws that
    # negates the statistic, hence the replicate distribution is symmetric about zero.
    issues = DateTime[]
    comparator = Float64[]
    candidate = Float64[]
    for pair in 1:60
        high = 100.0 + 2 * pair
        low = 81.0 - pair
        for (block, (c, d)) in zip((2 * pair - 1, 2 * pair), ((high, low), (low, high)))
            issue = _block_issue(block)
            for _ in 1:2               # two model steps share the issue time
                push!(issues, issue)
                push!(comparator, c)
                push!(candidate, d)
            end
        end
    end
    @test sum(comparator) == sum(candidate)

    result = v23_block_bootstrap(comparator, candidate, issues)
    @test result.n_rows == 240
    @test result.n_blocks == 120
    @test result.replicates == 10_000
    @test result.point == 0.0
    @test 0.45 <= result.p_one_sided <= 0.55
    # The replicate spread must be real, otherwise the p-value above would be meaningless.
    @test result.lower < -0.1

    # The reported bound and p-value are the alpha-quantile and the corrected tail fraction of the
    # replicate statistics.
    draws = SolarSINDy._v23_block_bootstrap_replicates(comparator, candidate, issues)
    @test result.lower == quantile(draws.statistics, 0.05)
    @test result.p_one_sided == (count(<=(0.0), draws.statistics) + 1) / 10_001

    # Wider tail mass gives a higher (less conservative) bound.
    narrow = v23_block_bootstrap(comparator, candidate, issues; alpha=0.01)
    wide = v23_block_bootstrap(comparator, candidate, issues; alpha=0.5)
    @test narrow.lower <= result.lower <= wide.lower
end

@testset "V2.3 block bootstrap resamples whole issues, never single rows" begin
    # Four issues, each in its own 168 h block, each with three model steps carrying distinct
    # squared errors. Whole-issue resampling can only produce the 35 statistics enumerated by
    # _reachable_statistics; a bootstrap that resampled rows would leave that finite support.
    per_block = [(3.0, 9.0, 25.0), (4.0, 16.0, 36.0), (1.0, 49.0, 81.0), (64.0, 2.0, 100.0)]
    issues = DateTime[]
    comparator = Float64[]
    candidate = Float64[]
    for block in 1:4, step in 1:3
        push!(issues, _block_issue(2 * block - 1))     # blocks 1, 3, 5, 7 -> four distinct blocks
        push!(comparator, per_block[block][step])
        push!(candidate, per_block[block][step] / 2 + block)
    end

    counts = fill(3, 4)
    comparator_sse = [sum(per_block[block]) for block in 1:4]
    candidate_sse = [sum(per_block[block]) / 2 + 3 * block for block in 1:4]
    reachable = _reachable_statistics(counts, comparator_sse, candidate_sse)
    @test length(reachable) == 35          # multiplicity vectors of 4 blocks summing to 4

    draws = SolarSINDy._v23_block_bootstrap_replicates(comparator, candidate, issues;
                                                       replicates=2_000)
    @test draws.n_blocks == 4
    @test draws.n_rows == 12
    @test all(statistic -> any(value -> isapprox(statistic, value; atol=1e-12), reachable),
              draws.statistics)
    # Non-vacuous: the replicates actually visit many distinct whole-issue multisets.
    @test length(unique(round.(draws.statistics; digits=12))) > 10

    # Discrimination witness: a multiset that splits issues (one step of each issue) lands outside
    # the whole-issue support, so the check above would fail for a row-level bootstrap.
    split_comparator = [per_block[block][1] for block in 1:4]
    split_candidate = [per_block[block][1] / 2 + block for block in 1:4]
    split_statistic = sqrt(sum(split_comparator) / 4) - sqrt(sum(split_candidate) / 4)
    @test !any(value -> isapprox(split_statistic, value; atol=1e-12), reachable)

    # Rows sharing an issue time, and issues sharing a 168 h window, stay in one block.
    fused = [V23_BOOTSTRAP_EPOCH, V23_BOOTSTRAP_EPOCH + Hour(167), V23_BOOTSTRAP_EPOCH + Hour(168)]
    fused_result = v23_block_bootstrap(fill(9.0, 3), fill(4.0, 3), fused; replicates=50)
    @test fused_result.n_blocks == 2
    @test fused_result.n_rows == 3

    # The block grid is floor-aligned on the epoch, so an issue one hour before the epoch belongs
    # to the preceding block rather than being fused with the block that starts at the epoch.
    straddling = [V23_BOOTSTRAP_EPOCH - Hour(1), V23_BOOTSTRAP_EPOCH + Hour(1)]
    straddle_result = v23_block_bootstrap(fill(9.0, 2), fill(4.0, 2), straddling; replicates=50)
    @test straddle_result.n_blocks == 2

    # Doubling the block length halves the number of blocks on evenly spaced issues.
    spread = [_block_issue(b) for b in 1:12]
    default_blocks = v23_block_bootstrap(fill(9.0, 12), fill(4.0, 12), spread;
                                         replicates=50).n_blocks
    doubled = v23_block_bootstrap(fill(9.0, 12), fill(4.0, 12), spread;
                                  replicates=50, block_hours=336).n_blocks
    @test default_blocks == 12
    @test doubled == 6
end

@testset "V2.3 block bootstrap is deterministic in its seed" begin
    rng = MersenneTwister(20260817)
    issues = [_block_issue(b) for b in 1:40 for _ in 1:2]
    comparator = abs2.(randn(rng, length(issues)) .* 4.0)
    candidate = abs2.(randn(rng, length(issues)) .* 3.0)

    first_run = v23_block_bootstrap(comparator, candidate, issues; replicates=500)
    second_run = v23_block_bootstrap(comparator, candidate, issues; replicates=500)
    @test first_run == second_run
    @test first_run.point == v23_block_bootstrap(comparator, candidate, issues;
                                                 replicates=500, seed=7).point

    other_seed = v23_block_bootstrap(comparator, candidate, issues; replicates=500, seed=7)
    @test other_seed.lower != first_run.lower
    @test other_seed.point == first_run.point   # the point statistic never touches the RNG
end

@testset "V2.3 block bootstrap rejects malformed input" begin
    issues = [_block_issue(b) for b in 1:4]
    comparator = fill(9.0, 4)
    candidate = fill(4.0, 4)

    @test_throws DimensionMismatch v23_block_bootstrap(comparator, candidate[1:3], issues)
    @test_throws DimensionMismatch v23_block_bootstrap(comparator, candidate, issues[1:3])
    @test_throws ArgumentError v23_block_bootstrap(Float64[], Float64[], DateTime[])
    @test_throws ArgumentError v23_block_bootstrap([9.0, -1.0, 4.0, 4.0], candidate, issues)
    @test_throws ArgumentError v23_block_bootstrap(comparator, [4.0, NaN, 4.0, 4.0], issues)
    @test_throws ArgumentError v23_block_bootstrap(comparator, [4.0, Inf, 4.0, 4.0], issues)
    @test_throws ArgumentError v23_block_bootstrap(comparator, candidate, issues; replicates=0)
    @test_throws ArgumentError v23_block_bootstrap(comparator, candidate, issues; block_hours=0)
    @test_throws ArgumentError v23_block_bootstrap(comparator, candidate, issues; alpha=0.0)
    @test_throws ArgumentError v23_block_bootstrap(comparator, candidate, issues; alpha=1.0)
    @test_throws ArgumentError v23_block_bootstrap(comparator, candidate, issues; min_blocks=5)
    # A single 168 h block cannot support a block bootstrap.
    single = fill(_block_issue(1), 4)
    @test_throws ArgumentError v23_block_bootstrap(comparator, candidate, single)
end

@testset "V2.3 Holm adjustment matches the package implementation" begin
    # Hand-computed families on exactly representable p-values.
    @test v23_holm([0.125, 0.5, 0.25]) == [0.375, 0.5, 0.5]
    @test v23_holm([0.5, 0.5, 0.5]) == [1.0, 1.0, 1.0]
    @test v23_holm([0.02, 0.5]) == [0.04, 0.5]
    @test v23_holm([0.25]) == [0.25]

    rng = MersenneTwister(22_022_026)
    for _ in 1:50
        family = rand(rng, 1:12)
        p = round.(rand(rng, family); digits=rand(rng, 1:6))   # digits low enough to create ties
        adjusted = v23_holm(p)
        reference = [record.holm_p_value for record in SolarSINDy.holm_adjust(p)]
        @test adjusted == reference
        # Structural properties the step-down must satisfy.
        @test all(adjusted .>= p)
        @test all(adjusted .<= 1.0)
        @test issorted(adjusted[sortperm(p)])
        # Order equivariance: permuting the family permutes the adjusted values.
        permutation = randperm(rng, family)
        @test v23_holm(p[permutation]) == adjusted[permutation]
    end

    @test_throws ArgumentError v23_holm(Float64[])
    @test_throws ArgumentError v23_holm([0.5, NaN])
    @test_throws ArgumentError v23_holm([0.5, 1.5])
    @test_throws ArgumentError v23_holm([-0.1])
end

@testset "V2.3 storm-cell labels follow plan §6-A2" begin
    # Hand-checked rows: (latest Dst nT, rate nT/h, coupling mV/m) -> expected cells.
    # V2.2 regime thresholds: disturbed Dst -30 nT, deepening rate -5 nT/h, coupling > 0.
    cases = [
        # Intense storm, strongly coupled and falling fast: every depth cell plus intense deepening.
        ((-120.0, -20.0, 3.0),
         [:all, :latest_le_m50, :latest_le_m100, :active_deepening, :intense_deepening]),
        # Same depth, falling too slowly for the intense cell.
        ((-120.0, -10.0, 2.0), [:all, :latest_le_m50, :latest_le_m100, :active_deepening]),
        # Zero coupling: deepening comes from the disturbed-and-falling branch, not from coupling.
        ((-100.0, -16.0, 0.0),
         [:all, :latest_le_m50, :latest_le_m100, :active_deepening, :intense_deepening]),
        # Rate exactly at the -15 nT/h boundary is excluded (strict inequality).
        ((-100.0, -15.0, 0.0), [:all, :latest_le_m50, :latest_le_m100, :active_deepening]),
        # Disturbed but recovering.
        ((-60.0, 2.0, 0.5), [:all, :latest_le_m50, :recovery]),
        # Dst exactly at -50 nT is inside the depth cell (non-strict inequality).
        ((-50.0, -1.0, 0.0), [:all, :latest_le_m50, :active_deepening]),
        # Just above the depth threshold: disturbed and falling, but no depth cell.
        ((-49.9, -3.0, 0.0), [:all, :active_deepening]),
        # Quiet.
        ((-10.0, 0.0, 0.0), [:all, :quiet]),
        # Quiet ring current but strongly driven: the V2.2 regime is active deepening.
        ((-10.0, -6.0, 1.0), [:all, :active_deepening]),
    ]
    for ((latest, rate, coupling), expected) in cases
        @test v23_regime_cells(latest, rate, coupling) == expected
    end

    rng = MersenneTwister(4_022_026)
    for _ in 1:200
        latest = -200.0 + 220.0 * rand(rng)
        rate = -30.0 + 60.0 * rand(rng)
        coupling = 8.0 * rand(rng)
        cells = v23_regime_cells(latest, rate, coupling)
        regime = operational_v22_regime(latest, rate, coupling)
        @test cells[1] == :all
        @test length(unique(cells)) == length(cells)
        @test all(cell -> cell in V23_CELL_LABELS, cells)
        # Exactly one V2.2 regime label, and it is the V2.2 classification of the same state.
        @test count(cell -> cell in (:active_deepening, :recovery, :quiet), cells) == 1
        @test regime in cells
        @test (:latest_le_m50 in cells) == (latest <= -50.0)
        @test (:latest_le_m100 in cells) == (latest <= -100.0)
        @test (:intense_deepening in cells) == (latest <= -100.0 && rate < -15.0)
        # Depth cells nest.
        @test !(:latest_le_m100 in cells) || (:latest_le_m50 in cells)
    end

    @test_throws ArgumentError v23_regime_cells(NaN, -1.0, 0.0)
    @test_throws ArgumentError v23_regime_cells(-100.0, Inf, 0.0)
    @test_throws ArgumentError v23_regime_cells(-100.0, -1.0, NaN)
    # Negative coupling is not a physical issue-time state; the V2.2 classifier rejects it.
    @test_throws ArgumentError v23_regime_cells(-100.0, -1.0, -1.0)
end

end # module
