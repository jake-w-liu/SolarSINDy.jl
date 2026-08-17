# operational_v23_stats.jl — Operational V2.3 inference primitives (block bootstrap, Holm, cells).
#
# The V2.3 protocol scores a single confirmatory TEST partition once. Its gates (research plan §6)
# are stated in terms of three primitives, and this file supplies exactly those three and nothing
# else: it does not read artifacts, does not fit anything, and does not decide gate outcomes.
#
#   1. a paired block bootstrap of the RMSE difference between a comparator and a candidate over
#      fixed non-overlapping 168 h calendar blocks of issue time (V2.2 convention), 10,000
#      replicates, seed 22,022,026 and a one-sided 95 % lower bound;
#   2. the Holm step-down family-wise adjustment applied across the families named by each gate;
#   3. the plan §6-A2 storm-cell membership of an issue, derived from causal issue-time state only.
#
# Squared errors are the bootstrap input rather than (observation, prediction) pairs because the
# development and confirmatory runners already materialise per-row squared errors for every variant
# and comparator; passing them in keeps this file independent of any table schema.

"Issue-time block length (hours) of the V2.3 fixed non-overlapping calendar-block bootstrap."
const V23_BOOTSTRAP_BLOCK_HOURS = 168

"Preregistered number of bootstrap replicates (research plan §6)."
const V23_BOOTSTRAP_REPLICATES = 10_000

"Preregistered bootstrap seed (research plan §6)."
const V23_BOOTSTRAP_SEED = 22_022_026

"Preregistered one-sided bootstrap tail mass; the reported bound is the 95 % lower bound."
const V23_BOOTSTRAP_ALPHA = 0.05

"""
    V23_BOOTSTRAP_EPOCH

Origin of the fixed 168 h block grid. Blocks are the calendar-aligned windows
`[epoch + 168 h·b, epoch + 168 h·(b+1))`, which is the convention of the V2.2 harness
(`validation/operational/v2_2_history_crossfit.jl` lines 511-516) and keeps the block partition of
a set of issues independent of which rows happen to be present in a particular scoring run.
"""
const V23_BOOTSTRAP_EPOCH = DateTime(2010, 1, 1)

"Dst threshold (nT) of the `latest_le_m50` storm cell."
const V23_CELL_DEEP_DST_NT = -50.0

"Dst threshold (nT) of the `latest_le_m100` and `intense_deepening` storm cells."
const V23_CELL_INTENSE_DST_NT = -100.0

"Dst rate threshold (nT/h, strict) of the `intense_deepening` storm cell."
const V23_CELL_INTENSE_RATE_NT_PER_H = -15.0

"""
    V23_CELL_LABELS

Every cell label [`v23_regime_cells`](@ref) can emit, in a fixed order so that reports iterate
cells deterministically. `:all` is the pooled cell; `:latest_le_m50` and `:latest_le_m100` are the
depth cells; `:active_deepening`, `:recovery` and `:quiet` are the V2.2 causal regimes
(`operational_v22_regime`), exactly one of which is always assigned; `:intense_deepening` is the
plan §6-A2 intense/extreme deepening cell.
"""
const V23_CELL_LABELS = (
    :all, :latest_le_m50, :latest_le_m100,
    :active_deepening, :recovery, :quiet,
    :intense_deepening,
)

function _v23_validate_squared_errors(values::AbstractVector{<:Real}, role::AbstractString)
    for value in values
        (value isa Bool) && throw(ArgumentError(
            "V2.3 bootstrap $role squared errors must be real numbers, not Bool",
        ))
        isfinite(value) || throw(ArgumentError(
            "V2.3 bootstrap $role squared errors must be finite",
        ))
        value >= 0 || throw(ArgumentError(
            "V2.3 bootstrap $role squared errors must be nonnegative",
        ))
    end
    return nothing
end

"""
    _v23_bootstrap_blocks(issue_times, block_hours, epoch)

Map each row to the index of its 168 h issue-time block. Rows sharing an issue time necessarily
share a block, so the resampling unit is a whole issue (with all of its model steps) and never a
single row. Returns `(block_of_row, n_blocks)` with block indices `1:n_blocks` ordered by time.
"""
function _v23_bootstrap_blocks(issue_times::AbstractVector{DateTime},
                               block_hours::Integer,
                               epoch::DateTime)
    block_ms = Int64(block_hours) * 3_600_000
    # `fld` rather than `div`: identical for issues at or after the epoch (the V2.2 convention and
    # the only case the study meets) and still a floor-aligned grid for earlier synthetic issues,
    # where truncation toward zero would fuse the two blocks adjacent to the epoch.
    keys = [fld(Dates.value(issue - epoch), block_ms) for issue in issue_times]
    ordered = sort!(unique(keys))
    lookup = Dict(key => index for (index, key) in enumerate(ordered))
    return [lookup[key] for key in keys], length(ordered)
end

"""
    _v23_block_bootstrap_replicates(se_comparator, se_candidate, issue_times; kwargs...)

Replicate statistics behind [`v23_block_bootstrap`](@ref). Returns
`(statistics, point, n_blocks, n_rows)` where `statistics[r]` is
`RMSE(comparator) − RMSE(candidate)` on replicate `r`'s resampled multiset of blocks. Exposed
(unexported) so that tests can check the replicate support directly.
"""
function _v23_block_bootstrap_replicates(se_comparator::AbstractVector{<:Real},
                                         se_candidate::AbstractVector{<:Real},
                                         issue_times::AbstractVector{DateTime};
                                         block_hours::Integer=V23_BOOTSTRAP_BLOCK_HOURS,
                                         replicates::Integer=V23_BOOTSTRAP_REPLICATES,
                                         seed::Integer=V23_BOOTSTRAP_SEED,
                                         epoch::DateTime=V23_BOOTSTRAP_EPOCH,
                                         min_blocks::Integer=2)
    n_rows = length(se_comparator)
    length(se_candidate) == n_rows || throw(DimensionMismatch(
        "V2.3 bootstrap comparator and candidate squared errors must have equal length",
    ))
    length(issue_times) == n_rows || throw(DimensionMismatch(
        "V2.3 bootstrap issue times must have one entry per scored row",
    ))
    n_rows > 0 || throw(ArgumentError("V2.3 bootstrap needs at least one scored row"))
    replicates > 0 || throw(ArgumentError("V2.3 bootstrap replicates must be positive"))
    block_hours > 0 || throw(ArgumentError("V2.3 bootstrap block length must be positive"))
    min_blocks >= 1 || throw(ArgumentError("V2.3 bootstrap min_blocks must be positive"))
    _v23_validate_squared_errors(se_comparator, "comparator")
    _v23_validate_squared_errors(se_candidate, "candidate")

    block_of_row, n_blocks = _v23_bootstrap_blocks(issue_times, block_hours, epoch)
    n_blocks >= min_blocks || throw(ArgumentError(string(
        "V2.3 bootstrap needs at least ", min_blocks, " blocks of ", block_hours,
        " h; the scored rows cover ", n_blocks,
    )))

    counts = zeros(Int, n_blocks)
    comparator_sse = zeros(Float64, n_blocks)
    candidate_sse = zeros(Float64, n_blocks)
    for row in 1:n_rows
        block = block_of_row[row]
        counts[block] += 1
        comparator_sse[block] += Float64(se_comparator[row])
        candidate_sse[block] += Float64(se_candidate[row])
    end

    point = sqrt(sum(comparator_sse) / n_rows) - sqrt(sum(candidate_sse) / n_rows)

    rng = MersenneTwister(seed)
    statistics = Vector{Float64}(undef, replicates)
    multiplicities = zeros(Int, n_blocks)
    for replicate in 1:replicates
        fill!(multiplicities, 0)
        for sampled in rand(rng, 1:n_blocks, n_blocks)
            multiplicities[sampled] += 1
        end
        n = sum(multiplicities .* counts)
        comparator_rmse = sqrt(sum(multiplicities .* comparator_sse) / n)
        candidate_rmse = sqrt(sum(multiplicities .* candidate_sse) / n)
        statistics[replicate] = comparator_rmse - candidate_rmse
    end
    return (statistics=statistics, point=point, n_blocks=n_blocks, n_rows=n_rows)
end

"""
    v23_block_bootstrap(se_comparator, se_candidate, issue_times;
                        block_hours=168, replicates=10_000, seed=22_022_026, alpha=0.05,
                        epoch=V23_BOOTSTRAP_EPOCH, min_blocks=2)

Paired block bootstrap of the RMSE gain `RMSE(comparator) − RMSE(candidate)` over fixed
non-overlapping 168 h calendar blocks (V2.2 convention), as preregistered in research plan §6.
The blocks are the fixed calendar grid described by [`V23_BOOTSTRAP_EPOCH`](@ref), not sliding
windows: no block overlaps another, and every block is drawn as a unit. `se_comparator` and `se_candidate` are per-row squared errors of
the two forecasts on the *same* rows, and `issue_times[i]` is the issue time of row `i`; several
rows may share an issue time (one per model step) and are then resampled together.

Blocks are the fixed 168 h windows of issue time described by [`V23_BOOTSTRAP_EPOCH`](@ref). Each
replicate draws `n_blocks` blocks with replacement and evaluates both RMSEs on the resampled
multiset, i.e. `sqrt(Σ_b m_b · SSE_b / Σ_b m_b · n_b)`; the statistic is their difference, so a
positive value means the candidate is better. This mirrors the V2.2 harness
`_v22_history_simultaneous_bootstrap` in `validation/operational/v2_2_history_crossfit.jl`: block
identifiers at lines 511-519, the `n_blocks`-draw multiplicity vector at lines 545-548, the
multiset RMSE at lines 549-557, the gain orientation (`comparator − candidate`) at line 558, and
the 5 % replicate quantile at line 563. The draw size is the block count, not the row count: a
replicate always contains `n_blocks` blocks, and the number of rows it pools varies with the sizes
of the blocks it happens to draw, which is why the multiset RMSE divides by the drawn row count
rather than by `n_rows`.

Returns `(point, lower, p_one_sided, replicates, n_blocks, n_rows)`:

  * `point` — the gain on the observed rows (every block at multiplicity one);
  * `lower` — the `alpha`-quantile of the replicate gains, i.e. the one-sided 95 % lower bound at
    the default `alpha = 0.05`; a gate passes only when this is positive;
  * `p_one_sided` — `(#{replicates with gain ≤ 0} + 1) / (replicates + 1)`, the continuity-corrected
    one-sided bootstrap p-value against "the candidate is no better than the comparator". The
    correction bounds the reported p-value away from zero, which keeps the Holm step-down of
    [`v23_holm`](@ref) well defined when no replicate crosses zero.

The result is deterministic given `seed`: the replicate draws come from `MersenneTwister(seed)` and
from nothing else.
"""
function v23_block_bootstrap(se_comparator::AbstractVector{<:Real},
                             se_candidate::AbstractVector{<:Real},
                             issue_times::AbstractVector{DateTime};
                             block_hours::Integer=V23_BOOTSTRAP_BLOCK_HOURS,
                             replicates::Integer=V23_BOOTSTRAP_REPLICATES,
                             seed::Integer=V23_BOOTSTRAP_SEED,
                             alpha::Real=V23_BOOTSTRAP_ALPHA,
                             epoch::DateTime=V23_BOOTSTRAP_EPOCH,
                             min_blocks::Integer=2)
    isfinite(alpha) && 0 < alpha < 1 || throw(ArgumentError(
        "V2.3 bootstrap alpha must lie strictly between 0 and 1",
    ))
    draws = _v23_block_bootstrap_replicates(
        se_comparator, se_candidate, issue_times;
        block_hours=block_hours, replicates=replicates, seed=seed, epoch=epoch,
        min_blocks=min_blocks,
    )
    lower = quantile(draws.statistics, Float64(alpha))
    p_one_sided = (count(<=(0.0), draws.statistics) + 1) / (length(draws.statistics) + 1)
    return (
        point=draws.point,
        lower=lower,
        p_one_sided=p_one_sided,
        replicates=length(draws.statistics),
        n_blocks=draws.n_blocks,
        n_rows=draws.n_rows,
    )
end

"""
    v23_holm(p) -> Vector{Float64}

Holm step-down family-wise adjustment of the p-values `p`, returned in input order. Sorting the
family ascending, the adjusted value of the `j`-th smallest is the running maximum of
`(m − j + 1)·p_(j)` capped at one; the running maximum enforces the monotonicity that makes the
step-down procedure coherent (an early non-rejection stops the family).

This is the vector-valued form of the package's `holm_adjust`, which returns labelled records; the
two are required to agree and the test suite checks that on random families.
"""
function v23_holm(p::AbstractVector{<:Real})
    isempty(p) && throw(ArgumentError("Holm adjustment requires at least one p-value"))
    for value in p
        (value isa Bool) && throw(ArgumentError("Holm p-values must be real numbers, not Bool"))
        isfinite(value) && 0 <= value <= 1 || throw(ArgumentError(
            "Holm p-values must be finite and lie in [0, 1]",
        ))
    end
    raw = Float64.(p)
    family_size = length(raw)
    # Ascending order with index tie-breaking: equal p-values then receive equal adjusted values,
    # because the running maximum is flat across them regardless of which one is visited first.
    order = sortperm(eachindex(raw); by=index -> (raw[index], index))
    adjusted = similar(raw)
    running = 0.0
    for (rank, index) in enumerate(order)
        running = max(running, (family_size - rank + 1) * raw[index])
        adjusted[index] = min(running, 1.0)
    end
    return adjusted
end

"""
    v23_regime_cells(latest_dst_nt, dst_rate_nt_per_h, coupling_active_mvm) -> Vector{Symbol}

Storm cells (research plan §6-A2) an issue belongs to, using causal issue-time state only:
`latest_dst_nt = Dst(t)`, `dst_rate_nt_per_h = Dst(t) − Dst(t−1)` and the issue-time coupling
`coupling_active_mvm`. Membership is not exclusive; a row is scored in every cell it returns.

  * `:all` — always;
  * `:latest_le_m50`, `:latest_le_m100` — depth cells, `Dst(t) ≤ −50` and `≤ −100` nT;
  * exactly one of `:active_deepening`, `:recovery`, `:quiet` — the V2.2 causal regime from
    `operational_v22_regime` with its deployed thresholds, so V2.3 cell reports partition issues
    the same way the V2.2 stack does;
  * `:intense_deepening` — the plan's intense/extreme deepening cell, `Dst(t) ≤ −100` nT *and*
    rate `< −15` nT/h (strict), which carries the mean-signed-error bound of gate A2.

The order of the returned labels follows [`V23_CELL_LABELS`](@ref).
"""
function v23_regime_cells(latest_dst_nt::Real,
                          dst_rate_nt_per_h::Real,
                          coupling_active_mvm::Real)
    latest = Float64(latest_dst_nt)
    rate = Float64(dst_rate_nt_per_h)
    coupling = Float64(coupling_active_mvm)
    all(isfinite, (latest, rate, coupling)) || throw(ArgumentError(
        "V2.3 cell assignment needs finite latest Dst, rate and coupling",
    ))
    cells = Symbol[:all]
    latest <= V23_CELL_DEEP_DST_NT && push!(cells, :latest_le_m50)
    latest <= V23_CELL_INTENSE_DST_NT && push!(cells, :latest_le_m100)
    push!(cells, operational_v22_regime(latest, rate, coupling))
    if latest <= V23_CELL_INTENSE_DST_NT && rate < V23_CELL_INTENSE_RATE_NT_PER_H
        push!(cells, :intense_deepening)
    end
    return cells
end
