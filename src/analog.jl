# analog.jl — Operational analog ensemble retrieval and analog driver continuation.
#
# Given the issue-time features of `analog_features.jl`, the analog ensemble tail replaces the served
# inertial-SINDy exponential relaxation with an ensemble of driver continuations copied from historical
# solar-wind origins whose issue-time state resembles the query. Two operations are needed and are
# both implemented here: the retrieval of the K nearest admissible archive origins, and the mapping
# from a retrieved origin to the driver tuple that the SINDy core integrates at each rollout step.
#
# Retrieval is a weighted Euclidean nearest-neighbour search on standardised features. It is
# evaluated blockwise through the expansion ‖q − a‖² = ‖q‖² + ‖a‖² − 2⟨q, a⟩ so the inner products
# come from one BLAS matrix product per query block; the per-query constant ‖q‖² is dropped because
# it cannot change the ordering of the archive. Origins inside a ±168 h window of the query are
# removed before selection: solar-wind structure is autocorrelated over several days, so a
# neighbouring hour would return the query's own storm and leak the answer into its own forecast.
#
# The continuation itself is anchored, not absolute. For an origin at issue time s and rollout step
# k, the archived increments of speed and log-density between the origin's own issue record (s−1)
# and its step record (s+k−1) are applied to the query's issue driver, while the magnetic field is
# copied directly, because the field has no meaningful multiplicative baseline to preserve and its
# sign structure is what drives the ring current. The ablation variant `direct = true` copies speed
# and density as well, which tests whether the anchoring matters.

"Exclusion half-window (hours) between a query issue time and an admissible analog origin."
const ANALOG_EXCLUSION_HOURS = 168

"Largest rollout step an analog origin must be able to supply."
const ANALOG_MAX_STEP = 7

"Physical floor on a continued solar-wind speed (km/s)."
const ANALOG_MEMBER_MIN_V_KMS = 200.0

"Physical bounds on a continued proton density (cm^-3)."
const ANALOG_MEMBER_MIN_N_CM3 = 0.05
"Upper projection bound on continued proton density, in cm^-3."
const ANALOG_MEMBER_MAX_N_CM3 = 200.0

"Default number of queries whose distances are formed per BLAS block."
const ANALOG_KNN_BLOCK = 256

const _ANALOG_HOUR_EPOCH = DateTime(2000, 1, 1)

"Convert a query/archive time vector to hours on a common axis."
function _analog_hours(times)
    isempty(times) && return Float64[]
    first_value = first(times)
    if first_value isa DateTime
        return Float64[Dates.value(t - _ANALOG_HOUR_EPOCH) / 3_600_000 for t in times]
    elseif first_value isa AbstractString
        return Float64[
            Dates.value(DateTime(strip(String(t))) - _ANALOG_HOUR_EPOCH) / 3_600_000 for t in times
        ]
    elseif first_value isa Real
        return Float64[Float64(t) for t in times]
    end
    throw(ArgumentError(
        "analog times must be DateTime, ISO-8601 strings, or hours as Real, got $(typeof(first_value))",
    ))
end

"""
    analog_knn(Xq, tq, Xa, ta, K; weights = ANALOG_WEIGHTS.uniform,
            exclusion_hours = ANALOG_EXCLUSION_HOURS, block = ANALOG_KNN_BLOCK) -> Matrix{Int}

Return the `nq × K` matrix whose row `i` lists, in increasing weighted Euclidean distance, the
archive rows nearest to query `i`. `Xq` (`nq × 18`) and `Xa` (`na × 18`) hold standardised analog ensemble
features; `tq` and `ta` are the matching issue times (`DateTime`, ISO-8601 strings, or hours).
Archive rows within `exclusion_hours` of a query are inadmissible for that query and are never
returned. Ties in distance resolve toward the smaller archive index, so the result is a
deterministic function of the inputs alone.

The search rejects non-finite features rather than ignoring them: an incomplete history has no
defined distance, and callers must drop those rows using the `ok` mask from
[`analog_feature_matrix`](@ref) before retrieval.
"""
function analog_knn(Xq::AbstractMatrix{<:Real}, tq, Xa::AbstractMatrix{<:Real}, ta, K::Integer;
                 weights::AbstractVector{<:Real} = ANALOG_WEIGHTS.uniform,
                 exclusion_hours::Real = ANALOG_EXCLUSION_HOURS,
                 block::Integer = ANALOG_KNN_BLOCK)
    nq, p = size(Xq)
    na, pa = size(Xa)
    p == pa || throw(DimensionMismatch(
        "query features have $(p) columns but archive features have $(pa)",
    ))
    length(weights) == p || throw(DimensionMismatch(
        "distance weights have $(length(weights)) entries but features have $(p) columns",
    ))
    K >= 1 || throw(ArgumentError("K must be positive, got $K"))
    K <= na || throw(ArgumentError("K = $K exceeds the archive size $na"))
    block >= 1 || throw(ArgumentError("block must be positive, got $block"))
    isfinite(exclusion_hours) && exclusion_hours >= 0 || throw(ArgumentError(
        "exclusion_hours must be finite and non-negative",
    ))
    all(w -> isfinite(w) && w >= 0, weights) || throw(ArgumentError(
        "distance weights must be finite and non-negative",
    ))
    any(>(0), weights) || throw(ArgumentError("at least one distance weight must be positive"))
    all(isfinite, Xq) || throw(ArgumentError(
        "query features must be finite; drop incomplete-history rows before retrieval",
    ))
    all(isfinite, Xa) || throw(ArgumentError(
        "archive features must be finite; drop incomplete-history rows before retrieval",
    ))
    query_hours = _analog_hours(tq)
    archive_hours = _analog_hours(ta)
    length(query_hours) == nq || throw(DimensionMismatch(
        "query times have $(length(query_hours)) entries but query features have $(nq) rows",
    ))
    length(archive_hours) == na || throw(DimensionMismatch(
        "archive times have $(length(archive_hours)) entries but archive features have $(na) rows",
    ))
    all(isfinite, query_hours) && all(isfinite, archive_hours) ||
        throw(ArgumentError("query and archive times must be finite"))

    scale = sqrt.(Float64.(weights))
    all(isfinite, scale) || throw(ArgumentError("distance weights must be representable as Float64"))
    # Scaling both matrices by sqrt(w) turns the weighted Euclidean distance into a plain one, so a
    # single BLAS product supplies every inner product in the block. The block is laid out with the
    # archive along the leading dimension, so each query's distance vector is contiguous in memory:
    # the selection step touches every archive entry once, and a strided layout there costs far more
    # than the matrix product itself.
    A = Float64.(Xa) .* transpose(scale)                 # na × p
    Qt = permutedims(Float64.(Xq) .* transpose(scale))   # p × nq, contiguous
    archive_norms = vec(sum(abs2, A; dims = 2))
    query_norms = vec(sum(abs2, Qt; dims = 1))
    all(isfinite, archive_norms) && all(isfinite, query_norms) ||
        throw(ArgumentError("scaled features must have finite Float64 squared norms"))
    max_archive_norm = maximum(archive_norms)
    exclusion = Float64(exclusion_hours)
    isfinite(exclusion) || throw(ArgumentError("exclusion_hours must be representable as Float64"))
    sorted_archive = issorted(archive_hours)

    neighbours = Matrix{Int}(undef, nq, K)
    distances = Matrix{Float64}(undef, na, min(block, nq))
    order = Vector{Int}(undef, na)
    candidates = Int[]
    sizehint!(candidates, K)
    for lo in 1:block:nq
        hi = min(lo + block - 1, nq)
        columns = hi - lo + 1
        chunk = view(distances, :, 1:columns)
        mul!(chunk, A, view(Qt, :, lo:hi), -2.0, 0.0)
        chunk .+= archive_norms
        for (offset, q) in enumerate(lo:hi)
            row = view(chunk, :, offset)
            admissible = _analog_apply_exclusion!(row, archive_hours, query_hours[q], exclusion,
                                               sorted_archive)
            admissible >= K || throw(ArgumentError(
                "query $q has only $(admissible) archive origins outside the " *
                "±$(exclusion) h exclusion window but K = $K neighbours were requested",
            ))
            selected = partialsortperm!(order, row, 1:K)
            # Expanded distances lose small separations through cancellation, even
            # for ordinary standardized features near 3. Retain BLAS for the broad
            # search, then recompute every possible contender using differences.
            # The conservative dot-product error bound covers both sides of the
            # cutoff; ordinary well-separated searches refine only K origins.
            error_margin = (64 * p * eps(Float64)) *
                           max(max_archive_norm, query_norms[q])
            cutoff = row[selected[K]] + error_margin
            empty!(candidates)
            @inbounds for j in 1:na
                if isfinite(row[j]) && row[j] <= cutoff
                    push!(candidates, j)
                    distance = 0.0
                    for c in 1:p
                        distance += abs2(A[j, c] - Qt[c, q])
                    end
                    isfinite(distance) || throw(ArgumentError(
                        "weighted distance is not representable as Float64",
                    ))
                    row[j] = distance
                end
            end
            partialsort!(candidates, 1:K; by=j -> (row[j], j))
            @inbounds for j in 1:K
                neighbours[q, j] = candidates[j]
            end
        end
    end
    return neighbours
end

"""
    _analog_apply_exclusion!(row, archive_hours, query_hour, exclusion, sorted_archive) -> Int

Mark every archive entry within `exclusion` hours of `query_hour` as unreachable (`Inf`) and return
the number of admissible entries. When the archive times are sorted the excluded entries form one
contiguous range, which is located by binary search instead of a full scan.
"""
function _analog_apply_exclusion!(row::AbstractVector{Float64}, archive_hours::Vector{Float64},
                               query_hour::Float64, exclusion::Float64, sorted_archive::Bool)
    na = length(row)
    exclusion == 0 && return na
    if sorted_archive
        # The exclusion rule is strict (|Δt| < exclusion), so an origin exactly `exclusion` hours
        # away stays admissible: take the first index strictly above the lower endpoint and the
        # last index strictly below the upper one.
        lo = searchsortedlast(archive_hours, query_hour - exclusion) + 1
        hi = searchsortedfirst(archive_hours, query_hour + exclusion) - 1
        hi < lo && return na
        @inbounds for j in lo:hi
            row[j] = Inf
        end
        return na - (hi - lo + 1)
    end
    excluded = 0
    @inbounds for j in 1:na
        if abs(archive_hours[j] - query_hour) < exclusion
            row[j] = Inf
            excluded += 1
        end
    end
    return na - excluded
end

"Fetch the driver record tagged `t`, or `nothing` when the hourly archive has no such record."
@inline _analog_record(lookup, t::DateTime) = get(lookup, t, nothing)

"""
    analog_origin_ok(lookup, times; max_step = ANALOG_MAX_STEP) -> BitVector

Flag which candidate analog origins in `times` can supply a complete continuation: the origin's own
issue record (`s−1`) and every step record (`s+k−1`, `k = 1 … max_step`) must exist in the hourly
driver `lookup`. Ineligible origins must be removed from the archive before
[`analog_knn`](@ref), because an origin that cannot be continued would otherwise be retrieved and then
silently dropped, changing the ensemble size in a query-dependent way.
"""
function analog_origin_ok(lookup, times; max_step::Integer = ANALOG_MAX_STEP)
    max_step >= 1 || throw(ArgumentError("max_step must be positive, got $max_step"))
    stamps = _analog_datetimes(times)
    ok = falses(length(stamps))
    for (i, s) in enumerate(stamps)
        _analog_record(lookup, s - Hour(1)) === nothing && continue
        complete = true
        for k in 1:max_step
            if _analog_record(lookup, s + Hour(k - 1)) === nothing
                complete = false
                break
            end
        end
        ok[i] = complete
    end
    return ok
end

function _analog_datetimes(times)
    isempty(times) && return DateTime[]
    first_value = first(times)
    first_value isa DateTime && return collect(DateTime, times)
    first_value isa AbstractString && return DateTime.(strip.(String.(times)))
    throw(ArgumentError("analog origin times must be DateTime or ISO-8601 strings"))
end

"""
    analog_member_driver(query_issue_drv, analog_frame_lookup, s, k; direct = false) -> NamedTuple

Driver tuple `(V, Bz, By, n, Pdyn)` that the analog origin at issue time `s` prescribes for rollout
step `k` of a query whose issue driver is `query_issue_drv`. Speed and log-density are continued as
increments measured on the analog, `V = V_q + (V(s+k−1) − V(s−1))` and
`log n = log n_q + (log n(s+k−1) − log n(s−1))`, so the ensemble inherits the archived evolution
without importing the archive's absolute level; the magnetic components are copied from the analog
record. Speed is floored at $(ANALOG_MEMBER_MIN_V_KMS) km/s and density is clamped to
[$(ANALOG_MEMBER_MIN_N_CM3), $(ANALOG_MEMBER_MAX_N_CM3)] cm^-3, and the dynamic pressure is recomputed
from the continued values through the package's proton-only identity so the pressure term of the
Dst reconstruction stays consistent with them.

With `direct = true` the speed and density are taken as absolute analog values, the preregistered
T1a ablation.
"""
function analog_member_driver(query_issue_drv, analog_frame_lookup, s::DateTime, k::Integer;
                           direct::Bool = false)
    k >= 1 || throw(ArgumentError("rollout step k must be positive, got $k"))
    step = _analog_record(analog_frame_lookup, s + Hour(k - 1))
    step === nothing && throw(ArgumentError(
        "analog origin $s has no driver record for step $k (record $(s + Hour(k - 1)))",
    ))
    V, n = if direct
        step.V, step.n
    else
        origin = _analog_record(analog_frame_lookup, s - Hour(1))
        origin === nothing && throw(ArgumentError(
            "analog origin $s has no issue driver record $(s - Hour(1))",
        ))
        (
            Float64(query_issue_drv.V) + (step.V - origin.V),
            exp(log(Float64(query_issue_drv.n)) + (log(step.n) - log(origin.n))),
        )
    end
    V = max(Float64(V), ANALOG_MEMBER_MIN_V_KMS)
    n = clamp(Float64(n), ANALOG_MEMBER_MIN_N_CM3, ANALOG_MEMBER_MAX_N_CM3)
    return (V = V, Bz = Float64(step.Bz), By = Float64(step.By), n = n,
            Pdyn = dynamic_pressure(n, V))
end

"""
    analog_member(query_issue_drv, analog_frame_lookup, s; direct = false) -> Function

Wrap [`analog_member_driver`](@ref) as an ensemble member for the analog ensemble kernel: the returned callable
takes `(k, last_known, kΔ)` and supplies the analog continuation for rollout step `k`. The
L1-admitted steps `k ≤ kΔ` never reach the member, so the continuation is asked only for the hours
that the served inertial-SINDy tail would have relaxed.
"""
function analog_member(query_issue_drv, analog_frame_lookup, s::DateTime; direct::Bool = false)
    return (k, last_known, kΔ) -> analog_member_driver(
        query_issue_drv, analog_frame_lookup, s, k; direct = direct,
    )
end
