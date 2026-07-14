# Data cleaning, storm catalog, and real-data preparation

using Dates

const OMNI_OBSERVATION_COLUMNS = (:V, :Bz, :By, :n, :Pdyn, :T, :Dst, :AE, :AL, :AU)

"""
    add_original_observation_flags!(df; columns=OMNI_OBSERVATION_COLUMNS)

Snapshot which parsed OMNI values were finite before any interpolation or
nearest-neighbour filling. For each requested column `x`, add a Boolean
`x_observed` column. Call this immediately after [`parse_omni2`](@ref) and before
[`clean_omni_data!`](@ref). Existing flag columns are rejected so a later call
cannot silently relabel interpolated values as original observations.
"""
function add_original_observation_flags!(df::DataFrame;
        columns=OMNI_OBSERVATION_COLUMNS)
    requested = collect(columns)
    length(unique(requested)) == length(requested) ||
        throw(ArgumentError("original-observation columns must be unique"))
    for col in requested
        col in propertynames(df) || throw(ArgumentError("missing OMNI column: $col"))
        flag = Symbol(col, "_observed")
        flag in propertynames(df) &&
            throw(ArgumentError("original-observation flag already exists: $flag"))
    end
    for col in requested
        flag = Symbol(col, "_observed")
        df[!, flag] = Bool[!ismissing(x) && x isa Real && isfinite(x)
                           for x in df[!, col]]
    end
    return df
end

"""
    original_sindy_mask(df, rows; smooth_window=5, require_ae=false)

Return the strict original-observation mask for SINDy rows in one storm window.
The instantaneous library drivers (`V`, `Bz`, `By`, `n`) must be original at the
candidate row. In addition, every original `Dst`, `V`, and `n` observation used
by the centered smoothing-plus-finite-difference target stencil must be present;
`AE` is included in that stencil when `require_ae=true`. Consequently no target,
pressure correction, or current-row driver depends on an interpolated value.

The required `*_observed` columns are produced by
[`add_original_observation_flags!`](@ref). The returned vector is local to
`rows` and aligns with [`extract_storm_data`](@ref) and
[`prepare_sindy_data`](@ref) for the same window.
"""
function original_sindy_mask(df::DataFrame, rows;
                             smooth_window::Int=5, require_ae::Bool=false)
    smooth_window > 0 && isodd(smooth_window) ||
        throw(ArgumentError("smooth_window must be a positive odd integer"))
    row_idx = collect(Int, rows)
    isempty(row_idx) && return Bool[]
    all(i -> 1 <= i <= nrow(df), row_idx) || throw(BoundsError(df, row_idx))
    all(==(1), diff(row_idx)) ||
        throw(ArgumentError("rows must be a chronological contiguous window"))

    current_flags = [:V_observed, :Bz_observed, :By_observed, :n_observed]
    stencil_flags = [:Dst_observed, :V_observed, :n_observed]
    require_ae && push!(stencil_flags, :AE_observed)
    for flag in union(current_flags, stencil_flags)
        flag in propertynames(df) ||
            throw(ArgumentError("missing original-observation flag: $flag"))
    end

    n = length(row_idx)
    radius = div(smooth_window, 2) + 1
    mask = falses(n)
    for i in 1:n
        all(flag -> Bool(df[row_idx[i], flag]), current_flags) || continue
        lo = max(1, i - radius)
        hi = min(n, i + radius)
        mask[i] = all(flag -> all(Bool, @view(df[row_idx[lo:hi], flag])),
                      stencil_flags)
    end
    return mask
end

"""
    clean_omni_data!(df::DataFrame; causal::Bool=false)

In-place cleaning of raw OMNI2 DataFrame:
1. Remove rows with missing critical variables (V, Bz, Dst)
2. Fill short gaps (≤3 hours) in the measured columns
3. Compute derived quantities: Bs, θ_c, proton-only Pdyn, Dst*
4. Add quality flag column

Short-gap filling depends on `causal`:
- `causal=false` (default, offline/training preprocessing): centered linear
  interpolation, which uses the post-gap bound.
- `causal=true` (replay/serving inputs): forward-fill (last-observation-carried-
  forward). No gap hour is ever filled with a value measured *after* it, so
  issue-time replay inputs stay strictly causal. `Dst` is never causally filled
  (a missing target/anchor Dst is left NaN rather than persisted).

Dynamic pressure is always recomputed proton-only (`1.6726e-6·n·V²`) via
[`dynamic_pressure`](@ref); the OMNI word-29 alpha-inclusive pressure is not kept,
so training and serving share one convention. The Dst* pressure correction falls
back to a physically-defensible pressure (carried-forward last known Pdyn, else
the quiet-time default) when Pdyn is unavailable.

Returns the modified DataFrame.
"""
function clean_omni_data!(df::DataFrame; causal::Bool=false)
    n_raw = nrow(df)

    # --- Short-gap (≤3 h) filling of the measured columns. Pdyn is NOT filled as
    #     an independent column; it is recomputed from the filled n, V below so it
    #     keeps the n·V² identity (an independently interpolated Pdyn drifts from
    #     n·V² at gap edges because Pdyn is quadratic in V). In causal mode Dst is
    #     left unfilled so a missing anchor/target is dropped, never persisted. ---
    fill_cols = causal ? [:V, :Bz, :By, :n, :T, :AE, :AL, :AU] :
                         [:V, :Bz, :By, :n, :T, :Dst, :AE, :AL, :AU]
    for col in fill_cols
        map!(x -> isfinite(x) ? x : NaN, df[!, col], df[!, col])
        causal ? _ffill_short_gaps!(df[!, col], 3) : _interp_short_gaps!(df[!, col], 3)
    end

    # --- Dynamic pressure: always proton-only from the filled n, V (drop the OMNI
    #     word-29 alpha-inclusive value so training matches the proton-only serve
    #     convention; real-time SWPC feeds carry no alpha data). ---
    for i in 1:nrow(df)
        df.Pdyn[i] = dynamic_pressure(df.n[i], df.V[i])
    end

    # --- Quality flag: 1 = all critical vars present, 0 = any critical missing ---
    df[!, :quality] = ones(Int, nrow(df))
    for i in 1:nrow(df)
        if !isfinite(df.V[i]) || !isfinite(df.Bz[i]) || !isfinite(df.Dst[i]) ||
           !isfinite(df.n[i]) || !isfinite(df.Pdyn[i])
            df.quality[i] = 0
        end
    end

    # --- Derived quantities ---
    df[!, :Bs] = [isnan(bz) ? NaN : max(-bz, 0.0) for bz in df.Bz]
    df[!, :theta_c] = [isnan(by) || isnan(bz) ? NaN : atan(abs(by), bz) for (by, bz) in zip(df.By, df.Bz)]
    df[!, :BT] = [isnan(by) || isnan(bz) ? NaN : hypot(by, bz) for (by, bz) in zip(df.By, df.Bz)]
    for i in eachindex(df.BT)
        if isfinite(df.By[i]) && isfinite(df.Bz[i]) && !isfinite(df.BT[i])
            df.BT[i] = NaN
            df.quality[i] = 0
        end
    end

    # Dst*: pressure-correct where Pdyn is available; when it is missing use a
    # physically-defensible fallback (carry the last known Pdyn forward over a
    # short outage, else the climatological quiet-time default) rather than the
    # old +11-only fallback, which implied Pdyn=0 and left outage rows 7–30 nT
    # above their pressure-corrected neighbours.
    df[!, :Dst_star] = Vector{Float64}(undef, nrow(df))
    last_pdyn = NaN
    last_pdyn_age = PDYN_CARRY_MAX_AGE_H + 1
    for i in 1:nrow(df)
        if isfinite(df.Pdyn[i]) && df.Pdyn[i] >= 0
            last_pdyn = df.Pdyn[i]
            last_pdyn_age = 0
        else
            last_pdyn_age = min(last_pdyn_age + 1, PDYN_CARRY_MAX_AGE_H + 1)
        end
        if !isfinite(df.Dst[i])
            df.Dst_star[i] = NaN
        else
            pdyn_eff = resolve_pdyn(df.Pdyn[i], last_pdyn, last_pdyn_age)
            df.Dst_star[i] = dst_to_dst_star(df.Dst[i], pdyn_eff)
        end
    end

    n_valid = count(df.quality .== 1)
    n_dropped = n_raw - n_valid
    println("  Cleaning: $(n_raw) rows, $(n_valid) valid ($(n_dropped) with missing critical vars)")

    return df
end

"""
    _interp_short_gaps!(x, max_gap)

Linear interpolation for gaps of ≤ max_gap consecutive NaN values in-place.
Longer gaps remain NaN.
"""
function _interp_short_gaps!(x::AbstractVector{Float64}, max_gap::Int)
    n = length(x)
    i = 1
    while i <= n
        if isnan(x[i])
            # Find extent of gap
            gap_start = i
            while i <= n && isnan(x[i])
                i += 1
            end
            gap_end = i - 1
            gap_len = gap_end - gap_start + 1

            # Interpolate only if gap is short and bounded by valid values
            if gap_len <= max_gap && gap_start > 1 && gap_end < n &&
               !isnan(x[gap_start - 1]) && !isnan(x[gap_end + 1])
                v0 = x[gap_start - 1]
                v1 = x[gap_end + 1]
                for j in gap_start:gap_end
                    frac = (j - gap_start + 1) / (gap_len + 1)
                    x[j] = v0 + frac * (v1 - v0)
                end
            end
        else
            i += 1
        end
    end
end

"""
    _ffill_short_gaps!(x, max_gap)

Causal forward-fill for gaps of ≤ `max_gap` consecutive NaN values in-place:
each short gap is filled with the last finite value BEFORE the gap
(last-observation-carried-forward). Uses no post-gap (future) value, so replay/
serving inputs stay strictly causal. Longer gaps, and leading gaps with no prior
value, remain NaN.
"""
function _ffill_short_gaps!(x::AbstractVector{Float64}, max_gap::Int)
    n = length(x)
    i = 1
    while i <= n
        if isnan(x[i])
            gap_start = i
            while i <= n && isnan(x[i])
                i += 1
            end
            gap_end = i - 1
            gap_len = gap_end - gap_start + 1
            if gap_len <= max_gap && gap_start > 1 && !isnan(x[gap_start - 1])
                v0 = x[gap_start - 1]
                for j in gap_start:gap_end
                    x[j] = v0
                end
            end
        else
            i += 1
        end
    end
end

# ============================================================
# Storm catalog
# ============================================================

"""
    StormCatalogEntry

Metadata for a single geomagnetic storm event.
"""
struct StormCatalogEntry
    storm_id::Int
    onset_time::DateTime
    min_dst_star::Float64
    min_dst_star_time::DateTime
    recovery_end_time::DateTime
    duration_hr::Float64
    solar_cycle::Int
    split::String              # "train", "val", "test"
    onset_idx::Int             # extraction-window start index in cleaned DataFrame
    end_idx::Int
end

# Source compatibility for callers that consumed the pre-migration field names.
# Canonical artifacts and new code use the explicit Dst* names above.
function Base.getproperty(entry::StormCatalogEntry, name::Symbol)
    name === :min_dst && return getfield(entry, :min_dst_star)
    name === :min_dst_time && return getfield(entry, :min_dst_star_time)
    return getfield(entry, name)
end

function Base.propertynames(::StormCatalogEntry, private::Bool=false)
    canonical = fieldnames(StormCatalogEntry)
    return (canonical..., :min_dst, :min_dst_time)
end

"""
    _solar_cycle(time)

Assign the official SILSO cycle number from its month-level minimum boundary.
The boundaries are based on the World Data Center SILSO Version 2 table of
13-month-smoothed sunspot-number minima.
"""
function _solar_cycle(time::Union{Date,DateTime})
    day = Date(time)
    day < Date(1964, 10, 1) && return 19
    day < Date(1976, 3, 1) && return 20
    day < Date(1986, 9, 1) && return 21
    day < Date(1996, 8, 1) && return 22
    day < Date(2008, 12, 1) && return 23
    day < Date(2019, 12, 1) && return 24
    return 25
end

"""
    _assign_split(cycle)

Assign train/val/test split by solar cycle:
  Train: cycles 20–23 (1964-10 through 2008-11)
  Val:   cycle 24 (2008–2019)
  Test:  cycle 25 (2019–present, includes May 2024 superstorm)
  Exclude: pre-cycle-20 records
"""
function _assign_split(cycle::Int)
    cycle in 20:23 && return "train"
    cycle == 24 && return "val"
    cycle >= 25 && return "test"
    return "exclude"
end

"""
    build_storm_catalog(df; dst_thresh=-50.0, window_pre=24, window_post=144,
                           min_separation=48)

Identify geomagnetic storms in cleaned OMNI DataFrame and build a catalog.

A storm is defined by Dst* crossing below `dst_thresh`.
Windows: `window_pre` hours before onset, `window_post` hours after onset.
Storms must be separated by at least `min_separation` hours.

Returns Vector{StormCatalogEntry}.
"""
function build_storm_catalog(df::DataFrame;
                              dst_thresh::Real=-50.0,
                              window_pre::Int=24,
                              window_post::Int=144,
                              min_separation::Int=48)
    isfinite(dst_thresh) || throw(ArgumentError("dst_thresh must be finite"))
    window_pre >= 0 || throw(ArgumentError("window_pre must be nonnegative"))
    window_post >= 0 || throw(ArgumentError("window_post must be nonnegative"))
    min_separation >= 0 || throw(ArgumentError("min_separation must be nonnegative"))
    required = (:datetime, :Dst_star, :quality)
    missing_columns = filter(name -> !(name in propertynames(df)), required)
    isempty(missing_columns) || throw(ArgumentError(
        "storm catalog is missing required columns: $(join(string.(missing_columns), ", "))"
    ))
    catalog = StormCatalogEntry[]
    n = nrow(df)
    n >= 1 || throw(ArgumentError("storm catalog requires at least one hourly row"))
    all(value -> value isa DateTime, df.datetime) ||
        throw(ArgumentError("storm catalog datetimes must be DateTime values"))
    all(i -> df.datetime[i] - df.datetime[i - 1] == Hour(1), 2:n) ||
        throw(ArgumentError("storm catalog rows must be strictly contiguous and hourly"))
    all(value -> value isa Real && (isfinite(value) || isnan(value)), df.Dst_star) ||
        throw(ArgumentError("Dst_star must contain only finite values or NaN gaps"))
    all(value -> value isa Real && (value == 0 || value == 1), df.quality) ||
        throw(ArgumentError("quality must contain only 0/1 flags"))

    # For storm detection, only need Dst_star (Dst + Pdyn).
    # Full quality (V, Bz) checked separately for SINDy usability.
    valid_mask = isfinite.(df.Dst_star)
    # Precompute SINDy-usable mask (V, Bz, Dst_star all valid)
    sindy_mask = (df.quality .== 1) .& valid_mask

    storm_id = 0
    last_storm_end = 0

    i = 1
    while i <= n
        # Skip if not valid or above threshold
        if !valid_mask[i] || df.Dst_star[i] >= dst_thresh
            i += 1
            continue
        end

        # Enforce minimum separation from last storm
        if i <= last_storm_end + min_separation
            i += 1
            continue
        end

        # Found a storm crossing — find onset (first point below threshold)
        onset_idx = i

        # Walk backwards to find actual onset (last point above -20 nT before crossing)
        # Cap at window_pre to avoid walking back too far (prevents infinite loop
        # where search_end < i causing i to go backwards)
        j = onset_idx
        back_limit = max(1, onset_idx - window_pre)
        while j > back_limit && df.Dst_star[j-1] < -20.0 && valid_mask[j-1]
            j -= 1
        end
        onset_idx = j

        # Find minimum Dst* in the storm window (search from original detection point)
        search_end = min(n, i + window_post)
        min_dst_star = df.Dst_star[onset_idx]
        min_idx = onset_idx
        for k in onset_idx:search_end
            if valid_mask[k] && df.Dst_star[k] < min_dst_star
                min_dst_star = df.Dst_star[k]
                min_idx = k
            end
        end

        # Find recovery end: Dst* returns above -20 nT or end of window
        recovery_idx = search_end
        for k in min_idx:search_end
            if valid_mask[k] && df.Dst_star[k] >= -20.0
                recovery_idx = k
                break
            end
        end

        # Define extraction window
        win_start = max(1, onset_idx - window_pre)
        win_end = min(n, recovery_idx + 24)  # 24hr buffer after recovery

        # Check sufficient SINDy-usable data in window (>60% with V, Bz, Dst_star)
        n_valid_in_window = count(@view sindy_mask[win_start:win_end])
        n_window = win_end - win_start + 1
        if n_valid_in_window / n_window < 0.60
            # Skip storm with too much missing data
            last_storm_end = max(last_storm_end, win_end)
            i = max(i + 1, win_end + 1)
            continue
        end

        storm_id += 1
        cycle = _solar_cycle(df.datetime[onset_idx])
        duration = (win_end - win_start)  # hours (hourly data)

        entry = StormCatalogEntry(
            storm_id,
            df.datetime[onset_idx],
            min_dst_star,
            df.datetime[min_idx],
            df.datetime[recovery_idx],
            Float64(duration),
            cycle,
            _assign_split(cycle),
            win_start,
            win_end
        )
        push!(catalog, entry)

        last_storm_end = max(last_storm_end, win_end)
        i = max(i + 1, win_end + 1)
    end

    # Summary
    n_train = count(e -> e.split == "train", catalog)
    n_val   = count(e -> e.split == "val", catalog)
    n_test  = count(e -> e.split == "test", catalog)
    n_excluded = count(e -> e.split == "exclude", catalog)
    println("  Storm catalog: $(length(catalog)) storms (train=$(n_train), " *
            "val=$(n_val), test=$(n_test), exclude=$(n_excluded))")
    if !isempty(catalog)
        println("  Dst* range: $(round(minimum(e.min_dst_star for e in catalog), digits=1)) to $(round(maximum(e.min_dst_star for e in catalog), digits=1)) nT")
    end

    return catalog
end

"""
    extract_storm_data(df, entry::StormCatalogEntry)

Extract a single storm's data from the cleaned OMNI DataFrame as SolarWindData.
Replaces remaining NaN values with linear interpolation for ODE integration.
"""
function extract_storm_data(df::DataFrame, entry::StormCatalogEntry)
    rows = entry.onset_idx:entry.end_idx
    n_pts = length(rows)

    # Extract and fill any remaining NaN with nearest-neighbor for simulation
    V    = _fillnan(df.V[rows])
    Bz   = _fillnan(df.Bz[rows])
    By   = _fillnan(df.By[rows])
    n_d  = _fillnan(df.n[rows])
    Pdyn = _fillnan(df.Pdyn[rows])
    Dst  = _fillnan(df.Dst[rows])
    Dst_star = _fillnan(df.Dst_star[rows])

    t = Float64.(0:n_pts-1)  # hourly, starting at 0

    return SolarWindData(t, V, Bz, By, n_d, Pdyn, Dst, Dst_star)
end

"""
    extract_all_storms(df, catalog; split=nothing)

Extract SolarWindData for all storms in catalog (optionally filtered by split).
Returns (Vector{SolarWindData}, Vector{StormCatalogEntry}).
"""
function extract_all_storms(df::DataFrame, catalog::Vector{StormCatalogEntry};
                             split::Union{Nothing,String}=nothing)
    entries = split === nothing ? catalog : filter(e -> e.split == split, catalog)
    datasets = SolarWindData[]
    for entry in entries
        push!(datasets, extract_storm_data(df, entry))
    end
    return datasets, entries
end

"""
    _fillnan(x)

Fill NaN values by nearest-neighbor interpolation (forward then backward fill).
"""
function _fillnan(x::AbstractVector)
    out = collect(Float64, x)
    n = length(out)

    # Forward fill
    for i in 2:n
        if isnan(out[i]) && !isnan(out[i-1])
            out[i] = out[i-1]
        end
    end
    # Backward fill
    for i in (n-1):-1:1
        if isnan(out[i]) && !isnan(out[i+1])
            out[i] = out[i+1]
        end
    end
    return out
end

"""
    save_storm_catalog(catalog, filepath)

Save storm catalog to CSV.
"""
function save_storm_catalog(catalog::Vector{StormCatalogEntry}, filepath::String)
    _require_regular_output_target(filepath)
    df = DataFrame(
        storm_id = [e.storm_id for e in catalog],
        onset_time = [e.onset_time for e in catalog],
        min_dst_star = [e.min_dst_star for e in catalog],
        min_dst_star_time = [e.min_dst_star_time for e in catalog],
        recovery_end_time = [e.recovery_end_time for e in catalog],
        duration_hr = [e.duration_hr for e in catalog],
        solar_cycle = [e.solar_cycle for e in catalog],
        split = [e.split for e in catalog],
        onset_idx = [e.onset_idx for e in catalog],
        end_idx = [e.end_idx for e in catalog]
    )
    parent = dirname(filepath)
    !isempty(parent) && mkpath(parent)
    temporary, io = mktemp(parent; cleanup=false)
    close(io)
    try
        CSV.write(temporary, df)
        _atomic_replace_regular(temporary, filepath)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    println("  Saved storm catalog: $filepath ($(nrow(df)) storms)")
end

"""
    load_storm_catalog(filepath)

Load storm catalog from CSV.
"""
function load_storm_catalog(filepath::String)
    df = CSV.read(filepath, DataFrame)
    min_col = :min_dst_star in propertynames(df) ? :min_dst_star :
              :min_dst in propertynames(df) ? :min_dst : nothing
    min_time_col = :min_dst_star_time in propertynames(df) ? :min_dst_star_time :
                   :min_dst_time in propertynames(df) ? :min_dst_time : nothing
    min_col === nothing && throw(ArgumentError(
        "storm catalog must contain min_dst_star (or legacy min_dst)"
    ))
    min_time_col === nothing && throw(ArgumentError(
        "storm catalog must contain min_dst_star_time (or legacy min_dst_time)"
    ))
    required = (:storm_id, :onset_time, :recovery_end_time, :duration_hr,
                :solar_cycle, :split, :onset_idx, :end_idx)
    missing_columns = filter(name -> !(name in propertynames(df)), required)
    isempty(missing_columns) || throw(ArgumentError(
        "storm catalog is missing required columns: $(join(string.(missing_columns), ", "))"
    ))
    catalog = StormCatalogEntry[]
    for row in eachrow(df)
        storm_id = Int(row.storm_id)
        min_dst_star = Float64(row[min_col])
        duration_hr = Float64(row.duration_hr)
        solar_cycle = Int(row.solar_cycle)
        split = String(row.split)
        onset_idx = Int(row.onset_idx)
        end_idx = Int(row.end_idx)
        storm_id >= 1 || throw(ArgumentError("storm_id must be positive"))
        isfinite(min_dst_star) || throw(ArgumentError("min_dst_star must be finite"))
        isfinite(duration_hr) && duration_hr >= 0 ||
            throw(ArgumentError("duration_hr must be finite and nonnegative"))
        split in ("train", "val", "test", "exclude") ||
            throw(ArgumentError("invalid storm split: $split"))
        1 <= onset_idx <= end_idx ||
            throw(ArgumentError("storm indices must satisfy 1 <= onset_idx <= end_idx"))
        onset_time = DateTime(row.onset_time)
        min_dst_star_time = DateTime(row[min_time_col])
        recovery_end_time = DateTime(row.recovery_end_time)
        onset_time <= min_dst_star_time <= recovery_end_time ||
            throw(ArgumentError("storm times must satisfy onset <= minimum Dst* <= recovery"))
        isapprox(duration_hr, end_idx - onset_idx; rtol=0.0, atol=eps(Float64)) ||
            throw(ArgumentError("duration_hr must equal end_idx - onset_idx for hourly rows"))
        push!(catalog, StormCatalogEntry(
            storm_id,
            onset_time,
            min_dst_star,
            min_dst_star_time,
            recovery_end_time,
            duration_hr,
            solar_cycle,
            split,
            onset_idx,
            end_idx,
        ))
    end
    ids = getfield.(catalog, :storm_id)
    length(unique(ids)) == length(ids) ||
        throw(ArgumentError("storm catalog contains duplicate storm ids"))
    return catalog
end
