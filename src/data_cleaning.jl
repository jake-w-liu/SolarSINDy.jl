# Data cleaning, storm catalog, and real-data preparation

using Dates

"""
    clean_omni_data!(df::DataFrame)

In-place cleaning of raw OMNI2 DataFrame:
1. Remove rows with missing critical variables (V, Bz, Dst)
2. Interpolate short gaps (≤3 hours) in secondary variables
3. Compute derived quantities: Bs, θ_c, Pdyn (if missing), Dst*
4. Add quality flag column

Returns the modified DataFrame.
"""
function clean_omni_data!(df::DataFrame)
    n_raw = nrow(df)

    # --- Compute Pdyn from n and V where Pdyn is missing but n, V available ---
    for i in 1:nrow(df)
        if isnan(df.Pdyn[i]) && !isnan(df.n[i]) && !isnan(df.V[i])
            df.Pdyn[i] = 1.6726e-6 * df.n[i] * df.V[i]^2
        end
    end

    # --- Interpolate short gaps (≤3 consecutive NaN) for each column ---
    for col in [:V, :Bz, :By, :n, :Pdyn, :T, :Dst, :AE, :AL, :AU]
        _interp_short_gaps!(df[!, col], 3)
    end

    # --- Quality flag: 1 = all critical vars present, 0 = any critical missing ---
    df[!, :quality] = ones(Int, nrow(df))
    for i in 1:nrow(df)
        if isnan(df.V[i]) || isnan(df.Bz[i]) || isnan(df.Dst[i])
            df.quality[i] = 0
        end
    end

    # --- Derived quantities ---
    df[!, :Bs] = [isnan(bz) ? NaN : max(-bz, 0.0) for bz in df.Bz]
    df[!, :theta_c] = [isnan(by) || isnan(bz) ? NaN : atan(abs(by), bz) for (by, bz) in zip(df.By, df.Bz)]
    df[!, :BT] = [isnan(by) || isnan(bz) ? NaN : sqrt(by^2 + bz^2) for (by, bz) in zip(df.By, df.Bz)]

    # Dst*: use pressure correction where Pdyn available, raw Dst otherwise
    df[!, :Dst_star] = Vector{Float64}(undef, nrow(df))
    for i in 1:nrow(df)
        if isnan(df.Dst[i])
            df.Dst_star[i] = NaN
        elseif isnan(df.Pdyn[i])
            df.Dst_star[i] = df.Dst[i]  # fallback: uncorrected Dst
        else
            df.Dst_star[i] = df.Dst[i] - 7.26 * sqrt(max(df.Pdyn[i], 0.0)) + 11.0
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
    min_dst::Float64
    min_dst_time::DateTime
    recovery_end_time::DateTime
    duration_hr::Float64
    solar_cycle::Int
    split::String              # "train", "val", "test"
    onset_idx::Int             # index into cleaned DataFrame
    end_idx::Int
end

"""
    _solar_cycle(year)

Approximate solar cycle number from year.
Cycles: 20 (1964–1976), 21 (1976–1986), 22 (1986–1996),
        23 (1996–2008), 24 (2008–2019), 25 (2019–present)
"""
function _solar_cycle(year::Int)
    year < 1976 && return 20
    year < 1986 && return 21
    year < 1996 && return 22
    year < 2008 && return 23
    year < 2019 && return 24
    return 25
end

"""
    _assign_split(cycle)

Assign train/val/test split by solar cycle:
  Train: cycles 20–23 (~1964–2008)
  Val:   cycle 24 (2008–2019)
  Test:  cycle 25 (2019–present, includes May 2024 superstorm)
"""
function _assign_split(cycle::Int)
    cycle <= 23 && return "train"
    cycle == 24 && return "val"
    return "test"
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
    catalog = StormCatalogEntry[]
    n = nrow(df)

    # For storm detection, only need Dst_star (Dst + Pdyn).
    # Full quality (V, Bz) checked separately for SINDy usability.
    valid_mask = .!isnan.(df.Dst_star)
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
        min_dst = df.Dst_star[onset_idx]
        min_idx = onset_idx
        for k in onset_idx:search_end
            if valid_mask[k] && df.Dst_star[k] < min_dst
                min_dst = df.Dst_star[k]
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
        year = Dates.year(df.datetime[onset_idx])
        cycle = _solar_cycle(year)
        duration = (win_end - win_start)  # hours (hourly data)

        entry = StormCatalogEntry(
            storm_id,
            df.datetime[onset_idx],
            min_dst,
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
    println("  Storm catalog: $(length(catalog)) storms (train=$(n_train), val=$(n_val), test=$(n_test))")
    if !isempty(catalog)
        println("  Dst range: $(round(minimum(e.min_dst for e in catalog), digits=1)) to $(round(maximum(e.min_dst for e in catalog), digits=1)) nT")
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
    df = DataFrame(
        storm_id = [e.storm_id for e in catalog],
        onset_time = [e.onset_time for e in catalog],
        min_dst = [e.min_dst for e in catalog],
        min_dst_time = [e.min_dst_time for e in catalog],
        recovery_end_time = [e.recovery_end_time for e in catalog],
        duration_hr = [e.duration_hr for e in catalog],
        solar_cycle = [e.solar_cycle for e in catalog],
        split = [e.split for e in catalog],
        onset_idx = [e.onset_idx for e in catalog],
        end_idx = [e.end_idx for e in catalog]
    )
    CSV.write(filepath, df)
    println("  Saved storm catalog: $filepath ($(nrow(df)) storms)")
end

"""
    load_storm_catalog(filepath)

Load storm catalog from CSV.
"""
function load_storm_catalog(filepath::String)
    df = CSV.read(filepath, DataFrame)
    catalog = StormCatalogEntry[]
    for row in eachrow(df)
        push!(catalog, StormCatalogEntry(
            row.storm_id,
            DateTime(row.onset_time),
            row.min_dst,
            DateTime(row.min_dst_time),
            DateTime(row.recovery_end_time),
            row.duration_hr,
            row.solar_cycle,
            row.split,
            row.onset_idx,
            row.end_idx
        ))
    end
    return catalog
end
