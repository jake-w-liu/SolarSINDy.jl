# Utility functions for numerical operations

"""
    numerical_derivative(x, dt)

Central finite difference derivative with forward/backward at boundaries.
"""
function numerical_derivative(x::AbstractVector{<:Real}, dt::Real)
    n = length(x)
    n >= 2 || throw(ArgumentError("numerical_derivative requires length ≥ 2, got $n"))
    isfinite(dt) && dt > 0 ||
        throw(ArgumentError("numerical_derivative requires finite dt > 0, got $dt"))
    all(value -> !isinf(value), x) ||
        throw(ArgumentError("numerical_derivative does not accept infinite samples"))
    dx = similar(x, float(eltype(x)))
    dt_float = float(dt)
    function difference(a, b, divisor_factor)
        (isnan(a) || isnan(b)) && return convert(eltype(dx), NaN)
        denominator = divisor_factor * dt_float
        value = (float(a) - float(b)) / denominator
        if !isfinite(denominator) || !isfinite(value) || (iszero(value) && a != b)
            value = (BigFloat(a) - BigFloat(b)) /
                    (BigFloat(divisor_factor) * BigFloat(dt_float))
        end
        converted = convert(eltype(dx), value)
        isfinite(converted) || throw(ArgumentError(
            "numerical_derivative result exceeded the output range",
        ))
        iszero(converted) && !iszero(value) && throw(ArgumentError(
            "numerical_derivative result underflowed the output range",
        ))
        return converted
    end
    # Forward difference at start
    dx[1] = difference(x[2], x[1], 1)
    # Central differences
    for k in 2:n-1
        dx[k] = difference(x[k+1], x[k-1], 2)
    end
    # Backward difference at end
    dx[n] = difference(x[n], x[n-1], 1)
    return dx
end

function _newell_coupling_value(speed::Real, transverse::Real, sin_half::Real)
    speed_float = Float64(speed)
    transverse_float = Float64(transverse)
    sin_float = Float64(sin_half)
    all(isfinite, (speed_float, transverse_float, sin_float)) ||
        throw(ArgumentError("Newell coupling inputs must be finite"))
    speed_float >= 0 && transverse_float >= 0 && sin_float >= 0 ||
        throw(ArgumentError("Newell coupling inputs must be nonnegative"))
    (iszero(speed_float) || iszero(transverse_float) || iszero(sin_float)) &&
        return 0.0

    value = speed_float^(4 / 3) * transverse_float^(2 / 3) * sin_float^(8 / 3)
    isfinite(value) && !iszero(value) && return value

    wide = BigFloat(speed_float)^(BigFloat(4) / 3) *
           BigFloat(transverse_float)^(BigFloat(2) / 3) *
           BigFloat(sin_float)^(BigFloat(8) / 3)
    converted = Float64(wide)
    isfinite(converted) && !iszero(converted) || throw(ArgumentError(
        iszero(converted) ? "Newell coupling underflowed the supported range" :
                            "Newell coupling exceeded the supported range",
    ))
    return converted
end

"""
    smooth_moving_average(x, window)

Simple moving average smoothing with centered window. Window must be odd.
"""
function smooth_moving_average(x::AbstractVector{<:Real}, window::Int)
    window > 0 && isodd(window) ||
        throw(ArgumentError("smoothing window must be a positive odd integer, got $window"))
    # A window wider than the series collapses every output to the global mean,
    # silently destroying the signal; reject it rather than returning a constant.
    length(x) >= window ||
        throw(ArgumentError("smoothing window ($window) exceeds series length ($(length(x)))"))
    n = length(x)
    half = div(window, 2)
    xs = similar(x, float(eltype(x)))
    for k in 1:n
        lo = max(1, k - half)
        hi = min(n, k + half)
        window_values = @view x[lo:hi]
        direct = mean(window_values)
        if isfinite(direct) || any(isnan, window_values)
            xs[k] = direct
            continue
        end
        all(isfinite, window_values) || throw(ArgumentError(
            "smoothing window contains an infinite value",
        ))
        scale = maximum(abs, window_values)
        scaled_mean = iszero(scale) ? zero(direct) :
            scale * (sum(value -> value / scale, window_values) / length(window_values))
        isfinite(scaled_mean) || throw(ArgumentError(
            "smoothing mean exceeded the supported range",
        ))
        xs[k] = scaled_mean
    end
    return xs
end

# ============================================================
# Canonical Dst* / dynamic-pressure conventions (single source of truth).
# Every training, replay, and serving path must reuse these helpers so the
# pressure-correction convention is identical at fit and serve time.
# ============================================================

"""
Proton-only solar-wind dynamic pressure coefficient. With the proton mass
`m_p = 1.6726e-27` kg, density `n` in cm⁻³ (=1e6 m⁻³) and speed `V` in km/s
(=1e3 m/s),

    Pdyn = m_p·n·V² = 1.6726e-27·(1e6)·(1e3)²·n·V² Pa = 1.6726e-6·n·V² nPa.

Real-time SWPC feeds carry no alpha-particle abundance, so proton-only is the
only convention that can be served live; training uses the same form for
train/serve parity (the OMNI word-29 alpha-inclusive pressure is not used).
"""
const PROTON_PDYN_COEFF = 1.6726e-6

"""
Pressure-correction constants for `Dst* = Dst - b√Pdyn + c`. Values
`b = 7.26` nT·nPa^(-1/2) and `c = 11.0` nT are from O'Brien & McPherron (2000);
Burton et al. (1975) used the different pair `b = 0.20`, `c = 20` nT.
"""
const DST_STAR_B = 7.26
const DST_STAR_C = 11.0

"""
Fallback dynamic pressure when the plasma feed is unavailable. `QUIET_PDYN_NPA`
is the climatological quiet-time solar-wind dynamic pressure; a stale finite
Pdyn is carried forward for up to `PDYN_CARRY_MAX_AGE_H` hours before reverting
to the quiet default. This replaces the physically impossible `Pdyn = 0`
(flat `Dst + 11`) fallback.
"""
const QUIET_PDYN_NPA = 2.0
const PDYN_CARRY_MAX_AGE_H = 6

"""
    dynamic_pressure(n, V)

Proton-only dynamic pressure [nPa] from density `n` [cm⁻³] and speed `V` [km/s].
Returns `NaN` if either input is `NaN`.
"""
function dynamic_pressure(n::Real, V::Real)
    (!isfinite(n) || !isfinite(V) || n < 0 || V < 0) && return NaN
    # Promote fixed-width integers before the square so `typemax(Int)^2` cannot
    # wrap to a small positive pressure. Keep the ordinary-range evaluation order;
    # only use the factored fallback when the square itself overflows.
    density = float(n)
    speed = float(V)
    iszero(density) && return zero(PROTON_PDYN_COEFF * density * speed)
    speed_squared = speed * speed
    pressure = PROTON_PDYN_COEFF * density * speed_squared
    if isfinite(speed_squared) && isfinite(pressure) &&
       (!iszero(pressure) || iszero(speed))
        return pressure
    end

    # An overflowing square can still have a finite product when density is
    # tiny; conversely, multiplying the tiny density first can underflow before
    # the two speed factors restore the scale. Evaluate only this exceptional
    # path with enough exponent/mantissa range to decide whether the Float result
    # is representable. The ordinary path above remains byte-for-byte unchanged.
    work_precision = max(
        256,
        density isa BigFloat ? precision(density) : 0,
        speed isa BigFloat ? precision(speed) : 0,
    )
    wide_pressure = setprecision(BigFloat, work_precision) do
        BigFloat(PROTON_PDYN_COEFF) * BigFloat(density) * BigFloat(speed)^2
    end
    result_type = promote_type(Float64, typeof(density), typeof(speed))
    converted = convert(result_type, wide_pressure)
    if !isfinite(converted) || (!iszero(wide_pressure) && iszero(converted))
        return NaN
    end
    return converted
end

"""
    dst_to_dst_star(dst, pdyn; b=DST_STAR_B, c=DST_STAR_C)

Pressure-corrected ring-current index `Dst* = Dst - b√max(Pdyn,0) + c` (scalar).
"""
dst_to_dst_star(dst::Real, pdyn::Real; b::Real=DST_STAR_B, c::Real=DST_STAR_C) =
    dst - b * sqrt(max(pdyn, 0.0)) + c

"""
    dst_star_to_dst(dst_star, pdyn; b=DST_STAR_B, c=DST_STAR_C)

Inverse of [`dst_to_dst_star`](@ref): recover `Dst` from `Dst*` and the
(target-time) dynamic pressure `Pdyn` (scalar).
"""
dst_star_to_dst(dst_star::Real, pdyn::Real; b::Real=DST_STAR_B, c::Real=DST_STAR_C) =
    dst_star + b * sqrt(max(pdyn, 0.0)) - c

"""
    resolve_pdyn(pdyn, last_pdyn, age; max_age=PDYN_CARRY_MAX_AGE_H, quiet=QUIET_PDYN_NPA)

Physically-defensible dynamic pressure for a single row/bin. Returns `pdyn`
when finite; otherwise carries `last_pdyn` forward when it is finite and no
older than `max_age` hours; otherwise returns the quiet-time default `quiet`.
"""
function resolve_pdyn(pdyn::Real, last_pdyn::Real, age::Integer;
                      max_age::Integer=PDYN_CARRY_MAX_AGE_H, quiet::Real=QUIET_PDYN_NPA)
    age >= 0 || throw(ArgumentError("dynamic-pressure age must be nonnegative, got $age"))
    max_age >= 0 || throw(ArgumentError("max_age must be nonnegative, got $max_age"))
    pdyn_float = Float64(pdyn)
    last_float = Float64(last_pdyn)
    quiet_float = Float64(quiet)
    isfinite(pdyn) && !isfinite(pdyn_float) && throw(ArgumentError(
        "dynamic pressure exceeds the supported Float64 range",
    ))
    isfinite(last_pdyn) && !isfinite(last_float) && throw(ArgumentError(
        "carried dynamic pressure exceeds the supported Float64 range",
    ))
    isfinite(quiet_float) && quiet_float >= 0 ||
        throw(ArgumentError("quiet fallback pressure must be finite and nonnegative, got $quiet"))
    (isfinite(pdyn_float) && pdyn_float >= 0) && return pdyn_float
    (isfinite(last_float) && last_float >= 0 && age <= max_age) && return last_float
    return quiet_float
end

"""
    pressure_correct_dst(dst, pdyn; b=DST_STAR_B, c=DST_STAR_C)

Pressure-correct Dst index: `Dst* = Dst - b√Pdyn + c` (vector form of
[`dst_to_dst_star`](@ref)). Constants `b = 7.26`, `c = 11.0` are from
O'Brien & McPherron (2000).
"""
function pressure_correct_dst(dst::AbstractVector, pdyn::AbstractVector;
                               b::Real=DST_STAR_B, c::Real=DST_STAR_C)
    return dst_to_dst_star.(dst, pdyn; b=b, c=c)
end

"""
    halfwave_rectify(bz)

Southward IMF component: Bs = max(-Bz, 0) following Burton convention.
"""
function halfwave_rectify(bz::AbstractVector)
    # `-typemin(Int)` wraps in fixed-width integer arithmetic. Promote first.
    return max.(-float.(bz), 0.0)
end

"""
    imf_clock_angle(by, bz)

IMF clock angle θ_c = atan(|By|, Bz) in radians.
"""
function imf_clock_angle(by::AbstractVector, bz::AbstractVector)
    # `abs(typemin(Int))` wraps negative and would produce a negative clock angle.
    return atan.(abs.(float.(by)), float.(bz))
end

"""
    get_data_dir()

Return the path to the SolarSINDy data directory.

The `data/` directory is bundled with the package and available both when
running from a cloned repo and when installed via `Pkg.add`.

# Returns
- `::String`: Absolute path to the data directory
"""
function get_data_dir()::String
    data_dir = joinpath(@__DIR__, "..", "data")
    isdir(data_dir) || error("Data directory not found at $data_dir")
    return data_dir
end
