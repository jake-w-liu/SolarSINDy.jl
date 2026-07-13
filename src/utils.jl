# Utility functions for numerical operations

"""
    numerical_derivative(x, dt)

Central finite difference derivative with forward/backward at boundaries.
"""
function numerical_derivative(x::AbstractVector, dt::Real)
    n = length(x)
    n >= 2 || throw(ArgumentError("numerical_derivative requires length ≥ 2, got $n"))
    dx = similar(x)
    # Forward difference at start
    dx[1] = (x[2] - x[1]) / dt
    # Central differences
    for k in 2:n-1
        dx[k] = (x[k+1] - x[k-1]) / (2dt)
    end
    # Backward difference at end
    dx[n] = (x[n] - x[n-1]) / dt
    return dx
end

"""
    smooth_moving_average(x, window)

Simple moving average smoothing with centered window. Window must be odd.
"""
function smooth_moving_average(x::AbstractVector, window::Int)
    @assert isodd(window) "Window must be odd"
    # A window wider than the series collapses every output to the global mean,
    # silently destroying the signal; reject it rather than returning a constant.
    length(x) >= window ||
        throw(ArgumentError("smoothing window ($window) exceeds series length ($(length(x)))"))
    n = length(x)
    half = div(window, 2)
    xs = similar(x)
    for k in 1:n
        lo = max(1, k - half)
        hi = min(n, k + half)
        xs[k] = mean(@view x[lo:hi])
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
dynamic_pressure(n::Real, V::Real) =
    (isnan(n) || isnan(V)) ? NaN : PROTON_PDYN_COEFF * n * V^2

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
    isfinite(pdyn) && return Float64(pdyn)
    (isfinite(last_pdyn) && age <= max_age) && return Float64(last_pdyn)
    return Float64(quiet)
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
    return max.(-bz, 0.0)
end

"""
    imf_clock_angle(by, bz)

IMF clock angle θ_c = atan(|By|, Bz) in radians.
"""
function imf_clock_angle(by::AbstractVector, bz::AbstractVector)
    return atan.(abs.(by), bz)
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
