# Utility functions for numerical operations

"""
    numerical_derivative(x, dt)

Central finite difference derivative with forward/backward at boundaries.
"""
function numerical_derivative(x::AbstractVector, dt::Real)
    n = length(x)
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

"""
    pressure_correct_dst(dst, pdyn; b=7.26, c=11.0)

Pressure-correct Dst index: Dst* = Dst - b√Pdyn + c
Burton et al. (1975) correction.
"""
function pressure_correct_dst(dst::AbstractVector, pdyn::AbstractVector;
                               b::Real=7.26, c::Real=11.0)
    return dst .- b .* sqrt.(max.(pdyn, 0.0)) .+ c
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

When installed via package manager, users can download data separately.
When running from source (cloned repo), uses the local `data` directory.

# Returns
- `::String`: Absolute path to the data directory
"""
function get_data_dir()::String
    # Check for local data/ directory first (for development/cloned repos)
    local_data = joinpath(@__DIR__, "..", "data")
    if isdir(local_data)
        return local_data
    end
    
    # Could check for artifact here in future, but for now fallback only
    error("Data directory not found. Clone the repository to access example data: " *
          "https://github.com/user/SolarSINDy.jl")
end
