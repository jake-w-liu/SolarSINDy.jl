# Data structures and synthetic data generation

"""
    SolarWindData

Container for time-aligned solar wind and geomagnetic data.
All times in hours from epoch.
"""
struct SolarWindData
    t::Vector{Float64}       # Time [hours]
    V::Vector{Float64}       # Solar wind velocity [km/s]
    Bz::Vector{Float64}      # IMF Bz [nT]
    By::Vector{Float64}      # IMF By [nT]
    n::Vector{Float64}       # Proton density [cm⁻³]
    Pdyn::Vector{Float64}    # Dynamic pressure [nPa]
    Dst::Vector{Float64}     # Dst index [nT]
    Dst_star::Vector{Float64}# Pressure-corrected Dst [nT]
end

"""
    StormEvent

Labeled storm event with phase boundaries.
"""
struct StormEvent
    data::SolarWindData
    onset_idx::Int           # Storm onset index
    main_end_idx::Int        # End of main phase
    recovery_end_idx::Int    # End of recovery phase
    min_dst_star::Float64    # Minimum pressure-corrected Dst* reached
end

# Compatibility for exploratory callers written before the minimum was named
# explicitly. Synthetic events have always stored the minimum of the generated
# Dst* trajectory, not the pressure-uncorrected Dst series.
function Base.getproperty(event::StormEvent, name::Symbol)
    name === :min_dst && return getfield(event, :min_dst_star)
    return getfield(event, name)
end

function Base.propertynames(::StormEvent, private::Bool=false)
    return (fieldnames(StormEvent)..., :min_dst)
end

"""
    generate_synthetic_storm(; seed=42, dt=1.0, duration=120.0,
                              noise_level=0.05, α=5.4e-3, τ=7.7)

Generate a synthetic geomagnetic storm event using the Burton model.
The solar wind drivers are prescribed analytically, Dst* is computed by
forward integration, and Dst is reconstructed with the declared pressure
correction.

Returns (SolarWindData, StormEvent).
"""
function generate_synthetic_storm(; seed::Int=42, dt::Real=1.0,
                                    duration::Real=120.0,
                                    noise_level::Real=0.05,
                                    α::Real=5.4e-3, τ::Real=7.7)
    seed >= 0 || throw(ArgumentError("seed must be nonnegative"))
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive"))
    isfinite(duration) && duration >= dt ||
        throw(ArgumentError("duration must be finite and at least one dt"))
    isfinite(noise_level) && noise_level >= 0 ||
        throw(ArgumentError("noise_level must be finite and nonnegative"))
    isfinite(α) && α >= 0 || throw(ArgumentError("α must be finite and nonnegative"))
    isfinite(τ) && τ > 0 || throw(ArgumentError("τ must be finite and positive"))
    step_ratio = Float64(duration) / Float64(dt)
    n_steps = round(Int, step_ratio)
    isapprox(step_ratio, n_steps; rtol=0.0,
             atol=sqrt(eps(Float64)) * max(1.0, abs(step_ratio))) ||
        throw(ArgumentError("duration must be an integer multiple of dt"))
    rng = MersenneTwister(seed)
    n_pts = n_steps + 1
    t = Float64(dt) .* collect(0:n_steps)

    # --- Prescribe solar wind drivers ---
    # Quiet → sudden onset → main phase → recovery

    # Velocity: step-up at onset, gradual decay
    V_quiet = 400.0  # km/s
    V_storm = 600.0  # km/s
    t_onset = 24.0   # hours
    t_main_end = 48.0
    V = V_quiet .+ (V_storm - V_quiet) .* (0.5 .* (1.0 .+ tanh.((t .- t_onset) ./ 2.0))) .*
        (0.5 .* (1.0 .+ tanh.((t_main_end .+ 24.0 .- t) ./ 6.0)))

    # IMF Bz: southward turning during main phase
    Bz_quiet = 2.0   # nT (slightly northward)
    Bz_storm = -15.0  # nT (strongly southward)
    Bz = Bz_quiet .+ (Bz_storm - Bz_quiet) .*
         exp.(-0.5 .* ((t .- 36.0) ./ 8.0).^2)  # Gaussian southward pulse

    # IMF By: fluctuations
    By = 3.0 .* sin.(2π .* t ./ 12.0) .+ randn(rng, n_pts) .* 1.0

    # Density: enhancement during storm
    n_density = 5.0 .+ 10.0 .* exp.(-0.5 .* ((t .- 30.0) ./ 6.0).^2) .+
                randn(rng, n_pts) .* 0.5

    # Dynamic pressure: Pdyn = n * m_p * V² ≈ 1.6726e-6 * n * V² [nPa]
    # (m_p = 1.6726e-27 kg; consistent with real data pipeline in data_cleaning.jl)
    Pdyn = 1.6726e-6 .* n_density .* V.^2

    # --- Compute Bs and derived quantities ---
    Bs = halfwave_rectify(Bz)

    # --- Forward integrate Burton model ---
    Dst_star = simulate_burton(V, Bs, dt; α=α, τ=τ, Dst0=0.0)

    # Add measurement noise
    Dst_star_noisy = Dst_star .+ noise_level .* std(Dst_star) .* randn(rng, n_pts)

    # Reverse pressure correction to get observed Dst
    Dst_obs = Dst_star_noisy .+ 7.26 .* sqrt.(max.(Pdyn, 0.0)) .- 11.0

    # --- Identify storm phases ---
    min_dst_val, min_idx = findmin(Dst_star)
    onset_idx = findfirst(Dst_star .< -10.0)
    onset_idx = onset_idx === nothing ? 1 : onset_idx
    recovery_end_idx = min(n_pts, min_idx + round(Int, 48.0 / dt))

    swd = SolarWindData(t, V, Bz, By, n_density, Pdyn, Dst_obs, Dst_star_noisy)
    event = StormEvent(swd, onset_idx, min_idx, recovery_end_idx, min_dst_val)
    return swd, event
end

"""
    generate_multistorm_dataset(; n_storms=5, seed=42)

Generate multiple synthetic storms with varying intensities.

**Note:** This function varies the ground-truth Burton parameters (α, τ) across storms for
synthetic diversity. The canonical single-storm generator uses the published Burton
injection slope (α=5.4e-3) and decay time (τ=7.7 hr).
This function is provided for exploratory use only and was
NOT used to generate any published figures or tables.
"""
function generate_multistorm_dataset(; n_storms::Int=5, seed::Int=42)
    n_storms >= 1 || throw(ArgumentError("n_storms must be at least 1"))
    seed >= 0 || throw(ArgumentError("seed must be nonnegative"))
    seed <= typemax(Int) - n_storms || throw(ArgumentError("seed range overflows Int"))
    storms = StormEvent[]
    datasets = SolarWindData[]

    # Vary storm intensity via injection efficiency
    α_range = range(3e-3, 6e-3, length=n_storms)
    τ_range = range(6.0, 9.0, length=n_storms)

    for k in 1:n_storms
        swd, event = generate_synthetic_storm(
            seed=seed + k, dt=1.0, duration=120.0,
            α=α_range[k], τ=τ_range[k], noise_level=0.05
        )
        push!(datasets, swd)
        push!(storms, event)
    end

    return datasets, storms
end

"""
    prepare_sindy_data(swd::SolarWindData, dt::Real; smooth_window=5)

Prepare data dictionary for SINDy from SolarWindData.
Computes numerical derivative of Dst* and all library inputs.
"""
function prepare_sindy_data(swd::SolarWindData, dt::Real; smooth_window::Int=5)
    n = length(swd.t)
    all(length(v) == n for v in
        (swd.V, swd.Bz, swd.By, swd.n, swd.Pdyn, swd.Dst, swd.Dst_star)) ||
        throw(DimensionMismatch("SolarWindData fields must have equal length"))
    # Smooth Dst* before differentiation
    Dst_smooth = smooth_moving_average(swd.Dst_star, smooth_window)
    dDst_dt = numerical_derivative(Dst_smooth, dt)

    Bs = halfwave_rectify(swd.Bz)
    theta_c = imf_clock_angle(swd.By, swd.Bz)
    BT = hypot.(swd.By, swd.Bz)

    data = Dict{String,Vector{Float64}}(
        "V" => swd.V,
        "Bz" => swd.Bz,
        "By" => swd.By,
        "Bs" => Bs,
        "n" => swd.n,
        "Pdyn" => swd.Pdyn,
        "Dst_star" => Dst_smooth,
        "theta_c" => theta_c,
        "BT" => BT
    )

    return data, dDst_dt
end

"""
    identify_storm_phases(Dst_star, dDst_dt; quiet_thresh=-20.0,
                          deriv_thresh=-2.0)

Classify each time step into storm phases:
1 = quiet, 2 = main phase, 3 = recovery
"""
function identify_storm_phases(Dst_star::AbstractVector,
                               dDst_dt::AbstractVector;
                               quiet_thresh::Real=-20.0,
                               deriv_thresh::Real=-2.0)
    length(Dst_star) == length(dDst_dt) ||
        throw(DimensionMismatch("Dst_star and dDst_dt must have equal length"))
    all(isfinite, Dst_star) && all(isfinite, dDst_dt) ||
        throw(ArgumentError("phase inputs must contain only finite values"))
    isfinite(quiet_thresh) && isfinite(deriv_thresh) ||
        throw(ArgumentError("phase thresholds must be finite"))
    n = length(Dst_star)
    phases = ones(Int, n)  # default: quiet
    for k in 1:n
        if Dst_star[k] < quiet_thresh && dDst_dt[k] < deriv_thresh
            phases[k] = 2  # main phase: Dst dropping
        elseif Dst_star[k] < quiet_thresh && dDst_dt[k] >= deriv_thresh
            phases[k] = 3  # recovery: Dst below threshold but recovering
        end
    end
    return phases
end
