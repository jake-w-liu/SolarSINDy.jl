# Baseline coupling function models

function _validate_baseline_vectors(name::AbstractString, vectors::AbstractVector...)
    isempty(vectors) && throw(ArgumentError("$name requires at least one input vector"))
    n = length(first(vectors))
    n >= 1 || throw(ArgumentError("$name requires at least one sample"))
    all(length(v) == n for v in vectors) ||
        throw(DimensionMismatch("$name input vectors must have equal length"))
    all(v -> all(x -> x isa Real && isfinite(x), v), vectors) ||
        throw(ArgumentError("$name inputs must contain only finite real values"))
    for vector in vectors, value in vector
        converted = Float64(value)
        isfinite(converted) || throw(ArgumentError(
            "$name input exceeds the supported Float64 range",
        ))
    end
    return n
end

function _validate_nonnegative_baseline_drivers(name::AbstractString,
                                                V::AbstractVector,
                                                Bs::AbstractVector)
    all(>=(0), V) || throw(ArgumentError("$name speed must be nonnegative"))
    all(>=(0), Bs) || throw(ArgumentError("$name southward IMF must be nonnegative"))
    return nothing
end

function _validate_baseline_parameters(name::AbstractString, dt::Real, α::Real, τ::Real)
    isfinite(dt) && dt > 0 || throw(ArgumentError("$name dt must be finite and positive, got $dt"))
    isfinite(α) && α >= 0 || throw(ArgumentError("$name α must be finite and nonnegative, got $α"))
    isfinite(τ) && τ > 0 || throw(ArgumentError("$name τ must be finite and positive, got $τ"))
    converted = Float64.((dt, α, τ))
    all(isfinite, converted) ||
        throw(ArgumentError("$name parameters exceed the supported Float64 range"))
    return converted
end

function _finite_baseline_derivative(value::Real, name::AbstractString)
    isfinite(value) || throw(ArgumentError("$name derivative became non-finite"))
    return Float64(value)
end

"""
    burton_model(V, Bs, Dst_star; α=5.4e-3, τ=7.7)

Threshold-free simplification of the Burton et al. (1975) ring-current model:

  dDst*/dt = -α·V·Bs - Dst*/τ

where `Dst* = Dst - 7.26√Pdyn + 11` is the pressure-corrected index (the
pressure-correction constants 7.26 nT·nPa^(-1/2) and 11 nT are from
O'Brien & McPherron (2000), not Burton 1975) and `V·Bs` is the rectified
solar-wind driving (V in km/s, Bs in nT).

This variant drops Burton's injection threshold (Ey > 0.5 mV/m); the published,
threshold-continuous injection is in `burton_model_full`.

Parameters:
- `α = 5.4e-3` nT/hr per (km/s·nT): Burton's injection slope
  `d = 1.5e-3` nT/s per mV/m = 5.4 nT/h per mV/m, expressed in V·Bs units.
- `τ = 7.7` hr: ring-current decay timescale (Burton 1975 `a = 3.6e-5 s⁻¹`),
  so the decay rate is `1/τ ≈ 0.13` hr⁻¹.

`Dst*` is supplied directly (already pressure-corrected); `Pdyn` is not an
argument here.
"""
function burton_model(V::AbstractVector, Bs::AbstractVector,
                      Dst_star::AbstractVector;
                      α::Real=5.4e-3, τ::Real=7.7)
    _validate_baseline_vectors("burton_model", V, Bs, Dst_star)
    _validate_nonnegative_baseline_drivers("burton_model", V, Bs)
    _, α64, τ64 = _validate_baseline_parameters("burton_model", 1.0, α, τ)
    # dDst*/dt = -α * V * Bs - Dst* / τ
    # α in nT/(km/s · nT) per hour, τ in hours
    return map(V, Bs, Dst_star) do speed, southward, dst
        derivative = -α64 * Float64(speed) * Float64(southward) - Float64(dst) / τ64
        _finite_baseline_derivative(derivative, "burton_model")
    end
end

"""
    simulate_burton(V, Bs, dt; α=5.4e-3, τ=7.7, Dst0=0.0)

Forward Euler simulation of the threshold-free Burton model
(see `burton_model`). dt in hours. Returns Dst* time series.
"""
function simulate_burton(V::AbstractVector, Bs::AbstractVector, dt::Real;
                         α::Real=5.4e-3, τ::Real=7.7, Dst0::Real=0.0)
    n = _validate_baseline_vectors("simulate_burton", V, Bs)
    _validate_nonnegative_baseline_drivers("simulate_burton", V, Bs)
    dt64, α64, τ64 = _validate_baseline_parameters("simulate_burton", dt, α, τ)
    isfinite(Dst0) || throw(ArgumentError("simulate_burton Dst0 must be finite, got $Dst0"))
    dst0 = Float64(Dst0)
    isfinite(dst0) || throw(ArgumentError("simulate_burton Dst0 exceeds the supported Float64 range"))
    Dst_star = zeros(n)
    Dst_star[1] = dst0
    for k in 1:n-1
        dDdt = -α64 * Float64(V[k]) * Float64(Bs[k]) - Dst_star[k] / τ64
        dDdt = _finite_baseline_derivative(dDdt, "simulate_burton")
        dDdt = clamp(dDdt, -200.0, 200.0)
        Dst_star[k+1] = clamp(Dst_star[k] + dDdt * dt64, -2000.0, 50.0)
    end
    return Dst_star
end

"""
    newell_coupling(V, BT, theta_c)

Newell et al. (2007) universal coupling function:
  dΦ/dt = V^(4/3) · B_T^(2/3) · sin^(8/3)(θ_c/2)

where B_T = √(By² + Bz²), θ_c = atan(|By|, Bz).
Returns coupling strength (proxy for energy input).
"""
function newell_coupling(V::AbstractVector, BT::AbstractVector,
                         theta_c::AbstractVector)
    _validate_baseline_vectors("newell_coupling", V, BT, theta_c)
    all(>=(0), V) || throw(ArgumentError("newell_coupling speed must be nonnegative"))
    all(>=(0), BT) || throw(ArgumentError("newell_coupling transverse IMF must be nonnegative"))
    all(θ -> 0 <= θ <= π, theta_c) ||
        throw(ArgumentError("newell_coupling clock angles must lie in [0, π]"))
    return map(V, BT, theta_c) do speed, transverse, angle
        _newell_coupling_value(speed, transverse, sin(Float64(angle) / 2))
    end
end

"""
    obrien_mcpherron_model(V, Bs, Dst_star; α=4.4, Ec_crit=0.49)

O'Brien & McPherron (2000) model with energy-dependent injection and decay:
  dDst*/dt = Q(Ec) - Dst*/τ(Ec)
  Q(Ec) = -α·(Ec - Ec_crit)   for Ec > Ec_crit mV/m, else 0
  τ(Ec) = 2.40·exp(9.74 / (4.69 + Ec))  hours

where Ec = V·Bs / 1000 (convection electric field in mV/m).
"""
function obrien_mcpherron_model(V::AbstractVector, Bs::AbstractVector,
                                Dst_star::AbstractVector;
                                α::Real=4.4, Ec_crit::Real=0.49)
    _validate_baseline_vectors("obrien_mcpherron_model", V, Bs, Dst_star)
    _validate_nonnegative_baseline_drivers("obrien_mcpherron_model", V, Bs)
    isfinite(α) && α >= 0 ||
        throw(ArgumentError("obrien_mcpherron_model α must be finite and nonnegative"))
    isfinite(Ec_crit) && Ec_crit >= 0 ||
        throw(ArgumentError("obrien_mcpherron_model Ec_crit must be finite and nonnegative"))
    α64 = Float64(α)
    threshold = Float64(Ec_crit)
    isfinite(α64) && isfinite(threshold) || throw(ArgumentError(
        "obrien_mcpherron_model parameters exceed the supported Float64 range",
    ))
    return map(V, Bs, Dst_star) do speed, southward, dst
        speed64 = Float64(speed)
        southward64 = Float64(southward)
        product = speed64 * southward64
        # Preserve the ordinary-range evaluation exactly; divide first only when
        # V*Bs overflows but the scaled electric field remains representable.
        Ec = isfinite(product) ? product / 1000.0 : speed64 * (southward64 / 1000.0)
        isfinite(Ec) || throw(ArgumentError(
            "obrien_mcpherron_model electric field became non-finite",
        ))
        Q = Ec > threshold ? -α64 * (Ec - threshold) : 0.0
        τ = 2.40 * exp(9.74 / (4.69 + Ec))
        _finite_baseline_derivative(Q - Float64(dst) / τ,
                                    "obrien_mcpherron_model")
    end
end

"""
    burton_model_full(V, Bs, Dst_star; α=5.4e-3, τ=7.7, vbs_crit=500.0)

Published Burton et al. (1975) Dst model with the threshold-continuous
injection F(Ey) = d·(Ey − 0.5) for Ey > 0.5 mV/m, else 0:

  Ey = V·Bs / 1000 [mV/m]
  Q  = V·Bs > 500 ? -α·(V·Bs − 500) : 0.0     (= -d·(Ey − 0.5), d = 5.4 nT/h per mV/m)
  dDst*/dt = Q - Dst*/τ

The `−500` offset (= 1000·0.5 mV/m in V·Bs units) makes the injection continuous
at the threshold; omitting it (the earlier `-α·V·Bs` form) introduced a spurious
~2.3 nT/h step at Ey = 0.5.
"""
function burton_model_full(V::AbstractVector, Bs::AbstractVector,
                           Dst_star::AbstractVector;
                           α::Real=5.4e-3, τ::Real=7.7, vbs_crit::Real=500.0)
    _validate_baseline_vectors("burton_model_full", V, Bs, Dst_star)
    _validate_nonnegative_baseline_drivers("burton_model_full", V, Bs)
    _, α64, τ64 = _validate_baseline_parameters("burton_model_full", 1.0, α, τ)
    isfinite(vbs_crit) && vbs_crit >= 0 ||
        throw(ArgumentError("burton_model_full vbs_crit must be finite and nonnegative"))
    threshold = Float64(vbs_crit)
    isfinite(threshold) || throw(ArgumentError(
        "burton_model_full vbs_crit exceeds the supported Float64 range",
    ))
    return map(V, Bs, Dst_star) do speed, southward, dst
        speed64 = Float64(speed)
        southward64 = Float64(southward)
        vbs = speed64 * southward64
        Q = if isfinite(vbs)
            vbs > threshold ? -α64 * (vbs - threshold) : 0.0
        elseif southward64 > 0 && speed64 > threshold / southward64
            # The raw product exceeds Float64, but the injection can remain
            # representable because α ≪ 1. Reorder only on this overflow path.
            -α64 * speed64 * southward64 + α64 * threshold
        else
            0.0
        end
        _finite_baseline_derivative(Q - Float64(dst) / τ64, "burton_model_full")
    end
end

"""
    simulate_burton_full(V, Bs, dt; α=5.4e-3, τ=7.7, vbs_crit=500.0, Dst0=0.0)

Forward Euler simulation of the published Burton model with the
threshold-continuous injection (Ey ≤ 0.5 mV/m → Q = 0; above threshold the
injection subtracts the threshold, F = d·(Ey − 0.5)).
"""
function simulate_burton_full(V::AbstractVector, Bs::AbstractVector, dt::Real;
                              α::Real=5.4e-3, τ::Real=7.7, vbs_crit::Real=500.0,
                              Dst0::Real=0.0)
    n = _validate_baseline_vectors("simulate_burton_full", V, Bs)
    _validate_nonnegative_baseline_drivers("simulate_burton_full", V, Bs)
    dt64, α64, τ64 = _validate_baseline_parameters("simulate_burton_full", dt, α, τ)
    isfinite(vbs_crit) && vbs_crit >= 0 ||
        throw(ArgumentError("simulate_burton_full vbs_crit must be finite and nonnegative"))
    isfinite(Dst0) || throw(ArgumentError("simulate_burton_full Dst0 must be finite"))
    threshold = Float64(vbs_crit)
    dst0 = Float64(Dst0)
    isfinite(threshold) && isfinite(dst0) || throw(ArgumentError(
        "simulate_burton_full parameters exceed the supported Float64 range",
    ))
    Dst_star = zeros(n)
    Dst_star[1] = dst0
    for k in 1:n-1
        speed = Float64(V[k])
        southward = Float64(Bs[k])
        vbs = speed * southward
        Q = if isfinite(vbs)
            vbs > threshold ? -α64 * (vbs - threshold) : 0.0
        elseif southward > 0 && speed > threshold / southward
            -α64 * speed * southward + α64 * threshold
        else
            0.0
        end
        dDdt = _finite_baseline_derivative(Q - Dst_star[k] / τ64,
                                           "simulate_burton_full")
        dDdt = clamp(dDdt, -200.0, 200.0)
        Dst_star[k+1] = clamp(Dst_star[k] + dDdt * dt64, -2000.0, 50.0)
    end
    return Dst_star
end

"""
    simulate_obrien(V, Bs, dt; α=4.4, Ec_crit=0.49, Dst0=0.0)

Forward Euler simulation of O'Brien-McPherron model.
"""
function simulate_obrien(V::AbstractVector, Bs::AbstractVector, dt::Real;
                         α::Real=4.4, Ec_crit::Real=0.49, Dst0::Real=0.0)
    n = _validate_baseline_vectors("simulate_obrien", V, Bs)
    _validate_nonnegative_baseline_drivers("simulate_obrien", V, Bs)
    isfinite(dt) && dt > 0 || throw(ArgumentError("simulate_obrien dt must be finite and positive"))
    isfinite(α) && α >= 0 || throw(ArgumentError("simulate_obrien α must be finite and nonnegative"))
    isfinite(Ec_crit) && Ec_crit >= 0 ||
        throw(ArgumentError("simulate_obrien Ec_crit must be finite and nonnegative"))
    isfinite(Dst0) || throw(ArgumentError("simulate_obrien Dst0 must be finite"))
    dt64 = Float64(dt)
    α64 = Float64(α)
    threshold = Float64(Ec_crit)
    dst0 = Float64(Dst0)
    all(isfinite, (dt64, α64, threshold, dst0)) || throw(ArgumentError(
        "simulate_obrien parameters exceed the supported Float64 range",
    ))
    Dst_star = zeros(n)
    Dst_star[1] = dst0
    for k in 1:n-1
        speed = Float64(V[k])
        southward = Float64(Bs[k])
        product = speed * southward
        Ec = isfinite(product) ? product / 1000.0 : speed * (southward / 1000.0)
        isfinite(Ec) || throw(ArgumentError("simulate_obrien electric field became non-finite"))
        Q = Ec > threshold ? -α64 * (Ec - threshold) : 0.0
        τ = 2.40 * exp(9.74 / (4.69 + Ec))
        dDdt = _finite_baseline_derivative(Q - Dst_star[k] / τ, "simulate_obrien")
        dDdt = clamp(dDdt, -200.0, 200.0)
        Dst_star[k+1] = clamp(Dst_star[k] + dDdt * dt64, -2000.0, 50.0)
    end
    return Dst_star
end
