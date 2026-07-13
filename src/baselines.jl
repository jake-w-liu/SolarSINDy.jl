# Baseline coupling function models

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
    # dDst*/dt = -α * V * Bs - Dst* / τ
    # α in nT/(km/s · nT) per hour, τ in hours
    return -α .* V .* Bs .- Dst_star ./ τ
end

"""
    simulate_burton(V, Bs, dt; α=5.4e-3, τ=7.7, Dst0=0.0)

Forward Euler simulation of the threshold-free Burton model
(see `burton_model`). dt in hours. Returns Dst* time series.
"""
function simulate_burton(V::AbstractVector, Bs::AbstractVector, dt::Real;
                         α::Real=5.4e-3, τ::Real=7.7, Dst0::Real=0.0)
    n = length(V)
    Dst_star = zeros(n)
    Dst_star[1] = Dst0
    for k in 1:n-1
        dDdt = -α * V[k] * Bs[k] - Dst_star[k] / τ
        dDdt = clamp(dDdt, -200.0, 200.0)
        Dst_star[k+1] = clamp(Dst_star[k] + dDdt * dt, -2000.0, 50.0)
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
    return V.^(4/3) .* max.(BT, 1e-10).^(2/3) .* sin.(theta_c ./ 2).^(8/3)
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
    Ec = V .* Bs ./ 1000.0  # mV/m
    Q = zeros(length(Ec))
    τ = zeros(length(Ec))
    for k in eachindex(Ec)
        if Ec[k] > Ec_crit
            Q[k] = -α * (Ec[k] - Ec_crit)
        end
        τ[k] = 2.40 * exp(9.74 / (4.69 + Ec[k]))
    end
    return Q .- Dst_star ./ τ
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
    Q = [(V[k] * Bs[k]) > vbs_crit ? -α * (V[k] * Bs[k] - vbs_crit) : 0.0
         for k in eachindex(V)]
    return Q .- Dst_star ./ τ
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
    n = length(V)
    Dst_star = zeros(n)
    Dst_star[1] = Dst0
    for k in 1:n-1
        vbs = V[k] * Bs[k]
        Q = vbs > vbs_crit ? -α * (vbs - vbs_crit) : 0.0
        dDdt = Q - Dst_star[k] / τ
        dDdt = clamp(dDdt, -200.0, 200.0)
        Dst_star[k+1] = clamp(Dst_star[k] + dDdt * dt, -2000.0, 50.0)
    end
    return Dst_star
end

"""
    simulate_obrien(V, Bs, dt; α=4.4, Ec_crit=0.49, Dst0=0.0)

Forward Euler simulation of O'Brien-McPherron model.
"""
function simulate_obrien(V::AbstractVector, Bs::AbstractVector, dt::Real;
                         α::Real=4.4, Ec_crit::Real=0.49, Dst0::Real=0.0)
    n = length(V)
    Dst_star = zeros(n)
    Dst_star[1] = Dst0
    for k in 1:n-1
        Ec = V[k] * Bs[k] / 1000.0
        Q = Ec > Ec_crit ? -α * (Ec - Ec_crit) : 0.0
        τ = 2.40 * exp(9.74 / (4.69 + Ec))
        dDdt = Q - Dst_star[k] / τ
        dDdt = clamp(dDdt, -200.0, 200.0)
        Dst_star[k+1] = clamp(Dst_star[k] + dDdt * dt, -2000.0, 50.0)
    end
    return Dst_star
end
