# Baseline coupling function models

"""
    burton_model(V, Bs, Pdyn, Dst; a=-4.4, b=3.6e-5, c=-3.6e-2, τ=7.7)

Burton et al. (1975) Dst model:
  dDst*/dt = a - b·V·Bs - c·Dst*

where Dst* = Dst - 7.26√Pdyn + 11 (pressure-corrected).

Parameters (original):
- a ≈ -4.4 nT/hr (quiet-time decay offset — often set to 0 for simplicity)
- b ≈ 3.6e-5 nT/(km/s · nT · hr) (injection efficiency, converted units)
- c ≈ 1/τ ≈ 0.13/hr (decay rate, τ ≈ 7.7 hr)

Note: We use the simplified form: dDst*/dt = -α·V·Bs - Dst*/τ
with α capturing injection and τ capturing recovery.
"""
function burton_model(V::AbstractVector, Bs::AbstractVector,
                      Dst_star::AbstractVector;
                      α::Real=4.559e-3, τ::Real=7.7)
    # dDst*/dt = -α * V * Bs - Dst* / τ
    # α in nT/(km/s · nT) per hour, τ in hours
    return -α .* V .* Bs .- Dst_star ./ τ
end

"""
    simulate_burton(V, Bs, dt; α=4.559e-3, τ=7.7, Dst0=0.0)

Forward Euler simulation of Burton model.
dt in hours. Returns Dst* time series.
"""
function simulate_burton(V::AbstractVector, Bs::AbstractVector, dt::Real;
                         α::Real=4.559e-3, τ::Real=7.7, Dst0::Real=0.0)
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
    burton_model_full(V, Bs, Dst_star; α=4.559e-3, τ=7.7)

Full Burton et al. (1975) Dst model with injection suppression threshold:
  Ey = V·Bs / 1000 [mV/m]
  Q = Ey > 0.5 ? -α·V·Bs : 0.0
  dDst*/dt = Q - Dst*/τ
"""
function burton_model_full(V::AbstractVector, Bs::AbstractVector,
                           Dst_star::AbstractVector;
                           α::Real=4.559e-3, τ::Real=7.7)
    Ey = V .* Bs ./ 1000.0  # mV/m
    Q = [Ey[k] > 0.5 ? -α * V[k] * Bs[k] : 0.0 for k in eachindex(V)]
    return Q .- Dst_star ./ τ
end

"""
    simulate_burton_full(V, Bs, dt; α=4.559e-3, τ=7.7, Dst0=0.0)

Forward Euler simulation of the full Burton model with injection suppression
threshold (Ey ≤ 0.5 mV/m → Q = 0).
"""
function simulate_burton_full(V::AbstractVector, Bs::AbstractVector, dt::Real;
                              α::Real=4.559e-3, τ::Real=7.7, Dst0::Real=0.0)
    n = length(V)
    Dst_star = zeros(n)
    Dst_star[1] = Dst0
    for k in 1:n-1
        Ey = V[k] * Bs[k] / 1000.0  # mV/m
        Q = Ey > 0.5 ? -α * V[k] * Bs[k] : 0.0
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
