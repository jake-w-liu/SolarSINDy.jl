# geoelectric.jl — geoelectric field (the actual GIC driver) from ground magnetic variations,
# via the plane-wave (magnetotelluric) method with a 1-D uniform half-space.
#
# Frequency domain (Viljanen/Pulkkinen plane-wave method):
#   surface impedance of a uniform half-space:  Z(w) = sqrt(i w mu0 rho)     [rho = 1/sigma]
#   E_x(w) =  Z(w) B_y(w) / mu0 ,   E_y(w) = -Z(w) B_x(w) / mu0
# The geoelectric field tracks dB/dt (high-pass character), leading B by 45 deg.
#
# Dependency-free: a small O(N^2) DFT/IDFT is used (windows are ~120 points, so this is
# instant). E is returned in V/km (1 V/m = 1000 V/km). This is a 1-D-conductivity ESTIMATE,
# not a calibrated geoelectric measurement; real GIC also depends on 3-D ground + grid topology.

using Statistics

const MU0 = 4e-7 * pi

function _dft(x::AbstractVector{<:Real})
    N = length(x); X = zeros(ComplexF64, N)
    @inbounds for k in 0:N-1
        s = 0.0 + 0.0im
        for n in 0:N-1
            s += x[n+1] * cis(-2pi * k * n / N)
        end
        X[k+1] = s
    end
    return X
end

function _idft(X::AbstractVector{ComplexF64})
    N = length(X); x = zeros(ComplexF64, N)
    @inbounds for n in 0:N-1
        s = 0.0 + 0.0im
        for k in 0:N-1
            s += X[k+1] * cis(2pi * k * n / N)
        end
        x[n+1] = s / N
    end
    return x
end

"""
    geoelectric_field(Bx_nt, By_nt, dt_s; rho_ohm_m=1000) -> (Ex, Ey)  [V/km]

Plane-wave geoelectric field for a uniform half-space of resistivity `rho_ohm_m`, from
horizontal magnetic field components [nT] sampled every `dt_s` seconds. The series mean is
removed (the method acts on variations). Hermitian symmetry of Z is enforced so E is real.
"""
function geoelectric_field(Bx_nt::AbstractVector{<:Real}, By_nt::AbstractVector{<:Real},
                           dt_s::Real; rho_ohm_m::Real = 1000.0)
    N = length(Bx_nt)
    bx = (Bx_nt .- mean(Bx_nt)) .* 1e-9          # nT -> T, detrended
    by = (By_nt .- mean(By_nt)) .* 1e-9
    BX = _dft(bx); BY = _dft(by)
    EX = zeros(ComplexF64, N); EY = zeros(ComplexF64, N)
    @inbounds for k in 0:N-1
        f = (k <= N / 2 ? k : k - N) / (N * dt_s)   # signed frequency [Hz]
        w = 2pi * f
        # Principal sqrt of i*w*mu0*rho is already Hermitian in w (Z(-w) = conj(Z(w)) since
        # i*w is positive/negative imaginary, never on the negative real branch cut), so the
        # reconstructed E is real automatically — do NOT manually conjugate (that double-flips
        # the phase, killing the 45-deg lead and a factor sqrt(2) of amplitude).
        Z = w == 0 ? (0.0 + 0.0im) : sqrt(im * w * MU0 * rho_ohm_m)
        EX[k+1] =  Z * BY[k+1] / MU0
        EY[k+1] = -Z * BX[k+1] / MU0
    end
    ex = real.(_idft(EX)) .* 1000.0                  # V/m -> V/km
    ey = real.(_idft(EY)) .* 1000.0
    return ex, ey
end

# Approximate geoelectric GIC-risk tiers [V/km]. Coarse — real risk depends on grid topology;
# for reference, the 1989 Québec collapse involved fields of order ~1–2 V/km in resistive ground.
const GEO_TIERS = ("Quiet", "Low", "Moderate", "High", "Severe")
const GEO_EDGES = (0.1, 0.5, 2.0, 5.0)
function geo_tier(e)
    (e === nothing || !(e isa Real) || !isfinite(e)) && return (level = nothing, label = "—")
    e < GEO_EDGES[1] && return (level = 0, label = GEO_TIERS[1])
    e < GEO_EDGES[2] && return (level = 1, label = GEO_TIERS[2])
    e < GEO_EDGES[3] && return (level = 2, label = GEO_TIERS[3])
    e < GEO_EDGES[4] && return (level = 3, label = GEO_TIERS[4])
    return (level = 4, label = GEO_TIERS[5])
end
