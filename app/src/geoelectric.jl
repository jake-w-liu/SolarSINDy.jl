# geoelectric.jl — geoelectric field from ground magnetic variations via the plane-wave
# (magnetotelluric) method. Local geoelectric field supplies the electric forcing for GIC, but
# current in a particular grid also depends on site conductivity and network geometry. Supports
# a 1-D uniform half-space (default) and a 1-D LAYERED earth (Wait recursion for the impedance).
#
# Frequency domain (Viljanen/Pulkkinen plane-wave method):
#   uniform half-space surface impedance:  Z(w) = sqrt(i w mu0 rho)          [rho = 1/sigma]
#   layered earth surface impedance:       Wait recursion over (rho_n, d_n) — see surface_impedance
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
    surface_impedance(w, rho, thick) -> ComplexF64

Plane-wave surface impedance ``Z(\\omega)`` of a 1-D LAYERED earth via the Wait recursion
(transmission-line form). `rho` are the layer resistivities [Ω·m] from the surface down
(`N` layers); `thick` are the thicknesses [m] of the top `N-1` layers (the bottom layer is a
half-space, so `length(thick) == length(rho) - 1`). A single layer recovers the uniform
half-space ``Z = \\sqrt{i\\,\\omega\\,\\mu_0\\,\\rho}``.

Limiting behavior (used as correctness oracles): identical layers leave `Z` unchanged; a thick
top layer → its own half-space impedance; a thin top layer → the impedance below it; high
frequency → top-layer resistivity, low frequency → bottom-layer resistivity.
"""
function surface_impedance(w::Real, rho::AbstractVector{<:Real}, thick::AbstractVector{<:Real})
    isfinite(w) || throw(ArgumentError("surface_impedance: frequency must be finite"))
    isempty(rho) && throw(ArgumentError("surface_impedance: at least one layer is required"))
    length(thick) == length(rho) - 1 ||
        throw(ArgumentError("surface_impedance: thickness count must equal resistivity count minus one"))
    all(r -> isfinite(r) && r > 0, rho) ||
        throw(ArgumentError("surface_impedance: resistivities must be finite and positive"))
    all(d -> isfinite(d) && d >= 0, thick) ||
        throw(ArgumentError("surface_impedance: thicknesses must be finite and nonnegative"))
    w == 0 && return 0.0 + 0.0im
    N = length(rho)
    Z = sqrt(im * w * MU0 * rho[N])                  # bottom half-space impedance
    @inbounds for n in (N-1):-1:1
        Z0 = sqrt(im * w * MU0 * rho[n])             # intrinsic impedance of layer n
        kn = sqrt(im * w * MU0 / rho[n])             # propagation constant (Re kn > 0 ⇒ decay)
        t  = tanh(kn * thick[n])
        Z  = Z0 * (Z + Z0 * t) / (Z0 + Z * t)        # layer n looking into the stack below it
    end
    return Z
end

"""
    geoelectric_field(Bx_nt, By_nt, dt_s; rho_ohm_m=1000, layers=nothing) -> (Ex, Ey)  [V/km]

Plane-wave geoelectric field from horizontal magnetic field components [nT] sampled every
`dt_s` seconds. The series mean is removed (the method acts on variations). Hermitian symmetry
of `Z` makes `E` real automatically.

Ground model: by default a uniform half-space of resistivity `rho_ohm_m`. Pass `layers` as a
vector of `(rho_ohm_m, thickness_m)` from the surface down (the bottom entry's thickness is
ignored — it is the terminating half-space) to use a 1-D LAYERED earth instead.
"""
function geoelectric_field(Bx_nt::AbstractVector{<:Real}, By_nt::AbstractVector{<:Real},
                           dt_s::Real; rho_ohm_m::Real = 1000.0, layers = nothing)
    isfinite(dt_s) && dt_s > 0 ||
        throw(ArgumentError("geoelectric_field: dt_s must be finite and positive"))
    rho, thick = layers === nothing ?
        (Float64[float(rho_ohm_m)], Float64[]) :
        (Float64[float(l[1]) for l in layers],
         Float64[float(layers[i][2]) for i in 1:length(layers)-1])
    # Bx and By must share a length: N drives both the loop and the per-frequency BX[k]/BY[k]
    # access under @inbounds below, so a mismatch would read past the shorter spectrum and
    # return silent garbage instead of failing.
    length(Bx_nt) == length(By_nt) ||
        throw(ArgumentError("geoelectric_field: Bx and By must have equal length"))
    N = length(Bx_nt)
    N >= 1 || throw(ArgumentError("geoelectric_field: input series must not be empty"))
    bx_values = Float64.(Bx_nt)
    by_values = Float64.(By_nt)
    all(isfinite, bx_values) && all(isfinite, by_values) ||
        throw(ArgumentError("geoelectric_field: magnetic inputs must be finite"))
    bx_mean = mean(bx_values)
    by_mean = mean(by_values)
    isfinite(bx_mean) && isfinite(by_mean) ||
        throw(ArgumentError("geoelectric_field: magnetic-input mean overflowed"))
    bx = (bx_values .- bx_mean) .* 1e-9          # nT -> T, detrended
    by = (by_values .- by_mean) .* 1e-9
    BX = _dft(bx); BY = _dft(by)
    EX = zeros(ComplexF64, N); EY = zeros(ComplexF64, N)
    @inbounds for k in 0:N-1
        f = (k <= N / 2 ? k : k - N) / (N * dt_s)   # signed frequency [Hz]
        w = 2pi * f
        # Principal sqrt of i*w*mu0*rho is already Hermitian in w (Z(-w) = conj(Z(w))), so the
        # reconstructed E is real automatically — do NOT manually conjugate (that double-flips
        # the phase, killing the 45-deg lead and a factor sqrt(2) of amplitude).
        Z = surface_impedance(w, rho, thick)
        EX[k+1] =  Z * BY[k+1] / MU0
        EY[k+1] = -Z * BX[k+1] / MU0
    end
    ex = real.(_idft(EX)) .* 1000.0                  # V/m -> V/km
    ey = real.(_idft(EY)) .* 1000.0
    return ex, ey
end

"""
    causal_halfspace_efield(Bx_nt, By_nt, dt_s; rho_ohm_m=1000) -> (Ex, Ey)  [V/km]

Time-domain causal geoelectric field for a uniform half-space — the exact time-domain
equivalent of the plane-wave surface impedance ``Z(\\omega)=\\sqrt{i\\,\\omega\\,\\mu_0\\,\\rho}``:

    E_x(t) =  sqrt(rho/(pi*mu0)) ∫_{t0}^{t} (dB_y/dt') (t - t')^{-1/2} dt'
    E_y(t) = -sqrt(rho/(pi*mu0)) ∫_{t0}^{t} (dB_x/dt') (t - t')^{-1/2} dt'

Unlike a windowed circular DFT of a finite record, this retains the sustained-ramp
(low-frequency) response: a constant `dB/dt` produces a growing ``\\sqrt{t-t_0}`` field instead of
being discarded as a trend. `dB/dt` is taken piecewise-constant on each sample interval and the
``(t-t')^{-1/2}`` kernel is integrated analytically per interval, so a constant ramp is reproduced
exactly. The convolution is truncated at the first sample (`t0`), so the earliest samples are
under-resolved; the value at the most recent sample integrates the full record and is the
physically meaningful causal nowcast. Uniform sampling `dt_s` [s] is required. The returned pair
uses the same component pairing and sign convention as [`geoelectric_field`](@ref).

For a purely periodic input this agrees with the frequency-domain [`geoelectric_field`] method in
its steady-state region; it is preferred for a real-time nowcast because it is causal, leakage-free,
and keeps the storm-time ramp response that a detrended DFT window removes.
"""
function causal_halfspace_efield(Bx_nt::AbstractVector{<:Real}, By_nt::AbstractVector{<:Real},
                                 dt_s::Real; rho_ohm_m::Real = 1000.0)
    isfinite(dt_s) && dt_s > 0 ||
        throw(ArgumentError("causal_halfspace_efield: dt_s must be finite and positive"))
    isfinite(rho_ohm_m) && rho_ohm_m > 0 ||
        throw(ArgumentError("causal_halfspace_efield: rho_ohm_m must be finite and positive"))
    length(Bx_nt) == length(By_nt) ||
        throw(ArgumentError("causal_halfspace_efield: Bx and By must have equal length"))
    N = length(Bx_nt)
    N >= 1 || throw(ArgumentError("causal_halfspace_efield: input series must not be empty"))
    bx = Float64.(Bx_nt); by = Float64.(By_nt)
    all(isfinite, bx) && all(isfinite, by) ||
        throw(ArgumentError("causal_halfspace_efield: magnetic inputs must be finite"))
    # C folds nT->T (1e-9), V/m->V/km (1e3), the 2*sqrt(dt_s) from the analytic ∫(t-t')^{-1/2}dt'
    # over one interval, and the 1/dt_s from a piecewise-constant dB/dt: 2*sqrt(dt_s)/dt_s = 2/sqrt(dt_s).
    C = sqrt(rho_ohm_m / (pi * MU0)) * 1e-6 * 2.0 / sqrt(dt_s)
    a = Vector{Float64}(undef, N)                    # kernel a_L = sqrt(L) - sqrt(L-1), decreasing in L
    @inbounds for L in 1:N
        a[L] = sqrt(float(L)) - sqrt(float(L - 1))
    end
    ex = zeros(N); ey = zeros(N)
    @inbounds for i in 2:N
        sx = 0.0; sy = 0.0
        for m in 1:(i - 1)
            L = i - m
            sx += (by[m + 1] - by[m]) * a[L]         # E_x from ∂B_y
            sy += (bx[m + 1] - bx[m]) * a[L]         # E_y from ∂B_x
        end
        ex[i] = C * sx
        ey[i] = -C * sy
    end
    return ex, ey
end

# Illustrative 1-D earth models (resistivity Ω·m, thickness m; bottom entry = half-space).
# Real GIC requires a site-specific ground model; these are coarse end-members for demonstration.
const EARTH_RESISTIVE   = [(2.0e4, 1.5e4), (5.0e2, 0.0)]   # resistive shield over a deep conductor
const EARTH_CONDUCTIVE  = [(2.0e1, 4.0e3), (1.0e3, 0.0)]   # conductive sediments over resistive basement
