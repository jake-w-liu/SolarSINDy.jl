# Tests for the operational app. Run with the research environment (has Test + JSON3 + HTTP):
#   julia --startup-file=no --project=../SolarSINDy.jl runtests.jl
# or from the project root:
#   julia --startup-file=no --project=SolarSINDy.jl app/test/runtests.jl
#
# The headline guarantee these tests protect is the forecaster<->export contract: the live
# dB/dt forecast must equal the paper3 conformal model recomputed independently from the
# exported JSON. A drift in feature order, standardization, scale, or quantile is caught here.

using Test, JSON3, Statistics, HTTP

const APPSRC = normpath(joinpath(@__DIR__, "..", "src"))
# server.jl transitively includes forecaster.jl, dbdt.jl, notify.jl, network.jl, ... exactly
# once each (including them again here would redefine `const _FORECASTER`). It does not
# auto-start the HTTP server on include.
include(joinpath(APPSRC, "server.jl"))

# ---- independent re-implementation of export_forecaster.jl's documented formula ----
# forecast = expm1(zhat + q * s(x)); P(y>thr) = mean(rn > (log1p(thr)-zhat)/s); cap at log(2001).
function golden_forecast(model, dbdt_recent::Vector{Float64}, V, Bz)
    w = dbdt_recent[max(1, end-29):end]
    dnow, dmean, dmax = w[end], mean(w), maximum(w)
    dstd = length(w) > 1 ? std(w) : 0.0
    Bs = max(-Bz, 0.0); VBs = V * Bs
    x = Float64[dnow, dmean, dmax, dstd, V, Bz, Bs, VBs]
    μ = Float64.(model.mu); σ = Float64.(model.sigma); β = Float64.(model.beta)
    ẑ = β[1] + sum(β[2:end] .* ((x .- μ) ./ σ))
    s = log1p(dmax) + 1.0
    rn = Float64.(model.rn_calib); n = length(rn)
    qexact = quantile(rn, 0.90)                       # exact 0.90 quantile of the residual grid
    zcap = log(2001.0)
    point = expm1(min(ẑ, zcap))
    ub90 = expm1(min(ẑ + qexact * s, zcap))
    exc = [(Int(thr), count(>((log1p(thr) - ẑ) / s), rn) / n) for thr in model.thresholds]
    return point, ub90, exc, ẑ, s
end

@testset verbose=true "operational app" begin

    @testset "forecaster <-> export golden-vector contract (FRD + CMO)" begin
        # a realistic active-but-not-extreme trailing dB/dt history
        recent = collect(range(2.0, 14.0; length=30)) .+ 0.0
        for (station, V, Bz) in (("FRD", 520.0, -8.0), ("CMO", 600.0, -12.0))
            m = load_forecaster(; station=station)
            @test m !== nothing
            fc = forecast_dbdt(recent, V, Bz; station=station)
            @test fc !== nothing
            gp, gub, gexc, ẑ, s = golden_forecast(m, recent, V, Bz)
            # live forecast must match the independently recomputed export formula
            @test isapprox(fc.point_dbdt, round(max(gp, 0.0); digits=2); atol=0.02)
            @test isapprox(fc.ub90_dbdt, round(gub; digits=2); atol=0.05)
            @test length(fc.exceedance) == length(gexc)
            for (k, e) in enumerate(fc.exceedance)
                @test e.threshold == gexc[k][1]
                @test isapprox(e.prob, round(gexc[k][2]; digits=3); atol=0.01)
            end
            @test fc.station == station
            @test fc.horizon_min == 30
            # interval sanity: upper bound is above the point forecast
            @test fc.ub90_dbdt >= fc.point_dbdt
        end
    end

    @testset "physical regimes: quiet < storm, auroral >= mid-latitude, no blow-up" begin
        quiet = fill(0.5, 30)
        storm = collect(range(20.0, 120.0; length=30))
        q = forecast_dbdt(quiet, 380.0, 1.0; station="FRD")
        s = forecast_dbdt(storm, 700.0, -20.0; station="FRD")
        @test q.point_dbdt < s.point_dbdt
        @test q.exceedance[1].prob <= s.exceedance[1].prob     # P(>18) rises with the storm
        # same drivers: auroral (CMO) bound >= mid-latitude (FRD) bound
        af = forecast_dbdt(storm, 700.0, -20.0; station="CMO")
        @test af.ub90_dbdt >= s.ub90_dbdt
        # out-of-range input must not overflow: physical cap ~2000 nT/min
        blow = fill(5000.0, 30)
        b = forecast_dbdt(blow, 1200.0, -60.0; station="CMO")
        @test isfinite(b.point_dbdt) && b.point_dbdt <= 2001.0
        @test isfinite(b.ub90_dbdt) && b.ub90_dbdt <= 2001.0
    end

    @testset "per-station cache returns the right model" begin
        # loading CMO then FRD must not return CMO's model for FRD (the prior Ref-vs-Dict bug)
        cmo = load_forecaster(; station="CMO")
        frd = load_forecaster(; station="FRD")
        @test cmo.station == "CMO"
        @test frd.station == "FRD"
        @test Float64.(cmo.beta) != Float64.(frd.beta)
    end

    @testset "static file serving is traversal-guarded" begin
        ok = serve_static("/index.html")
        @test ok.status == 200
        # Any path containing ".." is rejected with 403 before path resolution; an escape must
        # never be served (status 200). Both 403 (forbidden) and 404 (not found) mean "blocked".
        for esc in ("/../server.jl", "/../../etc/passwd", "/../src/forecaster.jl",
                    "/..%2f..%2fProject.toml", "/./../../README.md")
            r = serve_static(esc)
            @test r.status == 403            # ".." paths are forbidden outright
            @test r.status != 200            # the security invariant: never served
        end
        @test serve_static("/does-not-exist.html").status == 404   # no "..": just missing
    end

    @testset "exported model self-consistency (coverage proxy)" begin
        # the residual grid's exact 0.90 quantile is what the live forecaster targets;
        # the grid must be sorted/monotone and the 0.90 quantile finite and positive-ish.
        for station in ("FRD", "CMO")
            m = load_forecaster(; station=station)
            rn = Float64.(m.rn_calib)
            @test issorted(rn)
            @test isfinite(quantile(rn, 0.90))
            @test haskey(m, :cap_note)              # cap convention documented in the artifact
        end
    end

    @testset "geoelectric: layered-earth surface impedance (Wait recursion)" begin
        mu0 = 4e-7 * pi; w = 2pi * 1e-3
        ha(rho) = sqrt(im * w * mu0 * rho)                  # uniform half-space impedance
        rho_app(Z, ww) = abs2(Z) / (ww * mu0)               # MT apparent resistivity
        @test surface_impedance(w, [100.0], Float64[]) ≈ ha(100.0)                 # single = half-space
        @test surface_impedance(w, [100.0, 100.0], [5e3]) ≈ ha(100.0)              # identical interface invisible
        @test isapprox(surface_impedance(w, [100.0, 1.0], [1e7]), ha(100.0); rtol=1e-6)  # thick top → top
        @test isapprox(surface_impedance(w, [100.0, 5.0], [1.0]), ha(5.0); rtol=1e-3)    # thin top → below
        @test abs(rho_app(surface_impedance(2pi*1.0,  [10.0,1000.0], [5e3]), 2pi*1.0)  - 10.0)/10.0   < 0.25  # high f → top ρ
        @test abs(rho_app(surface_impedance(2pi*1e-6, [10.0,1000.0], [5e3]), 2pi*1e-6) - 1000.0)/1000.0 < 0.25  # low f → bottom ρ
        Bx = 50.0 .* sin.(2pi .* (1:120) ./ 30); By = 30.0 .* cos.(2pi .* (1:120) ./ 30)
        exL, eyL = geoelectric_field(Bx, By, 60.0; layers=EARTH_RESISTIVE)
        @test all(isfinite, exL) && all(isfinite, eyL) && maximum(abs, exL) > 0
        exU, _ = geoelectric_field(Bx, By, 60.0; rho_ohm_m=100.0)
        @test maximum(abs, exL) > maximum(abs, exU)         # resistive ground → larger geoelectric field
    end

    @testset "Phase D: storm-replay endpoint payload" begin
        dir = mktempdir()
        log_path = joinpath(dir, "forecast_log.csv")        # build_storm_replay reads siblings of this
        # No report yet -> available=false, never throws.
        r0 = build_storm_replay(log_path)
        @test r0.available == false

        write(joinpath(dir, "storm_replay_report.md"), "# Storm-time replay\n\nbody\n")
        write(joinpath(dir, "storm_replay_scored.csv"),
              "model_step_hours,storm\n1,\"May 2024 (Gannon, G5)\"\n3,\"May 2024 (Gannon, G5)\"\n1,\"Oct 2024\"\n")
        r = build_storm_replay(log_path)
        @test r.available == true
        @test r.n_scored == 3
        @test Set(r.storms) == Set(["May 2024 (Gannon, G5)", "Oct 2024"])
        @test occursin("Storm-time replay", r.report_markdown)
        @test r.report_age_min isa Real
    end

    @testset "dB/dt forecaster flags out-of-validated-range / saturated inputs" begin
        # A merely quiet-to-mild dB/dt history forecasts within the validated range.
        normal = forecast_dbdt(fill(2.0, 30), 420.0, -3.0; station="FRD")
        if normal !== nothing                      # only if the FRD artifact is present
            @test haskey(normal, :reliable)
            @test normal.reliable == true
            @test normal.saturated == false
            # An absurd dB/dt history (far outside the ~1.7 nT/min calibration mean) must be
            # flagged unreliable/saturated rather than surfaced as a confident Extreme forecast.
            extreme = forecast_dbdt(fill(5000.0, 30), 1200.0, -80.0; station="FRD")
            @test extreme !== nothing
            @test extreme.reliable == false
            @test (extreme.out_of_validated_range || extreme.saturated)
        end
    end
end
