# Tests for the operational app. Run with the research environment (has Test + JSON3 + HTTP):
#   julia --startup-file=no --project=../SolarSINDy.jl runtests.jl
# or from the project root:
#   julia --startup-file=no --project=SolarSINDy.jl app/test/runtests.jl
#
# The headline dB/dt contract is fail-closed serving plus offline replay/export equivalence:
# the API withholds the time-reference-mismatched forecast, while an explicitly requested replay
# must equal the exported ridge + historical-residual calculation. Feature, scale, and quantile
# drift is caught here.

using Test, JSON3, Statistics, HTTP

const APPSRC = normpath(joinpath(@__DIR__, "..", "src"))
# server.jl transitively includes forecaster.jl, dbdt.jl, notify.jl, network.jl, ... exactly
# once each (including them again here would redefine `const _FORECASTER`). It does not
# auto-start the HTTP server on include.
include(joinpath(APPSRC, "server.jl"))

struct _InterruptingSWPCText end
Base.String(::_InterruptingSWPCText) = throw(InterruptException())

# ---- independent re-implementation of export_forecaster.jl's documented formula ----
# forecast = expm1(zhat + q*s(x)); empirical score = mean(rn > cutoff); cap at log(2001).
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

function live_cycle_fixture(issue::DateTime;
                            vintage=issue, anchor_time=issue - Hour(1), latest_dst=-20.0,
                            model="v2.1", served_model="v2.1+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia",
                            interval="aci", observations=missing,
                            served_pred=-25.0, served_lo=-35.0, served_hi=-15.0,
                            audit_pred=served_pred, audit_lo=served_lo, audit_hi=served_hi)
    requested = collect(LIVE_CYCLE_HORIZONS)
    targets = floor(issue, Hour) .+ Hour.(requested)
    lead = [(target - issue) / Millisecond(3_600_000) for target in targets]
    expand(x) = x isa AbstractVector ? collect(x) : fill(x, length(requested))
    return DataFrame(
        issue_time_utc_dt=fill(issue, length(requested)),
        latest_solar_wind_utc_dt=fill(vintage, length(requested)),
        latest_dst_time_utc_dt=fill(anchor_time, length(requested)),
        target_time_utc_dt=targets,
        horizon_hours=lead,
        latest_dst_nt=fill(latest_dst, length(requested)),
        observation_dst_nt=expand(observations),
        served_pred_dst_nt=expand(served_pred),
        served_pred_dst_ci05_nt=expand(served_lo),
        served_pred_dst_ci95_nt=expand(served_hi),
        v2_pred_dst_nt=expand(audit_pred),
        v2_pred_dst_ci05_nt=expand(audit_lo),
        v2_pred_dst_ci95_nt=expand(audit_hi),
        interval_source=fill(interval, length(requested)),
        model_version=fill(model, length(requested)),
        sub_hourly_model_version=fill(served_model, length(requested)),
    )
end

@testset verbose=true "operational app" begin

    @testset "forecaster <-> export golden-vector contract (FRD + CMO)" begin
        # a realistic active-but-not-extreme trailing dB/dt history
        recent = collect(range(2.0, 14.0; length=30)) .+ 0.0
        for (station, V, Bz) in (("FRD", 520.0, -8.0), ("CMO", 600.0, -12.0))
            m = load_forecaster(; station=station)
            @test m !== nothing
            @test forecast_dbdt(recent, V, Bz; station=station) === nothing
            fc = forecast_dbdt(recent, V, Bz; station=station,
                               offline_replay=true)
            @test fc !== nothing
            gp, gub, gexc, ẑ, s = golden_forecast(m, recent, V, Bz)
            # Explicit offline replay must match the independently recomputed export formula.
            @test isapprox(fc.point_dbdt, round(max(gp, 0.0); digits=2); atol=0.02)
            @test isapprox(fc.ub90_dbdt, round(max(gub, 0.0); digits=2); atol=0.05)
            @test length(fc.exceedance) == length(gexc)
            for (k, e) in enumerate(fc.exceedance)
                @test e.threshold == gexc[k][1]
                @test isapprox(e.empirical_score, round(gexc[k][2]; digits=3); atol=0.01)
            end
            @test fc.station == station
            @test fc.horizon_min == 30
            @test fc.source == "offline ridge + historical residual sample"
            @test fc.training_ground_data_product == "quasi-definitive"
            @test fc.live_ground_data_product == "adjusted (provisional)"
            @test fc.training_driver_product == "OMNI_HRO_1MIN"
            @test fc.training_driver_propagation == "bow-shock-nose time shifted"
            @test fc.driver_time_reference_aligned == false
            @test fc.serving_enabled == false
            # interval sanity: upper bound is above the point forecast
            @test fc.ub90_dbdt >= fc.point_dbdt
        end
    end

    @testset "offline artifact regression vectors and numerical cap" begin
        quiet = fill(0.5, 30)
        storm = collect(range(20.0, 120.0; length=30))
        q = forecast_dbdt(quiet, 380.0, 1.0; station="FRD", offline_replay=true)
        s = forecast_dbdt(storm, 700.0, -20.0; station="FRD", offline_replay=true)
        @test q.point_dbdt < s.point_dbdt
        @test q.exceedance[1].empirical_score <= s.exceedance[1].empirical_score
        # Exercise the second station without imposing a universal cross-latitude ordering.
        af = forecast_dbdt(storm, 700.0, -20.0; station="CMO", offline_replay=true)
        @test isfinite(af.point_dbdt) && isfinite(af.ub90_dbdt)
        @test af.serving_enabled == false
        # Out-of-range input must not overflow the explicit numerical safety cap.
        blow = fill(5000.0, 30)
        b = forecast_dbdt(blow, 1200.0, -60.0; station="CMO", offline_replay=true)
        @test isfinite(b.point_dbdt) && b.point_dbdt <= 2001.0
        @test isfinite(b.ub90_dbdt) && b.ub90_dbdt <= 2001.0
    end

    @testset "dB/dt forecast requires a contiguous 30-minute history" begin
        @test forecast_dbdt(fill(2.0, 29), 420.0, -3.0;
                            station="FRD", offline_replay=true) === nothing
        invalid = fill(2.0, 30); invalid[10] = NaN
        @test forecast_dbdt(invalid, 420.0, -3.0;
                            station="FRD", offline_replay=true) === nothing

        start = DateTime(2026, 7, 16, 0, 0)
        rows = [(t=Dates.format(start + Minute(index - 1),
                                dateformat"yyyy-mm-ddTHH:MM:SS") * "Z",
                 dbdt=2.0 + index / 10) for index in 1:30]
        @test forecast_dbdt_observations(rows, 420.0, -3.0;
                                         station="FRD") === nothing
        @test forecast_dbdt_observations(rows, 420.0, -3.0;
                                         station="FRD", offline_replay=true) !== nothing
        gapped = copy(rows)
        gapped[15] = merge(gapped[15],
                           (t=Dates.format(start + Minute(30),
                                           dateformat"yyyy-mm-ddTHH:MM:SS") * "Z",))
        @test forecast_dbdt_observations(gapped, 420.0, -3.0;
                                         station="FRD", offline_replay=true) === nothing
        @test forecast_dbdt_observations(rows[1:29], 420.0, -3.0;
                                         station="FRD", offline_replay=true) === nothing
    end

    @testset "per-station cache returns the right model" begin
        # loading CMO then FRD must not return CMO's model for FRD (the prior Ref-vs-Dict bug)
        cmo = load_forecaster(; station="CMO")
        frd = load_forecaster(; station="FRD")
        @test cmo.station == "CMO"
        @test frd.station == "FRD"
        @test Float64.(cmo.beta) != Float64.(frd.beta)
    end

    @testset "forecaster thresholds are positive, integral, and Int-representable" begin
        artifact(thresholds) = (
            artifact_schema_version=3,
            station="FRD",
            training_ground_data_product="quasi-definitive",
            live_ground_data_product="adjusted (provisional)",
            training_driver_product="OMNI_HRO_1MIN",
            training_driver_propagation="bow-shock-nose time shifted",
            available_live_driver_product="NOAA SWPC real-time solar wind at L1",
            driver_time_reference_aligned=false,
            serving_enabled=false,
            serving_blocker="time-reference mismatch",
            forecast_horizon_minutes=30,
            issue_cadence_minutes=1,
            overlapping_targets=true,
            features=["dbdt_now", "dbdt_mean30", "dbdt_max30", "dbdt_std30",
                      "V", "Bz", "Bs", "VBs"],
            mu=zeros(8), sigma=ones(8), beta=zeros(9), rn_calib=[0.0, 1.0],
            thresholds=thresholds,
        )
        @test _valid_forecaster_artifact(artifact([18.0, 42.0]), "FRD")
        @test !_valid_forecaster_artifact(
            merge(artifact([18.0]), (artifact_schema_version=2,)), "FRD")
        @test !_valid_forecaster_artifact(
            merge(artifact([18.0]), (serving_enabled=true,)), "FRD")
        @test !_valid_forecaster_artifact(
            merge(artifact([18.0]), (driver_time_reference_aligned=true,)), "FRD")
        @test !_valid_forecaster_artifact(artifact([18.5]), "FRD")
        @test !_valid_forecaster_artifact(artifact([Float64(typemax(Int))]), "FRD")
        @test !_valid_forecaster_artifact(artifact([1.0e100]), "FRD")
        @test !_valid_forecaster_artifact(artifact([true]), "FRD")
        @test !_valid_forecaster_threshold(true)
    end

    @testset "SWPC row parsing tolerates partial public-feed rows" begin
        idx = Dict("speed" => 2, "density" => 3)
        @test _swpc_row_field(idx, ["2026-06-26T00:00:00Z", "460.5", "1.7"], "speed") == 460.5
        @test _swpc_row_field(idx, ["2026-06-26T00:00:00Z", "460.5"], "density") === nothing
        @test _swpc_row_field(idx, ["2026-06-26T00:00:00Z", "460.5"], "bz_gsm") === nothing
        @test_throws InterruptException _pf(_InterruptingSWPCText())
        @test_throws InterruptException _swpc_dt(_InterruptingSWPCText())
    end

    @testset "SWPC alert parsing is UTF-8 safe and isolates malformed records" begin
        # Byte-safe summary extraction: multibyte messages must not throw (StringIndexError) or
        # truncate mid-character. A raw byte index (nl-1) or a character-count slice (length) does.
        @test _alert_summary("WARNING: T² index high") == "WARNING: T² index high"   # no newline, multibyte tail
        @test _alert_summary("WARNING: 7°\nkp elevated") == "WARNING: 7°"            # multibyte before newline
        @test _alert_summary("ALERT: flux 10²² pfu") == "ALERT: flux 10²² pfu"       # multibyte tail preserved
        @test _alert_summary("WARNING: G3 storm\r\nmore") == "WARNING: G3 storm"     # CRLF handled
        @test _alert_summary("WARNING: K-index of 5\nvalid") == "WARNING: K-index of 5"
        @test _alert_summary("Space weather nominal.\nsecond") == "Space weather nominal. second"  # no keyword

        # Null message / product_id (JSON null -> nothing) must not throw MethodError.
        nullrec = JSON3.read("{\"product_id\":null,\"issue_datetime\":null,\"message\":null}")
        parsed = _parse_alert(nullrec)
        @test parsed.summary == "" && parsed.product_id == "" && parsed.issue_utc === nothing

        good = Dict(:product_id => "K05", :issue_datetime => "2026-06-20 10:00:00.000",
                    :message => "WARNING: Geomagnetic K-index of 5\nmore text")
        gp = _parse_alert(good)
        @test gp.product_id == "K05" && gp.summary == "WARNING: Geomagnetic K-index of 5"

        # Per-alert isolation: one pathological record cannot take down the whole snapshot.
        # An InterruptException inside a record must propagate (never swallowed).
        @test_throws InterruptException _alerts_from(Any[Dict(:message => _InterruptingSWPCText())], 6)
        # A non-Interrupt failure is skipped; surrounding good records survive.
        bad_record = Dict(:message => Dict("unexpected" => "object"))   # String(::Dict) throws MethodError
        kept = _alerts_from(Any[good, bad_record, Dict(:message => "ALERT: C")], 6)
        @test length(kept) == 2
        @test kept[1].summary == "WARNING: Geomagnetic K-index of 5"
        @test kept[2].summary == "ALERT: C"
        @test _alerts_from(nothing, 6) == NamedTuple[]
        @test _alerts_from(Any[], 6) == NamedTuple[]
    end

    @testset "forecast API exposes upgraded V2 as the product forecast" begin
        issue = now(UTC) - Minute(10)
        df = live_cycle_fixture(
            issue;
            latest_dst=-50.0,
            observations=[-60.0, -61.0, -62.0, -63.0],
            audit_pred=[-45.0, -46.0, -47.0, -48.0],
            audit_lo=[-55.0, -56.0, -57.0, -58.0],
            audit_hi=[-35.0, -36.0, -37.0, -38.0],
            served_pred=[-59.0, -60.0, -61.0, -62.0],
            served_lo=[-69.0, -70.0, -71.0, -72.0],
            served_hi=[-49.0, -50.0, -51.0, -52.0],
        )
        df[!, :persistence_dst_nt] = fill(-50.0, nrow(df))
        df[!, :obrien_dst_nt] = [-58.0, -59.0, -60.0, -61.0]
        cal = calibration_summary(df)
        expected_v2_rmse = round(sqrt(mean((df.observation_dst_nt .- df.served_pred_dst_nt).^2)); digits=2)
        expected_audit_rmse = round(sqrt(mean((df.observation_dst_nt .- df.v2_pred_dst_nt).^2)); digits=2)
        @test cal.v2_n_verified == 4
        @test cal.v2_rmse_nt == expected_v2_rmse
        @test cal.rmse_nt == expected_v2_rmse
        @test cal.audit_baseline_rmse_nt == expected_audit_rmse
        @test cal.frozen_tail_ablation_rmse_nt == expected_audit_rmse
        @test cal.v2_coverage_90 == 1.0
        hist = build_history(df, 24)
        @test hist.rmse_nt == cal.v2_rmse_nt
        @test hist.rows[1].pred_dst_nt == df.served_pred_dst_nt[1]
        @test hist.rows[1].audit_baseline_dst_nt == df.v2_pred_dst_nt[1]
        @test hist.rows[1].frozen_tail_ablation_dst_nt == df.v2_pred_dst_nt[1]
        fc = build_forecast(df)
        @test fc.available && length(fc.horizons) == 4
        @test fc.horizons[1].pred_dst_nt == df.served_pred_dst_nt[1]
        @test fc.horizons[1].audit_baseline_dst_nt == df.v2_pred_dst_nt[1]
        @test fc.horizons[1].frozen_tail_ablation_dst_nt == df.v2_pred_dst_nt[1]

        legacy = copy(df)
        legacy.model_version .= "v2"
        legacy.sub_hourly_model_version .= "v2+L1A+Bregime+Pinertia"
        @test nrow(verified_rows(legacy)) == 0
        @test calibration_summary(legacy).v2_n_verified == 0
        @test isempty(build_history(legacy, 24).rows)
        @test !build_forecast(legacy).available
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
        outside_dir = mktempdir()
        outside = joinpath(outside_dir, "secret.txt")
        write(outside, "must not be served")
        link = tempname(PUBLIC_DIR)
        symlink(outside, link)
        try
            @test serve_static("/" * basename(link)).status == 403
        finally
            rm(link; force=true)
            rm(outside_dir; recursive=true, force=true)
        end
    end

    @testset "exported offline model self-consistency" begin
        # The residual grid must be sorted and its empirical 0.90 quantile finite.
        for station in ("FRD", "CMO")
            m = load_forecaster(; station=station)
            rn = Float64.(m.rn_calib)
            @test issorted(rn)
            @test isfinite(quantile(rn, 0.90))
            @test haskey(m, :cap_note)              # cap convention documented in the artifact
            @test m.artifact_schema_version == 3
            @test m.training_ground_data_product == "quasi-definitive"
            @test m.live_ground_data_product == "adjusted (provisional)"
            @test m.training_driver_product == "OMNI_HRO_1MIN"
            @test m.driver_time_reference_aligned == false
            @test m.serving_enabled == false
            @test haskey(m, :dataset_sha256)
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
        @test_throws ArgumentError geoelectric_field(Float64[], Float64[], 60.0)
        @test_throws ArgumentError geoelectric_field([0.0], [0.0], 0.0)
        @test_throws ArgumentError geoelectric_field([0.0], [0.0], -60.0)
        @test_throws ArgumentError geoelectric_field([0.0], [0.0], Inf)
        @test_throws ArgumentError geoelectric_field([0.0, NaN], [0.0, 1.0], 60.0)
        @test_throws ArgumentError geoelectric_field([0.0, 1.0], [0.0, Inf], 60.0)
    end

    @testset "causal half-space E-field: exact ramp response and sinusoid amplitude" begin
        mu0 = 4e-7 * pi
        dt = 60.0; nwin = 121; rho = 1000.0
        tt = collect(0:nwin-1)
        # --- pure ramp: constant dB/dt must produce the exact 2*b*sqrt(t)/sqrt(pi*mu0*sigma) field.
        # This is precisely the sustained-storm component a detrended DFT window deletes. Independent
        # oracle: E_y(t) = -sqrt(rho/mu0) * D^{1/2}[B_x] with D^{1/2}[b*t] = 2*b*sqrt(t/pi).
        b_ps = 5.0                                  # nT per sample
        bx = Float64.(b_ps .* tt); by = zeros(nwin)
        exr, eyr = causal_halfspace_efield(bx, by, dt; rho_ohm_m=rho)
        b_SI = b_ps / dt * 1e-9                      # T/s
        t_cur = (nwin - 1) * dt
        ramp_analytic = 2 * b_SI * sqrt(t_cur) * sqrt(rho / (pi * mu0)) * 1000.0   # V/km
        ramp_got = sqrt(exr[end]^2 + eyr[end]^2)
        @test isapprox(ramp_got, ramp_analytic; rtol=1e-9)     # exact for piecewise-constant dB/dt
        @test exr[end] == 0.0                                   # By flat -> Ex zero
        # A detrended DFT window would return ~0 here; the causal field must be O(0.2 V/km).
        @test ramp_got > 0.2
        # --- pure sinusoid: amplitude must match the analytic |Z|/mu0 response A*sqrt(w*rho/mu0).
        A = 30.0; per = 10.0                         # 30 nT, 10-min period
        w = 2pi / (per * dt)
        xs = A .* sin.(2pi .* tt ./ per); ys = zeros(nwin)
        exs, eys = causal_halfspace_efield(xs, ys, dt; rho_ohm_m=rho)
        sin_got = maximum(sqrt.(exs[end-40:end].^2 .+ eys[end-40:end].^2))
        sin_analytic = A * 1e-9 * sqrt(w * rho / mu0) * 1000.0
        @test isapprox(sin_got, sin_analytic; rtol=0.03)        # piecewise-constant, 10 samples/period
        # --- sign/pairing: Ex from By, Ey from Bx (matches geoelectric_field convention).
        exb, eyb = causal_halfspace_efield(zeros(nwin), Float64.(b_ps .* tt), dt; rho_ohm_m=rho)
        @test eyb[end] == 0.0 && exb[end] > 0.2                 # ramp on By drives Ex, not Ey
        # --- error handling
        @test_throws ArgumentError causal_halfspace_efield([0.0], [0.0], 0.0)
        @test_throws ArgumentError causal_halfspace_efield([0.0], [0.0], -1.0)
        @test_throws ArgumentError causal_halfspace_efield(Float64[], Float64[], 60.0)
        @test_throws ArgumentError causal_halfspace_efield([0.0, NaN], [0.0, 1.0], 60.0)
        @test_throws ArgumentError causal_halfspace_efield([0.0, 1.0], [0.0], 60.0)
    end

    @testset "dB/dt bands preserve published numeric thresholds without risk labels" begin
        expected = [
            (0.0, 0, "Below 18 nT/min"),
            (17.999, 0, "Below 18 nT/min"),
            (18.0, 1, "At least 18 nT/min"),
            (42.0, 2, "At least 42 nT/min"),
            (66.0, 3, "At least 66 nT/min"),
            (90.0, 4, "At least 90 nT/min"),
        ]
        for (value, level, label) in expected
            band = dbdt_tier(value)
            @test band.level == level
            @test band.label == label
        end
        @test dbdt_tier(-1.0) == (level=nothing, label="—")
        @test dbdt_tier(true) == (level=nothing, label="—")
        @test dbdt_tier(NaN) == (level=nothing, label="—")
        @test !isdefined(@__MODULE__, :GEO_TIERS)
        @test !isdefined(@__MODULE__, :GEO_EDGES)
        @test !isdefined(@__MODULE__, :geo_tier)
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

        # A regenerated replay must immediately become the API/UI source without copying files
        # into the package snapshot.
        withenv("SOLARSINDY_OPERATIONAL_OUTPUT_DIR" => dir,
                "SOLARSINDY_OPERATIONAL_EVIDENCE_DIR" => nothing) do
            @test app_operational_evidence_dir(
                "storm_replay_report.md", "storm_replay_scored.csv",
            ) == abspath(dir)
            payload = JSON3.read(String(api_handler("/api/storm_replay", "", log_path).body))
            @test payload.available == true
            @test payload.n_scored == 3
        end
    end

    @testset "offline dB/dt replay flags out-of-validated-range / saturated inputs" begin
        # A merely quiet-to-mild dB/dt history replays within the validated range.
        normal = forecast_dbdt(fill(2.0, 30), 420.0, -3.0;
                               station="FRD", offline_replay=true)
        if normal !== nothing                      # only if the FRD artifact is present
            @test haskey(normal, :reliable)
            @test normal.reliable == true
            @test normal.saturated == false
            # An absurd dB/dt history (far outside the ~1.7 nT/min calibration mean) must be
            # flagged unreliable/saturated rather than surfaced as a confident estimate.
            extreme = forecast_dbdt(fill(5000.0, 30), 1200.0, -80.0;
                                    station="FRD", offline_replay=true)
            @test extreme !== nothing
            @test extreme.reliable == false
            @test (extreme.out_of_validated_range || extreme.saturated)
        end
    end

    @testset "RTSW solar-wind parser: named keys, active flag, null/out-of-bounds rejection" begin
        # Captured-schema sample of /json/rtsw/rtsw_mag_1m.json: array of OBJECTS, interleaved
        # spacecraft (SOLAR1/ACE), duplicate time_tags, deliberately out of order. The parser must
        # select by named keys + newest time_tag, prefer active=true, and skip null/out-of-bounds.
        payload = """
        [
          {"time_tag":"2026-07-13T02:38:00","active":false,"source":"ACE","bt":4.5,"bz_gsm":0.5},
          {"time_tag":"2026-07-13T02:40:00","active":true,"source":"SOLAR1","bt":4.7,"bz_gsm":-0.8},
          {"time_tag":"2026-07-13T02:40:00","active":false,"source":"ACE","bt":4.4,"bz_gsm":-0.3},
          {"time_tag":"2026-07-13T02:39:00","active":true,"source":"SOLAR1","bt":null,"bz_gsm":-0.5},
          {"time_tag":"2026-07-13T02:41:00","active":true,"source":"SOLAR1","bt":9.0e9,"bz_gsm":-1.0}
        ]"""
        arr = JSON3.read(payload)
        row = _rtsw_latest(arr, [:bz_gsm, :bt];
                           bounds = Dict(:bt => (0.0, 1.0e3), :bz_gsm => (-1.0e3, 1.0e3)))
        @test row !== nothing
        @test String(row.time_tag) == "2026-07-13T02:40:00"   # newest active, in-bounds, non-null
        @test _rtsw_field(row, :bz_gsm) == -0.8
        @test _rtsw_active(row) == true
        # only inactive rows present -> fall back to the newest valid inactive source
        only_inactive = JSON3.read("""[{"time_tag":"2026-07-13T02:30:00","active":false,"source":"ACE","bt":5.0,"bz_gsm":1.2}]""")
        r2 = _rtsw_latest(only_inactive, [:bz_gsm, :bt])
        @test r2 !== nothing && _rtsw_field(r2, :bz_gsm) == 1.2
        # empty / nothing payloads never throw
        @test _rtsw_latest(JSON3.read("[]"), [:bz_gsm]) === nothing
        @test _rtsw_latest(nothing, [:bz_gsm]) === nothing
        # wind schema uses different named keys; out-of-bounds speed is rejected
        wind = JSON3.read("""[{"time_tag":"2026-07-13T02:40:00","active":true,"proton_speed":461.0,"proton_density":2.7}]""")
        wrow = _rtsw_latest(wind, [:proton_speed, :proton_density];
                            bounds = Dict(:proton_speed => (50.0, 5.0e3)))
        @test _rtsw_field(wrow, :proton_speed) == 461.0
    end

    @testset "latest_cycle keys on issue epoch, not solar-wind vintage (L1 stall)" begin
        # Two hourly issue cycles that share ONE frozen solar-wind vintage (the L1-stall pattern):
        # keying on that vintage would merge them; keying on issue time must serve only the newest.
        sw_vintage = now(UTC) - Hour(3)
        iss_old = now(UTC) - Minute(90)
        iss_new = now(UTC) - Minute(30)
        old_cycle = live_cycle_fixture(
            iss_old; vintage=sw_vintage, anchor_time=iss_old - Hour(1), latest_dst=-10.0,
            served_pred=[-5.0, -6.0, -7.0, -8.0],
            served_lo=[-99.0, -98.0, -97.0, -96.0],
            served_hi=[0.0, -1.0, -2.0, -3.0],
        )
        new_cycle = live_cycle_fixture(
            iss_new; vintage=sw_vintage, anchor_time=iss_new - Hour(1), latest_dst=-20.0,
            served_pred=[-15.0, -16.0, -17.0, -18.0],
            served_lo=[-40.0, -42.0, -41.0, -43.0],
            served_hi=[-5.0, -6.0, -7.0, -8.0],
        )
        df = vcat(old_cycle, new_cycle)
        cyc = latest_cycle(df)
        @test nrow(cyc) == 4
        @test all(==(iss_new), cyc.issue_time_utc_dt)
        st = build_status(df)
        @test st.available == true
        @test st.forecast_issue_utc == string(iss_new) * "Z"          # newest issue, not superseded
        @test st.latest_observation.dst_nt == -20.0                   # anchor from newest cycle
        @test st.threat.interval_lower_edge_min_dst_nt ==
              st.threat.lower_bound_min_dst_nt ==
              st.threat.worst_credible_dst_nt == -43.0              # min over newest cycle, not -99
        @test st.threat.level == 0
        @test st.threat.watch == true
        @test st.threat.watch_level == 1
        alerts = build_alerts(df, st).alerts
        @test any(alert -> alert.kind == "watch" && alert.level == 1, alerts)
        watch = only(filter(alert -> alert.kind == "watch", alerts))
        @test occursin("displayed calibrated 90% interval", watch.message)
        @test !occursin("cannot be excluded", watch.message)
        fc = build_forecast(df)
        @test length(fc.horizons) == 4
        @test fc.issue_time_utc == string(iss_new) * "Z"
    end

    @testset "latest_cycle does not merge restart cycles across an hour boundary" begin
        boundary = floor(now(UTC), Hour)
        old_issue = boundary - Minute(1)
        new_issue = boundary + Minute(1)
        old_cycle = live_cycle_fixture(old_issue; latest_dst=-10.0)
        new_cycle = live_cycle_fixture(new_issue; latest_dst=-20.0)
        cyc = latest_cycle(vcat(old_cycle, new_cycle))
        @test nrow(cyc) == length(LIVE_CYCLE_HORIZONS)
        @test all(==(new_issue), cyc.issue_time_utc_dt)
        @test _valid_live_cycle(cyc)
    end

    @testset "live cycle rejects widely separated retries within one issue hour" begin
        issue = DateTime(2026, 7, 15, 12, 1)
        spread = live_cycle_fixture(issue)
        spread.issue_time_utc_dt[end] += Minute(6)
        spread.horizon_hours[end] =
            (spread.target_time_utc_dt[end] - spread.issue_time_utc_dt[end]) /
            Millisecond(3_600_000)
        @test nrow(latest_cycle(spread)) == length(LIVE_CYCLE_HORIZONS)
        @test !_valid_live_cycle(spread)
    end

    @testset "latest cycle requires the full horizon set and common metadata" begin
        issue = now(UTC) - Minute(10)
        valid = live_cycle_fixture(issue)
        @test _valid_live_cycle(valid)
        @test build_status(valid).available
        @test build_forecast(valid).available

        spread = copy(valid)
        for row in 2:nrow(spread)
            spread.issue_time_utc_dt[row] += Second(row)
            spread.horizon_hours[row] =
                (spread.target_time_utc_dt[row] - spread.issue_time_utc_dt[row]) /
                Millisecond(3_600_000)
        end
        @test _valid_live_cycle(spread)
        @test build_status(spread).forecast_issue_utc == jdt(maximum(spread.issue_time_utc_dt))
        @test build_forecast(spread).issue_time_utc == jdt(maximum(spread.issue_time_utc_dt))

        invalid = DataFrame[valid[1:3, :], vcat(valid, valid[end:end, :])]
        wrong_schedule = copy(valid)
        wrong_schedule.target_time_utc_dt[3] = floor(issue, Hour) + Hour(4)
        wrong_schedule.horizon_hours[3] =
            (wrong_schedule.target_time_utc_dt[3] - issue) / Millisecond(3_600_000)
        sort!(wrong_schedule, :target_time_utc_dt)
        push!(invalid, wrong_schedule)
        uniformly_historical = copy(valid)
        uniformly_historical.model_version .= "v2"
        uniformly_historical.sub_hourly_model_version .= "v2+L1A+Bregime+Pinertia"
        push!(invalid, uniformly_historical)
        for (field, value) in (
            (:model_version, "v3"),
            (:sub_hourly_model_version, "different-served-model"),
            (:interval_source, "different-interval"),
            (:latest_dst_time_utc_dt, issue - Hour(2)),
            (:latest_dst_nt, -21.0),
            (:latest_solar_wind_utc_dt, issue - Minute(1)),
        )
            mixed = copy(valid)
            mixed[end, field] = value
            push!(invalid, mixed)
        end
        for cycle in invalid
            @test !_valid_live_cycle(cycle)
            status = build_status(cycle)
            forecast = build_forecast(cycle)
            @test !status.available && !haskey(status, :model_version)
            @test !forecast.available && !haskey(forecast, :model_version)
            @test calibration_summary(cycle).current_interval_source == "unknown"
        end
    end

    @testset "staleness gate: expired cycle suppresses live status and alerts" begin
        old = now(UTC) - Day(10)
        df = DataFrame(
            issue_time_utc_dt = [old], latest_solar_wind_utc_dt = [old],
            latest_dst_time_utc_dt = [old - Hour(1)], target_time_utc_dt = [old + Hour(1)],
            horizon_hours = [1.0], latest_dst_nt = [-60.0], observation_dst_nt = [missing],
            served_pred_dst_nt = [-70.0], served_pred_dst_ci05_nt = [-90.0], served_pred_dst_ci95_nt = [-50.0],
            interval_source = ["aci"], model_version = ["v2.1"],
        )
        st = build_status(df)
        @test st.available == false
        @test st.stale == true
        @test st.expired == true
        @test st.age_hours > 200                     # ~240 h old
        @test build_alerts(df).active == false       # no forecast/watch alerts from an expired cycle
        cs = compute_alert_state(st, (available=false,), nothing)
        @test cs.level == 0
        @test cs.stale == true         # a blind/expired forecast is stale, not "quiet"
        # a fresh cycle is neither stale nor expired
        fresh = now(UTC) - Minute(20)
        df2 = live_cycle_fixture(fresh)
        st2 = build_status(df2)
        @test st2.available == true && st2.stale == false && st2.expired == false
    end

    @testset "served pipeline label exposed from sub_hourly_model_version" begin
        iss = now(UTC) - Minute(20)
        df = live_cycle_fixture(iss)
        fc = build_forecast(df); st = build_status(df)
        @test fc.served_model_version == "v2.1+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia"
        @test fc.model_version == "v2.1"
        @test st.served_model_version == "v2.1+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia"
        @test st.model_version == "v2.1"
        df2 = select(df, Not(:sub_hourly_model_version))
        @test !build_forecast(df2).available                         # served label is required
    end

    @testset "missing log degrades gracefully; NaN hours falls back to 72" begin
        missing_path = joinpath(mktempdir(), "does_not_exist.csv")
        _LOG_CACHE[] = nothing
        g = get_log(missing_path)
        @test g isa DataFrame && nrow(g) == 0                      # absent file -> empty frame, no throw
        h = make_handler(missing_path)
        # log-independent-but-log-backed endpoints must not 500 when the log is absent
        @test h(HTTP.Request("GET", "/api/forecast")).status == 200
        @test h(HTTP.Request("GET", "/api/history?hours=72")).status == 200
        health = h(HTTP.Request("GET", "/api/health"))
        health_body = JSON3.read(String(health.body))
        @test health.status == 200 && !haskey(health_body, :log_path)
        @test health_body.status == "no_log" && !health_body.cycle_complete
        @test !occursin(abspath(missing_path), String(health.body))
        # NaN hours reaches the crash path only with verified rows present -> use a real temp log
        now0 = floor(now(UTC), Hour)
        vdf = DataFrame(
            issue_time_utc = [string(now0 - Hour(2))],
            latest_solar_wind_utc = [string(now0 - Hour(2))],
            latest_dst_time_utc = [string(now0 - Hour(2))],
            target_time_utc = [string(now0 - Hour(1))],
            horizon_hours = [1.0], latest_dst_nt = [-20.0], observation_dst_nt = [-22.0],
            served_pred_dst_nt = [-21.0], served_pred_dst_ci05_nt = [-31.0], served_pred_dst_ci95_nt = [-11.0],
            v2_pred_dst_nt = [-21.0], v2_pred_dst_ci05_nt = [-31.0], v2_pred_dst_ci95_nt = [-11.0],
            model_version = ["v2.1"],
            sub_hourly_model_version = [CURRENT_V2_SERVED_MODEL_VERSION],
            interval_source = ["aci"],
        )
        logfile = joinpath(mktempdir(), "log.csv"); CSV.write(logfile, vdf)
        _LOG_CACHE[] = nothing
        h2 = make_handler(logfile)
        rn = h2(HTTP.Request("GET", "/api/history?hours=NaN"))
        @test rn.status == 200
        body = JSON3.read(String(rn.body))
        @test body.hours == 72.0 && body.n >= 1                    # NaN -> 72, verified row scored
        _LOG_CACHE[] = nothing                                      # leave the cache clean for other tests
    end

    @testset "health endpoint requires one complete, current product cycle" begin
        function write_cycle(path, issue; keep=1:length(LIVE_CYCLE_HORIZONS))
            raw = live_cycle_fixture(issue)[collect(keep), :]
            rename!(raw,
                :issue_time_utc_dt => :issue_time_utc,
                :latest_solar_wind_utc_dt => :latest_solar_wind_utc,
                :latest_dst_time_utc_dt => :latest_dst_time_utc,
                :target_time_utc_dt => :target_time_utc,
            )
            CSV.write(path, raw)
            return path
        end
        health(path) = JSON3.read(String(
            make_handler(path)(HTTP.Request("GET", "/api/health")).body,
        ))

        dir = mktempdir()
        current_issue = now(UTC) - Second(1)
        current_path = write_cycle(joinpath(dir, "current.csv"), current_issue)
        _LOG_CACHE[] = nothing
        current = health(current_path)
        @test current.status == "ok" && current.cycle_complete

        partial_path = write_cycle(
            joinpath(dir, "partial.csv"), current_issue; keep=1:3,
        )
        _LOG_CACHE[] = nothing
        partial = health(partial_path)
        @test partial.status == "incomplete" && !partial.cycle_complete

        stale_path = write_cycle(joinpath(dir, "stale.csv"), now(UTC) - Hour(4))
        _LOG_CACHE[] = nothing
        stale = health(stale_path)
        @test stale.status == "stale" && stale.cycle_complete
        _LOG_CACHE[] = nothing
    end

    @testset "warmup compiles and caches endpoint paths without throwing" begin
        # Present log: warm-up must prime the get_log cache so the first request after
        # the listener opens does not pay the CSV parse while holding _LOG_LOCK.
        dir = mktempdir()
        raw = live_cycle_fixture(now(UTC) - Minute(5))
        rename!(raw,
            :issue_time_utc_dt => :issue_time_utc,
            :latest_solar_wind_utc_dt => :latest_solar_wind_utc,
            :latest_dst_time_utc_dt => :latest_dst_time_utc,
            :target_time_utc_dt => :target_time_utc,
        )
        p = joinpath(dir, "log.csv"); CSV.write(p, raw)
        _LOG_CACHE[] = nothing
        _LATEST_CYCLE_CACHE[] = nothing; _HISTORY_CACHE[] = nothing
        secs = warmup(p)
        @test secs isa Real && secs >= 0
        c = _LOG_CACHE[]
        @test c !== nothing && c[1] == _log_file_identity(p)    # cache primed by warm-up
        @test nrow(c[2]) == length(LIVE_CYCLE_HORIZONS)
        # The payload builders must actually have run (a mutant warmup that only calls
        # get_log and skips the build_* lines leaves these downstream caches empty).
        @test _LATEST_CYCLE_CACHE[] !== nothing
        @test _HISTORY_CACHE[] !== nothing
        # Absent log (fresh install before the daemon's first write): must not throw and
        # must not poison the cache with a frame attributed to the missing path.
        _LOG_CACHE[] = nothing
        @test warmup(joinpath(dir, "missing.csv")) >= 0
        @test _LOG_CACHE[] === nothing                           # nothing cached for an absent file
        # "Never throws": an unreadable log must be absorbed by warmup's catch (get_log
        # raises when it has no previously cached frame to serve).
        if Sys.isunix() && ccall(:geteuid, Cint, ()) != 0
            locked = joinpath(dir, "locked.csv"); CSV.write(locked, raw); chmod(locked, 0o000)
            _LOG_CACHE[] = nothing
            @test (@test_logs (:warn, r"warm-up failed") warmup(locked)) >= 0
            chmod(locked, 0o644)
        end
        # Garbage bytes must not crash warm-up either (lenient parse or absorbed error).
        junk = joinpath(dir, "junk.csv"); write(junk, "\x00\x01not,a,log\n\xff\xfe")
        _LOG_CACHE[] = nothing
        @test warmup(junk) >= 0
        _LOG_CACHE[] = nothing
        _LATEST_CYCLE_CACHE[] = nothing; _HISTORY_CACHE[] = nothing
    end

    @testset "sub-hour trajectory served only for the matching cycle" begin
        dir = mktempdir(); logf = joinpath(dir, "log.csv")
        iss = DateTime("2026-06-30T23:59:50.122")
        write(joinpath(dir, "subhour_trajectory.json"),
              """{"points":[{"t":"2026-06-30T22:00:00","dst":1.0},{"t":"2026-06-30T23:15:00","dst":0.5},""" *
              """{"t":"2026-07-01T00:00:00","dst":-1.0}],"anchor_time_utc":"2026-06-30T23:00:00",""" *
              """"issue_time_utc":"2026-06-30T23:59:50.122","anchor_dst_nt":0.0}""")
        matched = _subhour_traj(logf; cycle_issue = iss)           # anchor drops the 22:00 point
        @test length(matched) == 2
        @test _subhour_traj(logf; cycle_issue = iss + Hour(1)) |> isempty   # log advanced -> stale sidecar
        @test length(_subhour_traj(logf)) == 2                     # no cycle context -> back-compat
    end

    @testset "geoelectric nowcast keeps the storm ramp and serves the real endpoint" begin
        tt = collect(1:120)
        xv = Vector{Any}(20.0 .* tt .+ 3.0 .* sin.(tt ./ 3.0))     # rising ramp + wiggle (storm main phase)
        yv = Vector{Any}(fill(5.0, 120))
        g = _geoe_nowcast(xv, yv, 60.0)
        @test g !== nothing
        # Wiring: the served value is the causal field at the real endpoint (no detrend, no edge trim).
        # The causal_halfspace_efield primitive itself is validated against analytic oracles above.
        xf = _interp_gaps([_num(xv[i]) for i in 1:120]); yf = _interp_gaps([_num(yv[i]) for i in 1:120])
        ex, ey = causal_halfspace_efield(xf, yf, 60.0; rho_ohm_m=1000.0)
        emag = sqrt.(ex.^2 .+ ey.^2); m = length(emag)
        @test g.trailing_gap == 0
        @test isapprox(g.current, emag[m]; atol=1e-9)                          # real endpoint, not emag[m-3]
        @test isapprox(g.max, maximum(emag[max(2, m - 30):m]); atol=1e-9)      # trailing ~30 min
        # A sustained 20 nT/min ramp yields a physically substantial field; the previous detrended
        # circular DFT collapsed it ~4x toward the fluctuation-only amplitude.
        @test g.current > 0.5
    end

    @testset "geoelectric current comes from the last real sample, not a flat-filled tail" begin
        tt = collect(1:120)
        ngap = 4
        xv = Vector{Any}(20.0 .* tt)                               # pure ramp -> monotone rise then decay
        for i in (120 - ngap + 1):120; xv[i] = nothing; end        # trailing nulls (USGS real-time latency)
        yv = Vector{Any}(fill(5.0, 120))
        g = _geoe_nowcast(xv, yv, 60.0)
        @test g !== nothing
        @test g.trailing_gap == ngap
        xf = _interp_gaps([_num(xv[i]) for i in 1:120]); yf = _interp_gaps([_num(yv[i]) for i in 1:120])
        ex, ey = causal_halfspace_efield(xf, yf, 60.0; rho_ohm_m=1000.0)
        emag = sqrt.(ex.^2 .+ ey.^2); m = length(emag); last_real = m - ngap
        @test isapprox(g.current, emag[last_real]; atol=1e-9)                  # served at last real sample
        @test g.current > emag[m] + 1e-6                                       # flat-filled tail is biased low
    end

    @testset "geoelectric payload exposes an honest observation time under trailing gaps" begin
        reference = DateTime(2026, 7, 14, 12); n = 40; ngap = 3
        times = [reference - Minute(n - 1) + Minute(i) for i in 0:n-1]         # 1-min cadence ending at reference
        xvals = Vector{Any}(10.0 .* collect(0:n-1))
        for i in (n - ngap + 1):n; xvals[i] = nothing; end                     # last 3 min null-filled
        payloadd = (times=jdt.(times),
                    values=[(metadata=(element="X",), values=xvals),
                            (metadata=(element="Y",), values=zeros(n))])
        nc = _compute_dbdt("TST", 120; fetch_fn=(s, m) -> payloadd, reference=reference)
        @test nc.available && nc.geoelectric !== nothing
        @test nc.geoelectric.current_time_utc == jdt_str(payloadd.times[n - ngap])  # real sample, not "now"
        @test nc.geoelectric.age_minutes ≈ 3.0
        @test nc.geoelectric.current_vkm > 0.0
    end

    @testset "forecast log cache is schema- and path-safe" begin
        dir = mktempdir(); p1 = joinpath(dir, "one.csv"); p2 = joinpath(dir, "two.csv")
        write(p1, "value\n1\n"); write(p2, "value\n2\n")
        _LOG_CACHE[] = nothing
        a = get_log(p1)
        @test a.value == [1]                         # absent configured time columns do not throw
        @test build_status(a).available == false
        @test build_forecast(a).available == false
        @test isempty(build_history(a).rows)
        cached_cycle = latest_cycle(a)
        cached_verified = verified_rows(a)
        cached_history = build_history(a)
        @test latest_cycle(a) === cached_cycle
        @test verified_rows(a) === cached_verified
        @test build_history(a).rows === cached_history.rows
        # Forge equal metadata to isolate path identity: p2 must never reuse p1's frame.
        _LOG_CACHE[] = (_log_file_identity(p1), a)
        @test get_log(p2).value == [2]
        @test nrow(get_log(joinpath(dir, "missing.csv"))) == 0  # another path gets no stale frame

        # Same-size in-place replacement with restored mtime must still invalidate
        # the cache through ctime/inode identity.
        _LOG_CACHE[] = nothing
        cached = get_log(p1)
        @test latest_cycle(cached) !== cached_cycle
        @test verified_rows(cached) !== cached_verified
        @test build_history(cached).rows !== cached_history.rows
        original = stat(p1)
        timestamp_reference = joinpath(dir, "timestamp-reference.csv")
        run(`cp -p $p1 $timestamp_reference`)
        sleep(0.002)
        write(p1, "value\n9\n")
        run(`touch -r $timestamp_reference $p1`)
        @test mtime(p1) == original.mtime
        @test get_log(p1).value == [9]
        @test cached.value == [1]
        _LOG_CACHE[] = nothing
    end

    @testset "invalid forecast cycles fail closed as unavailable" begin
        issue = now(UTC) - Minute(10)
        base = DataFrame(
            issue_time_utc_dt=[issue], latest_solar_wind_utc_dt=[issue],
            latest_dst_time_utc_dt=[issue - Hour(1)],
            target_time_utc_dt=[issue + Hour(1)], horizon_hours=[1.0],
            latest_dst_nt=[-20.0], observation_dst_nt=[missing],
            served_pred_dst_nt=Union{Missing,Float64}[missing],
            served_pred_dst_ci05_nt=Union{Missing,Float64}[missing],
            served_pred_dst_ci95_nt=Union{Missing,Float64}[missing],
        )
        @test build_status(base).available == false
        @test build_forecast(base).available == false
        @test build_alerts(base).active == false
        for bad in (NaN, Inf, -Inf)
            mutated = copy(base)
            mutated.served_pred_dst_nt[1] = bad
            mutated.served_pred_dst_ci05_nt[1] = bad
            mutated.served_pred_dst_ci95_nt[1] = bad
            @test build_status(mutated).available == false
            @test build_forecast(mutated).available == false
            @test build_alerts(mutated).active == false
        end
        reversed = copy(base)
        reversed.served_pred_dst_nt[1] = -30.0
        reversed.served_pred_dst_ci05_nt[1] = -20.0
        reversed.served_pred_dst_ci95_nt[1] = -40.0
        @test build_status(reversed).available == false
        @test build_forecast(reversed).available == false
    end

    @testset "history and calibration survive finite extreme residuals" begin
        @test _stable_rmse_or_nothing(
            fill(floatmax(Float64) / 2, 2), zeros(2),
        ) == floatmax(Float64) / 2
        @test _stable_rmse_or_nothing(
            [-floatmax(Float64)], [floatmax(Float64)],
        ) === nothing

        issue = now(UTC) - Hour(2)
        extreme = DataFrame(
            issue_time_utc_dt=[issue],
            latest_solar_wind_utc_dt=[issue],
            latest_dst_time_utc_dt=[issue],
            target_time_utc_dt=[issue + Hour(1)],
            horizon_hours=[1.0],
            latest_dst_nt=[-20.0],
            observation_dst_nt=[-floatmax(Float64)],
            served_pred_dst_nt=[floatmax(Float64)],
            served_pred_dst_ci05_nt=[-floatmax(Float64)],
            served_pred_dst_ci95_nt=[floatmax(Float64)],
            model_version=[CURRENT_V2_MODEL_VERSION],
            sub_hourly_model_version=[CURRENT_V2_SERVED_MODEL_VERSION],
            interval_source=["extreme_fixture"],
        )
        calibration = calibration_summary(extreme)
        @test calibration.n_verified == 1
        @test calibration.rmse_nt === nothing
        history = build_history(extreme, 72)
        @test history.n == 1
        @test history.rmse_nt === nothing
        @test history.rmse_nt_all === nothing
    end

    @testset "future forecast cycles fail closed" begin
        future = now(UTC) + Day(1)
        df = DataFrame(
            issue_time_utc_dt=[future], latest_solar_wind_utc_dt=[future],
            latest_dst_time_utc_dt=[future - Hour(1)], target_time_utc_dt=[future + Hour(1)],
            horizon_hours=[1.0], latest_dst_nt=[-20.0], observation_dst_nt=[missing],
            served_pred_dst_nt=[-80.0], served_pred_dst_ci05_nt=[-110.0],
            served_pred_dst_ci95_nt=[-50.0], interval_source=["aci"], model_version=["v2.1"],
        )
        st = build_status(df)
        @test st.available == false && st.stale == true && st.invalid_future == true
        @test st.age_hours < -23
        @test build_alerts(df).active == false
        @test build_forecast(df).invalid_future == true
    end

    @testset "dB/dt uses elapsed time and cache windows are isolated" begin
        @test USGS_LIVE_DATA_TYPE == "adjusted"
        reference = now(UTC)
        times = jdt.([reference - Minute(4), reference - Minute(2)])
        @test _dbdt_series(times, [0.0, 20.0], [0.0, 0.0])[2] == 10.0
        @test isnan(_dbdt_series([times[1], times[1]], [0.0, 20.0], [0.0, 0.0])[2])
        @test isnan(_dbdt_series(reverse(times), [0.0, 20.0], [0.0, 0.0])[2])
        d = (times=times,
             values=[(metadata=(element="X",), values=[0.0, 20.0]),
                     (metadata=(element="Y",), values=[0.0, 0.0])],
             metadata=(intermagnet=(imo=(coordinates=[-77.0, 39.0, 0.0], name="Test"),),))
        station_row = _station_parse("TST", d; reference=reference)
        @test station_row.current_dbdt == 10.0
        @test station_row.data_type == "adjusted"

        # The impedance transform uses the actual uniform cadence, and refuses
        # an irregular cadence instead of treating every row as one minute.
        long_times = [reference - Minute(38) + Minute(2i) for i in 0:19]
        long_x = sin.(range(0, 4pi; length=20)) .* 10
        long_payload = (times=jdt.(long_times),
            values=[(metadata=(element="X",), values=long_x),
                    (metadata=(element="Y",), values=zeros(20))])
        nc2 = _compute_dbdt(
            "TST", 120; fetch_fn=(s, m) -> long_payload, reference=reference,
        )
        expected_geoe = _geoe_nowcast(long_x, zeros(20), 120.0)
        @test nc2.geoelectric !== nothing
        @test nc2.geoelectric.current_vkm ≈ round(expected_geoe.current; digits=3)
        @test !hasproperty(nc2.geoelectric, :tier)

        irregular_times = [reference - Minute(21) + Minute(i) +
                           (i >= 10 ? Minute(1) : Minute(0)) for i in 0:19]
        irregular = (times=jdt.(irregular_times),
            values=[(metadata=(element="X",), values=long_x),
                    (metadata=(element="Y",), values=zeros(20))])
        irregular_result = _compute_dbdt(
            "TST", 120; fetch_fn=(s, m) -> irregular, reference=reference,
        )
        @test irregular_result.available
        @test irregular_result.geoelectric === nothing
        malformed = (times=times, values=[(metadata=(foo="bar",), values=[0.0, 1.0])])
        @test !_compute_dbdt(
            "TST", 120; fetch_fn=(s, m) -> malformed, reference=reference,
        ).available

        empty!(_DBDT_CACHE)
        sample_time = jdt(reference - Minute(1))
        a = (station="FRD", available=true, window=60, current_time_utc=sample_time)
        b = (station="FRD", available=true, window=120, current_time_utc=sample_time)
        _DBDT_CACHE[("FRD", 60)] = (time(), a)
        _DBDT_CACHE[("FRD", 120)] = (time(), b)
        @test usgs_dbdt(station="FRD", minutes=60; reference=reference).window == 60
        @test usgs_dbdt(station="FRD", minutes=120; reference=reference).window == 120
        empty!(_DBDT_CACHE)

        empty!(_NET_CACHE)
        frd = (station="FRD", time_utc=sample_time); cmo = (station="CMO", time_utc=sample_time)
        _NET_CACHE[("FRD",)] = (time(), [frd])
        _NET_CACHE[("CMO",)] = (time(), [cmo])
        @test only(usgs_network(stations=["FRD"], reference=reference).stations).station == "FRD"
        @test only(usgs_network(stations=["CMO"], reference=reference).stations).station == "CMO"
        empty!(_NET_CACHE)
    end

    @testset "live source freshness, timestamp windows, and bounded caches" begin
        reference = DateTime(2026, 7, 14, 12)
        payload(times, x) =
            (times=jdt.(times),
             values=[(metadata=(element="X",), values=x),
                     (metadata=(element="Y",), values=zeros(length(x)))],
             metadata=(intermagnet=(imo=(coordinates=[-77.0, 39.0, 0.0], name="Test"),),))

        # Thirty-one two-minute samples span 60 minutes. A spike 58 minutes ago must not
        # leak into the trailing 30-minute maximum merely because it is among the last 30 rows.
        times = [reference - Minute(60) + Minute(2i) for i in 0:30]
        increments = vcat(200.0, ones(29))
        x = vcat(0.0, cumsum(increments))
        d = payload(times, x)
        nc = _compute_dbdt("TST", 120; fetch_fn=(s, m) -> d, reference=reference)
        @test nc.available && nc.current_dbdt == 0.5
        @test nc.data_type == "adjusted"
        @test nc.max30_dbdt == 0.5
        @test nc.n_minutes == 30
        @test all(parse_dt(p.t) > reference - Minute(60) for p in nc.series)
        @test _station_parse("TST", d; reference=reference).max_dbdt == 0.5

        stale = payload([reference - Minute(22), reference - Minute(20)], [0.0, 20.0])
        sr = _compute_dbdt("TST", 120; fetch_fn=(s, m) -> stale, reference=reference)
        @test !sr.available && sr.stale && !sr.invalid_future && sr.age_minutes == 20.0
        @test _station_parse("TST", stale; reference=reference) === nothing
        future = payload([reference + Minute(3), reference + Minute(5)], [0.0, 20.0])
        fr = _compute_dbdt("TST", 120; fetch_fn=(s, m) -> future, reference=reference)
        @test !fr.available && fr.stale && fr.invalid_future && fr.age_minutes == -5.0
        @test _station_parse("TST", future; reference=reference) === nothing

        # An unavailable refresh preserves the last source observation while advancing the
        # retry clock, so repeated requests cannot hammer the failed upstream service.
        empty!(_DBDT_CACHE)
        good = (station="FRD", available=true, current_time_utc=jdt(reference - Minute(1)))
        old_fetch = time() - DBDT_TTL - 1
        _DBDT_CACHE[("FRD", 120)] = (old_fetch, good)
        fallback = usgs_dbdt(station="frd", minutes=120, reference=reference,
                             compute_fn=(s, m) -> (station=s, available=false),
                             wait_timeout=30.0)
        @test fallback.available && fallback.cached
        @test _DBDT_CACHE[("FRD", 120)][1] > old_fetch
        _DBDT_CACHE[("FRD", 120)] = (old_fetch,
            (station="FRD", available=true, current_time_utc=jdt(reference - Minute(20))))
        rejected = usgs_dbdt(station="FRD", minutes=120, reference=reference,
                             compute_fn=(s, m) -> (station=s, available=false),
                             wait_timeout=30.0)
        @test !rejected.available && rejected.stale

        # An overlapping slower refresh cannot overwrite a newer observation.
        empty!(_DBDT_CACHE)
        newer = (station="FRD", available=true,
                 current_time_utc=jdt(reference - Minute(1)))
        older = (station="FRD", available=true,
                 current_time_utc=jdt(reference - Minute(2)))
        raced = usgs_dbdt(
            station="FRD", minutes=120, reference=reference,
            compute_fn=(s, m) -> begin
                _DBDT_CACHE[("FRD", 120)] = (time(), newer)
                older
            end,
            wait_timeout=30.0,
        )
        @test raced.cached && raced.current_time_utc == newer.current_time_utc
        @test _DBDT_CACHE[("FRD", 120)][2].current_time_utc == newer.current_time_utc
        empty!(_DBDT_CACHE)
        @test_throws InterruptException usgs_dbdt(
            station="FRD", minutes=120, reference=reference,
            compute_fn=(s, m) -> throw(InterruptException()), wait_timeout=30.0)
        @test_throws ArgumentError usgs_dbdt(station="bad/station")
        @test_throws ArgumentError usgs_dbdt(station="FRD", minutes=1)

        empty!(_DBDT_CACHE)
        for i in 1:(DBDT_CACHE_MAX + 5)
            _bounded_time_cache_put!(_DBDT_CACHE, ("S$i", 120), (Float64(i), good), DBDT_CACHE_MAX)
        end
        @test length(_DBDT_CACHE) == DBDT_CACHE_MAX
        @test !haskey(_DBDT_CACHE, ("S1", 120))

        empty!(_NET_CACHE)
        row = (station="FRD", time_utc=jdt(reference - Minute(1)))
        old_net_fetch = time() - NET_TTL - 1
        _NET_CACHE[("FRD",)] = (old_net_fetch, [row])
        net = usgs_network(stations=["frd"], brief_fn=s -> nothing, reference=reference,
                           wait_timeout=30.0)
        @test net.available && net.cached && net.n_stations == 1
        @test _NET_CACHE[("FRD",)][1] > old_net_fetch
        _NET_CACHE[("FRD",)] = (old_net_fetch,
                                 [(station="FRD", time_utc=jdt(reference - Minute(20)))])
        net_stale = usgs_network(stations=["FRD"], brief_fn=s -> nothing,
                                 reference=reference, wait_timeout=30.0)
        @test !net_stale.available && net_stale.stale && isempty(net_stale.stations)

        # Partial and overlapping refreshes retain current missing stations and
        # never regress a station's observation timestamp.
        frd_old = (station="FRD", time_utc=jdt(reference - Minute(3)))
        frd_new = (station="FRD", time_utc=jdt(reference - Minute(1)))
        cmo_current = (station="CMO", time_utc=jdt(reference - Minute(2)))
        _NET_CACHE[("FRD", "CMO")] =
            (time() - NET_TTL - 1, [frd_new, cmo_current])
        partial = usgs_network(
            stations=["FRD", "CMO"], reference=reference,
            brief_fn=s -> s == "FRD" ? frd_old : nothing,
            wait_timeout=30.0,
        )
        @test partial.available && partial.cached && partial.n_stations == 2
        @test only(filter(r -> r.station == "FRD", partial.stations)).time_utc ==
              frd_new.time_utc
        @test only(filter(r -> r.station == "CMO", partial.stations)).time_utc ==
              cmo_current.time_utc
        empty!(_NET_CACHE)
        @test_throws InterruptException usgs_network(
            stations=["FRD"], brief_fn=s -> throw(InterruptException()), reference=reference,
            wait_timeout=30.0)
        @test_throws ArgumentError usgs_network(stations=String[])
        @test_throws ArgumentError usgs_network(stations=["bad/station"])

        empty!(_NET_CACHE)
        for i in 1:(NET_CACHE_MAX + 5)
            _bounded_time_cache_put!(_NET_CACHE, ("S$i",), (Float64(i), [row]), NET_CACHE_MAX)
        end
        @test length(_NET_CACHE) == NET_CACHE_MAX
        @test !haskey(_NET_CACHE, ("S1",))
        empty!(_DBDT_CACHE); empty!(_NET_CACHE)
    end

    @testset "USGS refreshes coalesce by key and retry after failure" begin
        reference = DateTime(2026, 7, 14, 12)
        sample_time = jdt(reference - Minute(1))

        lock(_DBDT_LOCK) do
            empty!(_DBDT_CACHE)
            empty!(_DBDT_REFRESH_TASKS)
        end
        calls = Threads.Atomic{Int}(0)
        release = Channel{Nothing}(6)
        compute = (station, minutes) -> begin
            Threads.atomic_add!(calls, 1)
            take!(release)
            (station=station, available=true, window=minutes,
             current_time_utc=sample_time)
        end
        requests = [Threads.@spawn usgs_dbdt(
            station="FRD", minutes=120, compute_fn=compute, reference=reference,
            wait_timeout=30.0,
        ) for _ in 1:6]
        started = Base.timedwait(() -> calls[] >= 1, 2.0; pollint=0.01)
        sleep(0.05) # let every caller join the published in-flight task
        foreach(_ -> put!(release, nothing), 1:6)
        results = fetch.(requests)
        @test started === :ok
        @test calls[] == 1
        @test all(r -> r.available && r.station == "FRD", results)
        @test lock(_DBDT_LOCK) do; isempty(_DBDT_REFRESH_TASKS); end

        # Distinct keys retain distinct in-flight identities, but their external work shares the
        # global upstream slot so blocking external work occupies at most one server thread.
        lock(_DBDT_LOCK) do
            empty!(_DBDT_CACHE)
            empty!(_DBDT_REFRESH_TASKS)
        end
        distinct_calls = Threads.Atomic{Int}(0)
        distinct_release = Channel{Nothing}(2)
        distinct_compute = (station, minutes) -> begin
            Threads.atomic_add!(distinct_calls, 1)
            take!(distinct_release)
            (station=station, available=true, window=minutes,
             current_time_utc=sample_time)
        end
        distinct = [
            Threads.@spawn(usgs_dbdt(station="FRD", minutes=60,
                compute_fn=distinct_compute, reference=reference, wait_timeout=30.0)),
            Threads.@spawn(usgs_dbdt(station="CMO", minutes=120,
                compute_fn=distinct_compute, reference=reference, wait_timeout=30.0)),
        ]
        published = Base.timedwait(
            () -> lock(_DBDT_LOCK) do; length(_DBDT_REFRESH_TASKS) == 2; end,
            2.0; pollint=0.01,
        )
        first_started = Base.timedwait(() -> distinct_calls[] == 1, 2.0; pollint=0.01)
        put!(distinct_release, nothing)
        second_started = Base.timedwait(() -> distinct_calls[] == 2, 2.0; pollint=0.01)
        put!(distinct_release, nothing)
        distinct_results = fetch.(distinct)
        @test published === :ok
        @test first_started === :ok
        @test second_started === :ok
        @test distinct_calls[] == 2
        @test Set(r.station for r in distinct_results) == Set(("FRD", "CMO"))

        # One failing worker is shared. Its cleanup removes the in-flight task, while the
        # short negative-cache TTL suppresses immediate hammering and permits a later retry.
        lock(_DBDT_LOCK) do
            empty!(_DBDT_CACHE)
            empty!(_DBDT_REFRESH_TASKS)
        end
        failed_calls = Threads.Atomic{Int}(0)
        failed_release = Channel{Nothing}(4)
        failing_compute = (station, minutes) -> begin
            Threads.atomic_add!(failed_calls, 1)
            take!(failed_release)
            error("injected dB/dt refresh failure")
        end
        failures = [Threads.@spawn usgs_dbdt(
            station="FRD", minutes=120, compute_fn=failing_compute, reference=reference,
            wait_timeout=30.0,
        ) for _ in 1:4]
        failure_started = Base.timedwait(() -> failed_calls[] >= 1, 2.0; pollint=0.01)
        sleep(0.05)
        foreach(_ -> put!(failed_release, nothing), 1:4)
        failed_results = fetch.(failures)
        @test failure_started === :ok
        @test failed_calls[] == 1
        @test all(r -> !r.available && occursin("injected", r.error), failed_results)
        @test lock(_DBDT_LOCK) do; isempty(_DBDT_REFRESH_TASKS); end
        retry_calls = Ref(0)
        retried = usgs_dbdt(
            station="FRD", minutes=120, reference=reference, wait_timeout=30.0,
            compute_fn=(station, minutes) -> begin
                retry_calls[] += 1
                (station=station, available=true, current_time_utc=sample_time)
            end,
        )
        @test !retried.available
        @test retry_calls[] == 0
        lock(_DBDT_LOCK) do
            cached = _DBDT_CACHE[("FRD", 120)]
            _DBDT_CACHE[("FRD", 120)] = (time() - DBDT_TTL - 1, cached[2])
        end
        retried = usgs_dbdt(
            station="FRD", minutes=120, reference=reference, wait_timeout=30.0,
            compute_fn=(station, minutes) -> begin
                retry_calls[] += 1
                (station=station, available=true, current_time_utc=sample_time)
            end,
        )
        @test retried.available
        @test retry_calls[] == 1

        # A stuck owner cannot pin request tasks; it may finish and clean up in the background.
        lock(_DBDT_LOCK) do
            empty!(_DBDT_CACHE)
            empty!(_DBDT_REFRESH_TASKS)
        end
        timeout_release = Channel{Nothing}(1)
        elapsed = @elapsed pending = usgs_dbdt(
            station="FRD", minutes=120, reference=reference, wait_timeout=0.02,
            compute_fn=(station, minutes) -> begin
                take!(timeout_release)
                (station=station, available=true, current_time_utc=sample_time)
            end,
        )
        @test !pending.available
        @test elapsed < 0.5
        put!(timeout_release, nothing)
        @test Base.timedwait(
            () -> lock(_DBDT_LOCK) do; isempty(_DBDT_REFRESH_TASKS); end,
            2.0; pollint=0.01,
        ) === :ok
        @test_throws ArgumentError usgs_dbdt(wait_timeout=-1)
        @test_throws ArgumentError usgs_dbdt(wait_timeout=Inf)

        lock(_NET_LOCK) do
            empty!(_NET_CACHE)
            empty!(_NET_REFRESH_TASKS)
        end
        network_calls = Threads.Atomic{Int}(0)
        brief = station -> begin
            Threads.atomic_add!(network_calls, 1)
            sleep(0.05)
            (station=station, time_utc=sample_time)
        end
        network_requests = [Threads.@spawn usgs_network(
            stations=["FRD", "CMO"], brief_fn=brief, reference=reference,
            wait_timeout=30.0,
        ) for _ in 1:5]
        network_results = fetch.(network_requests)
        @test network_calls[] == 2 # one call per station, not per HTTP requester
        @test all(r -> r.available && r.n_stations == 2, network_results)
        @test lock(_NET_LOCK) do; isempty(_NET_REFRESH_TASKS); end

        lock(_NET_LOCK) do
            empty!(_NET_CACHE)
            empty!(_NET_REFRESH_TASKS)
        end
        network_fail_calls = Threads.Atomic{Int}(0)
        network_release = Channel{Nothing}(4)
        bad_brief = station -> begin
            Threads.atomic_add!(network_fail_calls, 1)
            take!(network_release)
            error("injected network refresh failure")
        end
        network_failures = [Threads.@spawn usgs_network(
            stations=["FRD"], brief_fn=bad_brief, reference=reference,
            wait_timeout=30.0,
        ) for _ in 1:4]
        network_failure_started = Base.timedwait(
            () -> network_fail_calls[] >= 1, 2.0; pollint=0.01,
        )
        sleep(0.05)
        foreach(_ -> put!(network_release, nothing), 1:4)
        network_failed_results = fetch.(network_failures)
        @test network_failure_started === :ok
        @test network_fail_calls[] == 1
        @test all(r -> !r.available, network_failed_results)
        @test lock(_NET_LOCK) do; isempty(_NET_REFRESH_TASKS); end
        network_retry_calls = Ref(0)
        network_retry = usgs_network(
            stations=["FRD"], reference=reference, wait_timeout=30.0,
            brief_fn=station -> begin
                network_retry_calls[] += 1
                (station=station, time_utc=sample_time)
            end,
        )
        @test !network_retry.available
        @test network_retry_calls[] == 0
        lock(_NET_LOCK) do
            cached = _NET_CACHE[("FRD",)]
            _NET_CACHE[("FRD",)] = (time() - NET_TTL - 1, cached[2])
        end
        network_retry = usgs_network(
            stations=["FRD"], reference=reference, wait_timeout=30.0,
            brief_fn=station -> begin
                network_retry_calls[] += 1
                (station=station, time_utc=sample_time)
            end,
        )
        @test network_retry.available
        @test network_retry_calls[] == 1
        @test_throws ArgumentError usgs_network(wait_timeout=-1)
        @test_throws ArgumentError usgs_network(wait_timeout=Inf)

        lock(_DBDT_LOCK) do
            empty!(_DBDT_CACHE)
            empty!(_DBDT_REFRESH_TASKS)
        end
        lock(_NET_LOCK) do
            empty!(_NET_CACHE)
            empty!(_NET_REFRESH_TASKS)
        end
    end

    @testset "SWPC assessments require current source timestamps" begin
        reference = DateTime(2026, 7, 14, 12)
        snapshot_at(t) = (
            source="test", available=true,
            scales=(G="5", time_utc=jdt(t)),
            kp=(value=9.0, time_utc=jdt(t)),
            solar_wind=Dict{Symbol,Any}(
                :available=>true, :bz_gsm_nt=>-25.0, :speed_kms=>800.0,
                :mag_time_utc=>jdt(t), :plasma_time_utc=>jdt(t)),
            alerts=NamedTuple[])

        current = upstream_assessment(snapshot_at(reference - Minute(1)); reference=reference)
        @test current.available && current.elevated && length(current.reasons) == 4
        @test !current.solar_wind_stale && current.solar_wind_age_min == 1.0

        stale = upstream_assessment(snapshot_at(reference - Day(1)); reference=reference)
        @test !stale.available && !stale.elevated && isempty(stale.reasons)
        @test stale.scales_stale && stale.kp_stale && stale.solar_wind_stale
        @test !stale.invalid_future

        future = upstream_assessment(snapshot_at(reference + Minute(10)); reference=reference)
        @test !future.available && !future.elevated && isempty(future.reasons)
        @test future.invalid_future && future.mag_stale && future.plasma_stale
        @test future.mag_age_min == -10.0

        # A future-poisoned RTSW row must not displace the newest valid current row.
        rows = JSON3.read("""[
          {"time_tag":"2026-07-14T12:10:00","active":true,"bt":20.0,"bz_gsm":-30.0},
          {"time_tag":"2026-07-14T11:59:00","active":true,"bt":5.0,"bz_gsm":-2.0}
        ]""")
        chosen = _rtsw_latest(rows, [:bt, :bz_gsm]; reference=reference)
        @test String(chosen.time_tag) == "2026-07-14T11:59:00"

        malformed_rows = Any[1, Dict(:time_tag=>"bad"),
            Dict(:time_tag=>"2026-07-14T11:58:00", :active=>"false",
                 :bt=>4.0, :bz_gsm=>-1.0)]
        @test _rtsw_latest(malformed_rows, [:bt, :bz_gsm]; reference=reference) ==
              malformed_rows[3]

        kp_rows = [
            Dict(:time_tag=>"2026-07-14T09:00:00", :Kp=>"5.0"),
            Dict(:time_tag=>"2026-07-14T12:10:00", :Kp=>"9.0"),
            Dict(:time_tag=>"bad", :Kp=>"8.0"),
            Dict(:time_tag=>"2026-07-14T11:00:00", :Kp=>nothing),
        ]
        kp_current = _parse_swpc_kp(kp_rows; reference=reference)
        @test kp_current.value == 5.0
        @test parse_dt(kp_current.time_utc) == DateTime(2026, 7, 14, 9)

        # An unavailable refresh preserves the last source snapshot but advances the retry clock,
        # preventing every request from hammering an unavailable upstream service.
        good = snapshot_at(reference - Minute(1)); old_fetch = time() - SWPC_TTL - 1
        lock(_SWPC_LOCK) do
            _SWPC_CACHE[] = (old_fetch, good)
            _SWPC_REFRESH_TASK[] = current_task()
        end
        before_refresh = time()
        out = _run_swpc_refresh(build_fn=() -> (source="test", available=false))
        @test out == good
        @test before_refresh <= _SWPC_CACHE[][1] <= time()
        @test _SWPC_CACHE[][2] == good
        lock(_SWPC_LOCK) do; _SWPC_REFRESH_TASK[] = current_task(); end
        @test_throws InterruptException _run_swpc_refresh(
            build_fn=() -> throw(InterruptException()))
        @test _SWPC_REFRESH_TASK[] === nothing
        @test _SWPC_CACHE[][1] >= before_refresh
        lock(_SWPC_LOCK) do
            _SWPC_CACHE[] = nothing
            _SWPC_REFRESH_TASK[] = nothing
        end
    end

    @testset "earth model validation rejects unsafe layer shapes and values" begin
        @test_throws ArgumentError surface_impedance(1.0, Float64[], Float64[])
        @test_throws ArgumentError surface_impedance(1.0, [100.0, 10.0], Float64[])
        @test_throws ArgumentError surface_impedance(1.0, [0.0], Float64[])
        @test_throws ArgumentError surface_impedance(1.0, [NaN], Float64[])
        @test_throws ArgumentError surface_impedance(1.0, [100.0, 10.0], [-1.0])
        @test_throws ArgumentError surface_impedance(1.0, [100.0, 10.0], [Inf])
        @test surface_impedance(0.0, [100.0], Float64[]) == 0.0im
    end

    @testset "SWPC refresh entry points share one in-flight task" begin
        gate = Channel{Nothing}(0)
        sentinel = (source="test", available=false)
        held = Threads.@spawn (take!(gate); sentinel)
        lock(_SWPC_LOCK) do
            _SWPC_CACHE[] = nothing
            _SWPC_REFRESH_TASK[] = held
        end
        @test swpc_snapshot_cached_or_refresh().available == false
        @test lock(_SWPC_LOCK) do; _start_swpc_refresh_locked() === held; end
        elapsed = @elapsed bounded = swpc_snapshot(wait_timeout=0.02)
        @test bounded.available == false
        @test elapsed < 0.5
        @test _SWPC_REFRESH_TASK[] === held
        put!(gate, nothing)
        @test fetch(held) == sentinel
        lock(_SWPC_LOCK) do
            _SWPC_CACHE[] = (time(), sentinel)
        end
        @test swpc_snapshot(wait_timeout=0.1) == sentinel
        @test_throws ArgumentError swpc_snapshot(wait_timeout=-1)
        @test_throws ArgumentError swpc_snapshot(wait_timeout=Inf)
        lock(_SWPC_LOCK) do
            _SWPC_CACHE[] = nothing
            _SWPC_REFRESH_TASK[] = nothing
        end
    end

    @testset "live-source API routes never wait for third-party refreshes" begin
        # A held refresh is a deterministic oracle for the request contract: each route must
        # return its unavailable/cached payload promptly and leave the one in-flight worker intact.
        swpc_gate = Channel{Nothing}(0)
        swpc_held = Threads.@spawn (take!(swpc_gate); (source="test", available=false))
        lock(_SWPC_LOCK) do
            _SWPC_CACHE[] = nothing
            _SWPC_REFRESH_TASK[] = swpc_held
        end
        try
            elapsed = @elapsed response = api_handler("/api/swpc", "", "unused.csv")
            payload = JSON3.read(String(response.body))
            @test response.status == 200
            @test payload.available == false
            @test elapsed < 1.5
            @test _SWPC_REFRESH_TASK[] === swpc_held
        finally
            put!(swpc_gate, nothing)
            fetch(swpc_held)
            lock(_SWPC_LOCK) do
                _SWPC_CACHE[] = nothing
                _SWPC_REFRESH_TASK[] = nothing
            end
        end

        dbdt_key = ("FRD", 120)
        dbdt_gate = Channel{Nothing}(0)
        dbdt_held = Threads.@spawn (take!(dbdt_gate); nothing)
        lock(_DBDT_LOCK) do
            empty!(_DBDT_CACHE)
            empty!(_DBDT_REFRESH_TASKS)
            _DBDT_REFRESH_TASKS[dbdt_key] = dbdt_held
        end
        try
            elapsed = @elapsed response = api_handler("/api/dbdt", "station=FRD", "unused.csv")
            payload = JSON3.read(String(response.body))
            @test response.status == 200
            @test payload.available == false
            @test elapsed < 1.5
            @test _DBDT_REFRESH_TASKS[dbdt_key] === dbdt_held

            # An explicit unsupported or malformed station is rejected. It must never alias to
            # FRD, start a new worker, or increase the bounded cache/in-flight key sets.
            handler = make_handler("unused.csv")
            for query in (
                "station=ZZZ", "station=FRD%20", "station=bad%2Fstation", "station=",
                "station=%", "station=%GG", "station=%FF",
            )
                invalid = handler(HTTP.Request("GET", "/api/dbdt?$query"))
                invalid_payload = JSON3.read(String(invalid.body))
                @test invalid.status == 400
                @test invalid_payload.available == false
                expected_error = query in ("station=%", "station=%GG", "station=%FF") ?
                                 "malformed query" : "unsupported station"
                @test String(invalid_payload.error) == expected_error
                @test collect(keys(_DBDT_REFRESH_TASKS)) == [dbdt_key]
                @test isempty(_DBDT_CACHE)
            end
        finally
            put!(dbdt_gate, nothing)
            fetch(dbdt_held)
            lock(_DBDT_LOCK) do
                empty!(_DBDT_CACHE)
                empty!(_DBDT_REFRESH_TASKS)
            end
        end

        # The dashboard's implicit station may fall back only to the other calibrated station;
        # an explicit request above remains pinned to FRD.
        reference = now(UTC)
        cmo = (
            station="CMO", available=true, stale=false, invalid_future=false,
            age_minutes=0.0, current_dbdt=1.0,
            current_tier=(level=0, label="Below 18 nT/min"),
            current_time_utc=jdt(reference), max30_dbdt=2.0,
            max30_tier=(level=0, label="Below 18 nT/min"), thresholds=collect(PULK),
            exceedances=NamedTuple[], geoelectric=nothing, n_minutes=1,
            series=[(t=jdt(reference), dbdt=1.0)],
        )
        lock(_DBDT_LOCK) do
            _DBDT_CACHE[("FRD", 120)] = (time(), (station="FRD", available=false))
            _DBDT_CACHE[("CMO", 120)] = (time(), cmo)
        end
        lock(_SWPC_LOCK) do
            _SWPC_CACHE[] = (time(), _unavailable_swpc_snapshot())
        end
        fallback = JSON3.read(String(api_handler("/api/dbdt", "", "unused.csv").body))
        @test fallback.available == true
        @test String(fallback.station) == "CMO"
        @test fallback.forecast === nothing
        @test fallback.forecast_status.available == false
        @test occursin("not aligned", String(fallback.forecast_status.reason))
        lock(_DBDT_LOCK) do
            empty!(_DBDT_CACHE)
        end
        lock(_SWPC_LOCK) do
            _SWPC_CACHE[] = nothing
            _SWPC_REFRESH_TASK[] = nothing
        end

        network_key = Tuple(unique(NET_STATIONS))
        network_gate = Channel{Nothing}(0)
        network_held = Threads.@spawn (take!(network_gate); nothing)
        lock(_NET_LOCK) do
            empty!(_NET_CACHE)
            empty!(_NET_REFRESH_TASKS)
            _NET_REFRESH_TASKS[network_key] = network_held
        end
        try
            elapsed = @elapsed response = api_handler("/api/network", "", "unused.csv")
            payload = JSON3.read(String(response.body))
            @test response.status == 200
            @test payload.available == false
            @test payload.n_stations == 0
            @test elapsed < 1.5
            @test _NET_REFRESH_TASKS[network_key] === network_held
        finally
            put!(network_gate, nothing)
            fetch(network_held)
            lock(_NET_LOCK) do
                empty!(_NET_CACHE)
                empty!(_NET_REFRESH_TASKS)
            end
        end

        history_handler = make_handler("unused.csv")
        for query in ("hours=%", "hours=%GG", "hours=%FF")
            malformed = history_handler(HTTP.Request("GET", "/api/history?$query"))
            malformed_payload = JSON3.read(String(malformed.body))
            @test malformed.status == 400
            @test malformed_payload.available == false
            @test String(malformed_payload.error) == "malformed query"
        end
        nonfinite = history_handler(HTTP.Request("GET", "/api/history?hours=NaN"))
        @test nonfinite.status == 200
        @test JSON3.read(String(nonfinite.body)).hours == 72.0
    end

    @testset "NOAA and USGS workers share one upstream execution slot" begin
        lock(_SWPC_LOCK) do
            _SWPC_CACHE[] = nothing
            _SWPC_REFRESH_TASK[] = nothing
        end
        lock(_DBDT_LOCK) do
            empty!(_DBDT_CACHE)
            empty!(_DBDT_REFRESH_TASKS)
        end
        active = Threads.Atomic{Int}(0)
        maximum_active = Threads.Atomic{Int}(0)
        calls = Threads.Atomic{Int}(0)
        release = Channel{Nothing}(2)
        function gated_probe(value)
            Threads.atomic_add!(calls, 1)
            now_active = Threads.atomic_add!(active, 1) + 1
            while true
                prior = maximum_active[]
                prior >= now_active && break
                Threads.atomic_cas!(maximum_active, prior, now_active) == prior && break
            end
            try
                take!(release)
                return value
            finally
                Threads.atomic_add!(active, -1)
            end
        end

        swpc_task = Threads.@spawn _run_swpc_refresh(
            build_fn=() -> gated_probe((source="test", available=true)),
        )
        dbdt_task = Threads.@spawn usgs_dbdt(
            station="FRD", minutes=120, wait_timeout=30.0,
            compute_fn=(station, minutes) -> gated_probe((
                station=station, available=true, current_time_utc=jdt(now(UTC)),
            )),
        )
        @test Base.timedwait(() -> calls[] == 1, 2.0; pollint=0.01) === :ok
        @test maximum_active[] == 1
        put!(release, nothing)
        @test Base.timedwait(() -> calls[] == 2, 2.0; pollint=0.01) === :ok
        @test maximum_active[] == 1
        put!(release, nothing)
        @test fetch(swpc_task).available == true
        @test fetch(dbdt_task).available == true
        @test active[] == 0
        @test maximum_active[] == 1
        lock(_SWPC_LOCK) do
            _SWPC_CACHE[] = nothing
            _SWPC_REFRESH_TASK[] = nothing
        end
        lock(_DBDT_LOCK) do
            empty!(_DBDT_CACHE)
            empty!(_DBDT_REFRESH_TASKS)
        end
    end

    @testset "launchers reserve capacity for background refresh" begin
        app_root = normpath(joinpath(@__DIR__, ".."))
        for launcher in ("run.sh", "desktop.sh")
            source = read(joinpath(app_root, launcher), String)
            @test occursin(r"SWM_JULIA_THREADS:-4", source)
            @test occursin(r"--threads=\"\$JULIA_THREADS\"", source)
            @test occursin("VERSION >= v\"1.12.6\"", source)
        end
        @test !occursin("Pkg.instantiate()' >/dev/null 2>&1 || true",
                        read(joinpath(app_root, "desktop.sh"), String))
        desktop_source = read(joinpath(app_root, "desktop.sh"), String)
        @test occursin("Pkg.instantiate()' >/dev/null\n", desktop_source)
        @test occursin("press Ctrl-C here to stop the backend", desktop_source)
        @test !occursin("close the window or press Ctrl-C", desktop_source)
        dockerfile = read(joinpath(app_root, "Dockerfile"), String)
        @test occursin("FROM julia:1.12.6-bookworm", dockerfile)
        @test occursin(
            "julia = \"1.12.6\"",
            read(joinpath(app_root, "Project.toml"), String),
        )
        @test occursin(r"JULIA_NUM_THREADS=4", dockerfile)
        @test occursin("COPY data/operational_validation", dockerfile)
        @test occursin("COPY app/models ./models", dockerfile)
        @test occursin(
            "SOLARSINDY_OPERATIONAL_EVIDENCE_DIR=/app/data/operational_validation",
            dockerfile,
        )
        @test occursin("-f app/Dockerfile .", dockerfile)
    end

    @testset "threat watch remains in layout flow" begin
        css = read(joinpath(@__DIR__, "..", "public", "style.css"), String)
        js = read(joinpath(@__DIR__, "..", "public", "app.js"), String)
        html = read(joinpath(@__DIR__, "..", "public", "index.html"), String)
        readme = read(joinpath(@__DIR__, "..", "README.md"), String)
        served_source = join(
            (read(joinpath(APPSRC, name), String)
             for name in sort(filter(name -> endswith(name, ".jl"), readdir(APPSRC)))),
            "\n",
        )
        @test occursin("grid-template-areas: \"left right\" \"watch watch\"", css)
        @test occursin("grid-template-areas: \"left\" \"right\" \"watch\"", css)
        @test occursin("grid-area: watch; position: static", css)
        @test !occursin("position: absolute; right: 18px; bottom: 12px", css)
        @test occursin("justify-self: stretch", css)
        @test occursin("wf.textContent = \"\"", js)
        @test occursin("wf.classList.add(\"hidden\")", js)
        @test occursin("renderThreat(null);", js)
        @test count(occursin("delete upd.dataset.reltime", line)
                    for line in eachline(IOBuffer(js))) == 2
        @test count(occursin("upd.textContent = \"forecast unavailable\"", line)
                    for line in eachline(IOBuffer(js))) == 2
        @test occursin("history = history || { rows: [] };", js)
        @test occursin("if (!history || !Array.isArray(history.rows))", js)
        @test occursin("\$(\"dbdt-forecast\").innerHTML = \"\";", js)
        @test occursin("if (status.available === false) \$(\"health-dot\")", js)
        @test occursin("const lvl = Math.max(pointLevel, watchLevel);", js)
        @test occursin("const alertLabel = watchLevel > pointLevel ? th.watch_label : th.label;", js)
        @test occursin("const body = watchLevel > pointLevel", js)
        @test occursin("A displayed calibrated 90% interval extends to", js)
        @test occursin("interval_lower_edge_min_dst_nt", js)
        @test !occursin("? status.threat.level : 0", js)
        for label in ("Forecast: V2.1 (", "Verified V2.1", "V2.1 90% coverage",
                      "V2.1 RMSE nT", "V2.1 verified", "headline score is V2.1")
            @test occursin(label, js)
        end
        for retired_label in ("Forecast: V2 (", "name:\"Verified V2\"",
                              "V2 90% coverage", "V2 RMSE nT", "V2 verified",
                              "headline score is V2. Live")
            @test !occursin(retired_label, js)
        end
        @test occursin("package's V2.1 forecaster", readme)
        @test occursin("V2.1 and every baseline", readme)
        @test occursin("project's **V2.1** nowcaster", readme)
        @test !occursin("package's V2 forecaster", readme)
        @test !occursin("V2 and every baseline", readme)
        for unsupported in ("GIC driver", "GIC forecast", "GIC alert thresholds",
                            "Pulkkinen tier", "GIC risk", "storm cannot be excluded",
                            "90% lower bound")
            @test !occursin(unsupported, js)
            @test !occursin(unsupported, html)
            @test !occursin(unsupported, served_source)
        end
        @test occursin("GIC-hazard indicator", html)
        @test occursin("90% interval lower edge", html)
        for metric in ("cur-dst", "worst-dst", "horizon")
            @test occursin("\$(\"" * metric * "\").textContent = \"—\"", js)
        end
    end

    @testset "webhook failures do not expose credentials" begin
        reset_notify!()
        maybe_notify!((level=0, reasons=String[]))
        secret_url = "bogus://user:supersecret@localhost/hook?token=hunter2"
        r = @test_logs (:warn, r"alert webhook POST failed") begin
            maybe_notify!((level=1, reasons=["test"]); url=secret_url)
        end
        @test r.fired == false && r.error == "webhook delivery failed"
        @test !occursin("supersecret", JSON3.write(r)) && !occursin("hunter2", JSON3.write(r))
        @test _LAST_ALERT_LEVEL[] == 0               # failed transition remains retryable
        reset_notify!()
    end

    @testset "webhook fires once per level transition and never on a repeat" begin
        reset_notify!()
        calls = Any[]
        ok_post = (u, h, b) -> (push!(calls, JSON3.read(b)); nothing)
        url = "https://hooks.example/test"

        r = maybe_notify!((level=0, reasons=String[]); url=url, post_fn=ok_post)
        @test r.fired == false && r.reason == "baseline set"   # first call baselines, no fire
        @test isempty(calls) && _LAST_ALERT_LEVEL[] == 0

        r = maybe_notify!((level=1, reasons=["Dst forecast Minor storm"]); url=url, post_fn=ok_post)
        @test r.fired == true && r.level == 1                  # quiet -> L1 fires exactly once
        @test length(calls) == 1 && calls[end].level == 1 && calls[end].kind == "elevated"
        @test _LAST_ALERT_LEVEL[] == 1

        r = maybe_notify!((level=1, reasons=["Dst forecast Minor storm"]); url=url, post_fn=ok_post)
        @test r.fired == false && r.changed == false          # repeated level: no fire, no POST
        @test length(calls) == 1

        r = maybe_notify!((level=0, reasons=String[]); url=url, post_fn=ok_post)
        @test r.fired == true && r.level == 0                  # L1 -> quiet fires the all-clear
        @test length(calls) == 2 && calls[end].kind == "allclear"
        @test occursin("all clear", lowercase(calls[end].text))
        @test _LAST_ALERT_LEVEL[] == 0
        reset_notify!()
    end

    @testset "webhook commits the delivered level only on successful delivery" begin
        reset_notify!()
        maybe_notify!((level=0, reasons=String[]); url="https://h", post_fn=(u,h,b)->nothing)
        @test _LAST_ALERT_LEVEL[] == 0
        r = maybe_notify!((level=2, reasons=["storm"]); url="https://h",
                          post_fn=(u,h,b)->error("transient POST failure"))
        @test r.fired == false && r.error == "webhook delivery failed"
        @test _LAST_ALERT_LEVEL[] == 0                        # NOT committed -> transition retryable
        n = Ref(0)
        r = maybe_notify!((level=2, reasons=["storm"]); url="https://h",
                          post_fn=(u,h,b)->(n[] += 1; nothing))
        @test r.fired == true && n[] == 1 && _LAST_ALERT_LEVEL[] == 2   # retry delivers and commits
        reset_notify!()
    end

    @testset "stale forecast raises an outage alert, never a false all-clear" begin
        reset_notify!()
        calls = Any[]
        post = (u, h, b) -> (push!(calls, JSON3.read(b)); nothing)
        url = "https://h"

        maybe_notify!((level=2, reasons=["Dst forecast Moderate storm"], stale=false);
                      url=url, post_fn=post)                  # baseline: mid-storm at L2
        @test _LAST_ALERT_LEVEL[] == 2 && _LAST_ALERT_STALE[] == false

        # Daemon dies; 3 h later the forecast status is stale and the numeric level collapses to 0.
        r = maybe_notify!((level=0, reasons=String[], stale=true); url=url, post_fn=post)
        @test r.fired == true && r.stale == true && r.kind == "stale_onset"
        @test length(calls) == 1
        @test !occursin("all clear", lowercase(calls[end].text))   # the exact bug: no false all-clear
        @test occursin("stale", lowercase(calls[end].text))
        @test _LAST_ALERT_STALE[] == true

        r = maybe_notify!((level=0, reasons=String[], stale=true); url=url, post_fn=post)
        @test r.fired == false && length(calls) == 1          # still stale: no repeat, no all-clear

        r = maybe_notify!((level=0, reasons=String[], stale=false); url=url, post_fn=post)
        @test r.fired == true && r.kind == "restored"         # fresh quiet cycle -> restored notice
        @test length(calls) == 2 && occursin("restored", lowercase(calls[end].text))
        @test _LAST_ALERT_STALE[] == false && _LAST_ALERT_LEVEL[] == 0
        reset_notify!()
    end

    @testset "live-layer escalation fires even while the forecast feed is stale" begin
        reset_notify!()
        calls = Any[]
        post = (u, h, b) -> (push!(calls, JSON3.read(b)); nothing)
        url = "https://h"
        maybe_notify!((level=0, reasons=String[], stale=false); url=url, post_fn=post)   # baseline
        r = maybe_notify!((level=0, reasons=String[], stale=true); url=url, post_fn=post)
        @test r.fired == true && r.kind == "stale_onset"
        r = maybe_notify!((level=1, reasons=["SWPC upstream elevated"], stale=true);
                          url=url, post_fn=post)
        @test r.fired == true && r.kind == "stale_escalation" && r.level == 1
        r = maybe_notify!((level=1, reasons=["SWPC upstream elevated"], stale=true);
                          url=url, post_fn=post)
        @test r.fired == false                                # same stale level: no repeat
        reset_notify!()
    end

    @testset "webhook dedup baseline persists across a restart" begin
        reset_notify!()
        dir = mktempdir()
        sp = joinpath(dir, "alert_notify_state.json")
        n = Ref(0)
        post = (u, h, b) -> (n[] += 1; nothing)
        url = "https://h"
        maybe_notify!((level=0, reasons=String[]); url=url, state_path=sp, post_fn=post)  # baseline
        maybe_notify!((level=1, reasons=["storm"]); url=url, state_path=sp, post_fn=post)  # fires L1
        @test isfile(sp) && n[] == 1

        reset_notify!()                                       # simulate a process restart
        @test _LAST_ALERT_LEVEL[] == -1
        # Escalation to L2 happened during the downtime: the first poll after restart must deliver
        # it against the persisted baseline (1), not silently re-baseline at 2.
        r = maybe_notify!((level=2, reasons=["storm"]); url=url, state_path=sp, post_fn=post)
        @test r.fired == true && r.level == 2 && r.previous_level == 1 && n[] == 2
        reset_notify!()
        rm(dir; recursive=true, force=true)
    end

    @testset "static serving whitelists known asset extensions" begin
        @test serve_static("/index.html").status == 200
        @test serve_static("/style.css").status == 200
        stray = joinpath(PUBLIC_DIR, "stray_secret.txt")
        write(stray, "credentials must not leak through the static server")
        try
            # A real file that exists under public/ but whose extension is not whitelisted must be
            # treated as not found, not served as application/octet-stream.
            @test serve_static("/stray_secret.txt").status == 404
        finally
            rm(stray; force=true)
        end
    end

    @testset "get_log degrades to no-500 when the log is rotated away" begin
        dir = mktempdir()
        p = joinpath(dir, "live_forecast_log.csv")
        write(p, "issue_time_utc,target_time_utc\n2026-01-01T00:00:00,2026-01-01T01:00:00\n")
        _LOG_CACHE[] = nothing
        g1 = get_log(p)
        @test g1 isa DataFrame && nrow(g1) == 1
        rm(p; force=true)                                     # log rotation unlinks the file
        g2 = get_log(p)
        @test g2 isa DataFrame                                # cached frame served, no exception
        @test make_handler(p)(HTTP.Request("GET", "/api/forecast")).status == 200   # never a 500
        _LOG_CACHE[] = nothing
        rm(dir; recursive=true, force=true)
    end

    @testset "non-finite Dst threat is unknown" begin
        @test dst_threat_level(NaN) == (nothing, "Unknown")
        @test dst_threat_level(Inf) == (nothing, "Unknown")
        @test dst_threat_level(-Inf) == (nothing, "Unknown")
        @test jnum(true) === nothing
        @test jnum(false) === nothing
        @test jnum(big(10)^10_000) === nothing
    end
end
