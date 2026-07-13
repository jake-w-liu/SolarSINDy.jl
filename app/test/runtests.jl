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

    @testset "SWPC row parsing tolerates partial public-feed rows" begin
        idx = Dict("speed" => 2, "density" => 3)
        @test _swpc_row_field(idx, ["2026-06-26T00:00:00Z", "460.5", "1.7"], "speed") == 460.5
        @test _swpc_row_field(idx, ["2026-06-26T00:00:00Z", "460.5"], "density") === nothing
        @test _swpc_row_field(idx, ["2026-06-26T00:00:00Z", "460.5"], "bz_gsm") === nothing
    end

    @testset "forecast API exposes upgraded V2 as the product forecast" begin
        issue1 = now(UTC) - Hour(3)
        issue2 = now(UTC) - Hour(2)
        df = DataFrame(
            issue_time_utc_dt=[issue1, issue2],
            latest_solar_wind_utc_dt=[issue1, issue2],
            latest_dst_time_utc_dt=[issue1 - Hour(1), issue2 - Hour(1)],
            target_time_utc_dt=[issue1 + Hour(1), issue2 + Hour(1)],
            horizon_hours=[1.0, 1.0],
            latest_dst_nt=[-45.0, -50.0],
            observation_dst_nt=[-52.0, -60.0],
            v2_pred_dst_nt=[-40.0, -45.0],
            v2_pred_dst_ci05_nt=[-50.0, -55.0],
            v2_pred_dst_ci95_nt=[-30.0, -35.0],
            served_pred_dst_nt=[-51.0, -59.0],
            served_pred_dst_ci05_nt=[-61.0, -69.0],
            served_pred_dst_ci95_nt=[-41.0, -49.0],
            persistence_dst_nt=[-45.0, -50.0],
            obrien_dst_nt=[-50.0, -58.0],
            interval_source=["aci", "aci"],
            model_version=["v2", "v2"],
        )
        cal = calibration_summary(df)
        expected_v2_rmse = round(sqrt(mean((df.observation_dst_nt .- df.served_pred_dst_nt).^2)); digits=2)
        expected_audit_rmse = round(sqrt(mean((df.observation_dst_nt .- df.v2_pred_dst_nt).^2)); digits=2)
        @test cal.v2_n_verified == 2
        @test cal.v2_rmse_nt == expected_v2_rmse
        @test cal.rmse_nt == expected_v2_rmse
        @test cal.audit_baseline_rmse_nt == expected_audit_rmse
        @test cal.v2_coverage_90 == 1.0
        hist = build_history(df, 24)
        @test hist.rmse_nt == cal.v2_rmse_nt
        @test hist.rows[1].pred_dst_nt == df.served_pred_dst_nt[1]
        @test hist.rows[1].audit_baseline_dst_nt == df.v2_pred_dst_nt[1]
        fc = build_forecast(df)
        @test fc.horizons[1].pred_dst_nt == df.served_pred_dst_nt[2]
        @test fc.horizons[1].audit_baseline_dst_nt == df.v2_pred_dst_nt[2]
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
        df = DataFrame(
            issue_time_utc_dt = [iss_old, iss_old, iss_new, iss_new],
            latest_solar_wind_utc_dt = fill(sw_vintage, 4),
            latest_dst_time_utc_dt = [iss_old - Hour(1), iss_old - Hour(1), iss_new - Hour(1), iss_new - Hour(1)],
            target_time_utc_dt = [iss_old + Hour(1), iss_old + Hour(2), iss_new + Hour(1), iss_new + Hour(2)],
            horizon_hours = [1.0, 2.0, 1.0, 2.0],
            latest_dst_nt = [-10.0, -10.0, -20.0, -20.0],
            observation_dst_nt = [missing, missing, missing, missing],
            served_pred_dst_nt = [-5.0, -6.0, -15.0, -16.0],
            served_pred_dst_ci05_nt = [-99.0, -98.0, -40.0, -42.0],   # superseded cycle is DEEPER
            served_pred_dst_ci95_nt = [0.0, -1.0, -5.0, -6.0],
            interval_source = fill("aci", 4),
            model_version = fill("v2", 4),
        )
        cyc = latest_cycle(df)
        @test nrow(cyc) == 2
        @test all(==(iss_new), cyc.issue_time_utc_dt)
        st = build_status(df)
        @test st.available == true
        @test st.forecast_issue_utc == string(iss_new) * "Z"          # newest issue, not superseded
        @test st.latest_observation.dst_nt == -20.0                   # anchor from newest cycle
        @test st.threat.worst_credible_dst_nt == -42.0               # min over newest cycle, not -99
        fc = build_forecast(df)
        @test length(fc.horizons) == 2
        @test fc.issue_time_utc == string(iss_new) * "Z"
    end

    @testset "staleness gate: expired cycle suppresses live status and alerts" begin
        old = now(UTC) - Day(10)
        df = DataFrame(
            issue_time_utc_dt = [old], latest_solar_wind_utc_dt = [old],
            latest_dst_time_utc_dt = [old - Hour(1)], target_time_utc_dt = [old + Hour(1)],
            horizon_hours = [1.0], latest_dst_nt = [-60.0], observation_dst_nt = [missing],
            served_pred_dst_nt = [-70.0], served_pred_dst_ci05_nt = [-90.0], served_pred_dst_ci95_nt = [-50.0],
            interval_source = ["aci"], model_version = ["v2"],
        )
        st = build_status(df)
        @test st.available == false
        @test st.stale == true
        @test st.expired == true
        @test st.age_hours > 200                     # ~240 h old
        @test build_alerts(df).active == false       # no forecast/watch alerts from an expired cycle
        cs = compute_alert_state(st, (available=false,), nothing)
        @test cs.level == 0
        # a fresh cycle is neither stale nor expired
        fresh = now(UTC) - Minute(20)
        df2 = DataFrame(
            issue_time_utc_dt = [fresh], latest_solar_wind_utc_dt = [fresh],
            latest_dst_time_utc_dt = [fresh - Hour(1)], target_time_utc_dt = [fresh + Hour(2)],
            horizon_hours = [2.0], latest_dst_nt = [-20.0], observation_dst_nt = [missing],
            served_pred_dst_nt = [-25.0], served_pred_dst_ci05_nt = [-35.0], served_pred_dst_ci95_nt = [-15.0],
            interval_source = ["aci"], model_version = ["v2"],
        )
        st2 = build_status(df2)
        @test st2.available == true && st2.stale == false && st2.expired == false
    end

    @testset "served pipeline label exposed from sub_hourly_model_version" begin
        iss = now(UTC) - Minute(20)
        df = DataFrame(
            issue_time_utc_dt = [iss], latest_solar_wind_utc_dt = [iss],
            latest_dst_time_utc_dt = [iss - Hour(1)], target_time_utc_dt = [iss + Hour(1)],
            horizon_hours = [1.0], latest_dst_nt = [-20.0], observation_dst_nt = [missing],
            served_pred_dst_nt = [-25.0], served_pred_dst_ci05_nt = [-35.0], served_pred_dst_ci95_nt = [-15.0],
            interval_source = ["aci"], model_version = ["v2"],
            sub_hourly_model_version = ["v2+L1A+Bregime+Pinertia"],
        )
        fc = build_forecast(df); st = build_status(df)
        @test fc.served_model_version == "v2+L1A+Bregime+Pinertia"
        @test fc.model_version == "v2"                              # core-model contract unchanged
        @test st.served_model_version == "v2+L1A+Bregime+Pinertia"
        @test st.model_version == "v2"
        df2 = select(df, Not(:sub_hourly_model_version))           # legacy row -> "v2" fallback
        @test build_forecast(df2).served_model_version == "v2"
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
            model_version = ["v2"], interval_source = ["aci"],
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

    @testset "geoelectric current is edge-trimmed; max spans the trailing 30 min" begin
        tt = collect(1:120)
        xv = Vector{Any}(20.0 .* tt .+ 3.0 .* sin.(tt ./ 3.0))     # rising ramp + wiggle (storm onset)
        yv = Vector{Any}(fill(5.0, 120))
        g = _geoe_nowcast(xv, yv, 60.0)
        @test g !== nothing
        # recompute the pipeline independently and pin the two behavioral changes
        xs = [_num(xv[i]) for i in 1:120]; ys = [_num(yv[i]) for i in 1:120]
        xf = _interp_gaps(xs); yf = _interp_gaps(ys)
        ex, ey = geoelectric_field(_detrend(xf), _detrend(yf), 60.0; rho_ohm_m=1000.0)
        emag = sqrt.(ex.^2 .+ ey.^2); m = length(emag)
        @test isapprox(g.current, emag[max(1, m - 3)]; atol=1e-9)                       # trimmed, not emag[end]
        @test isapprox(g.max, maximum(emag[max(min(4, m), m - 32):max(1, m - 3)]); atol=1e-9)  # last ~30 min
    end
end
