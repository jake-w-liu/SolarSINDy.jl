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

struct _InterruptingSWPCText end
Base.String(::_InterruptingSWPCText) = throw(InterruptException())

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

function live_cycle_fixture(issue::DateTime;
                            vintage=issue, anchor_time=issue - Hour(1), latest_dst=-20.0,
                            model="v2", served_model="v2+L1A+Bregime+Pinertia",
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

    @testset "forecaster thresholds are positive, integral, and Int-representable" begin
        artifact(thresholds) = (
            station="FRD",
            features=["dbdt_now", "dbdt_mean30", "dbdt_max30", "dbdt_std30",
                      "V", "Bz", "Bs", "VBs"],
            mu=zeros(8), sigma=ones(8), beta=zeros(9), rn_calib=[0.0, 1.0],
            thresholds=thresholds,
        )
        @test _valid_forecaster_artifact(artifact([18.0, 42.0]), "FRD")
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
        @test cal.v2_coverage_90 == 1.0
        hist = build_history(df, 24)
        @test hist.rmse_nt == cal.v2_rmse_nt
        @test hist.rows[1].pred_dst_nt == df.served_pred_dst_nt[1]
        @test hist.rows[1].audit_baseline_dst_nt == df.v2_pred_dst_nt[1]
        fc = build_forecast(df)
        @test fc.available && length(fc.horizons) == 4
        @test fc.horizons[1].pred_dst_nt == df.served_pred_dst_nt[1]
        @test fc.horizons[1].audit_baseline_dst_nt == df.v2_pred_dst_nt[1]
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
        @test_throws ArgumentError geoelectric_field(Float64[], Float64[], 60.0)
        @test_throws ArgumentError geoelectric_field([0.0], [0.0], 0.0)
        @test_throws ArgumentError geoelectric_field([0.0], [0.0], -60.0)
        @test_throws ArgumentError geoelectric_field([0.0], [0.0], Inf)
        @test_throws ArgumentError geoelectric_field([0.0, NaN], [0.0, 1.0], 60.0)
        @test_throws ArgumentError geoelectric_field([0.0, 1.0], [0.0, Inf], 60.0)
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
        @test st.threat.worst_credible_dst_nt == -43.0               # min over newest cycle, not -99
        fc = build_forecast(df)
        @test length(fc.horizons) == 4
        @test fc.issue_time_utc == string(iss_new) * "Z"
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
        df2 = live_cycle_fixture(fresh)
        st2 = build_status(df2)
        @test st2.available == true && st2.stale == false && st2.expired == false
    end

    @testset "served pipeline label exposed from sub_hourly_model_version" begin
        iss = now(UTC) - Minute(20)
        df = live_cycle_fixture(iss)
        fc = build_forecast(df); st = build_status(df)
        @test fc.served_model_version == "v2+L1A+Bregime+Pinertia"
        @test fc.model_version == "v2"                              # core-model contract unchanged
        @test st.served_model_version == "v2+L1A+Bregime+Pinertia"
        @test st.model_version == "v2"
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

    @testset "forecast log cache is schema- and path-safe" begin
        dir = mktempdir(); p1 = joinpath(dir, "one.csv"); p2 = joinpath(dir, "two.csv")
        write(p1, "value\n1\n"); write(p2, "value\n2\n")
        _LOG_CACHE[] = nothing
        a = get_log(p1)
        @test a.value == [1]                         # absent configured time columns do not throw
        @test build_status(a).available == false
        @test build_forecast(a).available == false
        @test isempty(build_history(a).rows)
        # Forge equal metadata to isolate path identity: p2 must never reuse p1's frame.
        _LOG_CACHE[] = (_log_file_identity(p1), a)
        @test get_log(p2).value == [2]
        @test nrow(get_log(joinpath(dir, "missing.csv"))) == 0  # another path gets no stale frame

        # Same-size in-place replacement with restored mtime must still invalidate
        # the cache through ctime/inode identity.
        _LOG_CACHE[] = nothing
        cached = get_log(p1)
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
            served_pred_dst_ci95_nt=[-50.0], interval_source=["aci"], model_version=["v2"],
        )
        st = build_status(df)
        @test st.available == false && st.stale == true && st.invalid_future == true
        @test st.age_hours < -23
        @test build_alerts(df).active == false
        @test build_forecast(df).invalid_future == true
    end

    @testset "dB/dt uses elapsed time and cache windows are isolated" begin
        reference = now(UTC)
        times = jdt.([reference - Minute(4), reference - Minute(2)])
        @test _dbdt_series(times, [0.0, 20.0], [0.0, 0.0])[2] == 10.0
        @test isnan(_dbdt_series([times[1], times[1]], [0.0, 20.0], [0.0, 0.0])[2])
        @test isnan(_dbdt_series(reverse(times), [0.0, 20.0], [0.0, 0.0])[2])
        d = (times=times,
             values=[(metadata=(element="X",), values=[0.0, 20.0]),
                     (metadata=(element="Y",), values=[0.0, 0.0])],
             metadata=(intermagnet=(imo=(coordinates=[-77.0, 39.0, 0.0], name="Test"),),))
        @test _station_parse("TST", d; reference=reference).current_dbdt == 10.0

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

        # An unavailable refresh neither replaces nor re-dates the last good observation.
        empty!(_DBDT_CACHE)
        good = (station="FRD", available=true, current_time_utc=jdt(reference - Minute(1)))
        old_fetch = time() - DBDT_TTL - 1
        _DBDT_CACHE[("FRD", 120)] = (old_fetch, good)
        fallback = usgs_dbdt(station="frd", minutes=120, reference=reference,
                             compute_fn=(s, m) -> (station=s, available=false))
        @test fallback.available && fallback.cached
        @test _DBDT_CACHE[("FRD", 120)][1] == old_fetch
        _DBDT_CACHE[("FRD", 120)] = (old_fetch,
            (station="FRD", available=true, current_time_utc=jdt(reference - Minute(20))))
        rejected = usgs_dbdt(station="FRD", minutes=120, reference=reference,
                             compute_fn=(s, m) -> (station=s, available=false))
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
        )
        @test raced.cached && raced.current_time_utc == newer.current_time_utc
        @test _DBDT_CACHE[("FRD", 120)][2].current_time_utc == newer.current_time_utc
        empty!(_DBDT_CACHE)
        @test_throws InterruptException usgs_dbdt(
            station="FRD", minutes=120, reference=reference,
            compute_fn=(s, m) -> throw(InterruptException()))
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
        net = usgs_network(stations=["frd"], brief_fn=s -> nothing, reference=reference)
        @test net.available && net.cached && net.n_stations == 1
        @test _NET_CACHE[("FRD",)][1] == old_net_fetch
        _NET_CACHE[("FRD",)] = (old_net_fetch,
                                 [(station="FRD", time_utc=jdt(reference - Minute(20)))])
        net_stale = usgs_network(stations=["FRD"], brief_fn=s -> nothing, reference=reference)
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
        )
        @test partial.available && partial.cached && partial.n_stations == 2
        @test only(filter(r -> r.station == "FRD", partial.stations)).time_utc ==
              frd_new.time_utc
        @test only(filter(r -> r.station == "CMO", partial.stations)).time_utc ==
              cmo_current.time_utc
        empty!(_NET_CACHE)
        @test_throws InterruptException usgs_network(
            stations=["FRD"], brief_fn=s -> throw(InterruptException()), reference=reference)
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

        # An all-unavailable refresh returns the last good snapshot without overwriting its age.
        good = snapshot_at(reference - Minute(1)); old_fetch = time() - SWPC_TTL - 1
        lock(_SWPC_LOCK) do
            _SWPC_CACHE[] = (old_fetch, good)
            _SWPC_REFRESH_TASK[] = current_task()
        end
        out = _run_swpc_refresh(build_fn=() -> (source="test", available=false))
        @test out == good
        @test _SWPC_CACHE[][1] == old_fetch && _SWPC_CACHE[][2] == good
        lock(_SWPC_LOCK) do; _SWPC_REFRESH_TASK[] = current_task(); end
        @test_throws InterruptException _run_swpc_refresh(
            build_fn=() -> throw(InterruptException()))
        @test _SWPC_REFRESH_TASK[] === nothing
        @test _SWPC_CACHE[][1] == old_fetch
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
        held = @async (take!(gate); sentinel)
        lock(_SWPC_LOCK) do
            _SWPC_CACHE[] = nothing
            _SWPC_REFRESH_TASK[] = held
        end
        @test swpc_snapshot_cached_or_refresh().available == false
        @test lock(_SWPC_LOCK) do; _start_swpc_refresh_locked() === held; end
        waiter = @async swpc_snapshot()
        yield()
        @test _SWPC_REFRESH_TASK[] === held
        put!(gate, nothing)
        @test fetch(waiter) == sentinel
        lock(_SWPC_LOCK) do
            _SWPC_CACHE[] = nothing
            _SWPC_REFRESH_TASK[] = nothing
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

    @testset "non-finite Dst threat is unknown" begin
        @test dst_threat_level(NaN) == (nothing, "Unknown")
        @test dst_threat_level(Inf) == (nothing, "Unknown")
        @test dst_threat_level(-Inf) == (nothing, "Unknown")
        @test jnum(true) === nothing
        @test jnum(false) === nothing
        @test jnum(big(10)^10_000) === nothing
    end
end
