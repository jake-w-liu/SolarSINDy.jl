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
                            interval=nothing, observations=missing,
                            served_pred=-25.0, served_lo=-35.0, served_hi=-15.0,
                            audit_pred=served_pred, audit_lo=served_lo, audit_hi=served_hi,
                            v2_1_served_pred=nothing, v2_2_stack_pred=nothing,
                            v2_1_served_ci05=nothing, v2_2_stack_ci05=nothing,
                            driver_assumption=nothing, v2_2_status=nothing,
                            v24_status=nothing, v24_pred=nothing, v24_guard_applied=nothing,
                            v24_projection_applied=nothing, v24_regime_cell=nothing)
    requested = collect(LIVE_CYCLE_HORIZONS)
    targets = floor(issue, Hour) .+ Hour.(requested)
    lead = [(target - issue) / Millisecond(3_600_000) for target in targets]
    expand(x) = x isa AbstractVector ? collect(x) : fill(x, length(requested))
    # The engine sets `interval_source` to the depth-stratified conformal source inside the branch
    # that also records `v24_status = "ok"` — the status is returned only after the center and both
    # interval endpoints are finite — and leaves the batch policy's source on every other row. A
    # fixture that does not state the sources therefore derives them from the statuses, so a cycle
    # fixture is coherent the way the engine's own rows are. A test that needs an incoherent cycle
    # states the sources explicitly.
    statuses = v24_status === nothing ? nothing : expand(v24_status)
    sources = interval !== nothing ? expand(interval) :
              statuses === nothing ? fill("aci", length(requested)) :
              [s isa AbstractString && s == "ok" ? "v24_conformal_depth" : "aci" for s in statuses]
    frame = DataFrame(
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
        # The interval source and the served label are per-row fields: the super-learner stage acts
        # per horizon, so one batch can issue horizons under different stages and disclose it row by
        # row. Both accept a vector so a fixture can reproduce that cycle shape.
        interval_source=sources,
        model_version=fill(model, length(requested)),
        sub_hourly_model_version=expand(served_model),
    )
    # The V2.1 continuity column exists only in the stacked-served schema; a fixture that omits it
    # reproduces a pre-stack log, which must keep behaving exactly as before.
    v2_1_served_pred === nothing ||
        (frame[!, :v2_1_served_pred_dst_nt] = expand(v2_1_served_pred))
    # The static-stack continuity column exists only in the super-learner-served schema, and is the
    # second depth-safe partner of the published severity.
    v2_2_stack_pred === nothing ||
        (frame[!, :v2_2_stack_pred_dst_nt] = expand(v2_2_stack_pred))
    # The predecessor band edges: the lower edge each earlier stage would have published for this
    # issue. They exist only in the schema written after the watch edge became a minimum over the
    # partners' own edges; a fixture that omits them reproduces a row from before that change.
    v2_1_served_ci05 === nothing ||
        (frame[!, :v2_1_served_ci05_dst_nt] = expand(v2_1_served_ci05))
    v2_2_stack_ci05 === nothing ||
        (frame[!, :v2_2_stack_ci05_dst_nt] = expand(v2_2_stack_ci05))
    # The per-row driver assumption and served-stage statuses likewise exist only in newer schemas.
    driver_assumption === nothing ||
        (frame[!, :driver_assumption] = expand(driver_assumption))
    v2_2_status === nothing || (frame[!, :v2_2_status] = expand(v2_2_status))
    statuses === nothing || (frame[!, :v24_status] = statuses)
    v24_pred === nothing || (frame[!, :v24_pred_dst_nt] = expand(v24_pred))
    v24_guard_applied === nothing ||
        (frame[!, :v24_guard_applied] = expand(v24_guard_applied))
    v24_projection_applied === nothing ||
        (frame[!, :v24_projection_applied] = expand(v24_projection_applied))
    v24_regime_cell === nothing || (frame[!, :v24_regime_cell] = expand(v24_regime_cell))
    return frame
end

# Write a fixture cycle in the on-disk schema (raw ISO time columns, as the daemon appends them) so
# a test can drive the log-backed path end to end — load, parse, cycle selection, status — instead
# of handing an already-parsed frame to a builder.
function write_cycle_csv(path::AbstractString, frame::DataFrame)
    raw = copy(frame)
    rename!(raw,
        :issue_time_utc_dt => :issue_time_utc,
        :latest_solar_wind_utc_dt => :latest_solar_wind_utc,
        :latest_dst_time_utc_dt => :latest_dst_time_utc,
        :target_time_utc_dt => :target_time_utc,
    )
    CSV.write(path, raw)
    return path
end

# A log that exists but carries no forecast rows: the shape a truncation, a rotation, or a failed
# rewrite leaves behind, and the one that used to be read as "no data yet".
write_empty_log(path::AbstractString) =
    (write(path, "issue_time_utc,target_time_utc\n"); path)

# API payloads for the executed-render probe, written as the JSON the routes actually return.
# `@@MARKER@@` stands in for the injected string at every place the page renders text it did not
# author: the feed's alert identifier and summary, the NOAA scale token, the station code, and the
# log-written interval source and served label.
const RENDER_PROBE_PAYLOADS_JS = raw"""
const MARKER = "@@MARKER@@";
const PAYLOADS = {
  "/api/health": { status: "ok", cycle_complete: true },
  "/api/status": {
    available: true, generated_utc: "@@ISSUE@@", forecast_issue_utc: "@@ISSUE@@",
    latest_solar_wind_utc: "@@ISSUE@@", model_version: "v2.1",
    served_model_version: "@@SERVED@@", served_product: "V2.4" + MARKER,
    latest_observation: { dst_nt: -40.0, time_utc: "@@ISSUE@@" },
    threat: { level: 2, label: "Moderate storm", watch: true, watch_level: 3,
              watch_label: "Intense storm", point_min_dst_nt: -70.0,
              interval_lower_edge_min_dst_nt: -105.0,
              interval_lower_edge_source: "v2_1_served",
              basis: "Dst storm-intensity scale (-30/-50/-100/-200 nT)" },
    lead_time: { forecast_horizon_hours: 6.0, physical_upstream_lead_min: [30, 60],
                 driver_assumption: "Ballistically propagated L1 forcing" },
    calibration: { n_verified: 60, v2_n_verified: 60, v2_rmse_nt: 4.2, v2_coverage_90: 0.9,
                   comparison_n_verified: 60, live_skill_min_verified: 48,
                   live_skill_mature: true, current_interval_source: MARKER,
                   current_interval_sources: [MARKER, "aci"],
                   n_verified_current_source: 10, n_verified_current_served_model: 10,
                   v2_matched_rmse_nt: 4.2, persistence_matched_rmse_nt: 6.1,
                   by_source: [{ source: MARKER, n: 3, coverage_90: 0.9 },
                               { source: "aci", n: 4, coverage_90: 0.88 }],
                   by_served_model: [{ product: MARKER, n: 3, coverage_90: 0.9, rmse_nt: 4.0 },
                                     { product: "V2.4", n: 4, coverage_90: 0.88, rmse_nt: 4.4 }] },
    upstream: { available: true,
                solar_wind: { available: true, speed_kms: 520.0, bz_gsm_nt: -12.0, bt_nt: 14.0,
                              density_cm3: 6.0, mag_time_utc: "@@ISSUE@@" },
                kp: { value: 6.0, time_utc: "@@ISSUE@@" },
                scales: { G: "2" + MARKER, time_utc: "@@ISSUE@@" },
                alerts: [{ product_id: MARKER, issue_utc: "@@ISSUE@@",
                           summary: "geomagnetic storm " + MARKER }] },
    upstream_status: { available: true, elevated: true,
                       reasons: ["NOAA G" + MARKER + " geomagnetic storm"] },
  },
  "/api/forecast": {
    available: true, issue_time_utc: "@@ISSUE@@", latest_solar_wind_utc: "@@ISSUE@@",
    anchor_dst_nt: -40.0, anchor_dst_time_utc: "@@ISSUE@@", interval_source: MARKER,
    interval_sources: [MARKER, "v24_conformal_depth"], superseded_cycle_incomplete: false,
    served_product: "V2.4" + MARKER, served_model_version: "@@SERVED@@",
    recent_observed: [{ target_utc: "@@ISSUE@@", observed_dst_nt: -40.0 }],
    horizons: [{ target_utc: "2026-06-26T07:00:00Z", horizon_hours: 1.0, pred_dst_nt: -70.0,
                 ci05_dst_nt: -95.0, ci95_dst_nt: -45.0, severity_dst_nt: -75.0,
                 severity_ci05_dst_nt: -105.0, severity_ci05_source: "v2_1_served",
                 interval_source: MARKER }],
  },
  "/api/history": { hours: 72, coverage_90: 0.9, rmse_nt: 4.2,
                    rows: [{ target_utc: "@@ISSUE@@", observed_dst_nt: -40.0, pred_dst_nt: -42.0,
                             ci05_dst_nt: -60.0, ci95_dst_nt: -20.0, horizon_hours: 1.0,
                             inside_90ci: true }] },
  "/api/dbdt": { available: true, station: MARKER, current_dbdt: 12.0, max30_dbdt: 20.0,
                 current_tier: { level: 1, label: "elevated" },
                 max30_tier: { level: 1, label: "elevated" },
                 series: [{ t: "@@ISSUE@@", dbdt: 12.0 }],
                 geoelectric: { max_vkm: 0.4, rho_ohm_m: 1000.0 }, forecast: null },
  "/api/network": { n_stations: 1,
                    stations: [{ station: MARKER, name: MARKER, lat: 40.0, lon: -100.0,
                                 max_dbdt: 20.0, tier: { level: 1, label: "elevated" } }] },
};
"""

# The two machine tokens the engine writes for the served pipeline, and the reader-facing sentences
# they must map to. Restated here rather than imported so the payload contract is checked against an
# independent statement of it.
const SUPERLEARNER_DRIVER_TOKEN =
    "ballistically_propagated_l1_then_ten_expert_nnls_superlearner_with_sindy_family_floor_over_" *
    "served_frozen_analog_and_the_static_regime_stack_then_depth_stratified_conformal_interval"
const STACK_DRIVER_TOKEN =
    "ballistically_propagated_l1_then_regime_aware_relaxation_then_rate_projection_then_one_hour_" *
    "inertia_blend_then_state_inertia_then_extreme_inertia_guard_then_static_regime_stack"
const V2_1_DRIVER_TOKEN =
    "ballistically_propagated_l1_then_regime_aware_relaxation_then_rate_projection_then_one_hour_" *
    "inertia_blend_then_state_inertia_then_extreme_inertia_guard"

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
            # The compared quantities are both rounded to the same decimal, so the tolerance is
            # the rounding itself: an `atol` wider than that admits a real formula drift as
            # agreement, which is the whole thing these vectors exist to catch.
            @test isapprox(fc.point_dbdt, round(max(gp, 0.0); digits=2); atol=1e-9)
            @test isapprox(fc.ub90_dbdt, round(max(gub, 0.0); digits=2); atol=1e-9)
            @test length(fc.exceedance) == length(gexc)
            for (k, e) in enumerate(fc.exceedance)
                @test e.threshold == gexc[k][1]
                @test isapprox(e.empirical_score, round(gexc[k][2]; digits=3); atol=1e-9)
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
        df[!, :v1_pred_dst_nt] = [-55.0, -56.0, -57.0, -58.0]
        df[!, :persistence_dst_nt] = fill(-50.0, nrow(df))
        df[!, :burton_dst_nt] = [-57.0, -58.0, -59.0, -60.0]
        df[!, :burton_full_dst_nt] = [-56.0, -57.0, -58.0, -59.0]
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
        # Hand-derived constant residuals verify every displayed method and catch a
        # dropped comparator or a substitution of the unmatched product RMSE.
        @test cal.comparison_n_verified == 4
        @test cal.v2_matched_rmse_nt == 1.0
        @test cal.frozen_tail_ablation_matched_rmse_nt == 15.0
        @test cal.sindy_v1_matched_rmse_nt == 5.0
        @test cal.persistence_matched_rmse_nt == 11.55
        @test cal.burton_matched_rmse_nt == 3.0
        @test cal.burton_full_matched_rmse_nt == 4.0
        @test cal.obrien_matched_rmse_nt == 2.0
        @test cal.live_skill_min_verified == 48
        @test !cal.live_skill_mature
        @test cal.live_skill_rows_remaining == 44
        @test cal.served_interval_coverage_scope == "empirical_only"

        # One missing O'Brien value removes that target from every matched RMSE.
        # Persistence then has squared errors 100, 121, and 144: sqrt(365/3)=11.03
        # after the API's declared two-decimal presentation rounding.
        partial = copy(df)
        partial[!, :obrien_dst_nt] = Union{Missing,Float64}[-58.0, -59.0, -60.0, missing]
        partial_cal = calibration_summary(partial)
        @test partial_cal.comparison_n_verified == 3
        @test partial_cal.persistence_matched_rmse_nt == 11.03
        @test partial_cal.rmse_persistence_nt == 11.55

        # Boundary oracle for the live-skill gate: 12 complete four-horizon cycles
        # are exactly 48 matched targets; deleting one row must return to provisional.
        mature = vcat((live_cycle_fixture(issue - Hour(8 * k);
                      observations=[-40.0, -41.0, -42.0, -43.0],
                      served_pred=[-39.0, -40.0, -41.0, -42.0],
                      served_lo=[-49.0, -50.0, -51.0, -52.0],
                      served_hi=[-29.0, -30.0, -31.0, -32.0],
                      audit_pred=[-38.0, -39.0, -40.0, -41.0],
                      audit_lo=[-48.0, -49.0, -50.0, -51.0],
                      audit_hi=[-28.0, -29.0, -30.0, -31.0]) for k in 0:11)...)
        mature[!, :v1_pred_dst_nt] = mature.observation_dst_nt .+ 5.0
        mature[!, :persistence_dst_nt] = mature.observation_dst_nt .+ 6.0
        mature[!, :burton_dst_nt] = mature.observation_dst_nt .+ 3.0
        mature[!, :burton_full_dst_nt] = mature.observation_dst_nt .+ 4.0
        mature[!, :obrien_dst_nt] = mature.observation_dst_nt .+ 2.0
        mature_cal = calibration_summary(mature)
        @test mature_cal.comparison_n_verified == 48
        @test mature_cal.live_skill_mature
        @test mature_cal.live_skill_rows_remaining == 0
        almost_mature_cal = calibration_summary(mature[1:47, :])
        @test almost_mature_cal.comparison_n_verified == 47
        @test !almost_mature_cal.live_skill_mature
        @test almost_mature_cal.live_skill_rows_remaining == 1
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

        malformed_version = copy(df)
        malformed_version[!, :model_version] = Any[1, "v2.1", "v2.1", "v2.1"]
        @test nrow(verified_rows(malformed_version)) == 3

        missing_issue = copy(df)
        missing_issue[!, :issue_time_utc_dt] =
            Union{Missing,DateTime}[missing, df.issue_time_utc_dt[2:end]...]
        @test nrow(verified_rows(missing_issue)) == 3

        missing_anchor = copy(df)
        missing_anchor[!, :latest_dst_time_utc_dt] =
            Union{Missing,DateTime}[missing, df.latest_dst_time_utc_dt[2:end]...]
        @test nrow(verified_rows(missing_anchor)) == 3
    end

    @testset "bind settings reject an unrendered launchd placeholder" begin
        # A hand-copied plist whose `__SWM_PORT__` was never replaced reached
        # `parse(Int, "__SWM_PORT__")`, and launchd turned that ArgumentError into a sixty-second
        # crash loop diagnosable only from a stack trace. The failure has to name the setting and
        # the repair.
        withenv("SWM_PORT" => nothing, "SWM_HOST" => nothing) do
            @test port_from_env() == 8723
            @test bind_setting_from_env("SWM_HOST", "127.0.0.1") == "127.0.0.1"
        end
        withenv("SWM_PORT" => "9311", "SWM_HOST" => " 0.0.0.0 ") do
            @test port_from_env() == 9311
            @test bind_setting_from_env("SWM_HOST", "127.0.0.1") == "0.0.0.0"
        end
        for (name, value) in (("SWM_PORT", "__SWM_PORT__"), ("SWM_HOST", "__SWM_HOST__"))
            withenv(name => value) do
                err = try
                    name == "SWM_PORT" ? port_from_env() :
                        bind_setting_from_env("SWM_HOST", "127.0.0.1")
                    nothing
                catch e
                    e
                end
                @test err isa ErrorException
                @test occursin(name, err.msg)
                @test occursin("placeholder", err.msg)
                @test occursin("install_launchd.sh", err.msg)
            end
        end
        # A merely wrong value is still reported as a wrong value, not as a placeholder.
        for bad in ("nine", "0", "70000", "")
            withenv("SWM_PORT" => bad) do
                err = try
                    port_from_env()
                    nothing
                catch e
                    e
                end
                @test err isa ErrorException
                @test occursin("SWM_PORT", err.msg)
                @test !occursin("placeholder", err.msg)
            end
        end
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
            # Asserted separately, not as a disjunction: saturation is the expm1 cap being hit,
            # while the range flag is the standardized features leaving the calibration support.
            # Either one alone satisfied the pair, so the support bound itself was unpinned.
            @test extreme.out_of_validated_range == true
            @test extreme.saturated == true
            # An absurd history trips both flags, so it cannot separate them. A sustained
            # 14 nT/min history with ordinary drivers sits about 23 standard deviations outside the
            # calibration support while the ridge output stays below the numerical cap: the support
            # bound is then the only thing that can withhold confidence in the estimate, so a bound
            # widened past that distance would publish this as a reliable 700 nT/min forecast.
            outside = forecast_dbdt(fill(14.0, 30), 700.0, -15.0;
                                    station="FRD", offline_replay=true)
            @test outside !== nothing
            @test outside.saturated == false
            @test outside.out_of_validated_range == true
            @test outside.reliable == false
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

        # The live NOAA endpoint sometimes spells a missing measurement as bare NaN. The fetch
        # boundary accepts it, and the physical-field selector skips that row for a finite one.
        nonfinite_body = Vector{UInt8}(codeunits("""
            [{"time_tag":"2026-07-13T02:41:00","active":true,
              "proton_speed":NaN,"proton_density":3.0},
             {"time_tag":"2026-07-13T02:40:00","active":true,
              "proton_speed":461.0,"proton_density":2.7}]
            """))
        nonfinite = _swpc_get(
            "/json/rtsw/rtsw_wind_1m.json";
            http_get=(args...; kwargs...) -> (; body=nonfinite_body),
        )
        nonfinite_row = _rtsw_latest(
            nonfinite, [:proton_speed, :proton_density];
            bounds=Dict(:proton_speed => (50.0, 5.0e3)),
        )
        @test _rtsw_field(nonfinite_row, :proton_speed) == 461.0
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
        @test occursin("displayed 90% target interval", watch.message)
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

    @testset "a per-row-coherent cycle is served; an incoherent one is not" begin
        issue = now(UTC) - Minute(10)
        mixed_labels = [CURRENT_V2_SERVED_MODEL_VERSION, CURRENT_V2_SERVED_MODEL_VERSION,
                        CURRENT_V2_SERVED_MODEL_VERSION, STACK_V2_SERVED_MODEL_VERSION]
        mixed_status = ["ok", "ok", "ok", "fallback:serving_error"]
        depth = "v24_conformal_depth"
        mixed = live_cycle_fixture(issue; served_model=mixed_labels, v24_status=mixed_status,
                                   interval=[depth, depth, depth, "conformal"],
                                   served_pred=-60.0, served_lo=-80.0, served_hi=-40.0)
        # Three horizons served by the super-learner, one disclosed as a stack fallback: the log is
        # describing a stage that failed between the horizons of one batch, not an inconsistent
        # cycle. Requiring one interval source across the four rows rejected exactly this shape and
        # blanked every log-backed endpoint while the cycle itself was sound.
        @test _valid_live_cycle(mixed)
        st = build_status(mixed)
        fc = build_forecast(mixed)
        @test st.available && fc.available
        @test st.superseded_cycle_incomplete == false
        @test fc.superseded_cycle_incomplete == false
        @test st.served_model_version == STACK_V2_SERVED_MODEL_VERSION   # weakest stage, as before
        @test fc.interval_source == "conformal"        # the interval the reported stage issued
        @test fc.interval_sources == ["conformal", depth]
        @test [h.interval_source for h in fc.horizons] == [depth, depth, depth, "conformal"]
        @test build_alerts(mixed, st).active == true   # alerts are produced, not suppressed

        # Mutation guard: coherence is what is accepted, not variety. A row that claims the
        # depth-stratified band while its own status records a fallback would publish an interval
        # calibrated for a center it was not drawn around, and an acted row that disclaims it hides
        # which band was served. Both stay invalid, as does a batch that used two policies at once.
        incoherent = [
            [depth, depth, depth, depth],              # the fallback row claims the V2.4e band
            ["conformal", depth, depth, "conformal"],  # an acted row disclaims it
            [depth, depth, "aci", "conformal"],        # two batch policies in one cycle
        ]
        incoherent_status = [mixed_status, mixed_status,
                             ["ok", "ok", "fallback:serving_error", "fallback:serving_error"]]
        for (sources, statuses) in zip(incoherent, incoherent_status)
            bad = live_cycle_fixture(issue; served_model=mixed_labels, v24_status=statuses,
                                     interval=sources)
            @test !_valid_live_cycle(bad)
            @test !build_status(bad).available
            @test !build_forecast(bad).available
        end
    end

    @testset "the live interval method names the cycle that is published" begin
        # The calibration panel states the interval method the live product is issued under. It read
        # that as one common field across the cycle's horizons and from the newest issue hour, and
        # per-row disclosure invalidated both readings. A cycle whose super-learner stage acted on
        # some horizons and not others carries two methods by design, so the common-field reading was
        # empty and the panel published the method as the string "nothing"; and during the
        # superseded-cycle fallback the newest issue hour is the incomplete one, so the panel called
        # the method "unknown" while a complete forecast was on the page. Both readings must come
        # from the cycle the payloads actually publish.
        issue = now(UTC) - Minute(10)
        depth = "v24_conformal_depth"
        mixed_labels = [CURRENT_V2_SERVED_MODEL_VERSION, CURRENT_V2_SERVED_MODEL_VERSION,
                        CURRENT_V2_SERVED_MODEL_VERSION, STACK_V2_SERVED_MODEL_VERSION]
        mixed_status = ["ok", "ok", "ok", "fallback:serving_error"]
        mixed_sources = [depth, depth, depth, "aci"]

        # Three horizons under the depth-stratified conformal band and one under the batch policy.
        # The cycle is published under its weakest served stage, so its interval method is the one
        # that stage issued, and it must match the forecast payload exactly.
        mixed = live_cycle_fixture(issue; served_model=mixed_labels, v24_status=mixed_status,
                                   interval=mixed_sources,
                                   served_pred=-60.0, served_lo=-80.0, served_hi=-40.0)
        @test _valid_live_cycle(mixed)
        mixed_fc = build_forecast(mixed)
        mixed_cal = calibration_summary(mixed)
        @test mixed_fc.available
        @test mixed_cal.current_interval_source == "aci"
        @test mixed_cal.current_interval_source == mixed_fc.interval_source
        @test mixed_cal.current_interval_sources == ["aci", depth]
        @test mixed_cal.current_interval_sources == mixed_fc.interval_sources
        @test mixed_cal.current_served_model == mixed_fc.served_model_version
        @test mixed_cal.current_served_model == STACK_V2_SERVED_MODEL_VERSION
        # The two strings this panel used to publish for exactly this cycle.
        @test mixed_cal.current_interval_source != "nothing"
        @test mixed_cal.current_interval_source != "unknown"
        # The status payload carries the same summary, so the header line reads the same method.
        @test build_status(mixed).calibration.current_interval_source == "aci"

        # A cycle whose reported stage itself served two methods: no single one describes it, so
        # none is named and the set is what is published — the same set the forecast payload sends.
        two_sources = live_cycle_fixture(issue; served_model=CURRENT_V2_SERVED_MODEL_VERSION,
                                         v24_status=mixed_status, interval=mixed_sources,
                                         served_pred=-60.0, served_lo=-80.0, served_hi=-40.0)
        @test _valid_live_cycle(two_sources)
        two_cal = calibration_summary(two_sources)
        @test two_cal.current_interval_source === nothing
        @test two_cal.current_interval_source === build_forecast(two_sources).interval_source
        @test two_cal.current_interval_sources == ["aci", depth]

        # Superseded-cycle fallback: the newest issue hour is incomplete and the payloads publish the
        # newest complete one. The summary must describe that cycle, which has a method, rather than
        # the incomplete hour, which has none.
        previous = live_cycle_fixture(issue - Hour(1); v24_status="ok",
                                      served_model=CURRENT_V2_SERVED_MODEL_VERSION,
                                      served_pred=-40.0, served_lo=-55.0, served_hi=-25.0)
        partial = live_cycle_fixture(issue; v24_status="ok",
                                     served_model=CURRENT_V2_SERVED_MODEL_VERSION,
                                     served_pred=-70.0, served_lo=-95.0, served_hi=-45.0)[1:2, :]
        fallback = vcat(previous, partial)
        @test !_valid_live_cycle(latest_cycle(fallback))
        fallback_fc = build_forecast(fallback)
        fallback_cal = calibration_summary(fallback)
        @test fallback_fc.available && fallback_fc.superseded_cycle_incomplete == true
        @test fallback_cal.current_interval_source == depth
        @test fallback_cal.current_interval_source == fallback_fc.interval_source
        @test fallback_cal.current_interval_sources == [depth]
        @test fallback_cal.current_served_model == fallback_fc.served_model_version
        @test fallback_cal.current_interval_source != "unknown"

        # The count published beside the name counts the verified rows the name refers to: the
        # reported method when the cycle has one, and any method it carries when it does not.
        now0 = floor(now(UTC), Hour)
        history = DataFrame()
        for k in 1:3
            frame = live_cycle_fixture(now0 - Hour(30 + k); v24_status="ok",
                                       served_model=CURRENT_V2_SERVED_MODEL_VERSION,
                                       observations=-25.0 - k)
            history = isempty(history) ? frame : vcat(history, frame; cols=:union)
        end
        for k in 4:5
            history = vcat(history,
                           live_cycle_fixture(now0 - Hour(30 + k); interval="aci",
                                              served_model=PREVIOUS_V2_SERVED_MODEL_VERSION,
                                              observations=-25.0 - k);
                           cols=:union)
        end
        horizons = length(LIVE_CYCLE_HORIZONS)
        reported_cal = calibration_summary(vcat(history, mixed; cols=:union))
        @test reported_cal.n_verified == 5 * horizons
        @test reported_cal.n_verified_current_source == 2 * horizons      # the "aci" rows only
        set_cal = calibration_summary(vcat(history, two_sources; cols=:union))
        @test set_cal.current_interval_source === nothing
        @test set_cal.n_verified_current_source == 5 * horizons           # both methods it carries

        # A cycle that cannot be published at all still reports neither a method nor a set: the two
        # cases stay distinguishable, so "unknown" never stands in for a method that exists.
        unpublishable = calibration_summary(mixed[1:2, :])
        @test unpublishable.current_interval_source == "unknown"
        @test isempty(unpublishable.current_interval_sources)
    end

    @testset "an incomplete newest cycle falls back to the newest complete one" begin
        issue = now(UTC) - Minute(10)
        previous = live_cycle_fixture(issue - Hour(1);
                                      served_pred=-40.0, served_lo=-55.0, served_hi=-25.0)
        previous_issue = jdt(maximum(previous.issue_time_utc_dt))
        newest = live_cycle_fixture(issue; served_pred=-70.0, served_lo=-95.0, served_hi=-45.0)
        # Every shape the engine can leave in the newest issue hour without the previous hour
        # ceasing to be a complete, internally consistent product cycle.
        broken = Dict{String,DataFrame}(
            "partial write (2 of 4 horizons)" => newest[1:2, :],
            "duplicate retry row" => vcat(newest, newest[end:end, :]),
            "per-row anchor Dst" => (m = copy(newest); m[end, :latest_dst_nt] = -21.0; m),
            "per-row anchor hour" => (m = copy(newest);
                                      m[end, :latest_dst_time_utc_dt] = issue - Hour(2); m),
            "per-row driver vintage" => (m = copy(newest);
                                         m[end, :latest_solar_wind_utc_dt] = issue - Minute(3); m),
        )
        for (shape, newest_rows) in broken
            @testset "$shape" begin
                frame = vcat(previous, newest_rows)
                @test !_valid_live_cycle(latest_cycle(frame))      # the newest hour is not a cycle
                st = build_status(frame)
                fc = build_forecast(frame)
                @test st.available && fc.available                 # availability restored
                @test st.superseded_cycle_incomplete == true       # incompleteness disclosed
                @test fc.superseded_cycle_incomplete == true
                @test st.forecast_issue_utc == previous_issue      # served with its true age
                @test fc.issue_time_utc == previous_issue
                @test st.age_hours > 1.0
                @test all(h -> h.pred_dst_nt == -40.0, fc.horizons)   # the previous cycle's numbers
                @test build_alerts(frame, st).active == true
            end
        end

        # The disclosure reaches the health endpoint, and the newest issue hour is still reported as
        # incomplete there, so the issuance dead-man that reads it still trips.
        dir = mktempdir()
        path = write_cycle_csv(joinpath(dir, "log.csv"), vcat(previous, newest[1:2, :]))
        _LOG_CACHE[] = nothing
        health = JSON3.read(String(
            make_handler(path)(HTTP.Request("GET", "/api/health")).body))
        @test health.cycle_complete == false
        @test health.status == "incomplete"
        @test health.superseded_cycle_incomplete == true
        forecast = JSON3.read(String(
            make_handler(path)(HTTP.Request("GET", "/api/forecast")).body))
        @test forecast.available == true && forecast.superseded_cycle_incomplete == true
        # `/api/status` is not driven here: it refreshes the third-party upstream snapshot, and the
        # status payload over the same loaded log is what the route serves.
        loaded = build_status(get_log(path))
        @test loaded.available == true && loaded.superseded_cycle_incomplete == true
        _LOG_CACHE[] = nothing
        rm(dir; recursive=true, force=true)

        # The fallback restores availability; it must never manufacture it. With nothing complete in
        # the log, and with the newest complete cycle outside the freshness window, the endpoints
        # degrade exactly as they did before the fallback existed.
        only_broken = newest[1:2, :]
        @test build_status(only_broken).available == false
        @test build_forecast(only_broken).available == false
        stale_fallback = vcat(live_cycle_fixture(now(UTC) - Hour(5)), only_broken)
        stale_status = build_status(stale_fallback)
        @test stale_status.available == false
        @test stale_status.forecast_issue_utc == jdt(maximum(only_broken.issue_time_utc_dt))
        @test build_forecast(stale_fallback).available == false
    end

    @testset "the served cycle's published numbers are the logged ones" begin
        # The cycle-selection change alters WHICH cycle is published and WHETHER it is published. It
        # must not touch a published value: every number below is the exact Float64 the log carries.
        issue = now(UTC) - Minute(10)
        plain = live_cycle_fixture(issue; served_pred=-33.25, served_lo=-47.125, served_hi=-19.5)
        fc = build_forecast(plain)
        @test fc.available && fc.superseded_cycle_incomplete == false
        @test all(h -> h.pred_dst_nt === -33.25, fc.horizons)
        @test all(h -> h.ci05_dst_nt === -47.125, fc.horizons)
        @test all(h -> h.ci95_dst_nt === -19.5, fc.horizons)
        @test all(h -> h.severity_dst_nt === -33.25, fc.horizons)
        @test all(h -> h.severity_ci05_dst_nt === -47.125, fc.horizons)
        st = build_status(plain)
        @test st.threat.point_min_dst_nt === -33.25
        @test st.threat.interval_lower_edge_min_dst_nt === -47.125
        @test st.latest_observation.dst_nt === -20.0
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

    @testset "served pipeline labels and depth-safe severity" begin
        iss = now(UTC) - Minute(20)
        # The super-learner label is the current product label; the two earlier labels remain
        # acceptable because the served stage falls back through them and discloses it per row.
        for label in (CURRENT_V2_SERVED_MODEL_VERSION, STACK_V2_SERVED_MODEL_VERSION,
                      PREVIOUS_V2_SERVED_MODEL_VERSION)
            st = build_status(live_cycle_fixture(iss; served_model=label))
            @test st.available == true
            @test st.served_model_version == label
        end
        @test !build_status(live_cycle_fixture(iss; served_model="v2.9+made+up")).available
        @test CURRENT_V2_SERVED_MODEL_VERSION ==
              "v2.4+sindy20x11+superlearner10floor+conformal"
        @test STACK_V2_SERVED_MODEL_VERSION ==
              "v2.2+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia+staticstack(sindy60_fit407598)"
        @test V2_3_SHADOW_MODEL_VERSION ==
              "v2.3-shadow+sindy20x11+L1A+ADC(magnetic,K25)+T1rcal+LAT+E"
        # The chain is ordered strongest to weakest, which is how a mixed cycle picks its label.
        @test ACCEPTED_V2_SERVED_MODEL_VERSIONS ==
              (CURRENT_V2_SERVED_MODEL_VERSION, STACK_V2_SERVED_MODEL_VERSION,
               PREVIOUS_V2_SERVED_MODEL_VERSION)

        # Depth-safe severity: a shallower served center must not lower the published threat below
        # what either stage it replaced would have warned.
        shallow = live_cycle_fixture(iss; served_model=CURRENT_V2_SERVED_MODEL_VERSION,
                                     served_pred=-40.0, served_lo=-50.0, served_hi=-30.0,
                                     v2_2_stack_pred=-60.0, v2_1_served_pred=-120.0)
        st_shallow = build_status(shallow)
        @test st_shallow.threat.point_min_dst_nt == -120.0
        @test st_shallow.threat.level == 3
        # The stack partner counts on its own: a candidate shallower than the stack but deeper than
        # V2.1 is still held to the stack.
        stack_partner = live_cycle_fixture(iss; served_model=CURRENT_V2_SERVED_MODEL_VERSION,
                                           served_pred=-40.0, served_lo=-50.0, served_hi=-30.0,
                                           v2_2_stack_pred=-95.0, v2_1_served_pred=-60.0)
        @test build_status(stack_partner).threat.point_min_dst_nt == -95.0
        # A deeper served center still escalates past both partners.
        deep = live_cycle_fixture(iss; served_model=CURRENT_V2_SERVED_MODEL_VERSION,
                                  served_pred=-220.0, served_lo=-240.0, served_hi=-200.0,
                                  v2_2_stack_pred=-130.0, v2_1_served_pred=-120.0)
        @test build_status(deep).threat.point_min_dst_nt == -220.0
        @test build_status(deep).threat.level == 4
        # A cycle whose stack stage could not act has no stack column and is held to V2.1 alone.
        no_stack = live_cycle_fixture(iss; served_model=PREVIOUS_V2_SERVED_MODEL_VERSION,
                                      served_pred=-40.0, served_lo=-50.0, served_hi=-30.0,
                                      v2_1_served_pred=-120.0)
        @test build_status(no_stack).threat.point_min_dst_nt == -120.0
        # A pre-stack row without either column keeps the served center as the severity input.
        legacy = live_cycle_fixture(iss; served_model=PREVIOUS_V2_SERVED_MODEL_VERSION,
                                    served_pred=-40.0, served_lo=-50.0, served_hi=-30.0)
        @test build_status(legacy).threat.point_min_dst_nt == -40.0
    end

    @testset "the watch tier is taken on the depth-safe center, not the served band" begin
        # Reproduces the escalation the stack stage could otherwise drop. Same physics, two products:
        # the V2.1 operator warned at -95 nT with a [-105, -85] band, and the stack reports a shallower
        # -88 nT with the band shifted up to [-98, -78]. The point tier is depth-safe already, but a
        # watch taken on the shifted edge would fall from the intense tier to none, so the outbound
        # alert level would drop below the previous product on identical inputs.
        iss = now(UTC) - Minute(20)
        prior = build_status(live_cycle_fixture(iss;
            served_model=PREVIOUS_V2_SERVED_MODEL_VERSION,
            served_pred=-95.0, served_lo=-105.0, served_hi=-85.0))
        @test prior.threat.level == 2
        @test prior.threat.watch == true
        @test prior.threat.watch_level == 3
        @test compute_alert_state(prior, nothing, nothing).level == 3

        stacked = build_status(live_cycle_fixture(iss;
            served_model=CURRENT_V2_SERVED_MODEL_VERSION,
            served_pred=-88.0, served_lo=-98.0, served_hi=-78.0,
            v2_2_stack_pred=-90.0, v2_1_served_pred=-95.0))
        @test stacked.threat.level == 2
        @test stacked.threat.watch == true
        @test stacked.threat.watch_level == 3
        # The edge is the served edge lowered by exactly the amount the point was lowered.
        @test stacked.threat.interval_lower_edge_min_dst_nt == -105.0
        @test compute_alert_state(stacked, nothing, nothing).level == 3
        # The alert text quotes the same depth-safe edge it escalated on.
        stacked_df = live_cycle_fixture(iss;
            served_model=CURRENT_V2_SERVED_MODEL_VERSION,
            served_pred=-88.0, served_lo=-98.0, served_hi=-78.0,
            v2_2_stack_pred=-90.0, v2_1_served_pred=-95.0)
        watch_alerts = [a for a in build_alerts(stacked_df, stacked).alerts if a.kind == "watch"]
        @test length(watch_alerts) == 1
        @test occursin("-105", watch_alerts[1].message)

        # A deeper stacked center must not have its band pulled down: the shift is one-sided.
        deeper = build_status(live_cycle_fixture(iss;
            served_model=CURRENT_V2_SERVED_MODEL_VERSION,
            served_pred=-120.0, served_lo=-130.0, served_hi=-110.0,
            v2_2_stack_pred=-100.0, v2_1_served_pred=-95.0))
        @test deeper.threat.interval_lower_edge_min_dst_nt == -130.0
        # A pre-stack row without the continuity column keeps the served edge unchanged.
        legacy = build_status(live_cycle_fixture(iss;
            served_model=PREVIOUS_V2_SERVED_MODEL_VERSION,
            served_pred=-88.0, served_lo=-98.0, served_hi=-78.0))
        @test legacy.threat.interval_lower_edge_min_dst_nt == -98.0
        @test legacy.threat.watch == false

        # The severity rule is the package's own, loaded from the shared definition rather than
        # restated here: a second copy in the app is exactly how the two could drift.
        @test v22_serving_depth_safe_center(-88.0, -95.0) == -95.0
        @test v22_serving_depth_safe_center(-120.0, -95.0) == -120.0
        @test v22_serving_depth_safe_center(NaN, -95.0) == -95.0
        @test v22_serving_depth_safe_center(-88.0, NaN) == -88.0
        # The three-stage rule is the same shared definition, and it must reduce to the two-stage one.
        @test v24_serving_depth_safe_center(-88.0, -90.0, -95.0) == -95.0
        @test v24_serving_depth_safe_center(-88.0, -95.0, -90.0) == -95.0
        @test v24_serving_depth_safe_center(-120.0, -95.0, -90.0) == -120.0
        @test v24_serving_depth_safe_center(-88.0, NaN, -95.0) == -95.0
        @test v24_serving_depth_safe_center(-88.0, NaN, NaN) == -88.0
        @test v24_serving_depth_safe_center(-88.0, -95.0) ==
              v22_serving_depth_safe_center(-88.0, -95.0)
        @test count(f -> isfile(f), _DEPTH_SAFE_CANDIDATES) >= 1
        api_source = read(joinpath(APPSRC, "forecast_api.jl"), String)
        @test !occursin("min(Float64(served), Float64(previous))", api_source)
        # A second copy of the rule inside the application is exactly how the two could drift.
        @test !occursin("function v24_serving_depth_safe_center", api_source)
    end

    @testset "the watch edge is the deepest predecessor edge, not the served band shifted" begin
        # The band changed source with the super-learner: a V2.4e row carries the depth-stratified
        # conformal half-width, which in the shallow bins is narrower than every band the earlier
        # products published. Shifting the served edge down by the amount the point was lowered
        # therefore still publishes a shallower edge than the predecessor's own: the served center is
        # -88 nT with a +-4 nT band, while the V2.1 operator warned at -95 nT with a +-10 nT band, so
        # the shift gives -99 nT where the operator's edge was -105 nT — one storm tier shallower.
        iss = now(UTC) - Minute(20)
        narrow = live_cycle_fixture(iss; served_model=CURRENT_V2_SERVED_MODEL_VERSION,
            served_pred=-88.0, served_lo=-92.0, served_hi=-84.0,
            v2_2_stack_pred=-90.0, v2_1_served_pred=-95.0,
            v2_2_stack_ci05=-100.0, v2_1_served_ci05=-105.0)
        st = build_status(narrow)
        @test st.threat.interval_lower_edge_min_dst_nt == -105.0
        @test st.threat.watch == true
        @test st.threat.watch_level == 3
        # The payload says which stage's own edge is published, so an operator can see why the watch
        # is deeper than the served band.
        @test st.threat.interval_lower_edge_source == "v2_1_served"
        # The stack partner counts on its own, exactly as the center partner does.
        stack_edge = live_cycle_fixture(iss; served_model=CURRENT_V2_SERVED_MODEL_VERSION,
            served_pred=-88.0, served_lo=-92.0, served_hi=-84.0,
            v2_2_stack_pred=-90.0, v2_1_served_pred=-95.0,
            v2_2_stack_ci05=-112.0, v2_1_served_ci05=-105.0)
        @test build_status(stack_edge).threat.interval_lower_edge_min_dst_nt == -112.0
        @test build_status(stack_edge).threat.interval_lower_edge_source == "v2_2_stack"
        # A served edge already deeper than both predecessors is published unchanged: the rule is a
        # minimum, so it is idempotent and never widens a band the served product did not issue.
        deeper = live_cycle_fixture(iss; served_model=CURRENT_V2_SERVED_MODEL_VERSION,
            served_pred=-140.0, served_lo=-160.0, served_hi=-120.0,
            v2_2_stack_pred=-90.0, v2_1_served_pred=-95.0,
            v2_2_stack_ci05=-100.0, v2_1_served_ci05=-105.0)
        @test build_status(deeper).threat.interval_lower_edge_min_dst_nt == -160.0
        @test build_status(deeper).threat.interval_lower_edge_source == "served"
        # A non-finite predecessor edge is dropped rather than propagated, the same way a non-finite
        # center partner is.
        holed = live_cycle_fixture(iss; served_model=CURRENT_V2_SERVED_MODEL_VERSION,
            served_pred=-88.0, served_lo=-92.0, served_hi=-84.0,
            v2_2_stack_pred=-90.0, v2_1_served_pred=-95.0,
            v2_2_stack_ci05=NaN, v2_1_served_ci05=-105.0)
        @test build_status(holed).threat.interval_lower_edge_min_dst_nt == -105.0
        # Per horizon the payload carries the published edge, its source and both partner edges.
        fc = build_forecast(narrow)
        for h in fc.horizons
            @test h.ci05_dst_nt == -92.0
            @test h.severity_dst_nt == -95.0
            @test h.severity_ci05_dst_nt == -105.0
            @test h.severity_ci05_source == "v2_1_served"
            @test h.v2_1_served_ci05_dst_nt == -105.0
            @test h.v2_2_stack_ci05_dst_nt == -100.0
        end
        # Backwards compatibility: a row written before the predecessor edges were logged keeps the
        # earlier center-shift behaviour and discloses that it is the fallback rule.
        legacy = live_cycle_fixture(iss; served_model=CURRENT_V2_SERVED_MODEL_VERSION,
            served_pred=-88.0, served_lo=-92.0, served_hi=-84.0,
            v2_2_stack_pred=-90.0, v2_1_served_pred=-95.0)
        legacy_status = build_status(legacy)
        @test legacy_status.threat.interval_lower_edge_min_dst_nt == -99.0
        @test legacy_status.threat.interval_lower_edge_source == "legacy_center_shift"
        for h in build_forecast(legacy).horizons
            @test h.severity_ci05_source == "legacy_center_shift"
            @test h.v2_1_served_ci05_dst_nt === nothing
            @test h.v2_2_stack_ci05_dst_nt === nothing
        end
    end

    @testset "the forecast payload exposes the alerting center per horizon" begin
        iss = now(UTC) - Minute(20)
        df = live_cycle_fixture(iss; served_model=CURRENT_V2_SERVED_MODEL_VERSION,
                                served_pred=-88.0, served_lo=-98.0, served_hi=-78.0,
                                v2_2_stack_pred=-90.0, v2_1_served_pred=-95.0,
                                v2_2_status="ok", v24_status="ok", v24_pred=-88.0,
                                v24_guard_applied=false, v24_projection_applied=false,
                                v24_regime_cell="quiet/shallow")
        fc = build_forecast(df)
        @test fc.available == true
        @test fc.served_product == "V2.4"
        for h in fc.horizons
            @test h.pred_dst_nt == -88.0
            @test h.severity_dst_nt == -95.0
            @test h.severity_ci05_dst_nt == -105.0
            @test h.v2_1_served_pred_dst_nt == -95.0
            @test h.v2_2_stack_pred_dst_nt == -90.0
            @test h.served_model_version == CURRENT_V2_SERVED_MODEL_VERSION
            @test h.v2_2_status == "ok"
            # The served stage discloses itself per horizon: a fallback row says which stage acted.
            @test h.v24_status == "ok"
            @test h.v24_pred_dst_nt == -88.0
            @test h.v24_guard_applied == false
            # The physical projection is disclosed alongside the guard, so an operator can tell a
            # clamped center from a combination that happened to land on the same number.
            @test h.v24_projection_applied == false
            @test h.v24_regime_cell == "quiet/shallow"
        end
        # A projected row surfaces the flag as true, so the disclosure is read from the row rather
        # than reported as false for every cycle.
        projected = build_forecast(live_cycle_fixture(iss;
            served_model=CURRENT_V2_SERVED_MODEL_VERSION, served_pred=-88.0, served_lo=-98.0,
            served_hi=-78.0, v2_2_stack_pred=-90.0, v2_1_served_pred=-95.0,
            v2_2_status="ok", v24_status="ok", v24_pred=-88.0, v24_guard_applied=false,
            v24_projection_applied=true, v24_regime_cell="quiet/shallow"))
        for h in projected.horizons
            @test h.v24_projection_applied == true
        end
        # A row from a log that predates the column reports `nothing`, not `false`: an absent
        # disclosure is not the same statement as a projection that did not act.
        for h in build_forecast(live_cycle_fixture(iss;
                served_model=CURRENT_V2_SERVED_MODEL_VERSION, served_pred=-88.0,
                served_lo=-98.0, served_hi=-78.0, v2_2_stack_pred=-90.0,
                v2_1_served_pred=-95.0, v24_status="ok")).horizons
            @test h.v24_projection_applied === nothing
        end
        # A cycle served by the first fallback stage carries the stack label and its own status.
        fell_back = build_forecast(live_cycle_fixture(iss;
            served_model=STACK_V2_SERVED_MODEL_VERSION, served_pred=-88.0, served_lo=-98.0,
            served_hi=-78.0, v2_2_stack_pred=-88.0, v2_1_served_pred=-95.0,
            v2_2_status="ok", v24_status="fallback:deployment_absent"))
        @test fell_back.served_product == "V2.2"
        for h in fell_back.horizons
            @test h.v24_status == "fallback:deployment_absent"
            @test h.v24_pred_dst_nt === nothing
            @test h.severity_dst_nt == -95.0
        end
    end

    @testset "the product name and driver assumption come from the served row" begin
        iss = now(UTC) - Minute(20)
        served = build_status(live_cycle_fixture(iss;
            served_model=CURRENT_V2_SERVED_MODEL_VERSION,
            driver_assumption=SUPERLEARNER_DRIVER_TOKEN))
        @test served.served_product == "V2.4"
        @test occursin("fitted combination of ten causal forecasts",
                       served.lead_time.driver_assumption)
        @test occursin("includes the static regime stack among the combined forecasts",
                       served.lead_time.driver_assumption)
        @test occursin("conformal interval", served.lead_time.driver_assumption)

        stacked = build_status(live_cycle_fixture(iss;
            served_model=STACK_V2_SERVED_MODEL_VERSION,
            driver_assumption=STACK_DRIVER_TOKEN))
        @test stacked.served_product == "V2.2"
        @test occursin("static regime stack", stacked.lead_time.driver_assumption)
        @test occursin("Ballistically propagated L1 forcing", stacked.lead_time.driver_assumption)
        @test !occursin("ten causal forecasts", stacked.lead_time.driver_assumption)

        # A cycle whose stack stage could not act is served by the V2.1 operator, and the payload must
        # describe that operator rather than a stage that did not run.
        fell_back = build_status(live_cycle_fixture(iss;
            served_model=PREVIOUS_V2_SERVED_MODEL_VERSION,
            driver_assumption=V2_1_DRIVER_TOKEN,
            v2_2_status="fallback_v2_1:stack_absent"))
        @test fell_back.served_product == "V2.1"
        @test !occursin("static regime stack", fell_back.lead_time.driver_assumption)
        @test occursin("extreme-Dst inertia guard", fell_back.lead_time.driver_assumption)

        # An unrecorded assumption is reported as unrecorded, not as the full pipeline.
        silent = build_status(live_cycle_fixture(iss;
            served_model=CURRENT_V2_SERVED_MODEL_VERSION))
        @test silent.lead_time.driver_assumption == "unrecorded"

        @test served_product_name("v2.1+sindy20x11+L1A") == "V2.1"
        @test served_product_name(CURRENT_V2_SERVED_MODEL_VERSION) == "V2.4"
        @test served_product_name(nothing) == "unknown"
        # The dashboard names every stage of the served label it can be handed; an unknown token falls
        # back to the raw label rather than claiming a capability the pipeline does not have.
        app_js = read(joinpath(dirname(APPSRC), "public", "app.js"), String)
        for token in ("superlearner10floor", "conformal")
            @test occursin(token, app_js)
        end
        app_source = read(joinpath(dirname(APPSRC), "public", "app.js"), String)
        @test !occursin("Forecast: V2.1", app_source)
        @test !occursin("Product forecast: V2.1.", app_source)
        @test occursin("V2.1 core trajectory (display)", app_source)
    end

    @testset "a cycle whose stack stage healed mid-cycle stays available" begin
        # The stack is loaded per issuance, so the four horizons of one cycle can legitimately carry
        # different accepted labels. Refusing such a cycle would blank the dashboard and suppress its
        # alerts over a per-row degradation that the log discloses, which reads as an all-clear.
        iss = now(UTC) - Minute(20)
        mixed = live_cycle_fixture(iss; served_model=CURRENT_V2_SERVED_MODEL_VERSION,
                                   served_pred=-95.0, served_lo=-105.0, served_hi=-85.0)
        mixed.sub_hourly_model_version[2] = PREVIOUS_V2_SERVED_MODEL_VERSION
        st = build_status(mixed)
        @test st.available == true
        # The cycle is reported under the weakest label any of its rows carries.
        @test st.served_model_version == PREVIOUS_V2_SERVED_MODEL_VERSION
        @test st.served_product == "V2.1"
        @test length(st.served_model_versions) == 2
        # With all three stages present in one cycle the weakest still wins.
        three = live_cycle_fixture(iss; served_model=CURRENT_V2_SERVED_MODEL_VERSION,
                                  served_pred=-95.0, served_lo=-105.0, served_hi=-85.0)
        three.sub_hourly_model_version[2] = STACK_V2_SERVED_MODEL_VERSION
        three.sub_hourly_model_version[3] = PREVIOUS_V2_SERVED_MODEL_VERSION
        @test build_status(three).served_model_version == PREVIOUS_V2_SERVED_MODEL_VERSION
        # A cycle that lost only the super-learner is reported under the stack label.
        stack_only = live_cycle_fixture(iss; served_model=CURRENT_V2_SERVED_MODEL_VERSION,
                                       served_pred=-95.0, served_lo=-105.0, served_hi=-85.0)
        stack_only.sub_hourly_model_version[4] = STACK_V2_SERVED_MODEL_VERSION
        @test build_status(stack_only).served_model_version == STACK_V2_SERVED_MODEL_VERSION
        @test build_forecast(mixed).served_model_version == PREVIOUS_V2_SERVED_MODEL_VERSION
        @test build_alerts(mixed, st).threat_level == st.threat.level
        # A label this build does not know still fails closed, mixed in or not.
        unknown = live_cycle_fixture(iss; served_model=CURRENT_V2_SERVED_MODEL_VERSION)
        unknown.sub_hourly_model_version[3] = CURRENT_V2_SERVED_MODEL_VERSION * "+unpinned"
        @test build_status(unknown).available == false
        @test build_forecast(unknown).available == false
    end

    @testset "verified rows are counted per served pipeline" begin
        # Verified rows accumulate across served pipelines. Presenting them pooled reports a record
        # earned by the previous pipeline as the current product's record.
        now0 = floor(now(UTC), Hour)
        rows = DataFrame()
        for (k, label) in enumerate(vcat(fill(PREVIOUS_V2_SERVED_MODEL_VERSION, 5),
                                        fill(CURRENT_V2_SERVED_MODEL_VERSION, 3)))
            frame = live_cycle_fixture(now0 - Hour(30 + k);
                                       served_model=label, observations=-25.0 - k)
            rows = isempty(rows) ? frame : vcat(rows, frame)
        end
        current = live_cycle_fixture(now(UTC) - Minute(20);
                                    served_model=CURRENT_V2_SERVED_MODEL_VERSION)
        df = vcat(rows, current; cols=:union)
        cal = calibration_summary(df)
        @test cal.current_served_model == CURRENT_V2_SERVED_MODEL_VERSION
        by_label = Dict(b.served_model_version => b for b in cal.by_served_model)
        @test length(by_label) == 2
        @test by_label[PREVIOUS_V2_SERVED_MODEL_VERSION].n == 5 * length(LIVE_CYCLE_HORIZONS)
        @test by_label[CURRENT_V2_SERVED_MODEL_VERSION].n == 3 * length(LIVE_CYCLE_HORIZONS)
        @test by_label[CURRENT_V2_SERVED_MODEL_VERSION].product == "V2.4"
        @test cal.n_verified_current_served_model ==
              by_label[CURRENT_V2_SERVED_MODEL_VERSION].n
        @test cal.n_verified_current_served_model < cal.n_verified
        @test sum(b.n for b in cal.by_served_model) == cal.n_verified
    end

    @testset "the health endpoint states which product is served" begin
        # A fresh log that has silently reverted to the previous served pipeline is not a healthy
        # deployment, so the identity and the trailing fallback rate belong beside the freshness.
        now0 = floor(now(UTC), Hour)
        frames = DataFrame[]
        for k in 1:6
            label = k == 6 ? PREVIOUS_V2_SERVED_MODEL_VERSION : CURRENT_V2_SERVED_MODEL_VERSION
            push!(frames, live_cycle_fixture(now0 - Hour(6 - k) - Minute(47);
                                             served_model=label,
                                             v2_2_status="ok",
                                             v24_status=k == 6 ? "fallback:deployment_absent" : "ok"))
        end
        df = reduce(vcat, frames)
        df[!, :v23_status] = fill("ok:e_layer_pending", nrow(df))
        df[!, :v23_e_layer_applied] = fill(false, nrow(df))
        df[!, :v23_shadow_model_version] = fill(V2_3_SHADOW_MODEL_VERSION, nrow(df))
        health = build_served_health(df)
        @test health.cycles_considered == 6
        @test health.served_model_version == PREVIOUS_V2_SERVED_MODEL_VERSION
        @test health.served_product == "V2.1"
        @test health.served_fallback_cycles == 1
        @test health.served_fallback_rate == round(1 / 6; digits=4)
        @test health.newest_cycle_is_fallback == true
        # The chain is reported stage by stage: which fallback the window landed on is the operator's
        # first question, and a rate alone cannot answer it.
        @test health.served_v2_1_cycles == 1
        @test health.served_stack_cycles == 0
        # The shadow center is available even while its error layer is pending.
        @test health.shadow_available_rate == 1.0
        @test health.shadow_e_layer_rate == 0.0
        @test health.shadow_model_version == V2_3_SHADOW_MODEL_VERSION
        @test build_served_health(DataFrame()).cycles_considered == 0
    end

    @testset "health survives a trailing window that spans the shadow-schema change" begin
        # A build carrying the shadow columns is deployed onto a log that is already being appended
        # to, so for as long as the trailing window is, that window straddles the schema change and
        # the earlier cycles carry no shadow columns at all. Those fields read back as `missing`, and
        # `missing == 1` is three-valued: a predicate written that way makes `any` return `missing`,
        # the health summary throws, and the endpoint reports no served identity at all during the
        # first day of the deployment it is supposed to be reporting on.
        now0 = floor(now(UTC), Hour)
        legacy = reduce(vcat, [live_cycle_fixture(now0 - Hour(24 - k) - Minute(47);
                                                  served_model=PREVIOUS_V2_SERVED_MODEL_VERSION)
                               for k in 1:20])
        fresh = reduce(vcat, [live_cycle_fixture(now0 - Hour(4 - k) - Minute(47);
                                                 served_model=CURRENT_V2_SERVED_MODEL_VERSION,
                                                 v2_2_status="ok", v24_status="ok")
                              for k in 1:4])
        fresh[!, :v23_status] = fill("ok", nrow(fresh))
        fresh[!, :v23_e_layer_applied] = fill(true, nrow(fresh))
        fresh[!, :v23_shadow_model_version] = fill(V2_3_SHADOW_MODEL_VERSION, nrow(fresh))
        df = vcat(legacy, fresh; cols=:union)
        @test any(ismissing, df.v23_e_layer_applied)          # the schema change is in the window
        health = build_served_health(df)
        # Cycles written before the current served stage existed carry no `v24_status`; they are
        # disclosed as excluded, not counted as fallbacks, so the first day after a deployment onto
        # an existing log does not read as a 20/24 fallback rate (same rule as the readiness audit).
        @test health.cycles_considered == 4
        @test health.pre_stage_cycles_excluded == 20
        @test health.served_model_version == CURRENT_V2_SERVED_MODEL_VERSION
        @test health.served_product == "V2.4"
        @test health.served_fallback_cycles == 0
        @test health.served_fallback_rate == 0.0
        @test health.newest_cycle_is_fallback == false
        @test health.shadow_cycles_considered == 4
        @test health.shadow_available_cycles == 4
        @test health.shadow_available_rate == 1.0
        @test health.shadow_e_layer_cycles == 4
        @test health.shadow_model_version == V2_3_SHADOW_MODEL_VERSION
        # Legacy-only window: nothing staged yet -> no rate, no exception.
        legacy_only = build_served_health(legacy)
        @test legacy_only.cycles_considered == 0
        @test legacy_only.pre_stage_cycles_excluded == 20
        @test legacy_only.served_fallback_rate === nothing

        # A row with no served label at all is a fallback cycle with no reportable identity, which is
        # what the endpoint must say; it is not a reason to drop the whole summary.
        unlabelled = copy(df)
        unlabelled[!, :sub_hourly_model_version] =
            Vector{Union{Missing, String}}(unlabelled.sub_hourly_model_version)
        unlabelled.sub_hourly_model_version[end] = missing
        blind = build_served_health(unlabelled)
        @test blind.served_fallback_cycles == 1
        @test blind.newest_cycle_is_fallback == true
        @test blind.served_model_version === nothing
        @test blind.served_product === nothing
    end

    @testset "a mid-cycle stage change still names the stage the cycle was served by" begin
        # The four horizons of one cycle can carry different accepted labels and therefore different
        # driver-assumption tokens. Reading the assumption as a common field of the cycle finds no
        # single value and reports it as never recorded, which describes a logging failure rather than
        # the disclosed per-row degradation the log actually recorded. The cycle is published under its
        # weakest label, so the assumption must be the one that label was written with.
        iss = now(UTC) - Minute(20)
        mixed = live_cycle_fixture(iss; served_model=CURRENT_V2_SERVED_MODEL_VERSION,
                                   driver_assumption=STACK_DRIVER_TOKEN, v2_2_status="ok")
        mixed.sub_hourly_model_version[2] = PREVIOUS_V2_SERVED_MODEL_VERSION
        mixed.driver_assumption[2] = V2_1_DRIVER_TOKEN
        mixed.v2_2_status[2] = "fallback_v2_1:stack_absent"
        st = build_status(mixed)
        @test st.available == true
        @test st.served_model_version == PREVIOUS_V2_SERVED_MODEL_VERSION
        @test st.lead_time.driver_assumption != "unrecorded"
        @test !occursin("static regime stack", st.lead_time.driver_assumption)
        @test occursin("extreme-Dst inertia guard", st.lead_time.driver_assumption)
        @test occursin("Ballistically propagated L1 forcing", st.lead_time.driver_assumption)

        # The reverse mix reports the same weakest stage, whichever horizon fell back.
        other = live_cycle_fixture(iss; served_model=PREVIOUS_V2_SERVED_MODEL_VERSION,
                                   driver_assumption=V2_1_DRIVER_TOKEN,
                                   v2_2_status="fallback_v2_1:stack_absent")
        other.sub_hourly_model_version[4] = CURRENT_V2_SERVED_MODEL_VERSION
        other.driver_assumption[4] = STACK_DRIVER_TOKEN
        other.v2_2_status[4] = "ok"
        @test build_status(other).lead_time.driver_assumption ==
              st.lead_time.driver_assumption

        # A uniform stacked cycle still reports the stacked sentence.
        whole = live_cycle_fixture(iss; served_model=CURRENT_V2_SERVED_MODEL_VERSION,
                                   driver_assumption=STACK_DRIVER_TOKEN)
        @test occursin("static regime stack", build_status(whole).lead_time.driver_assumption)
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

    @testset "dashboard launch bounds cold-request compilation" begin
        template = read(joinpath(@__DIR__, "..", "..", "deploy",
                                 "com.example.solarsindy.dashboard.plist"), String)
        @test occursin("<string>--startup-file=no</string>", template)
        @test occursin("<string>--compile=min</string>", template)
        @test first(findfirst("<string>--compile=min</string>", template)) <
              first(findfirst("<string>--project=__APP_DIR__</string>", template))
    end

    @testset "launchd installer retries bootstrap without restart-killing the service" begin
        project_root = normpath(joinpath(@__DIR__, "..", ".."))
        installer_path = joinpath(project_root, "deploy", "install_launchd.sh")
        installer = read(installer_path, String)
        @test startswith(installer, "#!/bin/bash\n")
        @test occursin("for attempt in 1 2 3", installer)
        @test occursin("bootstrap_service \"\$label\" \"\$dst\"", installer)
        @test occursin("launchctl kickstart \"\$DOMAIN/\$label\"", installer)
        @test !occursin("launchctl kickstart -k", installer)

        if Sys.isapple()
            mktempdir() do dir
                fake_bin = joinpath(dir, "bin")
                fake_home = joinpath(dir, "home")
                event_log = joinpath(dir, "launchctl.log")
                mkpath(fake_bin)
                fake_launchctl = joinpath(fake_bin, "launchctl")
                write(fake_launchctl, raw"""#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >> "$SOLARSINDY_TEST_LAUNCHCTL_LOG"
case "$1" in
  bootout|enable|kickstart) exit 0 ;;
  bootstrap)
    [ "${SOLARSINDY_TEST_BOOTSTRAP_MODE:-transient}" != "persistent" ] || exit 5
    attempts=$(grep -c '^bootstrap ' "$SOLARSINDY_TEST_LAUNCHCTL_LOG" || true)
    [ "$attempts" -ge 3 ] || exit 5
    exit 0
    ;;
  *) exit 2 ;;
esac
""")
                chmod(fake_launchctl, 0o755)
                cmd = `/bin/bash $installer_path $project_root dashboard`
                installer_cmd(mode) = addenv(cmd,
                    "PATH" => string(fake_bin, ":", get(ENV, "PATH", "")),
                    "HOME" => fake_home,
                    "SOLARSINDY_TEST_LAUNCHCTL_LOG" => event_log,
                    "SOLARSINDY_TEST_BOOTSTRAP_MODE" => mode,
                    "SOLARSINDY_JULIA" => "/usr/bin/true",
                    "SOLARSINDY_MONITOR_DIR" => joinpath(dir, "monitor"),
                    "SOLARSINDY_ORG" => "installer-test",
                    "SOLARSINDY_LOAD" => "1",
                )
                run(pipeline(installer_cmd("transient"); stdout=devnull, stderr=devnull))
                events = readlines(event_log)
                @test count(startswith("bootstrap "), events) == 3
                @test count(startswith("enable "), events) == 1
                @test count(startswith("kickstart "), events) == 1
                @test all(!occursin("kickstart -k", event) for event in events)
                @test startswith(last(events), "kickstart gui/")

                # A persistent launchctl error must remain bounded and propagate as failure.
                write(event_log, "")
                persistent = run(pipeline(installer_cmd("persistent");
                                          stdout=devnull, stderr=devnull); wait=false)
                wait(persistent)
                @test !success(persistent)
                persistent_events = readlines(event_log)
                @test count(startswith("bootstrap "), persistent_events) == 3
                @test !any(startswith("enable "), persistent_events)
                @test !any(startswith("kickstart "), persistent_events)
            end
        end
    end

    @testset "external watchdog outage state machine" begin
        wd = normpath(joinpath(@__DIR__, "..", "..", "deploy", "watchdog.sh"))
        @test isfile(wd)
        wd_src = read(wd, String)
        # Structural pins (hold even where bash/curl differ): the data-route wedge class, the stable
        # kind-keyed dedup signature, and the sentinel-ownership marker the recovery branch keys on.
        @test occursin("dash_wedged", wd_src)
        @test occursin("PROBLEM: \$kinds", wd_src)
        @test occursin("Source: external watchdog", wd_src)

        bash = Sys.isunix() ? Sys.which("bash") : nothing
        if bash === nothing
            @warn "external watchdog harness skipped: POSIX bash required" wd
        else
            mktempdir() do dir
                # Shadow curl on PATH so probe reachability and webhook delivery are fully controllable
                # and no network is touched. The real stat/date/grep/rm the script needs stay resolvable
                # because the real PATH is appended.
                fake_bin = joinpath(dir, "bin"); mkpath(fake_bin)
                fake_curl = joinpath(fake_bin, "curl")
                write(fake_curl, raw"""#!/bin/bash
set -u
is_post=0
url=""
for a in "$@"; do
  [ "$a" = "POST" ] && is_post=1
  url="$a"
done
if [ "$is_post" = "1" ]; then
  printf '%s\n' "$*" >> "$WD_TEST_WEBHOOK_LOG"
  exit 0
fi
case "$url" in
  */api/health) [ "${WD_TEST_HEALTH_OK:-1}" = "1" ] && exit 0 || exit 7 ;;
  */api/forecast) [ "${WD_TEST_DATA_OK:-1}" = "1" ] && exit 0 || exit 7 ;;
  *) exit 0 ;;
esac
""")
                chmod(fake_curl, 0o755)

                webhook_log = joinpath(dir, "webhook.log")
                run_wd = function (mon; health, data)
                    cmd = addenv(`$bash $wd`,
                        "PATH" => string(fake_bin, ":", get(ENV, "PATH", "")),
                        "SOLARSINDY_MONITOR_DIR" => mon,
                        "SOLARSINDY_WATCHDOG_STALE_SEC" => "999999",  # isolate the dashboard branches
                        "SOLARSINDY_WATCHDOG_DASH_URL" => "http://127.0.0.1:65999/api/health",
                        "SWM_WEBHOOK_URL" => "http://webhook.test/post",
                        "WD_TEST_WEBHOOK_LOG" => webhook_log,
                        "WD_TEST_HEALTH_OK" => health ? "1" : "0",
                        "WD_TEST_DATA_OK" => data ? "1" : "0",
                    )
                    run(pipeline(cmd; stdout=devnull, stderr=devnull))
                end
                fresh_log = function (mon)
                    mkpath(mon)
                    write(joinpath(mon, "live_forecast_log.csv"),
                          "issue_time_utc\n2026-07-15T12:00\n")   # present + just-written => not stale
                end
                state_of = function (mon)
                    sf = joinpath(mon, "logs", "watchdog_state")
                    isfile(sf) ? read(sf, String) : ""
                end
                sentinel_of = mon -> joinpath(mon, "OUTAGE.md")
                webhook_lines = () -> isfile(webhook_log) ? readlines(webhook_log) : String[]

                # --- Cycle A: detect -> sentinel -> dedup -> recover --------------------------------
                monA = joinpath(dir, "monA"); fresh_log(monA)
                run_wd(monA; health=false, data=true)                       # A1 detect: health down
                @test isfile(sentinel_of(monA))
                @test occursin("Source: external watchdog", read(sentinel_of(monA), String))
                @test state_of(monA) == "PROBLEM: dash_down"
                @test length(webhook_lines()) == 1
                @test occursin("\"kind\":\"outage\"", last(webhook_lines()))

                run_wd(monA; health=false, data=true)                       # A2 dedup: same kind
                @test state_of(monA) == "PROBLEM: dash_down"
                @test length(webhook_lines()) == 1                          # no second POST

                run_wd(monA; health=true, data=true)                        # A3 recover
                @test !isfile(sentinel_of(monA))                            # watchdog sentinel cleared
                @test state_of(monA) == "OK"
                @test length(webhook_lines()) == 2
                @test occursin("\"kind\":\"recovery\"", last(webhook_lines()))

                # --- Cycle B: functional data-route wedge (health answers, /api/forecast hangs) -----
                monB = joinpath(dir, "monB"); fresh_log(monB)
                wh_b = length(webhook_lines())
                run_wd(monB; health=true, data=false)
                @test state_of(monB) == "PROBLEM: dash_wedged"   # only the data-route branch sets this kind
                @test occursin("Source: external watchdog", read(sentinel_of(monB), String))
                @test length(webhook_lines()) == wh_b + 1
                @test occursin("data route unresponsive", last(webhook_lines()))  # wedge message delivered

                # --- Cycle C: never clobber or clear a daemon-authored dead-man sentinel ------------
                monC = joinpath(dir, "monC"); fresh_log(monC)
                daemon_body = "# LIVE FORECAST ISSUANCE OUTAGE\n\nSource: daemon issuance dead-man\n"
                write(sentinel_of(monC), daemon_body)
                wh_c = length(webhook_lines())
                run_wd(monC; health=false, data=true)                       # C1 problem: don't clobber
                @test read(sentinel_of(monC), String) == daemon_body

                run_wd(monC; health=true, data=true)                        # C2 healthy: don't clear
                @test isfile(sentinel_of(monC))
                @test read(sentinel_of(monC), String) == daemon_body
                @test state_of(monC) == "DAEMON_OUTAGE"                      # held, not recovered
                @test !any(occursin("\"kind\":\"recovery\"", l)
                           for l in webhook_lines()[(wh_c + 1):end])
            end
        end
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
        # Compile the route and JSON parser against a fresh local sentinel before starting the
        # stopwatch. The latency assertion is an oracle for waiting on the held third-party task,
        # not for unrelated first-call Julia code generation under system load.
        lock(_SWPC_LOCK) do
            _SWPC_CACHE[] = (time(), _unavailable_swpc_snapshot())
            _SWPC_REFRESH_TASK[] = nothing
        end
        warm_swpc = api_handler("/api/swpc", "", "unused.csv")
        @test JSON3.read(String(warm_swpc.body)).available == false

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
        @test occursin("A displayed 90% target interval extends to", js)
        @test occursin("interval_lower_edge_min_dst_nt", js)
        # The alerting numbers are quoted by the alert text, so they must actually be displayed:
        # a number an operator is alerted on that appears nowhere on the page is not auditable.
        @test occursin("id=\"severity-line\"", html)
        @test occursin("function renderSeverityLine", js)
        @test occursin("renderSeverityLine(H);", js)
        @test occursin("renderSeverityLine(null);", js)
        @test occursin("severity_dst_nt", js)
        @test occursin("severity_ci05_dst_nt", js)
        @test occursin("severity_ci05_source", js)
        @test occursin("Depth-safe alerting values across these horizons", js)
        for token in ("the static regime stack", "the V2.1 operator")
            @test occursin(token, js)
        end
        # An unrecognised pipeline stage falls back to the raw label for the whole pipeline instead of
        # being dropped from the capability list, and the lookup cannot resolve Object.prototype names.
        @test occursin("Object.prototype.hasOwnProperty.call(PIPELINE_CAPS, p)", js)
        @test occursin("return String(label);", js)
        @test !occursin("}).filter(Boolean);", js)
        @test occursin("const PIPELINE_VERSION_TOKENS = new Set(", js)

        # The capability list is executed, not only read: the two defects here are behavioural and a
        # source scan cannot tell a reachable fallback from an unreachable one. The block is extracted
        # verbatim between its own sentinels, so the code under test is the shipped code.
        node = Sys.which("node")
        if node === nothing
            @test_skip "node is unavailable; the pipeline-capability behaviour test needs a JS runtime"
        else
            block = match(
                r"// ---- pipeline-capability block[^\n]*\n(.*?)// ---- end pipeline-capability block ----"s,
                js,
            )
            @test block !== nothing
            probe = block.captures[1] * """
            const out = [
              pipelineCapabilities("v2.4+sindy20x11+superlearner10floor+conformal"),
              pipelineCapabilities("v2.4+sindy20x11+superlearner10floor+quantumtail"),
              pipelineCapabilities("v2.4+sindy20x11+toString"),
              pipelineCapabilities("v2.4+sindy20x11+constructor"),
              pipelineCapabilities("v2.9+madeup"),
              pipelineCapabilities("v2.2+sindy20x11+L1A+staticstack(sindy60_fit407598)"),
              pipelineCapabilities(""),
            ];
            console.log(JSON.stringify(out));
            """
            caps = JSON3.read(read(pipeline(ignorestatus(`$node -e $probe`)), String))
            # Every stage recognised: the capability list is served.
            @test caps[1] == "ten-forecast fitted combination (SINDy-majority), conformal interval"
            # One unrecognised stage: the raw label is served for the whole pipeline rather than a
            # partial list that would describe a future build by this build's capabilities.
            @test caps[2] == "v2.4+sindy20x11+superlearner10floor+quantumtail"
            # A token that resolves through Object.prototype is not a capability.
            @test caps[3] == "v2.4+sindy20x11+toString"
            @test caps[4] == "v2.4+sindy20x11+constructor"
            @test caps[5] == "v2.9+madeup"
            @test caps[6] == "L1 look-ahead, static regime stack"
            @test caps[7] == ""

            # The alerting line is executed too, against a synthetic payload, so the numbers an
            # operator is alerted on are checked as rendered text rather than as source that mentions
            # the right field names.
            severity_block = match(
                r"// ---- severity-line block[^\n]*\n(.*?)// ---- end severity-line block ----"s,
                js,
            )
            @test severity_block !== nothing
            severity_probe = """
            const store = {};
            const el = { textContent: "", classList: { add: c => { store[c] = true; },
                                                       remove: c => { delete store[c]; } } };
            const \$ = () => el;
            const fmt = (x, d=0) => (x === null || x === undefined || Number.isNaN(x)) ? "\\u2014" : Number(x).toFixed(d);
            """ * severity_block.captures[1] * """
            const out = [];
            renderSeverityLine([
              { severity_dst_nt: -95.0, severity_ci05_dst_nt: -105.0, severity_ci05_source: "v2_1_served" },
              { severity_dst_nt: -90.0, severity_ci05_dst_nt: -100.0, severity_ci05_source: "v2_2_stack" },
            ]);
            out.push({ text: el.textContent, hidden: store.hidden === true });
            renderSeverityLine([]);
            out.push({ text: el.textContent, hidden: store.hidden === true });
            renderSeverityLine([
              { severity_dst_nt: -95.0, severity_ci05_dst_nt: -105.0, severity_ci05_source: "toString" },
            ]);
            out.push({ text: el.textContent, hidden: store.hidden === true });
            console.log(JSON.stringify(out));
            """
            rendered = JSON3.read(read(pipeline(ignorestatus(`$node -e $severity_probe`)), String))
            @test occursin("severity centre -95 nT", rendered[1].text)
            @test occursin("watch edge -105 nT", rendered[1].text)
            # The deepest edge belongs to the V2.1 operator, so that is the stage named.
            @test occursin("edge from the V2.1 operator", rendered[1].text)
            @test rendered[1].hidden == false
            # No alerting values in the payload leaves the line empty and hidden rather than showing
            # an em dash where a warning number belongs.
            @test rendered[2].text == ""
            @test rendered[2].hidden == true
            # A source token that resolves through Object.prototype is rendered as the token, never as
            # whatever that property happens to be: the stage that set a warning must be a name.
            @test occursin("edge from toString", rendered[3].text)
            @test !occursin("function", rendered[3].text)
        end
        # The pooled served row of the live-skill table is a mixed-pipeline record until every
        # verified row matured under the current served label; it must not carry the current
        # product's name alone before then. Executed, not pattern-matched.
        if node !== nothing
            label_block = match(
                r"// ---- served-row label helper[^\n]*\n(.*?)// ---- end served-row label helper ----"s,
                js,
            )
            @test label_block !== nothing
            @test occursin("const servedRowName = servedRowLabel(product, nServedLabel, nAllVerified);", js)
            @test occursin("[servedRowName, c.v2_matched_rmse_nt]", js)
            @test !occursin("[`\${product} served`, c.v2_matched_rmse_nt]", js)
            label_probe = label_block.captures[1] * """
            console.log(JSON.stringify([
              servedRowLabel("V2.4", 0, 212),
              servedRowLabel("V2.4", 47, 212),
              servedRowLabel("V2.4", 212, 212),
              servedRowLabel("V2.4", 250, 212),
              servedRowLabel("V2.4", null, 212),
            ]));
            """
            labels = JSON3.read(read(pipeline(ignorestatus(`$node -e $label_probe`)), String))
            @test labels[1] == "served pipelines (mixed record; 0 of 212 rows under V2.4)"
            @test labels[2] == "served pipelines (mixed record; 47 of 212 rows under V2.4)"
            @test labels[3] == "V2.4 served"
            @test labels[4] == "V2.4 served"
            @test labels[5] == "served pipelines (mixed record; 0 of 212 rows under V2.4)"

            # The superseded-cycle disclosure reaches all three payloads and was rendered nowhere:
            # the panel showed a forecast as current with no indication that the newest issuance had
            # not completed. It is a notice on the forecast panel and nowhere else, so it is checked
            # where it is written and executed against the three payload shapes it can receive.
            @test occursin("id=\"forecast-supersede\"", html)
            @test occursin("renderSupersededNotice(forecast);", js)
            @test count(occursin("forecast-supersede", line)
                        for line in eachline(IOBuffer(js))) == 1
            notice_block = match(
                r"// ---- superseded-cycle notice block[^\n]*\n(.*?)// ---- end superseded-cycle notice block ----"s,
                js,
            )
            @test notice_block !== nothing
            notice_probe = """
            const store = {};
            const el = { textContent: "", classList: { add: c => { store[c] = true; },
                                                       remove: c => { delete store[c]; } } };
            const \$ = (id) => (id === "forecast-supersede" ? el : null);
            """ * notice_block.captures[1] * """
            const out = [];
            const record = () => out.push({ text: el.textContent, hidden: store.hidden === true });
            renderSupersededNotice({ available: true, superseded_cycle_incomplete: true });
            record();
            renderSupersededNotice({ available: true, superseded_cycle_incomplete: false });
            record();
            renderSupersededNotice({ available: false });
            record();
            renderSupersededNotice(null);
            record();
            console.log(JSON.stringify(out));
            """
            notice = JSON3.read(read(pipeline(ignorestatus(`$node -e $notice_probe`)), String))
            # Stated, not alarming: the forecast on the panel is real and the situation is named.
            @test occursin("most recent hourly issuance has not completed", notice[1].text)
            @test occursin("most recent complete forecast cycle", notice[1].text)
            @test !occursin("<", notice[1].text)
            @test notice[1].hidden == false
            # And it is absent, not merely emptied, whenever the newest issuance did complete or the
            # payload does not say — a notice left standing would be its own false statement.
            for index in 2:4
                @test notice[index].text == ""
                @test notice[index].hidden == true
            end

            # The interval method the panel names comes from the published cycle, which can carry
            # more than one. Executed, because the failure the fix repairs was a rendered string.
            method_block = match(
                r"// ---- interval-method label block[^\n]*\n(.*?)// ---- end interval-method label block ----"s,
                js,
            )
            @test method_block !== nothing
            @test occursin("live interval method: \${intervalMethodText(st.calibration)}", js)
            @test occursin("const liveSrc = intervalMethodText(c);", js)
            @test !occursin("c.current_interval_source || \"—\"", js)
            method_probe = method_block.captures[1] * """
            console.log(JSON.stringify([
              intervalMethodText({ current_interval_source: "v24_conformal_depth",
                                   current_interval_sources: ["aci", "v24_conformal_depth"] }),
              intervalMethodText({ current_interval_source: null,
                                   current_interval_sources: ["aci", "v24_conformal_depth"] }),
              intervalMethodText({ current_interval_source: null,
                                   current_interval_sources: ["aci", "conformal", "v24_conformal_depth"] }),
              intervalMethodText({ current_interval_source: "unknown", current_interval_sources: [] }),
              intervalMethodText({ current_interval_source: null, current_interval_sources: [] }),
              intervalMethodText({}),
              intervalMethodText(null),
              intervalMethodIsSet({ current_interval_source: null,
                                    current_interval_sources: ["aci", "v24_conformal_depth"] }),
              intervalMethodIsSet({ current_interval_source: "aci",
                                    current_interval_sources: ["aci"] }),
              intervalMethodIsSet({}),
            ]));
            """
            methods = JSON3.read(read(pipeline(ignorestatus(`$node -e $method_probe`)), String))
            @test methods[1] == "v24_conformal_depth"
            @test methods[2] == "aci and v24_conformal_depth"
            @test methods[3] == "aci, conformal and v24_conformal_depth"
            @test methods[4] == "unknown"
            # No published cycle and no set: an em dash, never the word "nothing" and never a method.
            for index in 5:7
                @test methods[index] == "—"
            end
            @test methods[8] == true
            @test methods[9] == false
            @test methods[10] == false
        end
        @test !occursin("? status.threat.level : 0", js)
        # The product name is derived from the served label the log recorded, so no hardcoded version
        # string may remain in the rendering path; the frozen-tail ablation keeps its V2.1 name because
        # that comparator really is the V2.1 frozen-tail center.
        for label in ("Forecast: \${productName(st)} (", "Verified issued",
                      "Product forecast: \${esc(product)}.",
                      "served RMSE nT", "verified forecasts",
                      "V2.1 core trajectory (display)",
                      "V2.1 frozen-tail ablation", "SINDy v1", "Persistence",
                      "Burton full", "O'Brien–McPherron", "live_skill_mature",
                      "matched point forecast (n=\${matchedN})", "no best method is highlighted",
                      "by_served_model", "n_verified_current_served_model")
            @test occursin(label, js)
        end
        for retired_label in ("Forecast: V2 (", "name:\"Verified V2\"",
                              "V2 90% coverage", "V2 RMSE nT", "V2 verified",
                              "headline score", "(online, distribution-free)",
                              "calibrated 90% interval",
                              "Forecast: V2.1 (", "Product forecast: V2.1.",
                              "V2.1 RMSE nT", "V2.1 verified forecasts")
            @test !occursin(retired_label, js)
        end
        @test occursin("mature && v != null", js)
        @test !occursin("calibrated uncertainty", html)
        @test occursin("package's V2.1 forecaster", readme)
        @test occursin("exactly the same", readme)
        @test occursin("48 common rows", readme)
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
        @test occursin("90% target-interval lower edge", html)
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

    @testset "an emptied forecast log after a delivered alert is an outage, not an all-clear" begin
        # The status payload of a log that carries no rows at all — emptied, truncated, rotated
        # away, or unreadable — has no cycle and therefore no issue time. Read on payload shape
        # alone that is indistinguishable from a cold start, and the level collapse from a
        # delivered alert to zero was delivered as "Space weather returned to quiet (all clear)":
        # an explicit all-clear the physics never produced. The evaluation is driven through the
        # loop body so the log load, the status build, the outage sentinel and the delivery
        # decision are all exercised, with the upstream and ground layers held inert.
        url = "https://h"
        dir = mktempdir()
        logf = joinpath(dir, "live_forecast_log.csv")
        state_path = joinpath(dir, "alert_notify_state.json")
        calls = Any[]
        post = (u, h, b) -> (push!(calls, JSON3.read(b)); nothing)
        evaluate(path; sp = state_path) = notify_cycle!(
            path; url = url, state_path = sp, post_fn = post,
            snapshot_fn = () -> nothing, dbdt_fn = () -> nothing,
        )

        # Quiet: both the center and the interval lower edge stay above the minor-storm edge, so
        # neither the point level nor the watch tier raises the composed level.
        quiet_cycle(issue) = live_cycle_fixture(issue; served_pred = -10.0, served_lo = -20.0,
                                                served_hi = -2.0)

        reset_notify!()
        write_cycle_csv(logf, quiet_cycle(now(UTC) - Minute(20)))              # quiet, complete
        _LOG_CACHE[] = nothing
        quiet = evaluate(logf)
        @test quiet.state.level == 0 && quiet.state.stale == false
        @test quiet.result.fired == false && quiet.result.reason == "baseline set"

        write_cycle_csv(logf, live_cycle_fixture(now(UTC) - Minute(15);
                                                 served_pred = -70.0, served_lo = -90.0,
                                                 served_hi = -55.0))
        _LOG_CACHE[] = nothing
        storm = evaluate(logf)
        @test storm.state.level == 2 && storm.state.stale == false
        @test storm.result.fired == true && storm.result.kind == "elevated"
        @test length(calls) == 1

        write_empty_log(logf)                                  # the daemon's output disappears
        _LOG_CACHE[] = nothing
        blind = evaluate(logf)
        @test blind.state.stale == true                        # blind, not quiet
        @test blind.result.fired == true && blind.result.kind == "stale_onset"
        @test blind.result.kind != "allclear"
        @test length(calls) == 2
        @test !occursin("all clear", lowercase(calls[end].text))
        @test occursin("stale", lowercase(calls[end].text))

        # The marker is persisted, so a restart into the same outage still refuses the all-clear
        # instead of re-reading the empty log as a deployment that has never served anything.
        reset_notify!()
        @test _SEEN_DATA[] == false
        _LOG_CACHE[] = nothing
        restarted = evaluate(logf)
        @test _SEEN_DATA[] == true                             # restored from the state file
        @test restarted.state.stale == true
        @test restarted.result.fired == false                  # baseline restored at the stale state
        @test length(calls) == 2
        _LOG_CACHE[] = nothing
        held = evaluate(logf)
        @test held.result.fired == false && held.state.stale == true
        @test length(calls) == 2

        # An active outage sentinel forces the same verdict even while the log itself still looks
        # fresh and complete: a dead or unloaded daemon is exactly what the live layers cannot see.
        reset_notify!()
        sentinel_dir = mktempdir()
        sentinel_log = joinpath(sentinel_dir, "live_forecast_log.csv")
        sentinel_state = joinpath(sentinel_dir, "alert_notify_state.json")
        write_cycle_csv(sentinel_log, quiet_cycle(now(UTC) - Minute(20)))
        _LOG_CACHE[] = nothing
        empty!(calls)
        fresh = evaluate(sentinel_log; sp = sentinel_state)
        @test fresh.state.level == 0 && fresh.state.stale == false
        @test fresh.result.reason == "baseline set"
        write(joinpath(sentinel_dir, "OUTAGE.md"),
              "# LIVE FORECAST ISSUANCE OUTAGE\n\nSource: daemon issuance dead-man\n" *
              "Summary: no issuance for 6 cycles\n")
        _LOG_CACHE[] = nothing
        sentinel = evaluate(sentinel_log; sp = sentinel_state)
        @test sentinel.state.stale == true
        @test any(occursin("forecast monitor outage", r) for r in sentinel.state.reasons)
        @test sentinel.result.fired == true && sentinel.result.kind == "stale_onset"
        @test !occursin("all clear", lowercase(calls[end].text))

        # Cold-start control: a deployment that has never served a cycle must still baseline
        # silently, so a fresh install does not announce an outage before its first issuance.
        reset_notify!()
        cold_dir = mktempdir()
        cold_log = write_empty_log(joinpath(cold_dir, "live_forecast_log.csv"))
        cold_state = joinpath(cold_dir, "alert_notify_state.json")
        empty!(calls)
        _LOG_CACHE[] = nothing
        cold = evaluate(cold_log; sp = cold_state)
        @test cold.state.level == 0 && cold.state.stale == false
        @test cold.result.fired == false && cold.result.reason == "baseline set"
        @test isempty(calls)
        _LOG_CACHE[] = nothing
        cold_again = evaluate(cold_log; sp = cold_state)
        @test cold_again.state.stale == false && cold_again.result.fired == false
        @test isempty(calls)
        # ... and the persisted marker stays down until data actually arrives.
        @test JSON3.read(read(cold_state, String)).seen_data == false

        reset_notify!()
        _LOG_CACHE[] = nothing
        rm(dir; recursive = true, force = true)
        rm(sentinel_dir; recursive = true, force = true)
        rm(cold_dir; recursive = true, force = true)
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

    @testset "the served-health window states its own bounds and span" begin
        # The window is the last N issue hours PRESENT IN THE LOG. While issuance is unbroken that is
        # a day; after an outage the same N hours can span days, and a rate over them was published
        # as the trailing day's. The live log carries a real multi-day gap, so this is the shape the
        # endpoint has to describe, not a hypothetical one.
        now0 = floor(now(UTC), Hour)
        stack = reduce(vcat, [live_cycle_fixture(now0 - Hour(161) + Hour(k) - Minute(47);
                                                 served_model=STACK_V2_SERVED_MODEL_VERSION,
                                                 v2_2_status="ok",
                                                 v24_status="fallback:deployment_absent")
                              for k in 0:19])
        healthy = reduce(vcat, [live_cycle_fixture(now0 - Hour(4 - k) - Minute(47);
                                                   served_model=CURRENT_V2_SERVED_MODEL_VERSION,
                                                   v2_2_status="ok", v24_status="ok")
                                for k in 1:4])
        df = vcat(stack, healthy)
        health = build_served_health(df)
        @test health.cycles_considered == 24
        @test health.served_fallback_cycles == 20
        @test health.served_fallback_rate == round(20 / 24; digits=4)
        # The disclosure: 24 cycles, but 161 hours of them.
        @test health.served_fallback_window_cycles == 24
        @test health.window_span_hours == 161.0
        # The 47-minute offset puts each issue in the previous hour, which is the hour the window
        # keys on: the oldest cycle floors to now0-162 h and the newest to now0-1 h.
        @test health.window_start_utc == jdt(now0 - Hour(162))
        @test health.window_end_utc == jdt(now0 - Hour(1))
        # ... and the trailing day of issuance, which is what "trailing day" can honestly mean here,
        # is entirely healthy.
        @test health.served_fallback_window_24h_cycles == 4
        @test health.served_fallback_cycles_24h == 0
        @test health.served_fallback_rate_24h == 0.0

        # An unbroken day: the two rates agree and the span is the cycle count.
        unbroken = reduce(vcat, [live_cycle_fixture(now0 - Hour(6 - k) - Minute(47);
                                                     served_model=CURRENT_V2_SERVED_MODEL_VERSION,
                                                     v2_2_status="ok", v24_status="ok")
                                 for k in 1:6])
        steady = build_served_health(unbroken)
        @test steady.window_span_hours == 5.0
        @test steady.served_fallback_rate == 0.0
        @test steady.served_fallback_rate_24h == 0.0
        @test steady.served_fallback_window_24h_cycles == 6
    end

    @testset "alert text writes whole nanotesla, as the dashboard does" begin
        # The webhook and the dashboard are read together, so the same depth must not appear as
        # "-120.0 nT" in one and "-120" in the other.
        iss = now(UTC) - Minute(10)
        storm = live_cycle_fixture(iss; served_pred=-120.4, served_lo=-140.4, served_hi=-100.0)
        alerts = build_alerts(storm)
        @test alerts.active == true
        forecast_alert = only(filter(a -> a.kind == "forecast", alerts.alerts))
        @test occursin("-120 nT", forecast_alert.message)
        @test !occursin("-120.0", forecast_alert.message)
        @test !occursin(".0 nT", forecast_alert.message)

        watch = live_cycle_fixture(iss; served_pred=-60.4, served_lo=-206.4, served_hi=-40.0)
        watch_alerts = build_alerts(watch)
        watch_alert = only(filter(a -> a.kind == "watch", watch_alerts.alerts))
        @test occursin("-206 nT", watch_alert.message)
        @test !occursin("-206.0", watch_alert.message)
        # Ties round away from zero, matching the dashboard's fixed-decimal formatting.
        @test _alert_depth_nt(-120.5) == "-121"
        @test _alert_depth_nt(-119.5) == "-120"
        @test _alert_depth_nt(-120.4) == "-120"

        # Formatting an alerting depth is total. This runs inside the alert builder, so a depth it
        # cannot write as a whole number must degrade to the number itself, never to an exception:
        # a finite magnitude no machine integer can hold used to take `/api/alerts` and the webhook
        # down, replacing an obviously wrong number with no alert payload at all.
        @test _alert_depth_nt(-1.0e30) == "-1.0e30"
        @test _alert_depth_nt(1.0e30) == "1.0e30"
        @test _alert_depth_nt(NaN) == "NaN"
        @test _alert_depth_nt(Inf) == "Inf"
        @test _alert_depth_nt(-Inf) == "-Inf"
        # The machine boundary itself: exactly representable converts, past it is written out.
        @test _alert_depth_nt(Float64(typemin(Int))) == string(typemin(Int))
        @test _alert_depth_nt(-9.3e18) == "-9.3e18"
        @test _alert_depth_nt(9.3e18) == "9.3e18"

        # ... and the route stays up on a cycle carrying such a depth, end to end from the log.
        corrupt = live_cycle_fixture(iss; served_pred=-1.0e30, served_lo=-1.0e30,
                                     served_hi=-1.0e29)
        corrupt_alerts = build_alerts(corrupt)
        @test corrupt_alerts.active == true
        @test occursin("-1.0e30 nT",
                       only(filter(a -> a.kind == "forecast", corrupt_alerts.alerts)).message)
        dir = mktempdir()
        path = write_cycle_csv(joinpath(dir, "log.csv"), corrupt)
        _LOG_CACHE[] = nothing
        response = make_handler(path)(HTTP.Request("GET", "/api/alerts"))
        @test response.status == 200
        served_alerts = JSON3.read(String(response.body))
        @test served_alerts.active == true
        @test occursin("-1.0e30 nT",
                       only(filter(a -> a.kind == "forecast", served_alerts.alerts)).message)
        _LOG_CACHE[] = nothing
        rm(dir; recursive=true, force=true)
    end

    @testset "feed and log text reaches the page as text, not as markup" begin
        # The panels are assembled as HTML strings from values this page did not author: NOAA
        # product identifiers and alert summaries, the NOAA scale token, station codes, served
        # labels and interval sources written into the forecast log. Source scans cannot tell an
        # escaped interpolation from an unescaped one, so the shipped file is executed against a
        # stub DOM and every assigned innerHTML is recorded. The payload below is the marker: it
        # must never appear as markup in any sink, and it must appear escaped where it is rendered.
        node = Sys.which("node")
        if node === nothing
            @test_skip "node is unavailable; the dashboard rendering test needs a JS runtime"
        else
            js = read(joinpath(@__DIR__, "..", "public", "app.js"), String)
            marker = "<img src=x onerror=BOOM>"
            escaped = "&lt;img src=x onerror=BOOM&gt;"
            issue = "2026-06-26T06:00:00Z"
            payloads = replace(RENDER_PROBE_PAYLOADS_JS, "@@MARKER@@" => marker,
                               "@@SERVED@@" => CURRENT_V2_SERVED_MODEL_VERSION,
                               "@@ISSUE@@" => issue)
            prelude = raw"""
            // Stub DOM: records every innerHTML and textContent assignment by element id, so the
            // shipped renderers run unmodified and what they produce is the thing under test.
            const SINKS = {}, TEXTS = {}, ERRORS = [];
            const makeEl = (id) => {
              const el = { id, _html: "", _text: "", className: "", style: {}, dataset: {},
                           classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
                           querySelectorAll: () => [], appendChild() {} };
              Object.defineProperty(el, "innerHTML", {
                get: () => el._html,
                set: (v) => { el._html = String(v); SINKS[id] = String(v); } });
              Object.defineProperty(el, "textContent", {
                get: () => el._text,
                set: (v) => { el._text = String(v); TEXTS[id] = String(v); } });
              return el;
            };
            const ELEMENTS = {};
            const document = {
              getElementById: (id) => ELEMENTS[id] || (ELEMENTS[id] = makeEl(id)),
              querySelectorAll: () => [],
              createElement: () => ({ onerror: null }),
              head: { appendChild() {} },
              addEventListener() {},
            };
            // The plotting library is a markup sink too: it parses its own tag subset out of trace
            // text, hover text and legend names. The stub records what each plot was handed, so the
            // strings that reach that parser are checked as the values passed, not as source.
            const PLOTS = {}, CONFIGS = [];
            const Plotly = { react: async (id, traces, layout) => { PLOTS[id] = { traces, layout }; },
                             purge: () => {},
                             setPlotConfig: (config) => { CONFIGS.push(config); } };
            const window = { Plotly, __plotlyFailed: false };
            const console = { error: (...a) => ERRORS.push(a.map(String).join(" ")),
                              warn() {}, log() {} };
            // Only short waits run; the refresh loop's own reschedule and the ticker never fire.
            const setTimeout = (fn, ms) => { if ((ms || 0) < 1000) queueMicrotask(fn); return 0; };
            const clearTimeout = () => {};
            const setInterval = () => 0;
            class AbortController { constructor() { this.signal = {}; } abort() {} }
            """
            probe_payloads = payloads * raw"""
            const fetch = async (path) => ({ ok: true,
                                             json: async () => PAYLOADS[String(path).split("?")[0]] });
            """
            postlude = raw"""
            (async () => {
              await refresh();
              process.stdout.write(JSON.stringify({ sinks: SINKS, texts: TEXTS, errors: ERRORS,
                                                    plots: PLOTS, configs: CONFIGS }));
            })();
            """
            dir = mktempdir()
            script = joinpath(dir, "render_probe.js")
            write(script, prelude * probe_payloads * js * postlude)
            out = read(pipeline(ignorestatus(`$node $script`)), String)
            @test !isempty(out)
            rendered = JSON3.read(out)
            @test isempty(rendered.errors)          # every renderer completed against the payload

            sinks = Dict(String(k) => String(v) for (k, v) in pairs(rendered.sinks))
            # The renderers that assemble markup all ran: this is the executed coverage the suite
            # previously had only as source scans.
            for id in ("forecast-caption", "calib", "swpc-alerts", "upstream-caption",
                       "upstream-stats", "dbdt-caption", "dbdt-stats", "pipeline",
                       "history-caption", "network-caption")
                @test haskey(sinks, id)
            end
            # Not one sink carries the payload as markup.
            for (id, html) in sinks
                @test !occursin(marker, html)
                @test !occursin("<img", html)
                @test !occursin("<script", html)
            end
            # ... and it is rendered, escaped, everywhere it belongs: the feed's alert record and
            # scale token, the log's interval source and served label, the station code.
            for id in ("swpc-alerts", "upstream-caption", "upstream-stats", "calib",
                       "forecast-caption", "dbdt-caption")
                @test occursin(escaped, sinks[id])
            end
            @test occursin(escaped, sinks["swpc-alerts"])
            @test occursin("geomagnetic storm " * escaped, sinks["swpc-alerts"])
            # Text assignments were never a markup sink and must stay plain text.
            texts = Dict(String(k) => String(v) for (k, v) in pairs(rendered.texts))
            @test occursin(marker, texts["interval-badge"])

            # The plotting library is the page's other markup sink. Trace text, hover text and
            # legend names are parsed for its own tag subset — anchors, styled spans, breaks — so
            # the station codes and names the USGS feed supplies, and the served label the log
            # writes, are markup there exactly as they are in the document. Every string the
            # renderers handed the library is collected and checked as a group, so a new trace
            # carrying feed text is covered by the same assertion.
            plot_strings = String[]
            for (_, plot) in pairs(rendered.plots)
                for trace in plot.traces
                    for field in (:name, :hovertemplate, :text, :hovertext)
                        haskey(trace, field) || continue
                        value = trace[field]
                        if value isa JSON3.Array
                            append!(plot_strings, String[String(v) for v in value])
                        elseif value !== nothing
                            push!(plot_strings, String(value))
                        end
                    end
                end
            end
            @test !isempty(plot_strings)
            for value in plot_strings
                @test !occursin(marker, value)
                @test !occursin("<img", value)
                @test !occursin("<a ", value)
                @test !occursin("<span", value)
            end
            # ... and the marker is present, escaped, in the sinks that carry feed and log text: the
            # station labels drawn on the map, the hover records behind them, and the legend names
            # taken from the served label.
            network_trace = only(rendered.plots["network-plot"].traces)
            @test all(occursin(escaped, String(v)) for v in network_trace.text)
            @test all(occursin(escaped, String(v)) for v in network_trace.hovertext)
            forecast_names = String[String(t.name) for t in rendered.plots["forecast-plot"].traces
                                    if haskey(t, :name)]
            @test any(name -> occursin(escaped, name), forecast_names)

            # The base map is fetched from this origin, and the setting is applied once, before the
            # first plot is drawn — the library reads it when a geo subplot is created.
            @test length(rendered.configs) == 1
            @test String(rendered.configs[1].topojsonURL) == "/vendor/topojson/"
            rm(dir; recursive=true, force=true)
        end
    end

    @testset "responses carry the content-security policy and refuse writes" begin
        dir = mktempdir()
        path = write_cycle_csv(joinpath(dir, "log.csv"), live_cycle_fixture(now(UTC) - Minute(10)))
        _LOG_CACHE[] = nothing
        handler = make_handler(path)
        header(response, name) = HTTP.header(response, name, "")
        # The dashboard document and a JSON route: the policy travels with every response, so an
        # injected string cannot reach a script source, an object embed, or a framing context even
        # if an escaping sink were ever missed.
        for request in (HTTP.Request("GET", "/"), HTTP.Request("GET", "/api/health"),
                        HTTP.Request("GET", "/api/forecast"), HTTP.Request("GET", "/app.js"))
            response = handler(request)
            @test response.status == 200
            policy = header(response, "Content-Security-Policy")
            @test occursin("default-src 'self'", policy)
            @test occursin("object-src 'none'", policy)
            @test occursin("frame-ancestors 'none'", policy)
            @test occursin("connect-src 'self'", policy)
            @test header(response, "X-Content-Type-Options") == "nosniff"
            @test header(response, "Referrer-Policy") == "no-referrer"
        end
        # The page must not need inline script under that policy: the plotting fallback was an
        # inline onerror attribute and is now registered from the same-origin bundle.
        html = read(joinpath(@__DIR__, "..", "public", "index.html"), String)
        @test !occursin("onerror=", html)
        @test !occursin(r"<script(?![^>]*\ssrc=)", html)
        js = read(joinpath(@__DIR__, "..", "public", "app.js"), String)
        @test occursin("cdn.plot.ly", js)          # the documented offline fallback still exists
        @test occursin("cdn.plot.ly", CONTENT_SECURITY_POLICY)   # and the policy admits exactly it
        # The plot toolbar's PNG export renders the chart into a blob URL and loads it as an image
        # before the browser saves it, so a policy that admits only 'self' and data: makes the
        # download fail with the library's own notice and nothing else. The bundle is the evidence:
        # it creates the object URL and assigns it to an image source.
        vendor = joinpath(@__DIR__, "..", "public", "vendor", "plotly.min.js")
        if isfile(vendor)
            bundle = read(vendor, String)
            @test occursin("createObjectURL", bundle)
        end
        img = only(filter(d -> startswith(d, "img-src"), strip.(split(CONTENT_SECURITY_POLICY, ";"))))
        @test "blob:" in split(img)[2:end]
        @test occursin("displayModeBar", js)

        # A traversal rejection is a response too, and carries the same policy.
        forbidden = handler(HTTP.Request("GET", "/../secret.txt"))
        @test forbidden.status == 403
        @test occursin("default-src 'self'", header(forbidden, "Content-Security-Policy"))

        # This server publishes a log it does not own and holds no state a request can change. The
        # method was ignored, so a write verb was answered with the GET body and a 200.
        for method in ("POST", "PUT", "DELETE", "PATCH", "OPTIONS")
            response = handler(HTTP.Request(method, "/api/status"))
            @test response.status == 405
            @test header(response, "Allow") == "GET, HEAD"
            @test occursin("default-src 'self'", header(response, "Content-Security-Policy"))
        end
        @test handler(HTTP.Request("HEAD", "/api/health")).status == 200
        _LOG_CACHE[] = nothing
        rm(dir; recursive=true, force=true)
    end

    @testset "the network base map is fetched from an origin the policy admits" begin
        # `connect-src` governs one request the page does not write itself. A scattergeo has no
        # coastlines, borders or subunit outlines of its own: the plotting library fetches a topojson
        # file for the requested scope and resolution when the subplot is created, and its built-in
        # source is the plotting CDN. Closing `connect-src` to this origin without redirecting that
        # fetch leaves the network panel's markers floating over an empty frame, and nothing in the
        # page or in the policy says so. Everything below is derived — from the shipped page and, for
        # the parts that are the library's own behaviour, from the shipped bundle — so a bundle
        # upgrade or a panel change that moves the base map cannot pass this silently.
        public_dir = normpath(joinpath(@__DIR__, "..", "public"))
        js = read(joinpath(public_dir, "app.js"), String)

        directives = strip.(split(CONTENT_SECURITY_POLICY, ";"))
        connect = only(filter(d -> startswith(d, "connect-src"), directives))
        sources = split(connect)[2:end]

        # The page names exactly one base-map source, and it is a path on this origin.
        url_match = match(r"const TOPOJSON_URL = \"([^\"]+)\";", js)
        @test url_match !== nothing
        topojson_url = url_match.captures[1]
        @test startswith(topojson_url, "/") && !startswith(topojson_url, "//")
        @test endswith(topojson_url, "/")
        @test occursin("Plotly.setPlotConfig({ topojsonURL: TOPOJSON_URL });", js)
        # A same-origin path is admitted by 'self' and by nothing else in the directive, so the
        # directive has to carry it. An absolute URL here would have to name its own origin.
        @test "'self'" in sources

        # The map the panel needs, named by the library's own rule from the geo request the panel
        # actually makes. A panel that changes scope or resolution needs a different file.
        geo = match(r"geo:\s*\{\s*scope:\s*\"([^\"]+)\",\s*resolution:\s*(\d+)", js)
        @test geo !== nothing
        map_name = replace(geo.captures[1], " " => "-") * "_" * geo.captures[2] * "m"
        @test map_name == "north-america_50m"
        asset = joinpath(public_dir, splitpath(lstrip(topojson_url, '/'))...,
                         map_name * ".json")
        @test isfile(asset)
        topology = JSON3.read(read(asset, String))
        @test topology.type == "Topology"
        for layer in ("coastlines", "countries", "subunits", "land")
            @test haskey(topology.objects, Symbol(layer))
        end
        # It is served, as an asset of a whitelisted type, through the same guarded static route.
        served = serve_static(topojson_url * map_name * ".json")
        @test served.status == 200
        @test HTTP.header(served, "Content-Type", "") == "application/json; charset=utf-8"
        @test length(served.body) == filesize(asset)

        # Keyed to the shipped bundle: the default source the override exists to replace, and the
        # rule that turns the panel's geo request into that filename, are read from the bundle's own
        # bytes. The bundle is vendored, not versioned, so a checkout without it checks the rest.
        bundle = joinpath(public_dir, "vendor", "plotly.min.js")
        if !isfile(bundle)
            @test_skip "the plotting bundle is not vendored here; the base-map pairing is bundle-keyed"
        else
            bundle_source = read(bundle, String)
            default = match(r"topojsonURL:\{valType:\"string\",noBlank:!0,dflt:\"([^\"]+)\"\}",
                            bundle_source)
            @test default !== nothing
            default_url = default.captures[1]
            # The default really is cross-origin — this is why the override is not optional …
            @test startswith(default_url, "http")
            default_origin = match(r"^https?://[^/]+", default_url).match
            # … and the policy does not admit it as a fetch destination, so leaving the default in
            # place would block the map rather than merely route it off-origin.
            @test !any(source -> occursin(default_origin, source), sources)
            # The filename rule and the path rule, verbatim from the bundle: `map_name` above is
            # computed by exactly these two, so a bundle that changes either fails here.
            @test occursin(
                "getTopojsonName=function(t){return[t.scope.replace(/ /g,\"-\"),\"_\"," *
                "t.resolution.toString(),\"m\"].join(\"\")}", bundle_source)
            @test occursin("getTopojsonPath=function(t,e){return t+e+\".json\"}", bundle_source)
            # The bundle is the version the page's own CDN fallback names, so the vendored copy and
            # the fallback cannot drift apart into two different libraries.
            version = match(r"plotly\.js v([0-9]+\.[0-9]+\.[0-9]+)", bundle_source)
            @test version !== nothing
            @test occursin("cdn.plot.ly/plotly-" * version.captures[1] * ".min.js", js)
        end
    end

    @testset "an unreadable log is parsed once and reported as unreadable" begin
        dir = mktempdir()
        path = joinpath(dir, "live_forecast_log.csv")
        write_cycle_csv(path, live_cycle_fixture(now(UTC) - Minute(20)))
        _LOG_CACHE[] = nothing
        _LOG_PARSE_FAILURE[] = nothing
        @test nrow(get_log(path)) == length(LIVE_CYCLE_HORIZONS)
        @test log_parse_state(path).readable == true

        # New bytes the loader rejects. Serving the last good frame is right; re-running the parse
        # that produced the failure on every subsequent request is not — at the monitor's row cap
        # that is a multi-second parse, serialized behind the load lock, for as long as the
        # unreadable file sits there.
        sleep(0.05)
        write(path, "issue_time_utc,target_time_utc\n\"unterminated,2026-01-01T00:00:00\n")
        attempts = Ref(0)
        counting = p -> (attempts[] += 1; error("synthetic parse failure"))
        served = @test_logs (:warn, r"serving cached copy") match_mode = :any get_log(
            path; loader=counting)
        @test nrow(served) == length(LIVE_CYCLE_HORIZONS)      # the last good frame is served
        @test attempts[] == 1
        for _ in 1:5
            @test nrow(get_log(path; loader=counting)) == length(LIVE_CYCLE_HORIZONS)
        end
        @test attempts[] == 1                                  # the failing parse ran exactly once
        failed = log_parse_state(path)
        @test failed.readable == false
        @test failed.serving_cached_copy == true
        @test failed.error_type == "ErrorException"            # the type only, never the message
        @test failed.error_age_min !== nothing && failed.error_age_min >= 0.0

        # The negative cache is keyed on the file identity, never on the path, so repaired bytes are
        # read immediately rather than inheriting the failure.
        sleep(0.05)
        write_cycle_csv(path, live_cycle_fixture(now(UTC) - Minute(10)))
        @test nrow(get_log(path)) == length(LIVE_CYCLE_HORIZONS)
        @test log_parse_state(path).readable == true
        @test attempts[] == 1

        # Health is computed from the cached frame plus the file's mtime, so an unreadable log used
        # to report status "ok", a complete cycle and a zero-minute age — the mtime of the very file
        # that was rejected. The missing-log case was already honest; this one was not.
        corrupt = joinpath(dir, "corrupt.csv")
        write_cycle_csv(corrupt, live_cycle_fixture(now(UTC) - Minute(20)))
        _LOG_CACHE[] = nothing
        _LOG_PARSE_FAILURE[] = nothing
        health(p) = JSON3.read(String(
            make_handler(p)(HTTP.Request("GET", "/api/health")).body))
        before = health(corrupt)
        @test before.status == "ok" && before.log_readable == true
        @test before.cycle_complete == true && before.log_parse_error_age_min === nothing

        sleep(0.05)
        write(corrupt, "issue_time_utc,target_time_utc\n\"unterminated,2026-01-01T00:00:00\n")
        during = health(corrupt)
        @test during.status == "unreadable"
        @test during.log_readable == false
        @test during.cycle_complete == false            # the newest issuance cannot be verified
        @test during.serving_cached_log_copy == true
        @test during.log_parse_error_age_min !== nothing

        sleep(0.05)
        write_cycle_csv(corrupt, live_cycle_fixture(now(UTC) - Minute(15)))
        after = health(corrupt)
        @test after.status == "ok" && after.log_readable == true && after.cycle_complete == true
        _LOG_CACHE[] = nothing
        _LOG_PARSE_FAILURE[] = nothing
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

    @testset "the Dst threat scale is pinned at its band edges" begin
        # The published storm class is a step function of one number, and only its non-finite
        # inputs were asserted. Kyoto Dst is integer-valued, so the edges are hit constantly: a
        # one-character inclusivity change moves an exact -30 nT forecast from "Minor storm" to
        # "Quiet" — a whole tier of the alert an operator acts on — with the suite still green.
        # Each edge belongs to the storm side, and the value just above it does not.
        @test dst_threat_level(-29.999) == (0, "Quiet")
        @test dst_threat_level(-30.0) == (1, "Minor storm")
        @test dst_threat_level(-30.001) == (1, "Minor storm")
        @test dst_threat_level(-49.999) == (1, "Minor storm")
        @test dst_threat_level(-50.0) == (2, "Moderate storm")
        @test dst_threat_level(-50.001) == (2, "Moderate storm")
        @test dst_threat_level(-99.999) == (2, "Moderate storm")
        @test dst_threat_level(-100.0) == (3, "Intense storm")
        @test dst_threat_level(-100.001) == (3, "Intense storm")
        @test dst_threat_level(-199.999) == (3, "Intense storm")
        @test dst_threat_level(-200.0) == (4, "Extreme storm")
        @test dst_threat_level(-200.001) == (4, "Extreme storm")
        @test dst_threat_level(0.0) == (0, "Quiet")
        @test dst_threat_level(25.0) == (0, "Quiet")
        # The band edges the scale is defined by, read from the published constants rather than
        # restated, so the table above cannot drift away from the thresholds it is testing.
        @test THREAT_BANDS_NT == (-30.0, -50.0, -100.0, -200.0)
        for (index, edge) in enumerate(THREAT_BANDS_NT)
            @test dst_threat_level(edge) == (index, THREAT_LABELS[index + 1])
            @test dst_threat_level(nextfloat(edge)) == (index - 1, THREAT_LABELS[index])
        end

        # ... and the published status carries the same rule: a cycle whose depth-safe center sits
        # exactly on an edge is reported at the storm tier, not below it.
        iss = now(UTC) - Minute(10)
        edge_cycle = build_status(live_cycle_fixture(iss; served_pred=-50.0, served_lo=-60.0,
                                                     served_hi=-40.0))
        @test edge_cycle.available == true
        @test edge_cycle.threat.point_min_dst_nt === -50.0
        @test edge_cycle.threat.level == 2 && edge_cycle.threat.label == "Moderate storm"
        # The watch tier is taken on the same scale, so an interval edge exactly on -100 nT is an
        # intense-storm watch above a moderate-storm point forecast.
        watch_cycle = build_status(live_cycle_fixture(iss; served_pred=-60.0, served_lo=-100.0,
                                                      served_hi=-40.0))
        @test watch_cycle.threat.level == 2
        @test watch_cycle.threat.watch == true && watch_cycle.threat.watch_level == 3
    end
end
