# test_analog_serving.jl — analog ensemble serving machinery used by the super-learner expert.
#
# The analog center is only evidence about the scored candidate if the arithmetic on top of it is the
# scored arithmetic, so the ensemble raw center, the lead-aware blend, the capped error layer and the
# frozen-tail blend partner are each recomputed here from their definitions and compared.
#
# The fixture is a small synthetic in-process deployment so these paths run without the
# 87,000-origin archive.

module AnalogServingTests

using Test
using DataFrames
using Dates
using Statistics
using SolarSINDy

const FIXTURE_K = 5
const FIXTURE_LAT = [1.0, 1.0, 1.0, 1.0, 0.75, 0.75]

"Synthetic but physically admissible hourly frame: smooth, non-degenerate, strictly positive density."
function _fixture_frame(; first_hour::DateTime = DateTime(2010, 1, 1, 0), hours::Int = 480)
    times = [first_hour + Hour(k - 1) for k in 1:hours]
    speed = [380.0 + 60.0 * sin(2π * k / 37) + 0.05 * k for k in 1:hours]
    bz = [-3.0 + 6.0 * sin(2π * k / 23) for k in 1:hours]
    by = [2.0 * cos(2π * k / 19) for k in 1:hours]
    density = [5.0 + 2.0 * sin(2π * k / 29) for k in 1:hours]
    pdyn = [dynamic_pressure(density[k], speed[k]) for k in 1:hours]
    dst = [-15.0 - 30.0 * max(0.0, sin(2π * k / 41)) for k in 1:hours]
    return DataFrame(time_utc = times, V = speed, Bz = bz, By = by, n = density,
                     Pdyn = pdyn, Dst = dst)
end

"""
Build a self-consistent analog serving fixture in-process — the same way the super-learner
assembles its analog expert from the deployed bundle.
"""
function _fixture_artifacts(; with_e2::Bool = true)
    frame = _fixture_frame()
    lookup = analog_serving_frame_lookup(frame)
    candidates = frame.time_utc
    X, ok = analog_feature_matrix(frame, candidates)
    continuable = analog_origin_ok(lookup, candidates)
    keep = [j for j in eachindex(candidates) if ok[j] && continuable[j]]
    origins = candidates[keep]
    length(origins) > FIXTURE_K || error("fixture archive is too small")
    stats = analog_feature_stats(X[keep, :])
    archive = Matrix{Float64}(X[keep, :])
    standardised = analog_standardize(archive, stats.mean, stats.sd)

    # The correction layer is the deployed inertial-SINDy ridge artifact: a real 26-feature
    # calibration, so the fixture exercises the same feature contract the analog refit uses.
    calibration = read_inertial_sindy_calibration(
        operational_calibration_artifacts(OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION).point_csv)
    e2_names = vcat(String.(collect(ANALOG_SERVING_E_FEATURE_NAMES))[1:ANALOG_SERVING_E_INNOVATION_LAGS],
                    String.(collect(ANALOG_FEATURE_NAMES)), ["center_dst_nt"])
    e_layers = SolarSINDy.AnalogServingELayer[]
    for step in ANALOG_SERVING_MODEL_STEPS
        if with_e2 && step == 1
            rows = 200
            design = Matrix{Float64}(undef, rows, length(e2_names))
            for i in 1:rows, j in 1:length(e2_names)
                design[i, j] = sin(0.7 * i + 0.11 * j) + 0.01 * i
            end
            target = [0.3 * design[i, 1] - 0.2 * design[i, 7] for i in 1:rows]
            model = gbm_fit(design, target; max_depth = 3, nrounds = 8, min_weight = 1,
                            feature_names = e2_names)
            push!(e_layers, SolarSINDy.AnalogServingELayer(
                step, :E2, "(3, 8)", analog_serving_e_cap(step), e2_names, nothing, model))
        else
            push!(e_layers, SolarSINDy.AnalogServingELayer(
                step, :identity, "", analog_serving_e_cap(step), String[], nothing, nothing))
        end
    end

    art = SolarSINDy.AnalogServingArtifacts(
        "fixture", frame, lookup, origins, archive, standardised,
        stats.mean, stats.sd, :magnetic, analog_weights(:magnetic), FIXTURE_K,
        ANALOG_EXCLUSION_HOURS, calibration, FIXTURE_LAT, e_layers,
        Dict{String,Any}(),
        load_operational_core(OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION),
        analog_serving_identity(:magnetic, FIXTURE_K))
    return (art = art, frame = frame, origins = origins, stats = stats, lookup = lookup,
            e2_names = e2_names)
end

"A query anchor far outside the archive, plus its driver history and the two anchor Dst values."
function _fixture_query()
    anchor = DateTime(2026, 3, 1, 12)
    history = Any[(V = 420.0 + 3.0 * lag, Bz = -5.0 + 0.4 * lag, By = 1.5 - 0.1 * lag,
                   n = 6.0 + 0.2 * lag,
                   Pdyn = dynamic_pressure(6.0 + 0.2 * lag, 420.0 + 3.0 * lag))
                  for lag in 1:ANALOG_SOUTH_RUN_CAP_H]
    return (anchor = anchor, history = history, dst_anchor = -62.0, dst_previous = -55.0)
end

const FIXTURE_MEMORY = (dst_delta_1h_nt = -7.0, dst_delta_3h_nt = -18.0,
                        Bz_delta_1h_nt = -1.5, VBsouth_delta_1h_mvm = 0.4,
                        VBsouth_mean_3h_mvm = 1.9, Bsouth_mean_3h_nt = 4.5)
const FIXTURE_BASELINES = (persistence = -62.0, burton = -70.0, burton_full = -68.0,
                           obrien = -66.0)

@testset "Operational analog ensemble serving" begin
    @testset "published identity strings" begin
        @test ANALOG_SERVING_IDENTITY == "analog+sindy20x11+L1A+ADC(magnetic,K25)+T1rcal+LAT+E"
        @test analog_serving_identity(:uniform, 50) ==
              "analog+sindy20x11+L1A+ADC(uniform,K50)+T1rcal+LAT+E"
        @test collect(ANALOG_SERVING_MODEL_STEPS) == [1, 2, 3, 4, 6, 7]
        @test [analog_serving_e_cap(h) for h in ANALOG_SERVING_MODEL_STEPS] ==
              [10.0, 15.0, 20.0, 25.0, 35.0, 40.0]
        @test ANALOG_SERVING_STEP_SLOT[6] == 5
        @test ANALOG_SERVING_STEP_SLOT[5] == 0
    end

    @testset "ballistic transit helper" begin
        # Independent expectation: 1.5e6 km at 500 km/s is 3000 s, i.e. 5/6 h.
        @test analog_serving_transit_hours(500.0) ≈ 1.5e6 / 500.0 / 3600.0 atol=1e-12
        @test analog_serving_transit_hours(500.0) ≈ 5 / 6 atol=1e-12
        @test analog_serving_transit_hours(0.0) == 0.0
        @test analog_serving_transit_hours(NaN) == 0.0
        @test analog_serving_transit_hours(-400.0) == 0.0
    end

    @testset "innovation lag block is all-or-nothing" begin
        anchor = DateTime(2026, 3, 1, 12)
        full = Dict(anchor - Hour(lag) => Float64(lag) for lag in 1:8)
        block = analog_serving_innovation_lags(anchor, full)
        @test block.ok
        @test block.values == Float64.(1:6)
        holed = copy(full)
        delete!(holed, anchor - Hour(4))
        @test !analog_serving_innovation_lags(anchor, holed).ok
        nonfinite = copy(full)
        nonfinite[anchor - Hour(2)] = NaN
        @test !analog_serving_innovation_lags(anchor, nonfinite).ok
        @test !analog_serving_innovation_lags(anchor, Dict{DateTime,Float64}()).ok
    end

    @testset "innovations are formed from one-hour centers and matured observations" begin
        # The live engine has no one-hour issued row, so it records the one-hour pre-layer center of
        # every anchor and pairs it with the Dst observed one hour later. This is the definition the
        # offline identity oracle also uses; the arithmetic is checked here against hand values.
        anchor = DateTime(2026, 3, 1, 12)
        centers = Dict(anchor - Hour(lag) => -40.0 - Float64(lag) for lag in 1:6)
        observed = Dict(anchor - Hour(lag) + Hour(1) => -50.0 - Float64(lag) for lag in 1:6)
        innovations = analog_serving_innovations_from_step1_centers(centers, observed)
        @test length(innovations) == 6
        # Dst(a+1) - center(a) = (-50 - lag) - (-40 - lag) = -10 at every lag.
        @test all(≈(-10.0), collect(values(innovations)))
        block = analog_serving_innovation_lags(anchor, innovations)
        @test block.ok
        @test block.values == fill(-10.0, ANALOG_SERVING_E_INNOVATION_LAGS)

        # An anchor whose one-hour target has not been observed yet is not matured and must not enter:
        # imputing it would give the layer a residual no forecaster could have scored.
        unmatured = copy(observed)
        delete!(unmatured, anchor)
        @test length(analog_serving_innovations_from_step1_centers(centers, unmatured)) == 5
        @test !analog_serving_innovation_lags(
            anchor, analog_serving_innovations_from_step1_centers(centers, unmatured),
        ).ok

        # A non-finite center or observation is dropped rather than propagated as NaN.
        nan_center = copy(centers)
        nan_center[anchor - Hour(3)] = NaN
        @test length(analog_serving_innovations_from_step1_centers(nan_center, observed)) == 5
        nan_obs = copy(observed)
        nan_obs[anchor - Hour(3) + Hour(1)] = NaN
        @test length(analog_serving_innovations_from_step1_centers(centers, nan_obs)) == 5

        # The pairing is by anchor + 1 h, not by anchor: an observation series shifted by one hour must
        # not silently produce innovations against the wrong target.
        shifted = Dict(anchor - Hour(lag) => -50.0 - Float64(lag) for lag in 1:6)
        @test length(analog_serving_innovations_from_step1_centers(centers, shifted)) == 5
        wrong = analog_serving_innovations_from_step1_centers(centers, shifted)
        @test !haskey(wrong, anchor - Hour(1))
    end

    @testset "fixture artifacts carry a self-consistent archive" begin
        fixture = _fixture_artifacts()
        art = fixture.art
        @test art.identity == analog_serving_identity(:magnetic, FIXTURE_K)
        @test art.k == FIXTURE_K
        @test art.weight_set === :magnetic
        @test art.weights == analog_weights(:magnetic)
        @test art.origins == fixture.origins
        @test art.lat_weights == FIXTURE_LAT
        @test art.exclusion_hours == ANALOG_EXCLUSION_HOURS
        @test art.feature_mean ≈ fixture.stats.mean atol=0
        @test art.feature_sd ≈ fixture.stats.sd atol=0
        @test size(art.archive_standardised) == (length(fixture.origins), ANALOG_FEATURE_COUNT)
        @test all(isfinite, art.archive_standardised)
        # The standardised archive must be exactly the feature standardisation applied to the
        # rebuilt features; recomputing it independently catches a permuted feature order.
        expected = (art.archive_features .- transpose(art.feature_mean)) ./
                   transpose(art.feature_sd)
        @test art.archive_standardised == expected
        @test length(art.e_layers) == length(ANALOG_SERVING_MODEL_STEPS)
        @test art.e_layers[1].kind === :E2
        @test art.e_layers[1].cap == analog_serving_e_cap(1)
        @test art.e_layers[1].feature_names == fixture.e2_names
        @test all(l -> l.kind === :identity, art.e_layers[2:end])
        @test length(art.calibration.feature_names) == 26
    end

    @testset "analog key reports its own incompleteness" begin
        art = _fixture_artifacts().art
        q = _fixture_query()
        good = analog_serving_features(art, q.anchor, q.history, q.dst_anchor, q.dst_previous)
        @test good.ok
        @test good.reason == "ok"
        @test length(good.features) == ANALOG_FEATURE_COUNT
        @test all(isfinite, good.features)
        # Independent expectation for the features that are pure functions of the history.
        @test good.features[analog_feature_index(:bz0)] == q.history[1].Bz
        @test good.features[analog_feature_index(:v0)] == q.history[1].V
        @test good.features[analog_feature_index(:dv6)] ==
              q.history[1].V - q.history[7].V
        @test good.features[analog_feature_index(:bz_mean6)] ≈
              mean([q.history[j].Bz for j in 1:6]) atol=1e-12
        @test good.features[analog_feature_index(:dst0)] == q.dst_anchor
        @test good.features[analog_feature_index(:ddst1)] == q.dst_anchor - q.dst_previous
        @test good.features[analog_feature_index(:vbs0)] ≈
              q.history[1].V * max(0.0, -q.history[1].Bz) / 1000.0 atol=1e-12
        # A hole inside the mandatory seven-hour window names the missing lag.
        holed = copy(q.history)
        holed[4] = nothing
        bad = analog_serving_features(art, q.anchor, holed, q.dst_anchor, q.dst_previous)
        @test !bad.ok
        @test bad.reason == "missing_driver_lag4"
        @test all(!isfinite, bad.features)
        # A hole beyond the window only truncates the southward run length, which stays usable.
        truncated = copy(q.history)
        truncated[9] = nothing
        @test analog_serving_features(art, q.anchor, truncated, q.dst_anchor, q.dst_previous).ok
        @test !analog_serving_features(art, q.anchor, q.history, NaN, q.dst_previous).ok
        @test analog_serving_features(art, q.anchor, q.history, NaN, q.dst_previous).reason ==
              "missing_anchor_dst"
        @test analog_serving_features(art, q.anchor, q.history, q.dst_anchor, NaN).reason ==
              "missing_previous_dst"
        # Fewer than seven supplied lags is a programming error, not a fallback.
        @test_throws ArgumentError analog_serving_features(
            art, q.anchor, q.history[1:6], q.dst_anchor, q.dst_previous)
    end

    @testset "ensemble raw center is the member mean of the frozen core" begin
        art = _fixture_artifacts().art
        q = _fixture_query()
        key = analog_serving_features(art, q.anchor, q.history, q.dst_anchor, q.dst_previous)
        issue_drv = q.history[1]
        anchor_star = pressure_correct_dst([q.dst_anchor], [issue_drv.Pdyn])[1]
        # No L1-measured step: every rollout step takes the member's analog continuation.
        never_measured = (k, last_known) -> (driver = last_known, l1_measured = false)
        step = 4
        ensemble = analog_serving_members(
            art, key.features; anchor_time = q.anchor, issue_drv = issue_drv,
            anchor_dst_star = anchor_star, model_steps = step,
            step_driver = never_measured)
        @test length(ensemble.origins) == FIXTURE_K
        @test allunique(ensemble.origins)
        @test all(o -> o in art.origins, ensemble.origins)
        @test ensemble.k == FIXTURE_K
        @test ensemble.raw ≈ mean(ensemble.member_pred) atol=0
        @test ensemble.raw_reported == clamp(ensemble.raw, -2000.0, 50.0)
        # Independent recomputation of one member: continue the driver from the archive and roll
        # the frozen core by hand.
        member = 1
        origin = ensemble.origins[member]
        filter = init_assimilation(art.core.library, art.core.coefficients, Int[], anchor_star)
        final_drv = issue_drv
        for k in 1:step
            drv = analog_member_driver(issue_drv, art.lookup, origin, k)
            final_drv = drv
            assimilation_predict!(filter, drv)
            filter.mean[1] = clamp(filter.mean[1], -2000.0, 50.0)
        end
        expected = current_dst(filter) + 7.26 * sqrt(max(final_drv.Pdyn, 0.0)) - 11.0
        @test ensemble.member_pred[member] == expected
        # Retrieval must be ordered by distance, so the nearest origin is the first member.
        query_z = analog_standardize(reshape(collect(key.features), 1, ANALOG_FEATURE_COUNT),
                                  art.feature_mean, art.feature_sd)
        distances = [sum(art.weights .*
                         (vec(query_z) .- vec(art.archive_standardised[i, :])) .^ 2)
                     for i in eachindex(art.origins)]
        @test art.origins[argmin(distances)] == ensemble.origins[1]
        # An L1-measured step must bypass the member entirely.
        measured_drv = (V = 600.0, Bz = -12.0, By = 3.0, n = 8.0,
                        Pdyn = dynamic_pressure(8.0, 600.0))
        always_measured = (k, last_known) -> (driver = measured_drv, l1_measured = true)
        measured = analog_serving_members(
            art, key.features; anchor_time = q.anchor, issue_drv = issue_drv,
            anchor_dst_star = anchor_star, model_steps = step,
            step_driver = always_measured)
        @test all(p -> p == measured.member_pred[1], measured.member_pred)
        @test measured.raw != ensemble.raw
        # A non-finite key must be rejected rather than searched.
        @test_throws ArgumentError analog_serving_members(
            art, fill(NaN, ANALOG_FEATURE_COUNT); anchor_time = q.anchor,
            issue_drv = issue_drv, anchor_dst_star = anchor_star, model_steps = step,
            step_driver = never_measured)
    end

    @testset "correction, blend and error layer" begin
        art = _fixture_artifacts().art
        q = _fixture_query()
        key = analog_serving_features(art, q.anchor, q.history, q.dst_anchor, q.dst_previous)
        issue_drv = q.history[1]
        anchor_star = pressure_correct_dst([q.dst_anchor], [issue_drv.Pdyn])[1]
        raw = -58.25
        frozen = -49.5
        for (slot, step) in enumerate(ANALOG_SERVING_MODEL_STEPS)
            out = analog_serving_center(
                art; raw_reported = raw, latest_dst = q.dst_anchor,
                anchor_drivers = issue_drv, memory = FIXTURE_MEMORY,
                baselines = FIXTURE_BASELINES, model_steps = step, frozen_inertial_sindy = frozen,
                analog_features = key.features, anchor_time = q.anchor,
                innovations = nothing)
            # The correction is the deployed evaluator on the recomputed feature tuple.
            feats = analog_serving_t1r_features(art, raw, q.dst_anchor, issue_drv,
                                             FIXTURE_MEMORY, FIXTURE_BASELINES, step,
                                             q.anchor)
            @test out.correction == SolarSINDy.operational_inertial_sindy_correction(art.calibration, feats)
            @test out.t1r_center == clamp(raw + out.correction, -2000.0, 50.0)
            w = FIXTURE_LAT[slot]
            @test out.center == w * out.t1r_center + (1 - w) * frozen
            @test out.lat_weight == w
            @test out.model_step_hours == step
            # Without a matured innovation block every step keeps the identity layer.
            @test !out.e_layer_applied
            @test out.e_delta == 0.0
            @test out.final == out.center
        end
        # The five core-dependent features are the only ones that move with the raw core.
        a = analog_serving_t1r_features(art, -58.25, q.dst_anchor, issue_drv, FIXTURE_MEMORY,
                                     FIXTURE_BASELINES, 3, q.anchor)
        b = analog_serving_t1r_features(art, -40.0, q.dst_anchor, issue_drv, FIXTURE_MEMORY,
                                     FIXTURE_BASELINES, 3, q.anchor)
        moved = [c for c in keys(a) if getproperty(a, c) != getproperty(b, c)]
        @test Set(moved) == Set([:baseline_spread_nt, :v1_minus_persistence_nt,
                                 :obrien_minus_v1_nt, :burton_minus_v1_nt,
                                 :lead_v1_persistence_interaction])
        # Independent expectation for two of them.
        @test b.v1_minus_persistence_nt == -40.0 - FIXTURE_BASELINES.persistence
        @test b.lead_v1_persistence_interaction ==
              3.0 * (-40.0 - FIXTURE_BASELINES.persistence)

        # With a complete innovation block the boosted layer at one hour acts, capped.
        innovations = Float64[1.5, -2.0, 0.5, 3.0, -1.0, 0.25]
        acted = analog_serving_center(
            art; raw_reported = raw, latest_dst = q.dst_anchor,
            anchor_drivers = issue_drv, memory = FIXTURE_MEMORY,
            baselines = FIXTURE_BASELINES, model_steps = 1, frozen_inertial_sindy = frozen,
            analog_features = key.features, anchor_time = q.anchor,
            innovations = innovations)
        @test acted.e_layer_applied
        @test acted.e_layer == "E2"
        @test abs(acted.e_delta) <= analog_serving_e_cap(1)
        @test acted.final == acted.center + acted.e_delta
        design = vcat(innovations, collect(key.features), [acted.center])
        expected_delta = clamp(
            gbm_predict(art.e_layers[1].boosted, reshape(design, 1, length(design)))[1],
            -analog_serving_e_cap(1), analog_serving_e_cap(1))
        @test acted.e_delta == expected_delta
        # A step whose layer is the identity ignores the innovation block entirely.
        identity_step = analog_serving_center(
            art; raw_reported = raw, latest_dst = q.dst_anchor,
            anchor_drivers = issue_drv, memory = FIXTURE_MEMORY,
            baselines = FIXTURE_BASELINES, model_steps = 3, frozen_inertial_sindy = frozen,
            analog_features = key.features, anchor_time = q.anchor,
            innovations = innovations)
        @test !identity_step.e_layer_applied
        @test identity_step.final == identity_step.center
        # A non-finite innovation keeps the identity rather than imputing.
        holed = copy(innovations)
        holed[3] = NaN
        @test !analog_serving_center(
            art; raw_reported = raw, latest_dst = q.dst_anchor,
            anchor_drivers = issue_drv, memory = FIXTURE_MEMORY,
            baselines = FIXTURE_BASELINES, model_steps = 1, frozen_inertial_sindy = frozen,
            analog_features = key.features, anchor_time = q.anchor,
            innovations = holed).e_layer_applied
        # Unsupported steps and non-finite blend partners are programming errors.
        @test_throws ArgumentError analog_serving_center(
            art; raw_reported = raw, latest_dst = q.dst_anchor,
            anchor_drivers = issue_drv, memory = FIXTURE_MEMORY,
            baselines = FIXTURE_BASELINES, model_steps = 5, frozen_inertial_sindy = frozen,
            analog_features = key.features, anchor_time = q.anchor, innovations = nothing)
        @test_throws ArgumentError analog_serving_center(
            art; raw_reported = raw, latest_dst = q.dst_anchor,
            anchor_drivers = issue_drv, memory = FIXTURE_MEMORY,
            baselines = FIXTURE_BASELINES, model_steps = 3, frozen_inertial_sindy = NaN,
            analog_features = key.features, anchor_time = q.anchor, innovations = nothing)

        # The frozen-tail blend partner holds the issue driver for every step, which is what the
        # scored candidate blends against.
        cal = read_inertial_sindy_calibration(
            operational_calibration_artifacts(OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION).point_csv)
        for step in ANALOG_SERVING_MODEL_STEPS
            got = analog_serving_frozen_center(
                art; inertial_sindy_calibration = cal, issue_drv = issue_drv,
                anchor_dst_star = anchor_star, latest_dst = q.dst_anchor,
                memory = FIXTURE_MEMORY, baselines = FIXTURE_BASELINES,
                model_steps = step, anchor_time = q.anchor)
            trajectory = operational_core_forecast(art.core, anchor_star, issue_drv, step)
            @test got.raw == dst_star_to_dst(trajectory[step], issue_drv.Pdyn)
            @test got.center == clamp(got.raw + got.correction, -2000.0, 50.0)
        end
    end

    @testset "hourly step-driver policy restates the L1 admission gate" begin
        fixture = _fixture_artifacts()
        art = fixture.art
        anchor = fixture.origins[50]
        issue_drv = art.lookup[anchor - Hour(1)]
        policy = analog_serving_step_driver_from_frame(art.lookup, anchor, issue_drv)
        kdelta = floor(Int, analog_serving_transit_hours(issue_drv.V))
        for k in 1:7
            decision = policy(k, issue_drv)
            @test decision.l1_measured == (k <= kdelta)
            if k <= kdelta
                record = art.lookup[anchor + Hour(k - 1)]
                admitted = k <= analog_serving_transit_hours(record.V)
                @test decision.driver == (admitted ? record : issue_drv)
            else
                @test decision.driver == issue_drv
            end
        end
    end
end

end # module
