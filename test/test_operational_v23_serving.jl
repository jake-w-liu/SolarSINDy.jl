# test_operational_v23_serving.jl — V2.3 shadow deployment loading and center formation.
#
# The shadow center is only evidence about the scored candidate if the deployment it reads is the
# scored artifact and the arithmetic on top of it is the scored arithmetic. Two kinds of failure are
# therefore checked here against independent expectations rather than against the implementation:
#
#   * provenance — a tampered artifact, a missing digest row, an archive whose rebuilt standardisation
#     disagrees with the shipped one, or a wrong error-layer schema must all fail at load;
#   * arithmetic — the ensemble raw center, the lead-aware blend, the capped error layer and the
#     frozen-tail blend partner are each recomputed here from their definitions and compared.
#
# The fixture is a small synthetic deployment so these paths run without the 87,000-origin archive.
# The full-scale identity against the scored `V2_3_final` column lives in
# validation/operational/v2_3_serving_identity.jl.

module OperationalV23ServingTests

using Test
using CSV
using DataFrames
using Dates
using JSON3
using SHA
using Statistics
using SolarSINDy

const REAL_SHADOW_DIR = normpath(joinpath(@__DIR__, "..", "deploy", "v2_3_shadow"))
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

"Write a complete, self-consistent V2.3 deployment fixture into `dir`; returns its bookkeeping."
function _write_fixture(dir::AbstractString; with_e2::Bool = true)
    mkpath(dir)
    frame = _fixture_frame()
    lookup = v23_serving_frame_lookup(frame)
    candidates = frame.time_utc
    X, ok = v23_feature_matrix(frame, candidates)
    continuable = v23_analog_origin_ok(lookup, candidates)
    keep = [j for j in eachindex(candidates) if ok[j] && continuable[j]]
    origins = candidates[keep]
    length(origins) > FIXTURE_K || error("fixture archive is too small")
    stats = v23_feature_stats(X[keep, :])

    CSV.write(joinpath(dir, "analog_frame_2010_2019.csv"), frame)
    CSV.write(joinpath(dir, "analog_origins.csv"), DataFrame(origin_time_utc = origins))
    CSV.write(joinpath(dir, "analog_feature_stats.csv"),
              DataFrame(feature_name = String.(collect(V23_FEATURE_NAMES)),
                        feature_mean = stats.mean, feature_sd = stats.sd))
    # The correction layer is the deployed V2.1 ridge artifact: a real 26-feature calibration, so the
    # fixture exercises the same feature contract the analog refit uses.
    cp(operational_calibration_artifacts(OPERATIONAL_V2_1_MODEL_VERSION).point_csv,
       joinpath(dir, "t1r_calibration.csv"); force = true)
    chmod(joinpath(dir, "t1r_calibration.csv"), 0o644)
    CSV.write(joinpath(dir, "lat_weights.csv"),
              DataFrame(model_step_hours = collect(V23_SERVING_MODEL_STEPS),
                        n = fill(1000, length(FIXTURE_LAT)),
                        selected_weight = FIXTURE_LAT))

    e2_names = vcat(String.(collect(V23_SERVING_E_FEATURE_NAMES))[1:V23_SERVING_E_INNOVATION_LAGS],
                    String.(collect(V23_FEATURE_NAMES)), ["center_dst_nt"])
    steps = Any[]
    for (slot, step) in enumerate(V23_SERVING_MODEL_STEPS)
        if with_e2 && step == 1
            rows = 200
            design = Matrix{Float64}(undef, rows, length(e2_names))
            for i in 1:rows, j in 1:length(e2_names)
                design[i, j] = sin(0.7 * i + 0.11 * j) + 0.01 * i
            end
            target = [0.3 * design[i, 1] - 0.2 * design[i, 7] for i in 1:rows]
            model = v23_fit_gbm(design, target; max_depth = 3, nrounds = 8, min_weight = 1,
                                feature_names = e2_names)
            v23_save(model, joinpath(dir, "e2_step1.bson"))
            push!(steps, Dict{String,Any}(
                "model_step_hours" => step, "layer" => "E2", "param" => "(3, 8)",
                "development_gain_nt" => 0.1, "artifact" => "e2_step1.bson",
                "feature_names" => e2_names,
                "correction_cap_nt" => v23_serving_e_cap(step)))
        else
            push!(steps, Dict{String,Any}(
                "model_step_hours" => step, "layer" => "identity", "param" => nothing,
                "development_gain_nt" => 0.0, "artifact" => nothing,
                "feature_names" => String[],
                "correction_cap_nt" => v23_serving_e_cap(step)))
        end
    end
    open(joinpath(dir, "e_layers.json"), "w") do io
        JSON3.pretty(io, Dict{String,Any}(
            "selected_config" => "T1r_T1_magnetic_K$(FIXTURE_K)_Soff",
            "lat_weights" => FIXTURE_LAT, "steps" => steps))
    end
    open(joinpath(dir, "selected_configuration.json"), "w") do io
        JSON3.pretty(io, Dict{String,Any}(
            "family" => "T1r", "k" => FIXTURE_K, "safeguards" => false,
            "selected_config" => "T1r_T1_magnetic_K$(FIXTURE_K)_Soff",
            "model_steps" => collect(V23_SERVING_MODEL_STEPS),
            "lat_weights" => FIXTURE_LAT,
            "params" => Dict{String,Any}("weight_set" => "magnetic", "tail" => "T1r",
                                         "k" => FIXTURE_K, "direct" => false)))
    end
    _write_fixture_manifest(dir, origins)
    return (frame = frame, origins = origins, stats = stats, lookup = lookup,
            e2_names = e2_names)
end

"Digest manifest of a fixture directory; every shipped file gets a row, as the loader requires."
function _write_fixture_manifest(dir::AbstractString, origins::Vector{DateTime})
    files = [f for f in readdir(dir) if f != "manifest.csv"]
    rows = NamedTuple[]
    push!(rows, (entry_type = "analog_archive", name = "origins",
                 count = Float64(length(origins)),
                 value = "first=$(first(origins));last=$(last(origins))"))
    for f in sort(files)
        path = joinpath(dir, f)
        push!(rows, (entry_type = "sha256", name = f, count = Float64(filesize(path)),
                     value = v23_serving_file_sha256(path)))
    end
    CSV.write(joinpath(dir, "manifest.csv"), DataFrame(rows))
    return nothing
end

"A query anchor far outside the archive, plus its driver history and the two anchor Dst values."
function _fixture_query()
    anchor = DateTime(2026, 3, 1, 12)
    history = Any[(V = 420.0 + 3.0 * lag, Bz = -5.0 + 0.4 * lag, By = 1.5 - 0.1 * lag,
                   n = 6.0 + 0.2 * lag,
                   Pdyn = dynamic_pressure(6.0 + 0.2 * lag, 420.0 + 3.0 * lag))
                  for lag in 1:V23_SOUTH_RUN_CAP_H]
    return (anchor = anchor, history = history, dst_anchor = -62.0, dst_previous = -55.0)
end

const FIXTURE_MEMORY = (dst_delta_1h_nt = -7.0, dst_delta_3h_nt = -18.0,
                        Bz_delta_1h_nt = -1.5, VBsouth_delta_1h_mvm = 0.4,
                        VBsouth_mean_3h_mvm = 1.9, Bsouth_mean_3h_nt = 4.5)
const FIXTURE_BASELINES = (persistence = -62.0, burton = -70.0, burton_full = -68.0,
                           obrien = -66.0)

@testset "Operational V2.3 shadow serving" begin
    @testset "published identity strings" begin
        @test V23_SERVING_IDENTITY == "v2.3+sindy20x11+L1A+ADC(magnetic,K25)+T1rcal+LAT+E"
        @test V23_SERVING_SHADOW_IDENTITY ==
              "v2.3-shadow+sindy20x11+L1A+ADC(magnetic,K25)+T1rcal+LAT+E"
        @test v23_serving_identity(:uniform, 50) ==
              "v2.3+sindy20x11+L1A+ADC(uniform,K50)+T1rcal+LAT+E"
        @test collect(V23_SERVING_MODEL_STEPS) == [1, 2, 3, 4, 6, 7]
        @test [v23_serving_e_cap(h) for h in V23_SERVING_MODEL_STEPS] ==
              [10.0, 15.0, 20.0, 25.0, 35.0, 40.0]
        @test V23_SERVING_STEP_SLOT[6] == 5
        @test V23_SERVING_STEP_SLOT[5] == 0
    end

    @testset "ballistic transit helper" begin
        # Independent expectation: 1.5e6 km at 500 km/s is 3000 s, i.e. 5/6 h.
        @test v23_serving_transit_hours(500.0) ≈ 1.5e6 / 500.0 / 3600.0 atol=1e-12
        @test v23_serving_transit_hours(500.0) ≈ 5 / 6 atol=1e-12
        @test v23_serving_transit_hours(0.0) == 0.0
        @test v23_serving_transit_hours(NaN) == 0.0
        @test v23_serving_transit_hours(-400.0) == 0.0
    end

    @testset "innovation lag block is all-or-nothing" begin
        anchor = DateTime(2026, 3, 1, 12)
        full = Dict(anchor - Hour(lag) => Float64(lag) for lag in 1:8)
        block = v23_serving_innovation_lags(anchor, full)
        @test block.ok
        @test block.values == Float64.(1:6)
        holed = copy(full)
        delete!(holed, anchor - Hour(4))
        @test !v23_serving_innovation_lags(anchor, holed).ok
        nonfinite = copy(full)
        nonfinite[anchor - Hour(2)] = NaN
        @test !v23_serving_innovation_lags(anchor, nonfinite).ok
        @test !v23_serving_innovation_lags(anchor, Dict{DateTime,Float64}()).ok
    end

    @testset "innovations are formed from one-hour centers and matured observations" begin
        # The live engine has no one-hour issued row, so it records the one-hour pre-layer center of
        # every anchor and pairs it with the Dst observed one hour later. This is the definition the
        # offline identity oracle also uses; the arithmetic is checked here against hand values.
        anchor = DateTime(2026, 3, 1, 12)
        centers = Dict(anchor - Hour(lag) => -40.0 - Float64(lag) for lag in 1:6)
        observed = Dict(anchor - Hour(lag) + Hour(1) => -50.0 - Float64(lag) for lag in 1:6)
        innovations = v23_serving_innovations_from_step1_centers(centers, observed)
        @test length(innovations) == 6
        # Dst(a+1) - center(a) = (-50 - lag) - (-40 - lag) = -10 at every lag.
        @test all(≈(-10.0), collect(values(innovations)))
        block = v23_serving_innovation_lags(anchor, innovations)
        @test block.ok
        @test block.values == fill(-10.0, V23_SERVING_E_INNOVATION_LAGS)

        # An anchor whose one-hour target has not been observed yet is not matured and must not enter:
        # imputing it would give the layer a residual no forecaster could have scored.
        unmatured = copy(observed)
        delete!(unmatured, anchor)
        @test length(v23_serving_innovations_from_step1_centers(centers, unmatured)) == 5
        @test !v23_serving_innovation_lags(
            anchor, v23_serving_innovations_from_step1_centers(centers, unmatured),
        ).ok

        # A non-finite center or observation is dropped rather than propagated as NaN.
        nan_center = copy(centers)
        nan_center[anchor - Hour(3)] = NaN
        @test length(v23_serving_innovations_from_step1_centers(nan_center, observed)) == 5
        nan_obs = copy(observed)
        nan_obs[anchor - Hour(3) + Hour(1)] = NaN
        @test length(v23_serving_innovations_from_step1_centers(centers, nan_obs)) == 5

        # The pairing is by anchor + 1 h, not by anchor: an observation series shifted by one hour must
        # not silently produce innovations against the wrong target.
        shifted = Dict(anchor - Hour(lag) => -50.0 - Float64(lag) for lag in 1:6)
        @test length(v23_serving_innovations_from_step1_centers(centers, shifted)) == 5
        wrong = v23_serving_innovations_from_step1_centers(centers, shifted)
        @test !haskey(wrong, anchor - Hour(1))
    end

    @testset "fixture deployment loads and rebuilds its archive" begin
        mktempdir() do dir
            fixture = _write_fixture(dir)
            art = load_v23_serving_artifacts(dir)
            @test art.identity == v23_serving_identity(:magnetic, FIXTURE_K)
            @test art.k == FIXTURE_K
            @test art.weight_set === :magnetic
            @test art.weights == v23_weights(:magnetic)
            @test art.origins == fixture.origins
            @test art.lat_weights == FIXTURE_LAT
            @test art.exclusion_hours == V23_ANALOG_EXCLUSION_HOURS
            # The standardisation is recomputed from the frame, not read from the shipped table.
            @test art.feature_mean ≈ fixture.stats.mean atol=0
            @test art.feature_sd ≈ fixture.stats.sd atol=0
            @test size(art.archive_standardised) == (length(fixture.origins), V23_FEATURE_COUNT)
            @test all(isfinite, art.archive_standardised)
            # The standardised archive must be exactly the shipped standardisation applied to the
            # rebuilt features; recomputing it independently catches a permuted feature order.
            expected = (art.archive_features .- transpose(art.feature_mean)) ./
                       transpose(art.feature_sd)
            @test art.archive_standardised == expected
            @test length(art.e_layers) == length(V23_SERVING_MODEL_STEPS)
            @test art.e_layers[1].kind === :E2
            @test art.e_layers[1].cap == v23_serving_e_cap(1)
            @test art.e_layers[1].feature_names == fixture.e2_names
            @test all(l -> l.kind === :identity, art.e_layers[2:end])
            @test length(art.calibration.feature_names) == 26
        end
    end

    @testset "deployment provenance failures are load errors" begin
        mktempdir() do dir
            _write_fixture(dir)
            @test v23_serving_verify_manifest(dir) isa DataFrame
            # A tampered artifact must not load.
            open(joinpath(dir, "lat_weights.csv"), "a") do io
                write(io, "\n")
            end
            @test_throws ErrorException load_v23_serving_artifacts(dir)
        end
        mktempdir() do dir
            _write_fixture(dir)
            # A manifest that omits a required artifact leaves that file unverified.
            manifest = CSV.read(joinpath(dir, "manifest.csv"), DataFrame)
            trimmed = manifest[.!((manifest.entry_type .== "sha256") .&
                                 (manifest.name .== "t1r_calibration.csv")), :]
            CSV.write(joinpath(dir, "manifest.csv"), trimmed)
            @test_throws ErrorException load_v23_serving_artifacts(dir)
        end
        mktempdir() do dir
            fixture = _write_fixture(dir)
            # An archive whose origin count disagrees with the scoring run's record is a different
            # archive, and the distance metric would be standardised against different statistics.
            CSV.write(joinpath(dir, "analog_origins.csv"),
                      DataFrame(origin_time_utc = fixture.origins[1:end - 1]))
            _write_fixture_manifest(dir, fixture.origins)
            @test_throws ErrorException load_v23_serving_artifacts(dir)
        end
        mktempdir() do dir
            fixture = _write_fixture(dir)
            # A shipped standardisation that does not match the rebuilt archive means the deployment
            # is not the scored deployment.
            stats = CSV.read(joinpath(dir, "analog_feature_stats.csv"), DataFrame)
            stats.feature_mean[3] += 1e-3
            CSV.write(joinpath(dir, "analog_feature_stats.csv"), stats)
            _write_fixture_manifest(dir, fixture.origins)
            @test_throws ErrorException load_v23_serving_artifacts(dir)
        end
        mktempdir() do dir
            fixture = _write_fixture(dir)
            # The E-layer artifacts are named by e_layers.json rather than by the fixed required-file
            # list, so a manifest with their digest rows removed would load them unverified. Deleting
            # the E2 digest row must therefore be a load error, exactly like a tampered file.
            manifest = CSV.read(joinpath(dir, "manifest.csv"), DataFrame)
            @test "e2_step1.bson" in String.(manifest.name)
            trimmed = manifest[.!((manifest.entry_type .== "sha256") .&
                                 (manifest.name .== "e2_step1.bson")), :]
            CSV.write(joinpath(dir, "manifest.csv"), trimmed)
            @test !("e2_step1.bson" in v23_serving_manifest_hashed_names(
                CSV.read(joinpath(dir, "manifest.csv"), DataFrame; types = Dict("name" => String,
                                                                               "entry_type" => String,
                                                                               "value" => String))))
            @test_throws ErrorException load_v23_serving_artifacts(dir)
            # Without the digest check the same directory still loads, which is what makes the check
            # the thing that refuses the unverified artifact rather than an unrelated failure.
            @test load_v23_serving_artifacts(dir; verify_hashes = false) isa V23ServingArtifacts
            _write_fixture_manifest(dir, fixture.origins)
        end
        mktempdir() do dir
            fixture = _write_fixture(dir)
            # An error-layer cap that is not the published cap would silently widen the correction.
            payload = JSON3.read(read(joinpath(dir, "e_layers.json"), String), Dict{String,Any})
            payload["steps"][1]["correction_cap_nt"] = 99.0
            open(joinpath(dir, "e_layers.json"), "w") do io
                JSON3.pretty(io, payload)
            end
            _write_fixture_manifest(dir, fixture.origins)
            @test_throws ErrorException load_v23_serving_artifacts(dir)
        end
    end

    @testset "analog key reports its own incompleteness" begin
        mktempdir() do dir
            _write_fixture(dir)
            art = load_v23_serving_artifacts(dir)
            q = _fixture_query()
            good = v23_serving_features(art, q.anchor, q.history, q.dst_anchor, q.dst_previous)
            @test good.ok
            @test good.reason == "ok"
            @test length(good.features) == V23_FEATURE_COUNT
            @test all(isfinite, good.features)
            # Independent expectation for the features that are pure functions of the history.
            @test good.features[v23_feature_index(:bz0)] == q.history[1].Bz
            @test good.features[v23_feature_index(:v0)] == q.history[1].V
            @test good.features[v23_feature_index(:dv6)] ==
                  q.history[1].V - q.history[7].V
            @test good.features[v23_feature_index(:bz_mean6)] ≈
                  mean([q.history[j].Bz for j in 1:6]) atol=1e-12
            @test good.features[v23_feature_index(:dst0)] == q.dst_anchor
            @test good.features[v23_feature_index(:ddst1)] == q.dst_anchor - q.dst_previous
            @test good.features[v23_feature_index(:vbs0)] ≈
                  q.history[1].V * max(0.0, -q.history[1].Bz) / 1000.0 atol=1e-12
            # A hole inside the mandatory seven-hour window names the missing lag.
            holed = copy(q.history)
            holed[4] = nothing
            bad = v23_serving_features(art, q.anchor, holed, q.dst_anchor, q.dst_previous)
            @test !bad.ok
            @test bad.reason == "missing_driver_lag4"
            @test all(!isfinite, bad.features)
            # A hole beyond the window only truncates the southward run length, which stays usable.
            truncated = copy(q.history)
            truncated[9] = nothing
            @test v23_serving_features(art, q.anchor, truncated, q.dst_anchor, q.dst_previous).ok
            @test !v23_serving_features(art, q.anchor, q.history, NaN, q.dst_previous).ok
            @test v23_serving_features(art, q.anchor, q.history, NaN, q.dst_previous).reason ==
                  "missing_anchor_dst"
            @test v23_serving_features(art, q.anchor, q.history, q.dst_anchor, NaN).reason ==
                  "missing_previous_dst"
            # Fewer than seven supplied lags is a programming error, not a fallback.
            @test_throws ArgumentError v23_serving_features(
                art, q.anchor, q.history[1:6], q.dst_anchor, q.dst_previous)
        end
    end

    @testset "ensemble raw center is the member mean of the frozen core" begin
        mktempdir() do dir
            _write_fixture(dir)
            art = load_v23_serving_artifacts(dir)
            q = _fixture_query()
            key = v23_serving_features(art, q.anchor, q.history, q.dst_anchor, q.dst_previous)
            issue_drv = q.history[1]
            anchor_star = pressure_correct_dst([q.dst_anchor], [issue_drv.Pdyn])[1]
            # No L1-measured step: every rollout step takes the member's analog continuation.
            never_measured = (k, last_known) -> (driver = last_known, l1_measured = false)
            step = 4
            ensemble = v23_serving_members(
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
                drv = v23_member_driver(issue_drv, art.lookup, origin, k)
                final_drv = drv
                assimilation_predict!(filter, drv)
                filter.mean[1] = clamp(filter.mean[1], -2000.0, 50.0)
            end
            expected = current_dst(filter) + 7.26 * sqrt(max(final_drv.Pdyn, 0.0)) - 11.0
            @test ensemble.member_pred[member] == expected
            # Retrieval must be ordered by distance, so the nearest origin is the first member.
            query_z = v23_standardize(reshape(collect(key.features), 1, V23_FEATURE_COUNT),
                                      art.feature_mean, art.feature_sd)
            distances = [sum(art.weights .*
                             (vec(query_z) .- vec(art.archive_standardised[i, :])) .^ 2)
                         for i in eachindex(art.origins)]
            @test art.origins[argmin(distances)] == ensemble.origins[1]
            # An L1-measured step must bypass the member entirely.
            measured_drv = (V = 600.0, Bz = -12.0, By = 3.0, n = 8.0,
                            Pdyn = dynamic_pressure(8.0, 600.0))
            always_measured = (k, last_known) -> (driver = measured_drv, l1_measured = true)
            measured = v23_serving_members(
                art, key.features; anchor_time = q.anchor, issue_drv = issue_drv,
                anchor_dst_star = anchor_star, model_steps = step,
                step_driver = always_measured)
            @test all(p -> p == measured.member_pred[1], measured.member_pred)
            @test measured.raw != ensemble.raw
            # A non-finite key must be rejected rather than searched.
            @test_throws ArgumentError v23_serving_members(
                art, fill(NaN, V23_FEATURE_COUNT); anchor_time = q.anchor,
                issue_drv = issue_drv, anchor_dst_star = anchor_star, model_steps = step,
                step_driver = never_measured)
        end
    end

    @testset "correction, blend and error layer" begin
        mktempdir() do dir
            _write_fixture(dir)
            art = load_v23_serving_artifacts(dir)
            q = _fixture_query()
            key = v23_serving_features(art, q.anchor, q.history, q.dst_anchor, q.dst_previous)
            issue_drv = q.history[1]
            anchor_star = pressure_correct_dst([q.dst_anchor], [issue_drv.Pdyn])[1]
            raw = -58.25
            frozen = -49.5
            for (slot, step) in enumerate(V23_SERVING_MODEL_STEPS)
                out = v23_serving_center(
                    art; raw_reported = raw, latest_dst = q.dst_anchor,
                    anchor_drivers = issue_drv, memory = FIXTURE_MEMORY,
                    baselines = FIXTURE_BASELINES, model_steps = step, frozen_v2_1 = frozen,
                    analog_features = key.features, anchor_time = q.anchor,
                    innovations = nothing)
                # The correction is the deployed evaluator on the recomputed feature tuple.
                feats = v23_serving_t1r_features(art, raw, q.dst_anchor, issue_drv,
                                                 FIXTURE_MEMORY, FIXTURE_BASELINES, step,
                                                 q.anchor)
                @test out.correction == SolarSINDy.operational_v2_correction(art.calibration, feats)
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
            a = v23_serving_t1r_features(art, -58.25, q.dst_anchor, issue_drv, FIXTURE_MEMORY,
                                         FIXTURE_BASELINES, 3, q.anchor)
            b = v23_serving_t1r_features(art, -40.0, q.dst_anchor, issue_drv, FIXTURE_MEMORY,
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
            acted = v23_serving_center(
                art; raw_reported = raw, latest_dst = q.dst_anchor,
                anchor_drivers = issue_drv, memory = FIXTURE_MEMORY,
                baselines = FIXTURE_BASELINES, model_steps = 1, frozen_v2_1 = frozen,
                analog_features = key.features, anchor_time = q.anchor,
                innovations = innovations)
            @test acted.e_layer_applied
            @test acted.e_layer == "E2"
            @test abs(acted.e_delta) <= v23_serving_e_cap(1)
            @test acted.final == acted.center + acted.e_delta
            design = vcat(innovations, collect(key.features), [acted.center])
            expected_delta = clamp(
                v23_predict(art.e_layers[1].boosted, reshape(design, 1, length(design)))[1],
                -v23_serving_e_cap(1), v23_serving_e_cap(1))
            @test acted.e_delta == expected_delta
            # A step whose layer is the identity ignores the innovation block entirely.
            identity_step = v23_serving_center(
                art; raw_reported = raw, latest_dst = q.dst_anchor,
                anchor_drivers = issue_drv, memory = FIXTURE_MEMORY,
                baselines = FIXTURE_BASELINES, model_steps = 3, frozen_v2_1 = frozen,
                analog_features = key.features, anchor_time = q.anchor,
                innovations = innovations)
            @test !identity_step.e_layer_applied
            @test identity_step.final == identity_step.center
            # A non-finite innovation keeps the identity rather than imputing.
            holed = copy(innovations)
            holed[3] = NaN
            @test !v23_serving_center(
                art; raw_reported = raw, latest_dst = q.dst_anchor,
                anchor_drivers = issue_drv, memory = FIXTURE_MEMORY,
                baselines = FIXTURE_BASELINES, model_steps = 1, frozen_v2_1 = frozen,
                analog_features = key.features, anchor_time = q.anchor,
                innovations = holed).e_layer_applied
            # Unsupported steps and non-finite blend partners are programming errors.
            @test_throws ArgumentError v23_serving_center(
                art; raw_reported = raw, latest_dst = q.dst_anchor,
                anchor_drivers = issue_drv, memory = FIXTURE_MEMORY,
                baselines = FIXTURE_BASELINES, model_steps = 5, frozen_v2_1 = frozen,
                analog_features = key.features, anchor_time = q.anchor, innovations = nothing)
            @test_throws ArgumentError v23_serving_center(
                art; raw_reported = raw, latest_dst = q.dst_anchor,
                anchor_drivers = issue_drv, memory = FIXTURE_MEMORY,
                baselines = FIXTURE_BASELINES, model_steps = 3, frozen_v2_1 = NaN,
                analog_features = key.features, anchor_time = q.anchor, innovations = nothing)

            # The frozen-tail blend partner holds the issue driver for every step, which is what the
            # scored candidate blends against.
            cal = read_operational_v2_calibration(
                operational_calibration_artifacts(OPERATIONAL_V2_1_MODEL_VERSION).point_csv)
            for step in V23_SERVING_MODEL_STEPS
                got = v23_serving_frozen_center(
                    art; v2_1_calibration = cal, issue_drv = issue_drv,
                    anchor_dst_star = anchor_star, latest_dst = q.dst_anchor,
                    memory = FIXTURE_MEMORY, baselines = FIXTURE_BASELINES,
                    model_steps = step, anchor_time = q.anchor)
                trajectory = operational_core_forecast(art.core, anchor_star, issue_drv, step)
                @test got.raw == dst_star_to_dst(trajectory[step], issue_drv.Pdyn)
                @test got.center == clamp(got.raw + got.correction, -2000.0, 50.0)
            end
        end
    end

    @testset "hourly step-driver policy restates the L1 admission gate" begin
        mktempdir() do dir
            fixture = _write_fixture(dir)
            art = load_v23_serving_artifacts(dir)
            anchor = fixture.origins[50]
            issue_drv = art.lookup[anchor - Hour(1)]
            policy = v23_serving_step_driver_from_frame(art.lookup, anchor, issue_drv)
            kdelta = floor(Int, v23_serving_transit_hours(issue_drv.V))
            for k in 1:7
                decision = policy(k, issue_drv)
                @test decision.l1_measured == (k <= kdelta)
                if k <= kdelta
                    record = art.lookup[anchor + Hour(k - 1)]
                    admitted = k <= v23_serving_transit_hours(record.V)
                    @test decision.driver == (admitted ? record : issue_drv)
                else
                    @test decision.driver == issue_drv
                end
            end
        end
    end

    @testset "shipped shadow deployment verifies" begin
        if !isdir(REAL_SHADOW_DIR)
            @test_skip "deploy/v2_3_shadow is absent; run validation/operational/v2_3_build_deploy.jl"
        else
            art = load_v23_serving_artifacts(REAL_SHADOW_DIR)
            @test art.identity == V23_SERVING_IDENTITY
            @test art.k == 25
            @test art.weight_set === :magnetic
            @test art.lat_weights == [1.0, 1.0, 1.0, 1.0, 0.75, 0.75]
            # The selected configuration is the analog candidate the confirmatory run scored.
            @test String(art.selection["selected_config"]) == "T1r_T1_magnetic_K25_Soff"
            @test art.selection["safeguards"] == false
            # Error layers: boosted at one and two hours, identity at three and four, ridge at six
            # and seven, matching the development selection.
            @test [l.kind for l in art.e_layers] ==
                  [:E2, :E2, :identity, :identity, :E1, :E1]
            @test [l.cap for l in art.e_layers] == [10.0, 15.0, 20.0, 25.0, 35.0, 40.0]
            @test length(art.origins) >= 80_000
            @test first(art.origins) >= DateTime(2010, 1, 1)
            @test last(art.origins) <= DateTime(2019, 12, 24, 16)
        end
    end
end

end # module
