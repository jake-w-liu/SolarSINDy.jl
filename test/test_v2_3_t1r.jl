module V23T1rTests

# Unit oracles for the Amendment A1 correction layer
# (`validation/operational/v2_3_t1r.jl`).
#
# T1r refits the deployed V2.1 ridge correction on the analog raw core. Four
# things can silently break that and leave a plausible-looking number behind:
# the raw-core-dependent features can be recomputed wrongly (or not at all), the
# fit can drift off the deployed recipe, the leave-one-block-out cross-fit can
# leak the scored block's observations into its own correction, and the
# safeguard variant can stop being the published V2.1 point operator. Each test
# below fixes one of those with an expectation derived independently of the code
# under test.

using Test
using CSV
using DataFrames
using Dates
using LinearAlgebra
using Statistics
using SolarSINDy

include(normpath(joinpath(@__DIR__, "..", "validation", "operational", "v2_3_t1r.jl")))

const DEPLOYED = v23_t1r_deployed_calibration()

"""
Base-table-shaped frame: one issue per hour, the six scored model steps per
issue, and a raw core that is not a function of the baselines, so the expert
features carry real information. The calibration feature columns are built by
the package's own feature path, which is what the archived base table records,
and are therefore the reference the T1r recomputation must reproduce.
"""
function synthetic_base_rows(issues::Int; core_shift::Float64=0.0)
    start = DateTime(2011, 3, 1)
    rows = NamedTuple[]
    for i in 0:issues - 1
        t = start + Hour(i)
        latest = -35.0 + 45.0 * sin(2π * i / 53) - 12.0 * cos(2π * i / 17)
        speed = 380.0 + 160.0 * sin(2π * i / 29)
        bz = -8.0 * sin(2π * i / 23) + 2.5 * cos(2π * i / 11)
        by = 4.0 * cos(2π * i / 31)
        density = 5.5 + 3.0 * sin(2π * i / 19)
        pdyn = dynamic_pressure(density, speed)
        for (slot, h) in enumerate(V23_MODEL_STEPS)
            core = latest - 4.0 * sin(2π * (i + 3h) / 41) - 0.6 * h + core_shift
            push!(rows, (
                issue_time_utc=t,
                model_step_hours=h,
                cv_block=2013 + mod(div(i, 24), 3),
                latest_dst_nt=latest,
                V_kms=speed,
                Bz_nt=bz,
                By_nt=by,
                n_cm3=density,
                Pdyn_npa=pdyn,
                persistence_dst_nt=latest,
                burton_dst_nt=latest - 1.3 * h * max(-bz, 0.0) / 6,
                burton_full_dst_nt=latest - 1.1 * h * max(-bz, 0.0) / 6,
                obrien_dst_nt=latest - 0.9 * h * max(-bz, 0.0) / 6,
                observation_dst_nt=core + 3.0 * cos(2π * (i + slot) / 13),
                pred_dst_nt=core,
                pred_dst_ci05_nt=core - 9.0,
                pred_dst_ci95_nt=core + 9.0,
            ))
        end
    end
    df = DataFrame(rows)
    SolarSINDy.add_operational_v2_features!(df)
    return df
end

@testset "frozen features recomputed from the raw inputs reproduce the base table" begin
    base = synthetic_base_rows(240)
    oracle = v23_t1r_feature_oracle(base)
    @test oracle.worst_nt <= 1e-9
    @test length(oracle.columns) == 26
    @test all(c in oracle.columns for c in V23_T1R_RAW_CORE_FEATURES)

    # The oracle must be able to fail: with a shifted core exactly the five
    # raw-core-dependent features move, and every other feature is untouched.
    shifted = v23_t1r_features(base, Float64.(base.pred_dst_nt) .- 7.0)
    moved = Symbol[]
    for c in DEPLOYED.feature_names
        maximum(abs.(shifted[!, c] .- Float64.(base[!, c]))) > 1e-9 && push!(moved, c)
    end
    @test Set(moved) == Set(V23_T1R_RAW_CORE_FEATURES)
    @test all(isapprox.(shifted.v1_minus_persistence_nt,
                        Float64.(base.v1_minus_persistence_nt) .- 7.0; atol=1e-9))
    @test all(isapprox.(shifted.obrien_minus_v1_nt,
                        Float64.(base.obrien_minus_v1_nt) .+ 7.0; atol=1e-9))
    @test all(isapprox.(shifted.burton_minus_v1_nt,
                        Float64.(base.burton_minus_v1_nt) .+ 7.0; atol=1e-9))
    @test all(isapprox.(shifted.lead_v1_persistence_interaction,
                        Float64.(base.lead_v1_persistence_interaction) .-
                        7.0 .* max.(Float64.(base.model_step_hours), 1.0); atol=1e-9))
    @test v23_t1r_feature_oracle(base).worst_nt == 0.0

    # Same check against the archived base table when it exists: the first rows
    # of the real artifact, whose feature columns were written by the frozen V2.1
    # replay rather than by this session.
    if isfile(V23_BASE_TABLE)
        slice = CSV.read(V23_BASE_TABLE, DataFrame; limit=1200, ntasks=1,
                         types=Dict("issue_time_utc" => DateTime,
                                    "target_time_utc" => DateTime))
        archived = v23_t1r_feature_oracle(slice)
        @test archived.worst_nt <= 1e-9
        @test archived.n == nrow(slice)
    end
end

@testset "the T1r fit is the deployed V2.1 recipe on a different raw core" begin
    base = synthetic_base_rows(200)
    features = v23_t1r_features(base, Float64.(base.pred_dst_nt) .- 5.0)
    cal = v23_t1r_fit(features; label_stem="test_stem")

    @test cal.feature_names == DEPLOYED.feature_names
    @test length(cal.feature_names) == 26
    @test length(cal.coefficients) == 27
    @test cal.selected_component == DEPLOYED.selected_component
    @test cal.guard_margin_nt == DEPLOYED.guard_margin_nt
    @test cal.supported_model_steps == DEPLOYED.supported_model_steps
    @test occursin("_memory_expert_lead_ridge100.0_fit", cal.label)
    @test occursin("_memory_expert_lead_ridge100.0_fit", DEPLOYED.label)
    @test match(r"^test_stem_memory_expert_lead_ridge100\.0_fit(\d+)$", cal.label) !== nothing
    @test parse(Int, match(r"_fit(\d+)$", cal.label).captures[1]) == nrow(features)

    # Independent ridge solution: standardize by the fitted rows' mean/sd, leave
    # the intercept unpenalized, solve the normal equations at λ = 100.
    x = ones(nrow(features), 27)
    for (j, name) in enumerate(cal.feature_names)
        column = Float64.(features[!, name])
        scale = std(column)
        scale == 0.0 && (scale = 1.0)
        @test isapprox(cal.feature_mean[j], mean(column); atol=1e-12)
        @test isapprox(cal.feature_scale[j], scale; atol=1e-12)
        x[:, j + 1] = (column .- mean(column)) ./ scale
    end
    y = Float64.(features.observation_dst_nt) .- Float64.(features.pred_dst_nt)
    penalty = Matrix{Float64}(I, 27, 27)
    penalty[1, 1] = 0.0
    expected = (x' * x + 100.0 * penalty) \ (x' * y)
    @test isapprox(cal.coefficients, expected; rtol=1e-9, atol=1e-9)

    # The check discriminates: a different ridge strength moves the solution far
    # beyond the tolerance above.
    other = (x' * x + 10.0 * penalty) \ (x' * y)
    @test maximum(abs.(other .- expected)) > 1e-3

    # The declared fit-row count is the package's own finite-row rule.
    dirty = copy(features)
    dirty[1, :Bz_nt] = NaN
    dirty[2, :observation_dst_nt] = NaN
    dirty[3, :baseline_spread_nt] = NaN
    @test count(v23_t1r_finite_mask(dirty)) == nrow(dirty) - 3
    required = vcat(collect(V23_T1R_FIT_REQUIRED), DEPLOYED.feature_names)
    @test count(v23_t1r_finite_mask(dirty)) == nrow(SolarSINDy._finite_rows(dirty, required))
end

@testset "a zero correction leaves the analog raw core as the T1r center" begin
    base = synthetic_base_rows(40)
    features = v23_t1r_features(base, Float64.(base.pred_dst_nt) .- 2.0)
    # Physical-range boundaries, so the projection in the center is exercised.
    features[1, :pred_dst_nt] = -2500.0
    features[2, :pred_dst_nt] = 90.0
    zeroed = default_operational_v2_calibration(
        feature_names=copy(DEPLOYED.feature_names), label="zero_correction",
    )
    @test all(iszero, zeroed.coefficients)
    centers = v23_t1r_apply(zeroed, features)
    @test centers == clamp.(Float64.(features.pred_dst_nt), -2000.0, 50.0)
    @test centers[1] == -2000.0
    @test centers[2] == 50.0

    # And a nonzero intercept moves every center by exactly that intercept.
    shifted = OperationalV2Calibration(
        copy(zeroed.feature_names), copy(zeroed.feature_mean), copy(zeroed.feature_scale),
        vcat([2.5], zeros(26)), 1.0, "intercept_only",
    )
    interior = 3:nrow(features)
    @test all(isapprox.(v23_t1r_apply(shifted, features)[interior],
                        Float64.(features.pred_dst_nt)[interior] .+ 2.5; atol=1e-12))
end

@testset "the leave-one-block-out cross-fit cannot see the scored block" begin
    base = synthetic_base_rows(216)
    features = v23_t1r_features(base, Float64.(base.pred_dst_nt) .- 4.0)
    blocks = sort(unique(Int.(features.cv_block)))
    @test length(blocks) == 3
    reference = v23_t1r_crossfit(features, blocks; label_stem="test_xf")
    @test all(isfinite, reference.center_off)
    @test all(isfinite, reference.center_on)
    @test sort(collect(keys(reference.calibrations))) == blocks

    for b in blocks
        mutated = copy(features)
        inside = Int.(mutated.cv_block) .== b
        mutated[inside, :observation_dst_nt] .+= 250.0
        perturbed = v23_t1r_crossfit(mutated, blocks; label_stem="test_xf")
        # Block b's own targets never enter its correction.
        @test perturbed.center_off[inside] == reference.center_off[inside]
        @test perturbed.center_on[inside] == reference.center_on[inside]
        # The perturbation is large enough to be detected, so the invariance
        # above is a property of the split and not of an inert mutation.
        @test any(perturbed.center_off[.!inside] .!= reference.center_off[.!inside])
        @test perturbed.calibrations[b].coefficients == reference.calibrations[b].coefficients
    end

    # Excluding rows from every fit changes the layer but not which rows are scored.
    mask = trues(nrow(features))
    mask[1:600] .= false
    restricted = v23_t1r_crossfit(features, blocks; label_stem="test_xf", fit_mask=mask)
    @test all(isfinite, restricted.center_off)
    @test restricted.center_off != reference.center_off
end

@testset "the safeguard variant is the published V2.1 point operator" begin
    base = synthetic_base_rows(60)
    features = v23_t1r_features(base, Float64.(base.pred_dst_nt) .- 6.0)
    cal = v23_t1r_fit(features; label_stem="test_guard")
    sample = features[1:20, :]
    bare = v23_t1r_apply(cal, sample)
    guarded = v23_t1r_apply(cal, sample; safeguards=true)
    @test length(guarded) == 20

    columns = Vector{Float64}[Float64.(sample[!, c]) for c in cal.feature_names]
    for i in 1:20
        feats = NamedTuple{Tuple(cal.feature_names)}(Tuple(c[i] for c in columns))
        correction = SolarSINDy.operational_v2_correction(cal, feats)
        raw = Float64(sample.pred_dst_nt[i])
        latest = Float64(sample.latest_dst_nt[i])
        step = Int(sample.model_step_hours[i])
        rate = Float64(sample.dst_delta_1h_nt[i])
        @test bare[i] == clamp(raw + correction, -2000.0, 50.0)
        @test guarded[i] == _apply_v2_1_safeguards(bare[i], latest, step, rate)
        # The operator projects its argument first, so the runner's unprojected
        # sum and the projected center give the same guarded center.
        @test guarded[i] == v23_center(raw, correction, latest, step, rate, true)
        @test bare[i] == v23_center(raw, correction, latest, step, rate, false)
    end
    @test any(guarded .!= bare)
end

@testset "both safeguard variants come from one evaluation of the layer" begin
    base = synthetic_base_rows(80)
    features = v23_t1r_features(base, Float64.(base.pred_dst_nt) .- 3.0)
    cal = v23_t1r_fit(features; label_stem="test_both")
    sample = features[1:40, :]
    both = v23_t1r_centers(cal, sample)
    # Bit-for-bit, not approximately: the confirmatory runner scores both options
    # for every configuration, and a shared correction vector is only admissible
    # if it reproduces the single-variant path exactly.
    @test both.off == v23_t1r_apply(cal, sample; safeguards=false)
    @test both.on == v23_t1r_apply(cal, sample; safeguards=true)
    @test both.corrections == v23_t1r_correction_vector(cal, sample)
    @test length(both.corrections) == nrow(sample)
    # The safeguard operator actually moves something here, so the equality above
    # is not the trivial case of two identical vectors.
    @test any(both.on .!= both.off)
    # The correction is what separates the center from the raw core.
    @test all(isapprox.(both.off, clamp.(Float64.(sample.pred_dst_nt) .+ both.corrections,
                                         -2000.0, 50.0); atol=0.0))
    @test any(abs.(both.corrections) .> 1e-6)

    # Missing inputs fail closed instead of producing a plausible number.
    dropped = select(sample, Not(first(cal.feature_names)))
    @test_throws ArgumentError v23_t1r_correction_vector(cal, dropped)
    coreless = select(sample, Not(:pred_dst_nt))
    @test_throws ArgumentError v23_t1r_correction_vector(cal, coreless)
    @test_throws DimensionMismatch _v23_t1r_from_corrections(sample, both.corrections[1:5];
                                                             safeguards=false)
end

@testset "the base-table column selection covers every T1r input and oracle column" begin
    columns = v23_t1r_base_table_columns()
    @test length(columns) == length(unique(columns))
    for name in ("issue_time_utc", "model_step_hours", "partition", "pred_dst_nt")
        @test name in columns
    end
    for name in V23_T1R_PASSTHROUGH_INPUTS
        @test String(name) in columns
    end
    for name in DEPLOYED.feature_names
        @test String(name) in columns
    end
    for name in V23_T1R_RAW_CORE_FEATURES
        @test String(name) in columns
    end
    # 3 keys + 16 passthrough + 26 features + the raw core, less the passthrough
    # columns that are themselves calibration features.
    @test length(columns) == 34

    # A frame carrying exactly this selection is enough for the feature path and
    # for the frozen-feature oracle, which is the property the confirmatory
    # runner relies on when it re-reads the base table.
    base = synthetic_base_rows(60)
    base[!, :partition] = fill("TEST", nrow(base))
    slice = base[!, columns]
    @test v23_t1r_feature_oracle(slice).worst_nt <= 1e-9
    @test nrow(v23_t1r_features(slice, Float64.(slice.pred_dst_nt) .- 1.0)) == nrow(slice)

    if isfile(V23_BASE_TABLE)
        header = String.(propertynames(CSV.read(V23_BASE_TABLE, DataFrame; limit=1,
                                                ntasks=1)))
        for name in columns
            @test name in header
        end
    end
end

@testset "T1r artifacts carry a provenance key that a stale set cannot pass" begin
    directory = mktempdir()
    config = "T1_magnetic_K25"
    core = joinpath(directory, "oof_$(config).csv")
    write(core, "issue_time_utc,raw_dst_nt\n2015-01-01T00:00:00,-31.5\n")

    # The key is a function of the analog core, the base table, the V2.3 sources
    # and this file, so any of them moving must move the key.
    key = v23_t1r_signature(directory, config; table_sha="table-sha", code_sha="code-sha")
    @test key == v23_t1r_signature(directory, config; table_sha="table-sha",
                                   code_sha="code-sha")
    @test key != v23_t1r_signature(directory, config; table_sha="other-table",
                                   code_sha="code-sha")
    @test key != v23_t1r_signature(directory, config; table_sha="table-sha",
                                   code_sha="other-code")
    write(core, "issue_time_utc,raw_dst_nt\n2015-01-01T00:00:00,-99.0\n")
    @test key != v23_t1r_signature(directory, config; table_sha="table-sha",
                                   code_sha="code-sha")

    # An incomplete artifact set never validates, however good the key looks.
    write(v23_t1r_signature_path(directory, config),
          v23_t1r_signature(directory, config; table_sha="table-sha", code_sha="code-sha"))
    @test !v23_t1r_signature_ok(directory, config; table_sha="table-sha",
                                code_sha="code-sha")
    touch(v23_t1r_oof_path(directory, config))
    touch(v23_t1r_summary_path(directory, config))
    touch(v23_t1r_calibration_path(directory, config))
    @test v23_t1r_signature_ok(directory, config; table_sha="table-sha", code_sha="code-sha")
    @test v23_t1r_require_signature(directory, config; table_sha="table-sha",
                                    code_sha="code-sha")

    # A stale key and a missing key are both refused, with the reason named.
    @test !v23_t1r_signature_ok(directory, config; table_sha="table-sha",
                                code_sha="other-code")
    @test_throws ErrorException v23_t1r_require_signature(directory, config;
                                                          table_sha="table-sha",
                                                          code_sha="other-code")
    rm(v23_t1r_signature_path(directory, config))
    @test !v23_t1r_signature_ok(directory, config; table_sha="table-sha", code_sha="code-sha")
    @test_throws ErrorException v23_t1r_require_signature(directory, config;
                                                          table_sha="table-sha",
                                                          code_sha="code-sha")
    # A configuration whose analog core is absent cannot be keyed at all.
    @test_throws ErrorException v23_t1r_signature(directory, "T1_uniform_K200";
                                                  table_sha="table-sha", code_sha="code-sha")
end

end # module
