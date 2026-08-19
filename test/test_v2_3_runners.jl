module V23RunnerTests

# Unit oracles for the shared V2.3 runner machinery (`validation/operational/v2_3_common.jl`).
#
# The runners are judged by five properties that no amount of downstream
# reporting can repair: the multi-step ensemble rollout must reproduce the
# per-horizon kernel exactly, the nested-K prefix means must equal the
# corresponding ensembles, the composition stages must reduce to their declared
# identities, the direct comparator must be able to follow a level it never saw
# in training, and the resume path must refuse a stale artifact. Each test below
# fixes one of those with an independently derived expectation.

using Test
using CSV
using DataFrames
using Dates
using Statistics
using SolarSINDy

isdefined(@__MODULE__, :V23Context) || include(normpath(joinpath(@__DIR__, "..", "validation", "operational", "v2_3_common.jl")))

isdefined(Main, :LocalArtifactGate) ||
    Base.include(Main, normpath(joinpath(@__DIR__, "local_artifact_gate.jl")))
using Main.LocalArtifactGate: local_artifact_available

const CORE = load_operational_core(OPERATIONAL_V2_1_MODEL_VERSION)
const CAL = read_operational_v2_calibration(
    operational_calibration_artifacts(OPERATIONAL_V2_1_MODEL_VERSION).point_csv,
)
const LIB = CORE.library
const XI = CORE.coefficients

"""
Deterministic hourly frame with enough structure to exercise every branch: the
speed sweeps from 100 to 700 km/s so the L1-admitted window `kΔ` takes the
values 0 through 4, the field reverses sign, and the density stays positive.
"""
function synthetic_frame(hours::Int)
    start = DateTime(2001, 1, 1)
    times = [start + Hour(i) for i in 0:hours - 1]
    speed = [400.0 + 300.0 * sin(2π * i / 97) for i in 0:hours - 1]
    bz = [-9.0 * sin(2π * i / 37) + 3.0 * cos(2π * i / 7) for i in 0:hours - 1]
    by = [5.0 * cos(2π * i / 29) for i in 0:hours - 1]
    density = [5.0 + 3.0 * sin(2π * i / 23) + 1.5 * cos(2π * i / 5) for i in 0:hours - 1]
    pdyn = [dynamic_pressure(density[i], speed[i]) for i in eachindex(speed)]
    dst = [-40.0 + 35.0 * sin(2π * i / 61) - 10.0 * cos(2π * i / 11) for i in 0:hours - 1]
    return DataFrame(time_utc=times, V=speed, Bz=bz, By=by, n=density, Pdyn=pdyn, Dst=dst)
end

@testset "multi-step rollout and prefix means reproduce the per-horizon kernel" begin
    frame = synthetic_frame(600)
    lookup = v2_3_frame_driver_lookup(frame)
    dst = Dict(frame.time_utc[i] => frame.Dst[i] for i in 1:nrow(frame))
    origins = [frame.time_utc[i] for i in 300:307]          # eight distinct analog origins
    prefixes = [1, 2, 4, 8]
    checked = 0
    for issue in frame.time_utc[40:89]                       # fifty consecutive anchors
        driver = lookup[issue - Hour(1)]
        latest = dst[issue]
        rate = latest - dst[issue - Hour(1)]
        anchor_star = pressure_correct_dst([latest], [driver.Pdyn])[1]
        future = k -> get(lookup, issue + Hour(k - 1), nothing)
        members = [v23_analog_member(driver, lookup, s) for s in origins]
        raw = Matrix{Float64}(undef, length(members), V23_STEP_COUNT)
        v23_member_step_raw!(raw, LIB, XI, anchor_star, driver, future, members)
        prefix = Matrix{Float64}(undef, length(prefixes), V23_STEP_COUNT)
        v23_prefix_means!(prefix, raw, prefixes)
        for (row, k) in enumerate(prefixes), (slot, h) in enumerate(V23_MODEL_STEPS)
            features = _v2_features(latest, driver; v1_pred_dst=-100.0, model_steps=h)
            correction = v23_row_correction(features, CAL)
            guarded = _v23_forecast(LIB, XI, anchor_star, driver, future, latest, CAL, h, rate;
                                    tail_members=members[1:k], calibration_features=features)
            bare = _v23_forecast(LIB, XI, anchor_star, driver, future, latest, CAL, h, rate;
                                 tail_members=members[1:k], calibration_features=features,
                                 safeguards=false)
            @test clamp(prefix[row, slot], -2000.0, 50.0) == guarded[1]
            @test v23_center(prefix[row, slot], correction, latest, h, rate, true) == guarded[2]
            @test v23_center(prefix[row, slot], correction, latest, h, rate, false) == bare[2]
            checked += 1
        end
    end
    @test checked == 50 * length(prefixes) * V23_STEP_COUNT
end

@testset "post-issue Dst cannot move a V2.3 rollout" begin
    frame = synthetic_frame(400)
    lookup = v2_3_frame_driver_lookup(frame)
    issue = frame.time_utc[200]
    driver = lookup[issue - Hour(1)]
    latest = frame.Dst[200]
    anchor_star = pressure_correct_dst([latest], [driver.Pdyn])[1]
    future = k -> get(lookup, issue + Hour(k - 1), nothing)
    members = [v23_analog_member(driver, lookup, frame.time_utc[300])]
    before = Matrix{Float64}(undef, 1, V23_STEP_COUNT)
    v23_member_step_raw!(before, LIB, XI, anchor_star, driver, future, members)
    mutated = copy(frame)
    mutated.Dst[200:end] .= -900.0
    mutated_lookup = v2_3_frame_driver_lookup(mutated)
    after = Matrix{Float64}(undef, 1, V23_STEP_COUNT)
    v23_member_step_raw!(after, LIB, XI, anchor_star, driver,
                         k -> get(mutated_lookup, issue + Hour(k - 1), nothing), members)
    @test after == before
    # The same mutation applied to a driver channel must be visible, otherwise
    # the invariance above would be vacuous.
    driver_mutated = copy(frame)
    driver_mutated.Bz[200:end] .= -60.0
    driver_lookup = v2_3_frame_driver_lookup(driver_mutated)
    moved = Matrix{Float64}(undef, 1, V23_STEP_COUNT)
    v23_member_step_raw!(moved, LIB, XI, anchor_star, driver,
                         k -> get(driver_lookup, issue + Hour(k - 1), nothing),
                         [v23_analog_member(driver, driver_lookup, frame.time_utc[300])])
    @test moved != before
end

@testset "analog retrieval is nested in distance" begin
    frame = synthetic_frame(500)
    features, ok = v23_feature_matrix(frame, frame.time_utc[10:400])
    usable = findall(ok)
    stats = v23_feature_stats(features[usable, :])
    standardized = v23_standardize(features[usable, :], stats.mean, stats.sd)
    times = frame.time_utc[10:400][usable]
    archive = standardized[1:200, :]
    archive_times = times[1:200]
    queries = standardized[201:230, :]
    query_times = times[201:230]
    neighbours = v23_knn(queries, query_times, archive, archive_times, 20; exclusion_hours=0)
    for i in axes(neighbours, 1)
        distances = [sum(abs2, view(queries, i, :) .- view(archive, neighbours[i, j], :))
                     for j in axes(neighbours, 2)]
        @test issorted(distances)
    end
    smaller = v23_knn(queries, query_times, archive, archive_times, 5; exclusion_hours=0)
    @test smaller == neighbours[:, 1:5]
end

@testset "prefix means match explicit means and reject bad prefixes" begin
    raw = Float64[1 2; 3 5; 9 11; 13 17]
    destination = Matrix{Float64}(undef, 2, 2)
    v23_prefix_means!(destination, raw, [2, 4])
    @test destination[1, 1] == 2.0                     # (1 + 3) / 2
    @test destination[1, 2] == 3.5                     # (2 + 5) / 2
    @test destination[2, 1] == 6.5                     # (1 + 3 + 9 + 13) / 4
    @test destination[2, 2] == 8.75                    # (2 + 5 + 11 + 17) / 4
    @test_throws ArgumentError v23_prefix_means!(destination, raw, [4, 2])
    @test_throws ArgumentError v23_prefix_means!(Matrix{Float64}(undef, 1, 2), raw, [9])
end

"Anchor view with fully controlled numbers, for the composition and artifact tests."
function tiny_anchors(count::Int; obs, served, frozen, latest, rate, coupling)
    issues = [DateTime(2015, 1, 1) + Hour(i) for i in 0:count - 1]
    index = Dict(t => i for (i, t) in enumerate(issues))
    driver = [(V=400.0, Bz=-5.0, By=2.0, n=5.0, Pdyn=dynamic_pressure(5.0, 400.0))
              for _ in 1:count]
    present = trues(count, V23_STEP_COUNT)
    zeros_matrix() = zeros(count, V23_STEP_COUNT)
    return V23Anchors(
        issues, index, fill("DEV", count), fill(2015, count), driver,
        latest, rate, coupling, [-30.0 for _ in 1:count], present,
        obs, served, frozen, zeros_matrix(), zeros_matrix(), zeros_matrix(),
        zeros_matrix(), zeros_matrix(), zeros_matrix(), zeros_matrix(),
    )
end

function tiny_context(anchors::V23Anchors; outdir::AbstractString=mktempdir())
    plan = V23RunPlan("unit", outdir, true, [2015], 0, [:uniform], [25], [(4, 200)],
                      [(4, 200)])
    count = length(anchors)
    features = zeros(count, V23_FEATURE_COUNT)
    for i in 1:count
        features[i, v23_feature_index(:dst0)] = anchors.latest[i]
        features[i, v23_feature_index(:ddst1)] = anchors.rate[i]
    end
    return V23Context(plan, anchors, DataFrame(), Dict{DateTime,_V23_DRIVER_NT}(),
                      nothing, nothing, features, trues(count), trues(count),
                      v23_cell_masks(anchors), nothing, "table-sha", "frame-sha",
                      "code-sha", "unit", 0.0)
end

@testset "lead-aware composition reduces to its declared identities" begin
    count = 120
    obs = zeros(count, V23_STEP_COUNT)
    served = fill(1.0, count, V23_STEP_COUNT)
    frozen = fill(2.0, count, V23_STEP_COUNT)
    anchors = tiny_anchors(count; obs=obs, served=served, frozen=frozen,
                           latest=fill(-10.0, count), rate=zeros(count),
                           coupling=zeros(count))
    ctx = tiny_context(anchors)
    centers = fill(-2.0, count, V23_STEP_COUNT)
    rows = collect(1:count)
    # Blending 2 nT of frozen with −2 nT of tail gives 2 − 4w, so the pooled RMSE
    # |2 − 4w| is minimised exactly at w = 0.5 for every step.
    lat = v23_lat_weights(ctx, centers, rows)
    @test all(lat.weights .== 0.5)
    @test all(isapprox(row.rmse_selected_nt, 0.0; atol=1e-12) for row in lat.table)
    @test all(isapprox(row.rmse_w0_00_nt, 2.0; atol=1e-12) for row in lat.table)
    @test all(isapprox(row.rmse_w1_00_nt, 2.0; atol=1e-12) for row in lat.table)
    @test v23_apply_lat(ctx, centers, ones(V23_STEP_COUNT), rows)[rows, :] == centers[rows, :]
    @test v23_apply_lat(ctx, centers, zeros(V23_STEP_COUNT), rows)[rows, :] ==
          anchors.frozen[rows, :]
end

@testset "innovation error state is causal and capped" begin
    count = 900
    slot1 = V23_STEP_SLOT[1]
    obs = zeros(count, V23_STEP_COUNT)
    # A one-step innovation of ±50 nT at each anchor, and a residual at every
    # step that is exactly three times the previous hour's innovation: the ridge
    # would predict ±150 nT, far outside the ±(5 + 5h) cap.
    innovation = [iseven(i) ? 50.0 : -50.0 for i in 1:count]
    for i in 1:count
        obs[i, slot1] = innovation[i]
        for slot in 1:V23_STEP_COUNT
            slot == slot1 && continue
            obs[i, slot] = i > 1 ? 3.0 * innovation[i - 1] : 0.0
        end
    end
    anchors = tiny_anchors(count; obs=obs, served=zeros(count, V23_STEP_COUNT),
                           frozen=zeros(count, V23_STEP_COUNT),
                           latest=fill(-10.0, count), rate=zeros(count),
                           coupling=zeros(count))
    ctx = tiny_context(anchors)
    base = zeros(count, V23_STEP_COUNT)
    rows = collect(1:count)
    features, ok = v23_innovation_features(ctx, base, rows)
    @test !any(ok[1:V23_E_INNOVATION_LAGS])          # no complete history yet
    @test all(ok[V23_E_INNOVATION_LAGS + 1:end])
    for i in V23_E_INNOVATION_LAGS + 1:count, lag in 1:V23_E_INNOVATION_LAGS
        @test features[i, lag] == innovation[i - lag]
    end
    e_folds = [(name="second_half", train=collect(20:500), query=collect(520:count))]
    delta = v23_e_layer_predictions(ctx, e_folds, base, features, ok, :E1, 10.0)
    for (slot, step) in enumerate(V23_MODEL_STEPS)
        cap = v23_e_cap(step)
        @test cap == 5.0 + 5.0 * step
        applied = [delta[i, slot] for i in 520:count if isfinite(delta[i, slot])]
        @test !isempty(applied)
        @test maximum(abs.(applied)) <= cap + 1e-12
        # The uncapped signal is ±150 nT, so the cap must actually bind.
        @test maximum(abs.(applied)) >= cap - 1e-12
    end
    @test all(isnan, delta[1:519, :])
end

@testset "ridge fit recovers a known linear map and shrinks with the penalty" begin
    rows, columns = 400, 4
    design = [sin(i + 3j) + 0.5cos(2i - j) for i in 1:rows, j in 1:columns]
    truth = [1.5, -2.0, 0.75, 3.0]
    target = design * truth .+ 7.0
    weak = v23_ridge_fit(design, target, 1e-8)
    @test maximum(abs.(v23_ridge_predict(weak, design) .- target)) < 1e-5
    strong = v23_ridge_fit(design, target, 1.0e6)
    @test maximum(abs.(v23_ridge_predict(strong, design) .- mean(target))) <
          maximum(abs.(target .- mean(target)))
    @test_throws ArgumentError v23_ridge_fit(design, target, 0.0)
    @test_throws DimensionMismatch v23_ridge_predict(weak, design[:, 1:3])
end

@testset "storm-safety guards fire on the preregistered thresholds" begin
    row(cell, step, n, loss, bias) = (
        config="unit", model_step_hours=step, cell=String(cell), n=n, rmse_nt=10.0 + loss,
        bias_nt=bias, rmse_served_nt=10.0, bias_served_nt=0.0, loss_vs_served_nt=loss,
        fallback_fraction=0.0, family="unit", safeguards=true, params_json="{}", seconds=0.0,
    )
    clean = [row(:latest_le_m50, 3, 100, 0.49, 0.0),
             row(:intense_deepening, 6, 50, -0.10, 4.0)]
    @test v23_guard_flags(clean).guards_ok
    @test v23_guard_flags([row(:latest_le_m50, 3, 100, 0.51, 0.0)]).cell_guard_ok == false
    # Below the minimum cell size the loss guard does not apply.
    @test v23_guard_flags([row(:latest_le_m50, 3, 39, 5.0, 0.0)]).cell_guard_ok
    @test v23_guard_flags([row(:quiet, 3, 1000, 5.0, 0.0)]).cell_guard_ok
    @test v23_guard_flags([row(:intense_deepening, 6, 50, 0.01, 0.0)]).intense_guard_ok == false
    @test v23_guard_flags([row(:intense_deepening, 6, 50, -0.5, 10.01)]).intense_guard_ok == false
    @test v23_guard_flags([row(:intense_deepening, 6, 50, -0.5, -10.0)]).intense_guard_ok
    @test v23_guard_flags([row(:intense_deepening, 7, 50, 5.0, 40.0)]).intense_guard_ok
end

@testset "cell labelling matches the protocol thresholds" begin
    @test :intense_deepening in v23_cell_labels(-100.0, -15.01, 0.0)
    @test !(:intense_deepening in v23_cell_labels(-100.0, -15.0, 0.0))
    @test !(:intense_deepening in v23_cell_labels(-99.99, -40.0, 0.0))
    @test :latest_le_m50 in v23_cell_labels(-50.0, 1.0, 0.0)
    @test !(:latest_le_m50 in v23_cell_labels(-49.99, 1.0, 0.0))
    @test :latest_le_m100 in v23_cell_labels(-100.0, 1.0, 0.0)
    @test :all in v23_cell_labels(0.0, 0.0, 0.0)
    @test v23_regime_cells_source() isa String
end

@testset "selection rule ranks by score then ensemble size then safeguards" begin
    metric(config, family, safeguards, k, rmse) = [(
        config=config, model_step_hours=step, cell="all", n=500, rmse_nt=rmse,
        bias_nt=0.0, rmse_served_nt=rmse + 1.0, bias_served_nt=0.0,
        loss_vs_served_nt=-1.0, fallback_fraction=0.0, family=family,
        safeguards=safeguards, params_json="{\"k\":$(k)}", seconds=0.0,
    ) for step in V23_MODEL_STEPS]
    metrics = vcat(metric("T1_uniform_K200_Son", "T1", true, 200, 5.0),
                   metric("T1_uniform_K25_Son", "T1", true, 25, 5.0),
                   metric("T1_uniform_K25_Soff", "T1", false, 25, 5.0),
                   metric("T2_d4_r200_Son", "T2", true, 25, 4.0))
    candidates = v23_candidates(metrics)
    @test first(candidates).id == "T2_d4_r200_Son"
    @test v23_select(candidates).id == "T2_d4_r200_Son"
    tied = vcat(metric("T1_uniform_K200_Son", "T1", true, 200, 5.0),
                metric("T1_uniform_K25_Soff", "T1", false, 25, 5.0),
                metric("T1_uniform_K25_Son", "T1", true, 25, 5.0))
    @test v23_select(v23_candidates(tied)).id == "T1_uniform_K25_Son"
    # A guard failure removes the best-scoring configuration from the selection.
    failing = [merge(row, (cell="intense_deepening", model_step_hours=6, n=100,
                           loss_vs_served_nt=3.0, bias_nt=0.0))
               for row in metric("T2_d4_r200_Son", "T2", true, 25, 4.0)[1:1]]
    guarded = v23_candidates(vcat(tied, metric("T2_d4_r200_Son", "T2", true, 25, 4.0), failing))
    @test first(guarded).id == "T2_d4_r200_Son"
    @test !first(guarded).guards.guards_ok
    @test v23_select(guarded).id == "T1_uniform_K25_Son"
    # Relaxing the requirement changes nothing while some configuration passes.
    @test v23_select(guarded; require_guards=false).id == "T1_uniform_K25_Son"
    # It only acts when the whole grid fails, and then the score alone ranks.
    all_failing = v23_candidates(vcat(
        metric("T1_uniform_K25_Son", "T1", true, 25, 5.0),
        metric("T2_d4_r200_Son", "T2", true, 25, 4.0),
        failing,
        [merge(row, (cell="intense_deepening", model_step_hours=6, n=100,
                     loss_vs_served_nt=3.0, bias_nt=0.0))
         for row in metric("T1_uniform_K25_Son", "T1", true, 25, 5.0)[1:1]],
    ))
    @test v23_select(all_failing) === nothing
    @test v23_select(all_failing; require_guards=false).id == "T2_d4_r200_Son"
end

@testset "persisted configurations are reused only when the signature matches" begin
    outdir = mktempdir()
    count = 40
    obs = fill(1.0, count, V23_STEP_COUNT)
    anchors = tiny_anchors(count; obs=obs, served=zeros(count, V23_STEP_COUNT),
                           frozen=zeros(count, V23_STEP_COUNT),
                           latest=fill(-10.0, count), rate=zeros(count),
                           coupling=zeros(count))
    ctx = tiny_context(anchors; outdir=outdir)
    params = v23_analog_params("T1", :uniform, 25, false)
    result = V23ConfigResult("T1_uniform_K25", "T1", params,
                             fill(0.5, count, V23_STEP_COUNT), fill(false, count),
                             collect(1:count), false)
    @test !v23_config_is_current(ctx, result.id, params)
    v23_write_config!(ctx, result, 1.0)
    @test v23_config_is_current(ctx, result.id, params)
    @test !v23_config_is_current(ctx, result.id, v23_analog_params("T1", :uniform, 50, false))
    stored = v23_read_config(ctx, result.id)
    @test stored.rows == collect(1:count)
    @test all(stored.centers_on[1:count, :] .== 0.5)
    # A changed input hash must invalidate the artifact rather than reuse it.
    ctx.base_table_sha = "different-sha"
    @test !v23_config_is_current(ctx, result.id, params)
end

@testset "fallback anchors are served, counted and never silently dropped" begin
    count = 20
    obs = zeros(count, V23_STEP_COUNT)
    served = fill(-7.0, count, V23_STEP_COUNT)
    anchors = tiny_anchors(count; obs=obs, served=served,
                           frozen=fill(-3.0, count, V23_STEP_COUNT),
                           latest=fill(-10.0, count), rate=zeros(count),
                           coupling=zeros(count))
    ctx = tiny_context(anchors)
    fallback = [i <= 5 for i in 1:count]
    result = V23ConfigResult("unit", "T1", Dict{String,Any}("k" => 25),
                             fill(-1.0, count, V23_STEP_COUNT), fallback,
                             collect(1:count), false)
    centers = v23_config_centers(result, anchors, true)
    @test all(centers[1:5, :] .== -7.0)
    @test all(centers[6:count, :] .== -1.0)
    metrics = v23_metric_rows("unit", centers, anchors, ctx.masks, result.rows, fallback)
    pooled = [r for r in metrics if r.cell == "all"]
    @test length(pooled) == V23_STEP_COUNT
    @test all(r.n == count for r in pooled)
    @test all(isapprox(r.fallback_fraction, 0.25; atol=1e-12) for r in pooled)
    # A non-finite center is a lost row, and losing rows must fail closed.
    centers[7, 1] = NaN
    @test_throws ErrorException v23_metric_rows("unit", centers, anchors, ctx.masks,
                                                result.rows, fallback)
end

"""
Real hourly driver records, so the self-origin oracle is checked on the archive
the study actually rolls out over rather than on a smooth synthetic frame. Falls
back to the deterministic synthetic frame when the generated artifact is absent.
"""
const ORACLE_FRAME_PATH = normpath(joinpath(@__DIR__, "..", "validation", "output",
                                            "operational", "v2_3_base",
                                            "v2_3_hourly_frame.csv"))

function oracle_frame()
    isfile(ORACLE_FRAME_PATH) || return (frame=synthetic_frame(4000), real=false)
    frame = CSV.read(ORACLE_FRAME_PATH, DataFrame; types=Dict("time_utc" => DateTime))
    return (frame=frame, real=true)
end

@testset "a K=1 self-origin analog reproduces the realized-driver oracle" begin
    # Plan oracle (ii): continuing a query by its *own* archived increments is
    # the realized driver. The continuation reconstructs speed and density from
    # increments and recomputes the pressure, so the identity is exact only up to
    # floating point; 1e-9 nT is far below any quantity the study reports and far
    # above the reconstruction error.
    source = oracle_frame()
    frame = source.frame
    lookup = v2_3_frame_driver_lookup(frame)
    dst = Dict(frame.time_utc[i] => frame.Dst[i] for i in 1:nrow(frame))
    complete(issue) = haskey(lookup, issue - Hour(1)) && isfinite(get(dst, issue, NaN)) &&
                      all(haskey(lookup, issue + Hour(k - 1)) for k in 1:V23_MAX_STEP)
    step = max(1, div(nrow(frame), 40))
    anchors = [t for t in frame.time_utc[2:step:end] if complete(t)]
    @test length(anchors) >= 5
    checked = 0
    worst = 0.0
    worst_driver = 0.0
    for issue in anchors[1:min(12, length(anchors))]
        driver = lookup[issue - Hour(1)]
        anchor_star = pressure_correct_dst([dst[issue]], [driver.Pdyn])[1]
        future = k -> get(lookup, issue + Hour(k - 1), nothing)

        # Driver level: the k-th self-origin continuation is the realized record.
        for k in 1:V23_MAX_STEP
            continued = v23_member_driver(driver, lookup, issue, k)
            realized = lookup[issue + Hour(k - 1)]
            for field in (:V, :Bz, :By, :n, :Pdyn)
                deviation = abs(getproperty(continued, field) - getproperty(realized, field))
                worst_driver = max(worst_driver, deviation)
            end
        end

        # Center level: the six scored steps of a K = 1 self-origin ensemble and
        # of the realized-driver oracle.
        self_origin = Matrix{Float64}(undef, 1, V23_STEP_COUNT)
        oracle = Matrix{Float64}(undef, 1, V23_STEP_COUNT)
        v23_member_step_raw!(self_origin, LIB, XI, anchor_star, driver, future,
                             [v23_analog_member(driver, lookup, issue)])
        v23_member_step_raw!(oracle, LIB, XI, anchor_star, driver, future,
                             [v23_realized_driver_member(future)])
        for slot in 1:V23_STEP_COUNT
            @test isapprox(self_origin[1, slot], oracle[1, slot]; atol=1e-9, rtol=0.0)
            worst = max(worst, abs(self_origin[1, slot] - oracle[1, slot]))
            checked += 1
        end
    end
    @test checked >= 5 * V23_STEP_COUNT
    @test worst <= 1e-9
    @test worst_driver <= 1e-9
    # The identity must hold on the real archive, not only on a smooth frame. The
    # hourly frame is written by the offline base-table build and is not tracked,
    # so a clean checkout runs the identity on the synthetic frame and records
    # that the archive leg did not run.
    if local_artifact_available("V2.3 self-origin analog identity on the archived hourly frame",
                                ORACLE_FRAME_PATH)
        @test source.real
    else
        @test_skip "hourly frame absent at $(ORACLE_FRAME_PATH); run " *
                   "validation/operational/v2_3_base_table.jl"
    end
end

"""
Anchor view over a synthetic frame whose Dst is almost flat through the training
window and then sits on a deep plateau. The plateau is far below every level the
training rows carry, so it separates the two ways of posing the direct-GBM
regression: a fit on `Dst(t+h)` cannot leave the range of levels it was trained
on, while a fit on `Dst(t+h) - Dst(t)` only has to learn that a plateau does not
move and recovers the level from the issue-time Dst.

`quiet_amplitude` bounds the training increments, and therefore bounds what the
increment fit can predict on the plateau: with 2 nT of oscillation no training
target exceeds 4 nT in magnitude.
"""
function deep_plateau_fixture(; quiet_hours::Int=420, plateau_hours::Int=100,
                              plateau_dst::Float64=-250.0, quiet_level::Float64=-30.0,
                              quiet_amplitude::Float64=2.0)
    total = quiet_hours + plateau_hours
    frame = synthetic_frame(total)
    for i in 1:total
        frame.Dst[i] = i <= quiet_hours ?
            quiet_level + quiet_amplitude * sin(2π * (i - 1) / 13) : plateau_dst
    end
    # A training anchor needs its 24 h Dst ladder and its whole target window
    # inside the quiet stretch; a query anchor needs the same ladder and window
    # inside the plateau, so its features carry no memory of the quiet levels.
    train_rows = collect(26:quiet_hours - V23_MAX_STEP)
    query_rows = collect(quiet_hours + 25:total - V23_MAX_STEP)
    rows = vcat(train_rows, query_rows)
    n = length(rows)
    issues = collect(frame.time_utc[rows])
    index = Dict(t => i for (i, t) in enumerate(issues))
    lookup = v2_3_frame_driver_lookup(frame)
    driver = [lookup[frame.time_utc[a] - Hour(1)] for a in rows]
    latest = [frame.Dst[a] for a in rows]
    rate = [frame.Dst[a] - frame.Dst[a - 1] for a in rows]
    anchor_star = [pressure_correct_dst([latest[i]], [driver[i].Pdyn])[1] for i in 1:n]
    observation = [frame.Dst[rows[i] + V23_MODEL_STEPS[slot]]
                   for i in 1:n, slot in 1:V23_STEP_COUNT]
    persistence = [latest[i] for i in 1:n, slot in 1:V23_STEP_COUNT]
    zeros_matrix() = zeros(n, V23_STEP_COUNT)
    anchors = V23Anchors(
        issues, index, fill("DEV", n), fill(2015, n), driver, latest, rate, zeros(n),
        anchor_star, trues(n, V23_STEP_COUNT), observation, copy(observation),
        copy(observation), copy(observation), persistence, zeros_matrix(),
        zeros_matrix(), zeros_matrix(), zeros_matrix(), zeros_matrix(),
    )
    features, ok = v23_feature_matrix(frame, issues)
    plan = V23RunPlan("unit", mktempdir(), true, [2015], 0, [:uniform], [25], [(4, 200)],
                      [(4, 200)])
    ctx = V23Context(plan, anchors, frame, lookup, nothing, nothing, features, ok,
                     v23_analog_origin_ok(lookup, issues), v23_cell_masks(anchors),
                     nothing, "table-sha", "frame-sha", "code-sha", "unit", 0.0)
    fold = V23Fold("plateau", collect(1:length(train_rows)),
                   collect(length(train_rows) + 1:n))
    return (ctx=ctx, anchors=anchors, fold=fold, plateau_dst=plateau_dst)
end

@testset "the direct comparator inverts its increment target exactly" begin
    fixture = deep_plateau_fixture()
    ctx, anchors, fold = fixture.ctx, fixture.anchors, fixture.fold
    features, _ = v23_direct_features!(ctx)
    names = String.(v23_direct_feature_names())
    slot = V23_STEP_SLOT[6]
    increments = v23_direct_target(anchors, fold.train, slot)
    for (j, i) in enumerate(fold.train)
        @test increments[j] == anchors.obs[i, slot] - anchors.latest[i]
        @test v23_direct_center(anchors, i, increments[j]) ≈ anchors.obs[i, slot] atol=1e-9
    end
    @test maximum(abs.(increments)) < 5.0            # the flat training stretch
    model = v23_fit_gbm(features[fold.train, :], increments; max_depth=4, nrounds=200,
                        eta=V23_GBM_ETA, min_weight=V23_GBM_MIN_WEIGHT, seed=V23_SEED,
                        feature_names=names)
    fitted = v23_predict(model, features[fold.query, :])
    # The stage reports the fitted increment plus the issue-time Dst, bit for bit.
    result = v23_run_direct_stage!(ctx, [fold], 4, 200)
    for (j, i) in enumerate(fold.query)
        @test result.raw[i, slot] == fitted[j] + anchors.latest[i]
        @test result.raw[i, slot] == v23_direct_center(anchors, i, fitted[j])
    end
    # The recorded contract must say what a loader has to add back.
    params = v23_direct_params(4, 200)
    @test params["target"] == "increment"
    @test params["target_anchor"] == "latest_dst_nt"
    # The anchor the increment is measured from must be the design matrix's own
    # dst0 column, otherwise a persisted model cannot be inverted at all.
    @test v23_direct_check_anchor(features, anchors, fold.query) === nothing
    drifted = copy(features)
    drifted[fold.query[1], v23_feature_index(:dst0)] += 1.0
    @test_throws ErrorException v23_direct_check_anchor(drifted, anchors, fold.query)
end

@testset "the direct comparator follows a level it never saw in training" begin
    fixture = deep_plateau_fixture()
    ctx, anchors, fold = fixture.ctx, fixture.anchors, fixture.fold
    features, usable = v23_direct_features!(ctx)
    @test all(usable[fold.train])
    @test all(usable[fold.query])
    @test length(fold.train) >= 2 * V23_GBM_MIN_WEIGHT
    @test all(anchors.latest[i] == fixture.plateau_dst for i in fold.query)
    # The plateau is more than 200 nT below every level and every target the
    # training rows carry, which is what makes the two targets separable here.
    @test minimum(anchors.latest[fold.train]) - fixture.plateau_dst > 200.0
    @test minimum(anchors.obs[fold.train, :]) - fixture.plateau_dst > 200.0

    result = v23_run_direct_stage!(ctx, [fold], 4, 200)
    @test !any(result.fallback[i] for i in fold.query)
    for slot in 1:V23_STEP_COUNT
        deviation = maximum(abs(result.raw[i, slot] - anchors.latest[i]) for i in fold.query)
        @test deviation < 10.0
    end

    # The same stage posed on the level — the formulation this replaces — cannot
    # leave its training range, so the bound above is a real discriminator and
    # not a property every fit would satisfy.
    names = String.(v23_direct_feature_names())
    for slot in 1:V23_STEP_COUNT
        level = v23_fit_gbm(features[fold.train, :],
                            [anchors.obs[i, slot] for i in fold.train];
                            max_depth=4, nrounds=200, eta=V23_GBM_ETA,
                            min_weight=V23_GBM_MIN_WEIGHT, seed=V23_SEED,
                            feature_names=names)
        saturated = v23_predict(level, features[fold.query, :])
        @test maximum(abs.(saturated .- anchors.latest[fold.query])) > 100.0
        @test minimum(saturated) > fixture.plateau_dst + 100.0
    end
end

@testset "error-layer audit rows report a guard verdict only where one was evaluated" begin
    # The guards are evaluated on the best parameter alone. A rejected parameter
    # inheriting the winner's verdict made the audit claim a check that never ran.
    count = 400
    frame = synthetic_frame(count + 40)
    issues = frame.time_utc[20:20 + count - 1]
    index = Dict(t => i for (i, t) in enumerate(issues))
    driver = [lookup_driver for lookup_driver in
              [(V=400.0, Bz=-5.0, By=2.0, n=5.0, Pdyn=dynamic_pressure(5.0, 400.0))
               for _ in 1:count]]
    zeros_matrix() = zeros(count, V23_STEP_COUNT)
    observation = [(-30.0 + 12.0 * sin(2π * i / 17)) for i in 1:count,
                                                         _ in 1:V23_STEP_COUNT]
    anchors = V23Anchors(
        collect(issues), index, fill("DEV", count), fill(2015, count), driver,
        fill(-10.0, count), zeros(count), zeros(count), fill(-30.0, count),
        trues(count, V23_STEP_COUNT), observation, zeros_matrix(), zeros_matrix(),
        zeros_matrix(), zeros_matrix(), zeros_matrix(), zeros_matrix(), zeros_matrix(),
        zeros_matrix(), zeros_matrix(),
    )
    plan = V23RunPlan("unit", mktempdir(), true, [2015], 0, [:uniform], [25], [(4, 200)],
                      [(4, 200)])
    ctx = V23Context(plan, anchors, frame, v2_3_frame_driver_lookup(frame), nothing, nothing,
                     zeros(count, V23_FEATURE_COUNT), trues(count), trues(count),
                     v23_cell_masks(anchors), nothing, "t", "f", "c", "unit", 0.0)
    rows = collect(1:count)
    centers = fill(-25.0, count, V23_STEP_COUNT)
    folds = [(name="second", train=collect(1:250), query=collect(251:count))]
    audit = v23_run_e_layer(ctx, folds, centers, rows, :E1;
                            params=collect(V23_E1_LAMBDA_GRID), label="E1",
                            eval_rows=collect(251:count)).audit
    @test !isempty(audit)
    for row in audit
        if row.selected_param
            @test row.guard_failures != "not_evaluated"
        else
            @test row.guard_failures == "not_evaluated"
            @test row.guards_ok == false
            @test row.accepted == false
        end
    end
    # Exactly one parameter per step carries a verdict.
    for step in V23_MODEL_STEPS
        selected = [row for row in audit if row.model_step_hours == step && row.selected_param]
        @test length(selected) <= 1
    end
end

end # module
