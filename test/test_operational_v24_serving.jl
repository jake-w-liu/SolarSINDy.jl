module OperationalV24ServingTests

# Unit tests for the served Operational V2.4e center (`src/operational_v24_serving.jl`).
#
# Three things are checked, in order of what would hurt most if it were wrong:
#
#   * the arithmetic of the center — the cell chain, the weighted combination, the optional
#     point-forecast guard, the depth bins and the conformal interval — against independent expectations
#     computed here, not against the implementation's own output;
#   * the loader's refusals — every way a bundle can be wrong is injected one at a time by
#     `V24ServingFixture` and must fail to load, because a loader whose checks are never exercised is a
#     loader that does not check;
#   * the expert helpers — the direct-GBM design's construction from an hourly frame and the Dst ladder,
#     its agreement with the analog key, the increment inversion, and the climatology relaxation.
#
# The fixture bundle is synthetic on purpose: the real bundle's numbers are the study's evidence, and
# the offline identity oracle (`validation/operational/v2_4_serving_identity.jl`) is what ties this code
# to them.

using Test
using CSV
using DataFrames
using Dates
using JSON3
using SolarSINDy

include(joinpath(@__DIR__, "v2_4_serving_fixture.jl"))
using .V24ServingFixture

"Ten expert centers as a named tuple in the served order."
_experts(values) = NamedTuple{V24_SERVING_EXPERTS}(Tuple(Float64.(values)))

@testset "Operational V2.4e serving" begin
    @testset "published contract" begin
        @test V24_SERVED_IDENTITY == "v2.4+sindy20x11+superlearner10floor+conformal"
        @test V24_SERVED_VARIANT == "v2_4e"
        @test V24_SERVING_STACK_VARIANT == "L1e"
        @test V24_SERVING_MODEL_STEPS == (1, 2, 3, 4, 6, 7)
        @test V24_SERVING_EXPERTS == (:served_v2_1, :frozen_v2_1, :t1r_analog, :persistence,
                                      :burton, :burton_full, :obrien, :direct_gbm, :climatology,
                                      :static_v2_2)
        @test V24_SERVING_EXPERT_COUNT == 10
        # Amendment A3's floor group is served, frozen, analog and the static stack. The check is
        # positional in the weight columns, so the group's indices must name exactly those four.
        @test V24_SERVING_SINDY_FAMILY == (1, 2, 3, 10)
        @test all(j -> V24_SERVING_EXPERTS[j] in
                       (:served_v2_1, :frozen_v2_1, :t1r_analog, :static_v2_2),
                  V24_SERVING_SINDY_FAMILY)
        @test V24_SERVING_STATIC_EXPERT == 10
        @test V24_SERVING_EXPERTS[V24_SERVING_STATIC_EXPERT] === :static_v2_2
        @test V24_SERVING_GUARD_REFERENCE == "static_v2_2"
        @test V24_SERVING_GUARD_REFERENCE_NONE == "none"
        # The published identity must not claim a guard stage the deployed bundle does not carry.
        @test !occursin("depthguard", V24_SERVED_IDENTITY)
        @test V24_SERVING_SINDY_FLOOR == 0.60
        @test V24_SERVING_GUARD_RATE_NT_PER_H == -15.0
        @test V24_SERVING_GUARD_DEPTH_NT == -50.0
        @test V24_SERVING_DEPTH_MODERATE_NT == -30.0
        @test V24_SERVING_DEPTH_DEEP_NT == -70.0
        @test V24_SERVING_COVERAGE == 0.90
        @test V24_SERVING_DST_LAG_MAX_H == 24
        @test v24_serving_direct_file(6) == "direct_gbm_step6.bson"
        # The served identity must not be mistakable for the shadow product's.
        @test V24_SERVED_IDENTITY != V23_SERVING_IDENTITY
        @test V24_SERVING_ANALOG_IDENTITY != V23_SERVING_IDENTITY
    end

    @testset "depth bins are closed from below" begin
        @test v24_serving_depth_bin(0.0) === :shallow
        @test v24_serving_depth_bin(-29.999) === :shallow
        @test v24_serving_depth_bin(-30.0) === :moderate
        @test v24_serving_depth_bin(-69.999) === :moderate
        @test v24_serving_depth_bin(-70.0) === :deep
        @test v24_serving_depth_bin(-400.0) === :deep
        # A row that cannot be placed takes the coarsest bin rather than the deepest one, so an
        # unmeasured state can never claim the storm stratum's interval.
        @test v24_serving_depth_bin(NaN) === :shallow
    end

    @testset "cell chain and grid" begin
        chain = v24_serving_cell_chain(3, :active_deepening, :deep)
        @test chain == [(3, :active_deepening, :deep), (3, :active_deepening, :pooled),
                        (3, :pooled, :pooled)]
        grid = v24_serving_cell_grid(2)
        @test first(grid) == (2, :pooled, :pooled)
        @test length(grid) == 1 + length(V24_SERVING_REGIMES) * (1 + length(V24_SERVING_DEPTH_BINS))
        @test allunique(grid)
        for regime in V24_SERVING_REGIMES, depth in V24_SERVING_DEPTH_BINS
            @test (2, regime, depth) in grid
            # Every resolved cell's chain must end at a key the grid holds, or a row could find no cell.
            @test last(v24_serving_cell_chain(2, regime, depth)) in grid
        end
    end

    @testset "deepening cell and guard" begin
        # A one-hour fall steeper than the threshold is a deepening cell on its own.
        @test v24_serving_deepening(-15.001, 0.0, 0.0)
        @test !v24_serving_deepening(-15.0, 0.0, 0.0)
        # Active coupling with the ring current already at or below -50 nT is the other branch.
        @test v24_serving_deepening(-1.0, 0.5, -50.0)
        @test !v24_serving_deepening(-1.0, 0.5, -49.999)
        @test !v24_serving_deepening(-1.0, 0.0, -200.0)
        # A non-finite rate is neutral: a Dst gap must not invent a deepening cell.
        @test !v24_serving_deepening(NaN, 0.0, -10.0)
        @test v24_serving_deepening(NaN, 1.0, -80.0)
        # The coupling branch is a strict `> 0`, not "nonzero". The gated proxy returns zero when the
        # coupling is not active, so a sign error here would read an inactive or reversed coupling as
        # active deepening on a deep but recovering ring current. Rate strictly below zero and depth
        # already past the threshold isolate the coupling term as the only thing deciding the cell.
        @test !v24_serving_deepening(-1.0, -0.5, -80.0)
        @test !v24_serving_deepening(-1.0, -1e-9, -80.0)
        @test v24_serving_deepening(-1.0, 1e-9, -80.0)
        @test !v24_serving_deepening(-1.0, 0.0, -80.0)
        # And a non-finite coupling is neutral rather than active.
        @test !v24_serving_deepening(-1.0, NaN, -80.0)
        # In a deepening cell the guard takes the deeper of the two centers, and only there.
        @test v24_serving_guard(-90.0, -60.0, -20.0, 0.0, -10.0) == -90.0
        @test v24_serving_guard(-40.0, -60.0, -20.0, 0.0, -10.0) == -60.0
        @test v24_serving_guard(-90.0, -60.0, -1.0, 0.0, -10.0) == -60.0
        # The guard can only deepen: outside a deepening cell the reference is ignored entirely.
        for reference in (-500.0, 0.0, 25.0)
            @test v24_serving_guard(reference, -33.0, 0.0, 0.0, 0.0) == -33.0
        end
    end

    @testset "depth-safe severity over the fallback chain" begin
        # The published severity is the deepest of the served center and both continuity partners.
        @test v24_serving_depth_safe_center(-90.0, -120.0, -60.0) == -120.0
        @test v24_serving_depth_safe_center(-150.0, -120.0, -60.0) == -150.0
        @test v24_serving_depth_safe_center(-90.0) == -90.0
        # A non-finite partner is dropped, never propagated.
        @test v24_serving_depth_safe_center(-90.0, NaN, -95.0) == -95.0
        @test v24_serving_depth_safe_center(-90.0, NaN, NaN) == -90.0
        @test v24_serving_depth_safe_center(NaN, -95.0) == -95.0
        # The two-partner rule must agree with the deployed two-stage rule wherever both apply.
        for served in (-10.0, -90.0, -200.0), previous in (-5.0, -95.0, -210.0)
            @test v24_serving_depth_safe_center(served, previous) ==
                  v22_serving_depth_safe_center(served, previous)
        end
    end

    mktempdir() do root
        bundle = build_v24_fixture_bundle(joinpath(root, "good"))
        artifacts = load_v24_serving_artifacts(bundle.dir)

        @testset "a well-formed bundle loads and reports what it holds" begin
            @test artifacts.identity == V24_SERVED_IDENTITY
            @test artifacts.served_variant == V24_SERVED_VARIANT
            # The deployed product carries no point-forecast guard, and the loader must read that from
            # the bundle rather than assume it.
            @test artifacts.guard_applied == false
            @test artifacts.guard_reference == V24_SERVING_GUARD_REFERENCE_NONE
            @test artifacts.fold_year == bundle.fold_year
            @test artifacts.oof_pool_years == sort(bundle.oof_pool_years)
            @test artifacts.climatology_tau_hours == bundle.tau
            @test length(artifacts.analog.origins) == length(bundle.origins)
            @test artifacts.analog.k == V24_SERVING_ANALOG_K
            @test artifacts.analog.weight_set === V24_SERVING_ANALOG_WEIGHT_SET
            @test artifacts.analog.identity == V24_SERVING_ANALOG_IDENTITY
            @test artifacts.direct_feature_names == String.(v23_direct_feature_names())
            @test sort(collect(keys(artifacts.direct_models))) ==
                  sort(collect(V24_SERVING_MODEL_STEPS))
            @test artifacts.manifest_sha256 ==
                  v23_serving_file_sha256(joinpath(bundle.dir, "manifest.csv"))
            # The standardisation is rebuilt from the shipped frame, not trusted: it must reproduce the
            # table the fixture computed.
            @test maximum(abs.(artifacts.analog.feature_mean .- bundle.stats.mean)) <=
                  V24_SERVING_STATS_ATOL
            @test maximum(abs.(artifacts.analog.feature_sd .- bundle.stats.sd)) <=
                  V24_SERVING_STATS_ATOL
            for step in V24_SERVING_MODEL_STEPS
                @test haskey(artifacts.cells, (step, V24_SERVING_POOLED, V24_SERVING_POOLED))
                for depth in (V24_SERVING_POOLED, V24_SERVING_DEPTH_BINS...)
                    @test haskey(artifacts.conformal, (step, depth))
                end
            end
        end

        @testset "cell resolution walks the chain" begin
            # A resolved cell is used where it exists.
            own = v24_serving_cell(artifacts, 2, :active_deepening, :deep)
            @test own.cell.regime === :active_deepening && own.cell.depth === :deep
            @test !own.used_pooled_fallback
            @test own.cell.weights == bundle.deep_weights
            # A regime with only a depth-pooled cell falls back one step.
            quiet = v24_serving_cell(artifacts, 2, :quiet, :deep)
            @test quiet.cell.regime === :quiet && quiet.cell.depth === V24_SERVING_POOLED
            @test quiet.used_pooled_fallback
            # A regime with no cell at all falls back to the fully pooled one.
            pooled = v24_serving_cell(artifacts, 2, :recovery, :moderate)
            @test pooled.cell.regime === V24_SERVING_POOLED
            @test pooled.used_pooled_fallback
        end

        @testset "the served center is the weighted combination over the ten experts" begin
            experts = _experts([-60.0, -55.0, -70.0, -40.0, -50.0, -52.0, -48.0, -65.0, -30.0,
                                -150.0])
            quiet = v24_serving_center(artifacts; model_steps = 3, latest_dst = -10.0,
                                      dst_delta_1h_nt = 0.5, vbsouth_mvm = 0.0, experts = experts)
            expected = sum(bundle.weights[j] * values(experts)[j]
                           for j in 1:V24_SERVING_EXPERT_COUNT)
            @test quiet.l1_center ≈ expected atol = 1e-12
            @test quiet.center ≈ expected atol = 1e-12
            @test !quiet.guard_applied
            @test !quiet.guard_enabled
            @test !quiet.projection_applied
            @test !quiet.deepening_cell
            @test quiet.regime === :quiet
            @test quiet.depth_bin === :shallow
            @test quiet.cell_regime === :quiet
            @test quiet.sindy_family_mass ≈ V24_SERVING_SINDY_FLOOR atol = 1e-12
            # The static stack is a weighted input, not a post-hoc minimum: changing it alone must move
            # the center by exactly its own weight times the change.
            shifted = v24_serving_center(
                artifacts; model_steps = 3, latest_dst = -10.0, dst_delta_1h_nt = 0.5,
                vbsouth_mvm = 0.0,
                experts = merge(experts, (static_v2_2 = values(experts)[V24_SERVING_STATIC_EXPERT] -
                                                        10.0,)),
            )
            @test shifted.l1_center ≈ quiet.l1_center -
                  10.0 * bundle.weights[V24_SERVING_STATIC_EXPERT] atol = 1e-12
            # A shallow row must not be handed the deep stratum's width.
            @test quiet.half_width_nt == bundle.half_width(3, :shallow)
            @test quiet.ci05_nt ≈ quiet.center - quiet.half_width_nt atol = 1e-12
            @test quiet.ci95_nt ≈ quiet.center + quiet.half_width_nt atol = 1e-12

            # A deepening state resolves the deep cell, and with the deployed bundle's guard switch off
            # the published center is that cell's combination — the static stack cannot cap it.
            deep = v24_serving_center(artifacts; model_steps = 3, latest_dst = -120.0,
                                     dst_delta_1h_nt = -25.0, vbsouth_mvm = 3.0, experts = experts)
            @test deep.deepening_cell
            @test !deep.guard_applied
            @test deep.regime === :active_deepening
            @test deep.depth_bin === :deep
            @test deep.cell_depth === :deep
            @test deep.l1_center ≈ sum(bundle.deep_weights[j] * values(experts)[j]
                                       for j in 1:V24_SERVING_EXPERT_COUNT) atol = 1e-12
            @test deep.center ≈ deep.l1_center atol = 1e-12
            @test deep.half_width_nt == bundle.half_width(3, :deep)
            @test deep.regime_cell == "active_deepening/deep"

            # The gated coupling proxy is the stack's, not the raw proxy: a rising ring current with
            # positive coupling is not active deepening.
            rising = v24_serving_center(artifacts; model_steps = 3, latest_dst = -60.0,
                                       dst_delta_1h_nt = 4.0, vbsouth_mvm = 6.0, experts = experts)
            @test rising.coupling_active_mvm == 0.0
            @test rising.regime === :recovery
            @test !rising.deepening_cell
            @test !rising.guard_applied
        end

        @testset "the physical projection is reported when it acts" begin
            # The reported center is taken through `clamp(., -2000, 50)` nT. Every anchor the study
            # scored sits far from that ceiling, so the oracle can assert the projection inert there —
            # but a real ring current above roughly +52 nT with a panel to match does reach it, and a
            # projected center is indistinguishable from an unprojected one unless the stage says so.
            # The flag is what makes the `center == l1_center` identity of an unguarded bundle a
            # conditional statement instead of an invariant that quietly stops holding.
            high = _experts(fill(60.0, V24_SERVING_EXPERT_COUNT))
            projected = v24_serving_center(artifacts; model_steps = 3, latest_dst = 55.0,
                                          dst_delta_1h_nt = 0.5, vbsouth_mvm = 0.0,
                                          experts = high)
            @test projected.l1_center ≈ 60.0 atol = 1e-9
            @test projected.raw_center ≈ 60.0 atol = 1e-9
            @test projected.center == V24_SERVING_DST_CEIL_NT
            @test projected.projection_applied
            @test !projected.guard_applied
            # The interval is formed on the projected center, which is the number the row publishes.
            @test projected.ci05_nt ≈ projected.center - projected.half_width_nt atol = 1e-12
            @test projected.ci95_nt ≈ projected.center + projected.half_width_nt atol = 1e-12
            # The floor side of the same projection.
            floored = v24_serving_center(artifacts; model_steps = 3, latest_dst = -2400.0,
                                        dst_delta_1h_nt = 0.5, vbsouth_mvm = 0.0,
                                        experts = _experts(fill(-2500.0,
                                                                V24_SERVING_EXPERT_COUNT)))
            @test floored.center == V24_SERVING_DST_FLOOR_NT
            @test floored.projection_applied
            # A center strictly inside the physical range is not flagged, near either edge. The edges
            # themselves are not used: a convex combination whose weights sum to unity only within the
            # shipped tolerance lands a hair outside an exactly-boundary panel, which the projection
            # then legitimately moves.
            for value in (V24_SERVING_DST_CEIL_NT - 1.0, V24_SERVING_DST_FLOOR_NT + 1.0)
                inert = v24_serving_center(artifacts; model_steps = 3, latest_dst = value,
                                          dst_delta_1h_nt = 0.0, vbsouth_mvm = 0.0,
                                          experts = _experts(fill(value,
                                                                  V24_SERVING_EXPERT_COUNT)))
                @test !inert.projection_applied
                @test inert.center ≈ value atol = 1e-9
            end
        end

        @testset "a bundle that enables the guard is served with it" begin
            # The deployed product does not guard the point forecast, but the code path stays
            # configurable, so it is exercised through a bundle whose own record enables it.
            mktempdir() do guarded_root
                guarded_bundle = build_v24_fixture_bundle(joinpath(guarded_root, "guarded");
                                                          guard = true)
                guarded = load_v24_serving_artifacts(guarded_bundle.dir)
                @test guarded.guard_applied
                @test guarded.guard_reference == V24_SERVING_GUARD_REFERENCE
                experts = _experts([-60.0, -55.0, -70.0, -40.0, -50.0, -52.0, -48.0, -65.0, -30.0,
                                    -150.0])
                deep = v24_serving_center(guarded; model_steps = 3, latest_dst = -120.0,
                                         dst_delta_1h_nt = -25.0, vbsouth_mvm = 3.0,
                                         experts = experts)
                @test deep.guard_enabled
                @test deep.deepening_cell
                @test deep.guard_applied
                @test deep.center == -150.0
                @test deep.center < deep.l1_center
                # Outside a deepening cell the reference is ignored even when the guard is enabled.
                quiet = v24_serving_center(guarded; model_steps = 3, latest_dst = -10.0,
                                          dst_delta_1h_nt = 0.5, vbsouth_mvm = 0.0,
                                          experts = experts)
                @test quiet.guard_enabled
                @test !quiet.guard_applied
                @test quiet.center ≈ quiet.l1_center atol = 1e-12
                # A shallower static stack in a deepening cell cannot lift the center.
                lifted = v24_serving_center(
                    guarded; model_steps = 3, latest_dst = -120.0, dst_delta_1h_nt = -25.0,
                    vbsouth_mvm = 3.0, experts = merge(experts, (static_v2_2 = -20.0,)),
                )
                @test lifted.deepening_cell
                @test !lifted.guard_applied
                @test lifted.center ≈ lifted.l1_center atol = 1e-12
            end
        end

        @testset "the served center refuses inputs it cannot serve" begin
            experts = _experts(fill(-50.0, V24_SERVING_EXPERT_COUNT))
            @test_throws ArgumentError v24_serving_center(
                artifacts; model_steps = 5, latest_dst = -10.0, dst_delta_1h_nt = 0.0,
                vbsouth_mvm = 0.0, experts = experts)
            @test_throws ArgumentError v24_serving_center(
                artifacts; model_steps = 3, latest_dst = NaN, dst_delta_1h_nt = 0.0,
                vbsouth_mvm = 0.0, experts = experts)
            # The static stack is expert ten, so a row without it cannot be combined at all; the center
            # is refused rather than published over a nine-expert panel.
            @test_throws ArgumentError v24_serving_center(
                artifacts; model_steps = 3, latest_dst = -10.0, dst_delta_1h_nt = 0.0,
                vbsouth_mvm = 0.0, experts = merge(experts, (static_v2_2 = NaN,)))
            bad = _experts([-50.0, -50.0, NaN, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0])
            @test_throws ArgumentError v24_serving_center(
                artifacts; model_steps = 3, latest_dst = -10.0, dst_delta_1h_nt = 0.0,
                vbsouth_mvm = 0.0, experts = bad)
            @test_throws ArgumentError v24_serving_center(
                artifacts; model_steps = 3, latest_dst = -10.0, dst_delta_1h_nt = 0.0,
                vbsouth_mvm = 0.0, experts = (served_v2_1 = -50.0, frozen_v2_1 = -50.0))
        end

        @testset "climatology relaxes persistence toward zero" begin
            for step in V24_SERVING_MODEL_STEPS
                @test v24_serving_climatology_center(artifacts, -100.0, step) ≈
                      -100.0 * exp(-step / bundle.tau) atol = 1e-12
            end
            # Deeper leads relax further, and the sign never flips.
            values_by_step = [v24_serving_climatology_center(artifacts, -100.0, step)
                              for step in V24_SERVING_MODEL_STEPS]
            @test issorted(values_by_step)
            @test all(v -> v < 0.0, values_by_step)
            @test v24_serving_climatology_center(artifacts, 0.0, 6) == 0.0
        end

        @testset "the direct design comes from the frame and the Dst ladder" begin
            frame = bundle.frame
            index = Dict(frame.time_utc[i] => i for i in 1:nrow(frame))
            anchor = frame.time_utc[400]
            history = Any[(V = frame.V[index[anchor - Hour(lag)]],
                           Bz = frame.Bz[index[anchor - Hour(lag)]],
                           By = frame.By[index[anchor - Hour(lag)]],
                           n = frame.n[index[anchor - Hour(lag)]],
                           Pdyn = frame.Pdyn[index[anchor - Hour(lag)]])
                          for lag in 1:V24_SERVING_DRIVER_LAGS_H]
            dst_history = Dict(anchor - Hour(lag) => frame.Dst[index[anchor - Hour(lag)]]
                               for lag in 0:V24_SERVING_DST_LAG_MAX_H)

            mini = v24_serving_direct_frame(anchor, history, dst_history)
            @test nrow(mini) == V24_SERVING_DST_LAG_MAX_H + 1
            @test last(mini.time_utc) == anchor
            @test first(mini.time_utc) == anchor - Hour(V24_SERVING_DST_LAG_MAX_H)
            @test issorted(mini.time_utc)
            # The driver channels are carried only as deep as the history reaches; the Dst ladder is
            # carried the whole span, because `dst_lag24` reads its far end.
            @test isfinite(mini.V[end - 1])
            @test isnan(mini.V[1])
            @test all(isfinite, mini.Dst)

            design = v24_serving_direct_features(artifacts, anchor, history, dst_history)
            @test design.ok
            @test length(design.features) == V23_DIRECT_FEATURE_COUNT
            # The design must equal the package's own construction on the whole frame: that is what
            # makes the live block the block the study fitted on.
            reference, reference_ok = v23_direct_features(frame, [anchor])
            @test reference_ok[1]
            @test design.features == vec(Matrix{Float64}(reference)[1, :])
            # And its first 18 columns must be the analog key exactly; a disagreement is refused.
            key = v24_serving_analog_features(artifacts, anchor, history,
                                             frame.Dst[index[anchor]],
                                             frame.Dst[index[anchor - Hour(1)]])
            @test key.ok
            @test design.features[1:V23_FEATURE_COUNT] == key.features
            @test v24_serving_direct_features(artifacts, anchor, history, dst_history;
                                              analog_features = key.features).ok
            drifted = copy(key.features)
            drifted[1] += 0.5
            @test_throws ErrorException v24_serving_direct_features(
                artifacts, anchor, history, dst_history; analog_features = drifted)

            # The increment is inverted with the design's own anchor Dst.
            for step in V24_SERVING_MODEL_STEPS
                increment = v23_predict(artifacts.direct_models[step],
                                        reshape(design.features, 1, length(design.features)))[1]
                @test v24_serving_direct_center(artifacts, design.features, step) ≈
                      increment + design.features[v23_feature_index(:dst0)] atol = 1e-12
            end
            @test v24_serving_direct_center(artifacts, design.features, 1) !=
                  v24_serving_direct_center(artifacts, design.features, 7)

            # A gap anywhere in the ladder is reported by name rather than imputed.
            for lag in (1, 6, 12, 24)
                holed = copy(dst_history)
                delete!(holed, anchor - Hour(lag))
                result = v24_serving_direct_features(artifacts, anchor, history, holed)
                @test !result.ok
                @test result.reason == "missing_dst_lag$(lag)"
                @test all(!isfinite, result.features)
            end
            no_anchor = copy(dst_history)
            delete!(no_anchor, anchor)
            @test v24_serving_direct_features(artifacts, anchor, history, no_anchor).reason ==
                  "missing_anchor_dst"
            gapped_driver = copy(history)
            gapped_driver[3] = nothing
            @test !v24_serving_direct_features(artifacts, anchor, gapped_driver, dst_history).ok
            @test_throws ArgumentError v24_serving_direct_frame(anchor, history[1:3], dst_history)
        end

        @testset "the interval takes the row's stratum with a pooled fallback" begin
            for step in V24_SERVING_MODEL_STEPS, depth in V24_SERVING_DEPTH_BINS
                interval = v24_serving_interval(artifacts, step, depth, -100.0)
                @test interval.half_width_nt == bundle.half_width(step, depth)
                @test interval.lo_nt == -100.0 - interval.half_width_nt
                @test interval.hi_nt == -100.0 + interval.half_width_nt
            end
            # A depth label the calibration does not carry resolves to the step's pooled stratum
            # instead of erroring, which is the only path a future bin could take. The pooled width is
            # distinct from every depth bin's in this bundle, so the assertion discriminates the pooled
            # fallback from silently returning the shallow stratum — which is what it did while the two
            # widths were equal.
            for step in V24_SERVING_MODEL_STEPS
                pooled = bundle.half_width(step, V24_SERVING_POOLED)
                for depth in V24_SERVING_DEPTH_BINS
                    @test pooled != bundle.half_width(step, depth)
                end
                @test v24_serving_interval(artifacts, step, :unknown_bin, 0.0).half_width_nt ==
                      pooled
                @test v24_serving_interval(artifacts, step, :unknown_bin, 0.0).half_width_nt !=
                      bundle.half_width(step, :shallow)
                @test v24_serving_interval(artifacts, step, :unknown_bin, -100.0).stratum.depth ===
                      V24_SERVING_POOLED
            end
            @test_throws ErrorException v24_serving_interval(artifacts, 5, :shallow, 0.0)
        end

        @testset "manifest verification is a digest check, not a file listing" begin
            @test v24_serving_verify_manifest(bundle.dir) isa DataFrame
            hashed = v24_serving_manifest_hashed_names(
                CSV.read(joinpath(bundle.dir, "manifest.csv"), DataFrame))
            for name in V24_SERVING_REQUIRED_FILES
                @test name in hashed
            end
            for step in V24_SERVING_MODEL_STEPS
                @test v24_serving_direct_file(step) in hashed
            end
        end
    end

    @testset "every bundle defect is refused by the check it was written for" begin
        # One mutation at a time, and each one asserted against the message of the check it exists to
        # exercise. A bare "something threw" assertion passes when a defect is caught by an earlier,
        # unrelated check, which leaves the intended check untested while the suite looks green — and
        # two of these defects were in exactly that state before the expected-message table existed.
        for mutation in (:wrong_expert_set, :wrong_expert_count, :wrong_floor_group,
                         :wrong_served_variant, :wrong_stack_variant,
                         :guard_enabled_without_reference, :guard_disabled_with_reference,
                         :guard_switch_absent, :nine_expert_stack,
                         :conformal_variant_mismatch)
            @test mutation in V24_FIXTURE_MUTATIONS
        end
        @test length(V24_FIXTURE_MUTATIONS) == 28
        @test allunique(V24_FIXTURE_MUTATIONS)
        # Every defect names the check it must trip; a new mutation without an expected message would
        # otherwise be asserted only as "throws something".
        @test sort(collect(keys(V24_FIXTURE_EXPECTED_ERRORS))) ==
              sort(collect(V24_FIXTURE_MUTATIONS))
        for mutation in V24_FIXTURE_MUTATIONS
            mktempdir() do root
                bad = build_v24_fixture_bundle(joinpath(root, String(mutation));
                                               mutate = mutation)
                thrown = try
                    load_v24_serving_artifacts(bad.dir)
                    nothing
                catch err
                    err isa InterruptException && rethrow()
                    err
                end
                @test thrown !== nothing
                thrown === nothing && return
                message = sprint(showerror, thrown)
                @test occursin(V24_FIXTURE_EXPECTED_ERRORS[mutation], message)
            end
        end
    end

    @testset "the direct-GBM digest gate refuses a model the manifest does not cover" begin
        # The reader refuses a direct-GBM file that carries no verified manifest digest. Through a full
        # load that defect is caught earlier, by the manifest's required-artifact rule — which is what
        # `:unlisted_gbm_model` proves — so the gate itself is reached here, through the reader, rather
        # than left as a branch no test can enter.
        mktempdir() do root
            bundle = build_v24_fixture_bundle(joinpath(root, "direct"))
            labels = Dict{Int,String}()
            complete = Set{String}(v24_serving_direct_file(step)
                                   for step in V24_SERVING_MODEL_STEPS)
            full = SolarSINDy._v24_serving_read_direct(bundle.dir, complete, labels)
            @test sort(collect(keys(full.models))) == sort(collect(V24_SERVING_MODEL_STEPS))
            missing_name = v24_serving_direct_file(first(V24_SERVING_MODEL_STEPS))
            short = Set{String}(name for name in complete if name != missing_name)
            thrown = try
                SolarSINDy._v24_serving_read_direct(bundle.dir, short, labels)
                nothing
            catch err
                err isa InterruptException && rethrow()
                err
            end
            @test thrown !== nothing
            @test occursin("carries no verified manifest digest", sprint(showerror, thrown))
            @test occursin(missing_name, sprint(showerror, thrown))
            # With no manifest to check against the gate is skipped by design, which is the only way an
            # unverified bundle can be read at all.
            @test SolarSINDy._v24_serving_read_direct(bundle.dir, nothing, labels).models isa Dict
        end
    end

    @testset "a missing or unreadable bundle is refused rather than substituted" begin
        mktempdir() do root
            @test_throws ErrorException load_v24_serving_artifacts(joinpath(root, "absent"))
            empty_dir = joinpath(root, "empty")
            mkpath(empty_dir)
            @test_throws ErrorException load_v24_serving_artifacts(empty_dir)
        end
    end
end

end # module
