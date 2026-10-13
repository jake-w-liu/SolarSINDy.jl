module SuperlearnerServingTests

# Unit tests for the served Operational super-learner center (`src/superlearner_serving.jl`).
#
# Three things are checked, in order of what would hurt most if it were wrong:
#
#   * the arithmetic of the center — the cell chain, the weighted combination, the optional
#     point-forecast guard, the depth bins and the conformal interval — against independent expectations
#     computed here, not against the implementation's own output;
#   * the loader's refusals — every way a bundle can be wrong is injected one at a time by
#     `SuperlearnerServingFixture` and must fail to load, because a loader whose checks are never exercised is a
#     loader that does not check;
#   * the expert helpers — the direct-GBM design's construction from an hourly frame and the Dst ladder,
#     its agreement with the analog key, the increment inversion, and the climatology relaxation.
#
# The fixture bundle is synthetic on purpose: the real bundle's numbers are the study's evidence, and
# the offline identity oracle (`validation/operational/superlearner_serving_identity.jl`) is what ties this code
# to them.

using Test
using CSV
using DataFrames
using Dates
using JSON3
using SolarSINDy

include(joinpath(@__DIR__, "superlearner_serving_fixture.jl"))
using .SuperlearnerServingFixture

"Ten expert centers as a named tuple in the served order."
_experts(values) = NamedTuple{SUPERLEARNER_SERVING_EXPERTS}(Tuple(Float64.(values)))

@testset "Operational super-learner serving" begin
    @testset "published contract" begin
        @test SUPERLEARNER_SERVED_IDENTITY == "dst-superlearner+sindyfloor+conformal"
        @test SUPERLEARNER_SERVED_VARIANT == "superlearner_e"
        @test SUPERLEARNER_SERVING_STACK_VARIANT == "L1e"
        @test SUPERLEARNER_SERVING_MODEL_STEPS == (1, 2, 3, 4, 6, 7)
        @test SUPERLEARNER_SERVING_EXPERTS == (:served_inertial_sindy, :frozen_inertial_sindy, :t1r_analog, :persistence,
                                      :burton, :burton_full, :obrien, :direct_gbm, :climatology,
                                      :static_stack)
        @test SUPERLEARNER_SERVING_EXPERT_COUNT == 10
        # Amendment A3's floor group is served, frozen, analog and the static stack. The check is
        # positional in the weight columns, so the group's indices must name exactly those four.
        @test SUPERLEARNER_SERVING_SINDY_FAMILY == (1, 2, 3, 10)
        @test all(j -> SUPERLEARNER_SERVING_EXPERTS[j] in
                       (:served_inertial_sindy, :frozen_inertial_sindy, :t1r_analog, :static_stack),
                  SUPERLEARNER_SERVING_SINDY_FAMILY)
        @test SUPERLEARNER_SERVING_STATIC_EXPERT == 10
        @test SUPERLEARNER_SERVING_EXPERTS[SUPERLEARNER_SERVING_STATIC_EXPERT] === :static_stack
        @test SUPERLEARNER_SERVING_GUARD_REFERENCE == "static_stack"
        @test SUPERLEARNER_SERVING_GUARD_REFERENCE_NONE == "none"
        # The published identity must not claim a guard stage the deployed bundle does not carry.
        @test !occursin("depthguard", SUPERLEARNER_SERVED_IDENTITY)
        @test SUPERLEARNER_SERVING_SINDY_FLOOR == 0.60
        @test SUPERLEARNER_SERVING_GUARD_RATE_NT_PER_H == -15.0
        @test SUPERLEARNER_SERVING_GUARD_DEPTH_NT == -50.0
        @test SUPERLEARNER_SERVING_DEPTH_MODERATE_NT == -30.0
        @test SUPERLEARNER_SERVING_DEPTH_DEEP_NT == -70.0
        @test SUPERLEARNER_SERVING_COVERAGE == 0.90
        @test SUPERLEARNER_SERVING_DST_LAG_MAX_H == 24
        @test superlearner_serving_direct_file(6) == "direct_gbm_step6.bson"
        # The served identity must not be mistakable for the shadow product's.
        @test SUPERLEARNER_SERVED_IDENTITY != ANALOG_SERVING_IDENTITY
        @test SUPERLEARNER_SERVING_ANALOG_IDENTITY != ANALOG_SERVING_IDENTITY
    end

    @testset "conformal.csv without the optional source column loads with source \"\" (G07-01)" begin
        mktempdir() do dir
            path = joinpath(dir, "conformal.csv")
            rows = [(fold_year=2024, variant="superlearner_e", model_step_hours=step,
                     depth_bin=String(depth), half_width_nt=10.0 + step, n=50,
                     coverage_floor=0.9)
                    for step in SUPERLEARNER_SERVING_MODEL_STEPS
                    for depth in (SUPERLEARNER_SERVING_POOLED, SUPERLEARNER_SERVING_DEPTH_BINS...)]
            CSV.write(path, DataFrame(rows))
            @test !("source" in names(CSV.read(path, DataFrame)))
            strata = SolarSINDy._superlearner_serving_read_conformal(path, "superlearner_e", 2024)
            @test length(strata) == length(SUPERLEARNER_SERVING_MODEL_STEPS) *
                                    (1 + length(SUPERLEARNER_SERVING_DEPTH_BINS))
            @test all(stratum -> stratum.source == "", values(strata))
            @test strata[(1, SUPERLEARNER_SERVING_POOLED)].half_width_nt == 11.0
        end
    end

    @testset "depth bins are closed from below" begin
        @test superlearner_serving_depth_bin(0.0) === :shallow
        @test superlearner_serving_depth_bin(-29.999) === :shallow
        @test superlearner_serving_depth_bin(-30.0) === :moderate
        @test superlearner_serving_depth_bin(-69.999) === :moderate
        @test superlearner_serving_depth_bin(-70.0) === :deep
        @test superlearner_serving_depth_bin(-400.0) === :deep
        # A row that cannot be placed takes the coarsest bin rather than the deepest one, so an
        # unmeasured state can never claim the storm stratum's interval.
        @test superlearner_serving_depth_bin(NaN) === :shallow
    end

    @testset "cell chain and grid" begin
        chain = superlearner_serving_cell_chain(3, :active_deepening, :deep)
        @test chain == [(3, :active_deepening, :deep), (3, :active_deepening, :pooled),
                        (3, :pooled, :pooled)]
        grid = superlearner_serving_cell_grid(2)
        @test first(grid) == (2, :pooled, :pooled)
        @test length(grid) == 1 + length(SUPERLEARNER_SERVING_REGIMES) * (1 + length(SUPERLEARNER_SERVING_DEPTH_BINS))
        @test allunique(grid)
        for regime in SUPERLEARNER_SERVING_REGIMES, depth in SUPERLEARNER_SERVING_DEPTH_BINS
            @test (2, regime, depth) in grid
            # Every resolved cell's chain must end at a key the grid holds, or a row could find no cell.
            @test last(superlearner_serving_cell_chain(2, regime, depth)) in grid
        end
    end

    @testset "deepening cell and guard" begin
        # A one-hour fall steeper than the threshold is a deepening cell on its own.
        @test superlearner_serving_deepening(-15.001, 0.0, 0.0)
        @test !superlearner_serving_deepening(-15.0, 0.0, 0.0)
        # Active coupling with the ring current already at or below -50 nT is the other branch.
        @test superlearner_serving_deepening(-1.0, 0.5, -50.0)
        @test !superlearner_serving_deepening(-1.0, 0.5, -49.999)
        @test !superlearner_serving_deepening(-1.0, 0.0, -200.0)
        # A non-finite rate is neutral: a Dst gap must not invent a deepening cell.
        @test !superlearner_serving_deepening(NaN, 0.0, -10.0)
        @test superlearner_serving_deepening(NaN, 1.0, -80.0)
        # The coupling branch is a strict `> 0`, not "nonzero". The gated proxy returns zero when the
        # coupling is not active, so a sign error here would read an inactive or reversed coupling as
        # active deepening on a deep but recovering ring current. Rate strictly below zero and depth
        # already past the threshold isolate the coupling term as the only thing deciding the cell.
        @test !superlearner_serving_deepening(-1.0, -0.5, -80.0)
        @test !superlearner_serving_deepening(-1.0, -1e-9, -80.0)
        @test superlearner_serving_deepening(-1.0, 1e-9, -80.0)
        @test !superlearner_serving_deepening(-1.0, 0.0, -80.0)
        # And a non-finite coupling is neutral rather than active.
        @test !superlearner_serving_deepening(-1.0, NaN, -80.0)
        # In a deepening cell the guard takes the deeper of the two centers, and only there.
        @test superlearner_serving_guard(-90.0, -60.0, -20.0, 0.0, -10.0) == -90.0
        @test superlearner_serving_guard(-40.0, -60.0, -20.0, 0.0, -10.0) == -60.0
        @test superlearner_serving_guard(-90.0, -60.0, -1.0, 0.0, -10.0) == -60.0
        # The guard can only deepen: outside a deepening cell the reference is ignored entirely.
        for reference in (-500.0, 0.0, 25.0)
            @test superlearner_serving_guard(reference, -33.0, 0.0, 0.0, 0.0) == -33.0
        end
    end

    @testset "depth-safe severity over the fallback chain" begin
        # The published severity is the deepest of the served center and both continuity partners.
        @test superlearner_serving_depth_safe_center(-90.0, -120.0, -60.0) == -120.0
        @test superlearner_serving_depth_safe_center(-150.0, -120.0, -60.0) == -150.0
        @test superlearner_serving_depth_safe_center(-90.0) == -90.0
        # A non-finite partner is dropped, never propagated.
        @test superlearner_serving_depth_safe_center(-90.0, NaN, -95.0) == -95.0
        @test superlearner_serving_depth_safe_center(-90.0, NaN, NaN) == -90.0
        @test superlearner_serving_depth_safe_center(NaN, -95.0) == -95.0
        # The two-partner rule must agree with the deployed two-stage rule wherever both apply.
        for served in (-10.0, -90.0, -200.0), previous in (-5.0, -95.0, -210.0)
            @test superlearner_serving_depth_safe_center(served, previous) ==
                  static_stack_serving_depth_safe_center(served, previous)
        end
    end

    mktempdir() do root
        bundle = build_superlearner_fixture_bundle(joinpath(root, "good"))
        artifacts = load_superlearner_serving_artifacts(bundle.dir)

        @testset "a well-formed bundle loads and reports what it holds" begin
            @test artifacts.identity == SUPERLEARNER_SERVED_IDENTITY
            @test artifacts.served_variant == SUPERLEARNER_SERVED_VARIANT
            # The deployed product carries no point-forecast guard, and the loader must read that from
            # the bundle rather than assume it.
            @test artifacts.guard_applied == false
            @test artifacts.guard_reference == SUPERLEARNER_SERVING_GUARD_REFERENCE_NONE
            @test artifacts.fold_year == bundle.fold_year
            @test artifacts.oof_pool_years == sort(bundle.oof_pool_years)
            @test artifacts.climatology_tau_hours == bundle.tau
            @test length(artifacts.analog.origins) == length(bundle.origins)
            @test artifacts.analog.k == SUPERLEARNER_SERVING_ANALOG_K
            @test artifacts.analog.weight_set === SUPERLEARNER_SERVING_ANALOG_WEIGHT_SET
            @test artifacts.analog.identity == SUPERLEARNER_SERVING_ANALOG_IDENTITY
            @test artifacts.direct_feature_names == String.(direct_gbm_feature_names())
            @test sort(collect(keys(artifacts.direct_models))) ==
                  sort(collect(SUPERLEARNER_SERVING_MODEL_STEPS))
            @test artifacts.manifest_sha256 ==
                  analog_serving_file_sha256(joinpath(bundle.dir, "manifest.csv"))
            # The standardisation is rebuilt from the shipped frame, not trusted: it must reproduce the
            # table the fixture computed.
            @test maximum(abs.(artifacts.analog.feature_mean .- bundle.stats.mean)) <=
                  SUPERLEARNER_SERVING_STATS_ATOL
            @test maximum(abs.(artifacts.analog.feature_sd .- bundle.stats.sd)) <=
                  SUPERLEARNER_SERVING_STATS_ATOL
            for step in SUPERLEARNER_SERVING_MODEL_STEPS
                @test haskey(artifacts.cells, (step, SUPERLEARNER_SERVING_POOLED, SUPERLEARNER_SERVING_POOLED))
                for depth in (SUPERLEARNER_SERVING_POOLED, SUPERLEARNER_SERVING_DEPTH_BINS...)
                    @test haskey(artifacts.conformal, (step, depth))
                end
            end
        end

        @testset "cell resolution walks the chain" begin
            # A resolved cell is used where it exists.
            own = superlearner_serving_cell(artifacts, 2, :active_deepening, :deep)
            @test own.cell.regime === :active_deepening && own.cell.depth === :deep
            @test !own.used_pooled_fallback
            @test own.cell.weights == bundle.deep_weights
            # A regime with only a depth-pooled cell falls back one step.
            quiet = superlearner_serving_cell(artifacts, 2, :quiet, :deep)
            @test quiet.cell.regime === :quiet && quiet.cell.depth === SUPERLEARNER_SERVING_POOLED
            @test quiet.used_pooled_fallback
            # A regime with no cell at all falls back to the fully pooled one.
            pooled = superlearner_serving_cell(artifacts, 2, :recovery, :moderate)
            @test pooled.cell.regime === SUPERLEARNER_SERVING_POOLED
            @test pooled.used_pooled_fallback
        end

        @testset "the served center is the weighted combination over the ten experts" begin
            experts = _experts([-60.0, -55.0, -70.0, -40.0, -50.0, -52.0, -48.0, -65.0, -30.0,
                                -150.0])
            quiet = superlearner_serving_center(artifacts; model_steps = 3, latest_dst = -10.0,
                                      dst_delta_1h_nt = 0.5, vbsouth_mvm = 0.0, experts = experts)
            expected = sum(bundle.weights[j] * values(experts)[j]
                           for j in 1:SUPERLEARNER_SERVING_EXPERT_COUNT)
            @test quiet.l1_center ≈ expected atol = 1e-12
            @test quiet.center ≈ expected atol = 1e-12
            @test !quiet.guard_applied
            @test !quiet.guard_enabled
            @test !quiet.projection_applied
            @test !quiet.deepening_cell
            @test quiet.regime === :quiet
            @test quiet.depth_bin === :shallow
            @test quiet.cell_regime === :quiet
            @test quiet.sindy_family_mass ≈ SUPERLEARNER_SERVING_SINDY_FLOOR atol = 1e-12
            # The static stack is a weighted input, not a post-hoc minimum: changing it alone must move
            # the center by exactly its own weight times the change.
            shifted = superlearner_serving_center(
                artifacts; model_steps = 3, latest_dst = -10.0, dst_delta_1h_nt = 0.5,
                vbsouth_mvm = 0.0,
                experts = merge(experts, (static_stack = values(experts)[SUPERLEARNER_SERVING_STATIC_EXPERT] -
                                                        10.0,)),
            )
            @test shifted.l1_center ≈ quiet.l1_center -
                  10.0 * bundle.weights[SUPERLEARNER_SERVING_STATIC_EXPERT] atol = 1e-12
            # A shallow row must not be handed the deep stratum's width.
            @test quiet.half_width_nt == bundle.half_width(3, :shallow)
            @test quiet.ci05_nt ≈ quiet.center - quiet.half_width_nt atol = 1e-12
            @test quiet.ci95_nt ≈ quiet.center + quiet.half_width_nt atol = 1e-12

            # A deepening state resolves the deep cell, and with the deployed bundle's guard switch off
            # the published center is that cell's combination — the static stack cannot cap it.
            deep = superlearner_serving_center(artifacts; model_steps = 3, latest_dst = -120.0,
                                     dst_delta_1h_nt = -25.0, vbsouth_mvm = 3.0, experts = experts)
            @test deep.deepening_cell
            @test !deep.guard_applied
            @test deep.regime === :active_deepening
            @test deep.depth_bin === :deep
            @test deep.cell_depth === :deep
            @test deep.l1_center ≈ sum(bundle.deep_weights[j] * values(experts)[j]
                                       for j in 1:SUPERLEARNER_SERVING_EXPERT_COUNT) atol = 1e-12
            @test deep.center ≈ deep.l1_center atol = 1e-12
            @test deep.half_width_nt == bundle.half_width(3, :deep)
            @test deep.regime_cell == "active_deepening/deep"

            # The gated coupling proxy is the stack's, not the raw proxy: a rising ring current with
            # positive coupling is not active deepening.
            rising = superlearner_serving_center(artifacts; model_steps = 3, latest_dst = -60.0,
                                       dst_delta_1h_nt = 4.0, vbsouth_mvm = 6.0, experts = experts)
            @test rising.coupling_active_mvm == 0.0
            @test rising.regime === :recovery
            @test !rising.deepening_cell
            @test !rising.guard_applied
        end

        @testset "every expert occupies its own slot: any exchange moves the center" begin
            # An expert wired into a neighbouring slot is a silent defect while the two slots carry
            # the same weight. In the shipped bundle 27 of 60 cells have `w_burton != w_burton_full`,
            # so such a swap would move the published center; the fixture must therefore give every
            # slot its own weight, and the combination must be sensitive to each one separately.
            @test allunique(bundle.weights)
            @test allunique(bundle.deep_weights)
            @test all(w -> w > 0.0, bundle.weights)
            @test all(w -> w > 0.0, bundle.deep_weights)
            # No two slots' weights coincide with another pair's sum split either way, so an exchange
            # cannot cancel.
            @test sum(bundle.weights) ≈ 1.0 atol = 1e-12
            @test sum(bundle.weights[j] for j in SUPERLEARNER_SERVING_SINDY_FAMILY) ≈
                  SUPERLEARNER_SERVING_SINDY_FLOOR atol = 1e-12

            # Ten distinct expert centers, so an exchange changes the combination by
            # (w_a - w_b) * (x_b - x_a) — nonzero exactly when both differ.
            centers = [-30.0 - 7.0 * k for k in 1:SUPERLEARNER_SERVING_EXPERT_COUNT]
            experts = _experts(centers)
            base = superlearner_serving_center(artifacts; model_steps = 3, latest_dst = -10.0,
                                      dst_delta_1h_nt = 0.5, vbsouth_mvm = 0.0, experts = experts)
            for a in 1:SUPERLEARNER_SERVING_EXPERT_COUNT, b in (a + 1):SUPERLEARNER_SERVING_EXPERT_COUNT
                swapped_centers = copy(centers)
                swapped_centers[a], swapped_centers[b] = swapped_centers[b], swapped_centers[a]
                swapped = superlearner_serving_center(artifacts; model_steps = 3, latest_dst = -10.0,
                                             dst_delta_1h_nt = 0.5, vbsouth_mvm = 0.0,
                                             experts = _experts(swapped_centers))
                @test swapped.l1_center != base.l1_center
                # The size of the move is the weight difference times the center difference, which is
                # what makes this an arithmetic check rather than an inequality that any change passes.
                @test swapped.l1_center - base.l1_center ≈
                      (bundle.weights[a] - bundle.weights[b]) * (centers[b] - centers[a]) atol = 1e-12
            end
        end

        @testset "the physical projection is reported when it acts" begin
            # The reported center is taken through `clamp(., -2000, 50)` nT. Every anchor the study
            # scored sits far from that ceiling, so the oracle can assert the projection inert there —
            # but a real ring current above roughly +52 nT with a panel to match does reach it, and a
            # projected center is indistinguishable from an unprojected one unless the stage says so.
            # The flag is what makes the `center == l1_center` identity of an unguarded bundle a
            # conditional statement instead of an invariant that quietly stops holding.
            high = _experts(fill(60.0, SUPERLEARNER_SERVING_EXPERT_COUNT))
            projected = superlearner_serving_center(artifacts; model_steps = 3, latest_dst = 55.0,
                                          dst_delta_1h_nt = 0.5, vbsouth_mvm = 0.0,
                                          experts = high)
            @test projected.l1_center ≈ 60.0 atol = 1e-9
            @test projected.raw_center ≈ 60.0 atol = 1e-9
            @test projected.center == SUPERLEARNER_SERVING_DST_CEIL_NT
            @test projected.projection_applied
            @test !projected.guard_applied
            # The interval is formed on the projected center, which is the number the row publishes.
            @test projected.ci05_nt ≈ projected.center - projected.half_width_nt atol = 1e-12
            @test projected.ci95_nt ≈ projected.center + projected.half_width_nt atol = 1e-12
            # The floor side of the same projection.
            floored = superlearner_serving_center(artifacts; model_steps = 3, latest_dst = -2400.0,
                                        dst_delta_1h_nt = 0.5, vbsouth_mvm = 0.0,
                                        experts = _experts(fill(-2500.0,
                                                                SUPERLEARNER_SERVING_EXPERT_COUNT)))
            @test floored.center == SUPERLEARNER_SERVING_DST_FLOOR_NT
            @test floored.projection_applied
            # A center strictly inside the physical range is not flagged, near either edge. The edges
            # themselves are not used: a convex combination whose weights sum to unity only within the
            # shipped tolerance lands a hair outside an exactly-boundary panel, which the projection
            # then legitimately moves.
            for value in (SUPERLEARNER_SERVING_DST_CEIL_NT - 1.0, SUPERLEARNER_SERVING_DST_FLOOR_NT + 1.0)
                inert = superlearner_serving_center(artifacts; model_steps = 3, latest_dst = value,
                                          dst_delta_1h_nt = 0.0, vbsouth_mvm = 0.0,
                                          experts = _experts(fill(value,
                                                                  SUPERLEARNER_SERVING_EXPERT_COUNT)))
                @test !inert.projection_applied
                @test inert.center ≈ value atol = 1e-9
            end
        end

        @testset "a bundle that enables the guard is served with it" begin
            # The deployed product does not guard the point forecast, but the code path stays
            # configurable, so it is exercised through a bundle whose own record enables it.
            mktempdir() do guarded_root
                guarded_bundle = build_superlearner_fixture_bundle(joinpath(guarded_root, "guarded");
                                                          guard = true)
                guarded = load_superlearner_serving_artifacts(guarded_bundle.dir)
                @test guarded.guard_applied
                @test guarded.guard_reference == SUPERLEARNER_SERVING_GUARD_REFERENCE
                # A guard moves the published center, so the bundle must name that stage in its
                # identity, and it must be banded by the guarded variant's own strata rather than by
                # the quantiles of the unguarded residuals.
                @test guarded.identity == SUPERLEARNER_SERVED_GUARDED_IDENTITY
                @test guarded.identity != SUPERLEARNER_SERVED_IDENTITY
                @test endswith(guarded.identity, SUPERLEARNER_SERVED_GUARD_IDENTITY_TOKEN)
                @test superlearner_served_identity(true) == SUPERLEARNER_SERVED_GUARDED_IDENTITY
                @test superlearner_served_conformal_variant(true) == SUPERLEARNER_SERVED_GUARDED_VARIANT
                for step in SUPERLEARNER_SERVING_MODEL_STEPS,
                    depth in (SUPERLEARNER_SERVING_POOLED, SUPERLEARNER_SERVING_DEPTH_BINS...)
                    @test guarded.conformal[(step, depth)].half_width_nt ≈
                          guarded_bundle.half_width(step, depth) *
                          SUPERLEARNER_FIXTURE_GUARDED_WIDTH_SCALE atol = 1e-12
                    # The unguarded width is present in the same file and must NOT have been read.
                    @test guarded.conformal[(step, depth)].half_width_nt !=
                          guarded_bundle.half_width(step, depth)
                end
                experts = _experts([-60.0, -55.0, -70.0, -40.0, -50.0, -52.0, -48.0, -65.0, -30.0,
                                    -150.0])
                deep = superlearner_serving_center(guarded; model_steps = 3, latest_dst = -120.0,
                                         dst_delta_1h_nt = -25.0, vbsouth_mvm = 3.0,
                                         experts = experts)
                @test deep.guard_enabled
                @test deep.deepening_cell
                @test deep.guard_applied
                @test deep.center == -150.0
                @test deep.center < deep.l1_center
                # Outside a deepening cell the reference is ignored even when the guard is enabled.
                quiet = superlearner_serving_center(guarded; model_steps = 3, latest_dst = -10.0,
                                          dst_delta_1h_nt = 0.5, vbsouth_mvm = 0.0,
                                          experts = experts)
                @test quiet.guard_enabled
                @test !quiet.guard_applied
                @test quiet.center ≈ quiet.l1_center atol = 1e-12
                # A shallower static stack in a deepening cell cannot lift the center.
                lifted = superlearner_serving_center(
                    guarded; model_steps = 3, latest_dst = -120.0, dst_delta_1h_nt = -25.0,
                    vbsouth_mvm = 3.0, experts = merge(experts, (static_stack = -20.0,)),
                )
                @test lifted.deepening_cell
                @test !lifted.guard_applied
                @test lifted.center ≈ lifted.l1_center atol = 1e-12
            end
        end

        @testset "a non-finite one-hour rate is neutralised, not fatal" begin
            # `static_stack_regime` throws on a NaN rate. Without the neutralisation in
            # `superlearner_serving_center` every row with a missing one-hour Dst rate — an interior Kyoto gap
            # — would drop out of super-learner into the fallback chain, which is an availability loss the
            # suite could not see. The static stack path pins the identical guard; this is its super-learner mirror.
            experts = _experts([-60.0, -55.0, -70.0, -40.0, -50.0, -52.0, -48.0, -65.0, -30.0,
                                -150.0])
            for (latest, vbsouth) in ((-80.0, 3.0), (-10.0, 0.0), (-120.0, 6.0), (-45.0, 1.5))
                zero_rate = superlearner_serving_center(
                    artifacts; model_steps = 3, latest_dst = latest, dst_delta_1h_nt = 0.0,
                    vbsouth_mvm = vbsouth, experts = experts)
                for bad_rate in (NaN, Inf, -Inf)
                    neutral = superlearner_serving_center(
                        artifacts; model_steps = 3, latest_dst = latest,
                        dst_delta_1h_nt = bad_rate, vbsouth_mvm = vbsouth, experts = experts)
                    # A neutralised rate must reproduce the zero-rate row exactly: same regime, same
                    # deepening label, same cell, same center and same band.
                    @test neutral.regime === zero_rate.regime
                    @test neutral.deepening_cell === zero_rate.deepening_cell
                    @test neutral.coupling_active_mvm == zero_rate.coupling_active_mvm
                    @test neutral.regime_cell == zero_rate.regime_cell
                    @test neutral.center == zero_rate.center
                    @test neutral.half_width_nt == zero_rate.half_width_nt
                end
            end
            # A deepening state is reached through the rate, so a neutralised rate must NOT be
            # labelled active deepening: this is what distinguishes neutralisation from a guess.
            neutral_deep = superlearner_serving_center(
                artifacts; model_steps = 3, latest_dst = -120.0, dst_delta_1h_nt = NaN,
                vbsouth_mvm = 3.0, experts = experts)
            @test !neutral_deep.deepening_cell
            @test neutral_deep.regime !== :active_deepening
            # The depth bin comes from the anchor Dst alone and is unaffected by the missing rate.
            @test neutral_deep.depth_bin === :deep
        end

        @testset "the served center refuses inputs it cannot serve" begin
            experts = _experts(fill(-50.0, SUPERLEARNER_SERVING_EXPERT_COUNT))
            @test_throws ArgumentError superlearner_serving_center(
                artifacts; model_steps = 5, latest_dst = -10.0, dst_delta_1h_nt = 0.0,
                vbsouth_mvm = 0.0, experts = experts)
            @test_throws ArgumentError superlearner_serving_center(
                artifacts; model_steps = 3, latest_dst = NaN, dst_delta_1h_nt = 0.0,
                vbsouth_mvm = 0.0, experts = experts)
            # The static stack is expert ten, so a row without it cannot be combined at all; the center
            # is refused rather than published over a nine-expert panel.
            @test_throws ArgumentError superlearner_serving_center(
                artifacts; model_steps = 3, latest_dst = -10.0, dst_delta_1h_nt = 0.0,
                vbsouth_mvm = 0.0, experts = merge(experts, (static_stack = NaN,)))
            bad = _experts([-50.0, -50.0, NaN, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0, -50.0])
            @test_throws ArgumentError superlearner_serving_center(
                artifacts; model_steps = 3, latest_dst = -10.0, dst_delta_1h_nt = 0.0,
                vbsouth_mvm = 0.0, experts = bad)
            @test_throws ArgumentError superlearner_serving_center(
                artifacts; model_steps = 3, latest_dst = -10.0, dst_delta_1h_nt = 0.0,
                vbsouth_mvm = 0.0, experts = (served_inertial_sindy = -50.0, frozen_inertial_sindy = -50.0))
        end

        @testset "climatology relaxes persistence toward zero" begin
            for step in SUPERLEARNER_SERVING_MODEL_STEPS
                @test superlearner_serving_climatology_center(artifacts, -100.0, step) ≈
                      -100.0 * exp(-step / bundle.tau) atol = 1e-12
            end
            # Deeper leads relax further, and the sign never flips.
            values_by_step = [superlearner_serving_climatology_center(artifacts, -100.0, step)
                              for step in SUPERLEARNER_SERVING_MODEL_STEPS]
            @test issorted(values_by_step)
            @test all(v -> v < 0.0, values_by_step)
            @test superlearner_serving_climatology_center(artifacts, 0.0, 6) == 0.0
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
                          for lag in 1:SUPERLEARNER_SERVING_DRIVER_LAGS_H]
            dst_history = Dict(anchor - Hour(lag) => frame.Dst[index[anchor - Hour(lag)]]
                               for lag in 0:SUPERLEARNER_SERVING_DST_LAG_MAX_H)

            mini = superlearner_serving_direct_frame(anchor, history, dst_history)
            @test nrow(mini) == SUPERLEARNER_SERVING_DST_LAG_MAX_H + 1
            @test last(mini.time_utc) == anchor
            @test first(mini.time_utc) == anchor - Hour(SUPERLEARNER_SERVING_DST_LAG_MAX_H)
            @test issorted(mini.time_utc)
            # The driver channels are carried only as deep as the history reaches; the Dst ladder is
            # carried the whole span, because `dst_lag24` reads its far end.
            @test isfinite(mini.V[end - 1])
            @test isnan(mini.V[1])
            @test all(isfinite, mini.Dst)

            design = superlearner_serving_direct_features(artifacts, anchor, history, dst_history)
            @test design.ok
            @test length(design.features) == DIRECT_GBM_FEATURE_COUNT
            # The design must equal the package's own construction on the whole frame: that is what
            # makes the live block the block the study fitted on.
            reference, reference_ok = direct_gbm_features(frame, [anchor])
            @test reference_ok[1]
            @test design.features == vec(Matrix{Float64}(reference)[1, :])
            # And its first 18 columns must be the analog key exactly; a disagreement is refused.
            key = superlearner_serving_analog_features(artifacts, anchor, history,
                                             frame.Dst[index[anchor]],
                                             frame.Dst[index[anchor - Hour(1)]])
            @test key.ok
            @test design.features[1:ANALOG_FEATURE_COUNT] == key.features
            @test superlearner_serving_direct_features(artifacts, anchor, history, dst_history;
                                              analog_features = key.features).ok
            drifted = copy(key.features)
            drifted[1] += 0.5
            @test_throws ErrorException superlearner_serving_direct_features(
                artifacts, anchor, history, dst_history; analog_features = drifted)

            # The increment is inverted with the design's own anchor Dst.
            for step in SUPERLEARNER_SERVING_MODEL_STEPS
                increment = gbm_predict(artifacts.direct_models[step],
                                        reshape(design.features, 1, length(design.features)))[1]
                @test superlearner_serving_direct_center(artifacts, design.features, step) ≈
                      increment + design.features[analog_feature_index(:dst0)] atol = 1e-12
            end
            @test superlearner_serving_direct_center(artifacts, design.features, 1) !=
                  superlearner_serving_direct_center(artifacts, design.features, 7)

            # A gap anywhere in the ladder is reported by name rather than imputed.
            for lag in (1, 6, 12, 24)
                holed = copy(dst_history)
                delete!(holed, anchor - Hour(lag))
                result = superlearner_serving_direct_features(artifacts, anchor, history, holed)
                @test !result.ok
                @test result.reason == "missing_dst_lag$(lag)"
                @test all(!isfinite, result.features)
            end
            no_anchor = copy(dst_history)
            delete!(no_anchor, anchor)
            @test superlearner_serving_direct_features(artifacts, anchor, history, no_anchor).reason ==
                  "missing_anchor_dst"
            gapped_driver = copy(history)
            gapped_driver[3] = nothing
            @test !superlearner_serving_direct_features(artifacts, anchor, gapped_driver, dst_history).ok
            @test_throws ArgumentError superlearner_serving_direct_frame(anchor, history[1:3], dst_history)
        end

        @testset "the interval takes the row's stratum with a pooled fallback" begin
            for step in SUPERLEARNER_SERVING_MODEL_STEPS, depth in SUPERLEARNER_SERVING_DEPTH_BINS
                interval = superlearner_serving_interval(artifacts, step, depth, -100.0)
                @test interval.half_width_nt == bundle.half_width(step, depth)
                @test interval.lo_nt == -100.0 - interval.half_width_nt
                @test interval.hi_nt == -100.0 + interval.half_width_nt
            end
            # A depth label the calibration does not carry resolves to the step's pooled stratum
            # instead of erroring, which is the only path a future bin could take. The pooled width is
            # distinct from every depth bin's in this bundle, so the assertion discriminates the pooled
            # fallback from silently returning the shallow stratum — which is what it did while the two
            # widths were equal.
            for step in SUPERLEARNER_SERVING_MODEL_STEPS
                pooled = bundle.half_width(step, SUPERLEARNER_SERVING_POOLED)
                for depth in SUPERLEARNER_SERVING_DEPTH_BINS
                    @test pooled != bundle.half_width(step, depth)
                end
                @test superlearner_serving_interval(artifacts, step, :unknown_bin, 0.0).half_width_nt ==
                      pooled
                @test superlearner_serving_interval(artifacts, step, :unknown_bin, 0.0).half_width_nt !=
                      bundle.half_width(step, :shallow)
                @test superlearner_serving_interval(artifacts, step, :unknown_bin, -100.0).stratum.depth ===
                      SUPERLEARNER_SERVING_POOLED
            end
            @test_throws ErrorException superlearner_serving_interval(artifacts, 5, :shallow, 0.0)
        end

        @testset "manifest verification is a digest check, not a file listing" begin
            @test superlearner_serving_verify_manifest(bundle.dir) isa DataFrame
            hashed = superlearner_serving_manifest_hashed_names(
                CSV.read(joinpath(bundle.dir, "manifest.csv"), DataFrame))
            for name in SUPERLEARNER_SERVING_REQUIRED_FILES
                @test name in hashed
            end
            for step in SUPERLEARNER_SERVING_MODEL_STEPS
                @test superlearner_serving_direct_file(step) in hashed
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
                         :conformal_variant_mismatch,
                         # Cross-record defects: each file parses on its own, but they describe
                         # different fits.
                         :fold_year_mismatch, :conformal_fold_mismatch, :tau_fold_mismatch,
                         :tau_cutoff_mismatch, :tau_record_drift, :guard_records_disagree,
                         :selection_guard_switch_absent, :expert_set_absent, :n_experts_absent,
                         :archive_past_training_cutoff,
                         # A guard-on bundle may not go out under the unguarded identity, and may not
                         # be banded by the unguarded variant's strata.
                         :guard_plain_identity, :guard_missing_guarded_strata)
            @test mutation in SUPERLEARNER_FIXTURE_MUTATIONS
        end
        @test length(SUPERLEARNER_FIXTURE_MUTATIONS) == 40
        @test allunique(SUPERLEARNER_FIXTURE_MUTATIONS)
        # Every defect names the check it must trip; a new mutation without an expected message would
        # otherwise be asserted only as "throws something".
        @test sort(collect(keys(SUPERLEARNER_FIXTURE_EXPECTED_ERRORS))) ==
              sort(collect(SUPERLEARNER_FIXTURE_MUTATIONS))
        for mutation in SUPERLEARNER_FIXTURE_MUTATIONS
            mktempdir() do root
                bad = build_superlearner_fixture_bundle(joinpath(root, String(mutation));
                                               mutate = mutation)
                thrown = try
                    load_superlearner_serving_artifacts(bad.dir)
                    nothing
                catch err
                    err isa InterruptException && rethrow()
                    err
                end
                @test thrown !== nothing
                thrown === nothing && return
                message = sprint(showerror, thrown)
                @test occursin(SUPERLEARNER_FIXTURE_EXPECTED_ERRORS[mutation], message)
            end
        end
    end

    @testset "the direct-GBM digest gate refuses a model the manifest does not cover" begin
        # The reader refuses a direct-GBM file that carries no verified manifest digest. Through a full
        # load that defect is caught earlier, by the manifest's required-artifact rule — which is what
        # `:unlisted_gbm_model` proves — so the gate itself is reached here, through the reader, rather
        # than left as a branch no test can enter.
        mktempdir() do root
            bundle = build_superlearner_fixture_bundle(joinpath(root, "direct"))
            labels = Dict{Int,String}()
            complete = Set{String}(superlearner_serving_direct_file(step)
                                   for step in SUPERLEARNER_SERVING_MODEL_STEPS)
            full = SolarSINDy._superlearner_serving_read_direct(bundle.dir, complete, labels)
            @test sort(collect(keys(full.models))) == sort(collect(SUPERLEARNER_SERVING_MODEL_STEPS))
            missing_name = superlearner_serving_direct_file(first(SUPERLEARNER_SERVING_MODEL_STEPS))
            short = Set{String}(name for name in complete if name != missing_name)
            thrown = try
                SolarSINDy._superlearner_serving_read_direct(bundle.dir, short, labels)
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
            @test SolarSINDy._superlearner_serving_read_direct(bundle.dir, nothing, labels).models isa Dict
        end
    end

    @testset "a missing or unreadable bundle is refused rather than substituted" begin
        mktempdir() do root
            @test_throws ErrorException load_superlearner_serving_artifacts(joinpath(root, "absent"))
            empty_dir = joinpath(root, "empty")
            mkpath(empty_dir)
            @test_throws ErrorException load_superlearner_serving_artifacts(empty_dir)
        end
    end

    @testset "the super-learner cell regime uses the same thresholds the deployed stack was cut on" begin
        # `superlearner_serving_center` labels a row's issue-time regime with `static_stack_regime`'s
        # package defaults, while expert E10 — the static stack — labels its own cell with the
        # thresholds recorded in the stack file. The two are the same regime split today, and nothing
        # but the deployed file's SHA pin says so. This asserts the equality directly, so a restacked
        # bundle cut on different edges cannot silently put the two experts in different regimes.
        stack_path = joinpath(@__DIR__, "..", "deploy", "static_stack.csv")
        @test isfile(stack_path)
        recorded = CSV.read(stack_path, DataFrame)
        for column in ("disturbed_dst_threshold_nt", "deepening_rate_threshold_nt_per_h",
                       "coupling_threshold_mvm", "sindy_mass_floor", "minimum_cell_rows")
            @test column in names(recorded)
        end
        @test length(unique(recorded.disturbed_dst_threshold_nt)) == 1
        @test length(unique(recorded.deepening_rate_threshold_nt_per_h)) == 1
        @test length(unique(recorded.coupling_threshold_mvm)) == 1
        @test Float64(recorded.disturbed_dst_threshold_nt[1]) ==
              SolarSINDy.STATIC_STACK_DEFAULT_DISTURBED_DST_NT
        @test Float64(recorded.deepening_rate_threshold_nt_per_h[1]) ==
              SolarSINDy.STATIC_STACK_DEFAULT_DEEPENING_RATE_NT_PER_H
        @test Float64(recorded.coupling_threshold_mvm[1]) ==
              SolarSINDy.STATIC_STACK_DEFAULT_COUPLING_THRESHOLD_MVM
        @test Float64(recorded.sindy_mass_floor[1]) ==
              SolarSINDy.STATIC_STACK_DEFAULT_SINDY_MASS_FLOOR
        @test Int(recorded.minimum_cell_rows[1]) ==
              SolarSINDy.STATIC_STACK_DEFAULT_MINIMUM_CELL_ROWS
        # The equality is what makes the two labellings agree; assert that consequence too, over the
        # regime boundaries the deployed edges sit on.
        deployed = load_static_stack(stack_path)
        for latest in (-100.0, -70.0, -50.0, -30.0, -29.0, -10.0, 0.0),
            rate in (-20.0, -5.1, -5.0, -4.9, 0.0, 6.0),
            vbsouth in (0.0, 2.5)
            coupling = static_stack_serving_coupling_active(vbsouth, rate)
            @test static_stack_regime(deployed, latest, rate, coupling) ===
                  static_stack_regime(latest, rate, coupling)
        end
    end
end

end # module
