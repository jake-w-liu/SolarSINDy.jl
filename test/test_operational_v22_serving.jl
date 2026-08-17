# test_operational_v22_serving.jl — served V2.2 static-stack serving contract.
#
# The served point center is the fitted static regime stack applied to six point components. Three
# things can silently break it: the coupling gate (which decides the regime), the component order
# (which decides which weight multiplies which forecast), and the provenance pin (which decides
# whether the served identity means anything). Each is checked here against an independent
# expectation rather than against the implementation's own output, and the archived
# `static_v2_2_dst_nt` column supplies a real-data oracle for the whole application.

module OperationalV22ServingTests

using Test
using CSV
using DataFrames
using Dates
using SolarSINDy

const DEPLOY_STACK = normpath(joinpath(@__DIR__, "..", "deploy", V22_SERVED_STACK_FILE))
const DEPLOY_MANIFEST = normpath(joinpath(@__DIR__, "..", "deploy", V22_SERVED_STACK_MANIFEST))
const BASE_TABLE = normpath(joinpath(
    @__DIR__, "..", "validation", "output", "operational", "v2_3_base", "v2_3_base_table.csv"))

"Hand-rolled weighted sum: the independent expectation for one stack cell."
function _hand_stack_sum(stack::OperationalV22Stack, step::Int, regime::Symbol, components)
    cells = [c for c in stack.cells if c.model_step_hours == step && c.regime === regime]
    isempty(cells) && return nothing
    cell = cells[1]
    total = 0.0
    for (i, name) in enumerate(OPERATIONAL_V22_COMPONENTS)
        total += cell.weights[i] * Float64(getproperty(components, name))
    end
    return total
end

@testset "Operational V2.2 static-stack serving" begin
    @testset "published identity strings" begin
        # The label is the operator's only machine-readable record of which stages produced the
        # published center, so it is pinned literally rather than rebuilt from parts.
        @test V22_SERVED_IDENTITY ==
              "v2.2+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia+staticstack(sindy60_fit407598)"
        @test V22_SERVED_STACK_LABEL == "operational_v2_2_primary_sindy60_fit407598"
        @test V22_SERVED_STACK_FILE == "operational_v2_2_stack.csv"
        @test occursin("static_regime_stack", V22_SERVED_DRIVER_ASSUMPTION)
        @test startswith(V22_SERVED_DRIVER_ASSUMPTION, "ballistically_propagated_l1")
    end

    @testset "coupling gate disengages outside the storm main phase" begin
        # Independent statement of the archived definition: rectified coupling counts only while the
        # wind drives southward AND the ring current deepens.
        @test v22_serving_coupling_active(2.5, -4.0) == 2.5
        @test v22_serving_coupling_active(2.5, -1e-12) == 2.5
        @test v22_serving_coupling_active(2.5, 0.0) == 0.0
        @test v22_serving_coupling_active(2.5, 3.0) == 0.0
        @test v22_serving_coupling_active(0.0, -4.0) == 0.0
        @test v22_serving_coupling_active(-1.0, -4.0) == 0.0
        @test v22_serving_coupling_active(NaN, -4.0) == 0.0
        @test v22_serving_coupling_active(2.5, NaN) == 0.0
    end

    @testset "depth-safe alerting center takes the deeper forecast" begin
        @test v22_serving_depth_safe_center(-90.0, -120.0) == -120.0
        @test v22_serving_depth_safe_center(-150.0, -120.0) == -150.0
        @test v22_serving_depth_safe_center(-150.0, NaN) == -150.0
        @test v22_serving_depth_safe_center(NaN, -120.0) == -120.0
    end

    @testset "deployment provenance is pinned" begin
        @test isfile(DEPLOY_STACK)
        @test v22_serving_stack_sha256(DEPLOY_STACK) == V22_SERVED_STACK_SHA256
        stack = load_v22_serving_stack(DEPLOY_STACK)
        @test stack.label == V22_SERVED_STACK_LABEL
        @test collect(stack.supported_model_steps) == [1, 2, 3, 4, 6, 7]
        # A tampered file must fail closed rather than serve unpinned weights.
        mktempdir() do dir
            copy = joinpath(dir, V22_SERVED_STACK_FILE)
            cp(DEPLOY_STACK, copy)
            @test_throws ErrorException load_v22_serving_stack(
                copy; expect_sha256 = repeat("0", 64))
            @test_throws ErrorException load_v22_serving_stack(
                copy; expect_label = "operational_v2_2_not_this_fit")
            # A digest override of "" is the documented fixture escape hatch and must still check
            # the label, so a fixture cannot masquerade as the published stack.
            @test load_v22_serving_stack(copy; expect_sha256 = "").label == V22_SERVED_STACK_LABEL
        end
        @test isfile(DEPLOY_MANIFEST)
        manifest = CSV.read(DEPLOY_MANIFEST, DataFrame)
        digest_rows = manifest[manifest.entry_type .== "sha256", :]
        @test nrow(digest_rows) == 1
        @test String(digest_rows.name[1]) == V22_SERVED_STACK_FILE
        @test String(digest_rows.value[1]) == V22_SERVED_STACK_SHA256
        label_rows = manifest[(manifest.entry_type .== "stack") .&
                              (manifest.name .== "label"), :]
        @test String(label_rows.value[1]) == V22_SERVED_STACK_LABEL
    end

    @testset "served center is the stack cell applied to the six components" begin
        stack = load_v22_serving_stack(DEPLOY_STACK)
        components = (served_v2_1 = -140.0, frozen_v2_1 = -130.0, persistence = -120.0,
                      burton = -150.0, burton_full = -145.0, obrien = -135.0)
        for step in (1, 2, 3, 4, 6, 7)
            # Active deepening: coupling positive and Dst falling fast, Dst well below the quiet edge.
            result = v22_serving_center(
                stack; model_steps = step, latest_dst = -120.0, dst_delta_1h_nt = -12.0,
                vbsouth_mvm = 3.2, components...)
            @test result.regime === :active_deepening
            @test result.coupling_active_mvm == 3.2
            expected = _hand_stack_sum(stack, step, :active_deepening, components)
            @test expected !== nothing
            @test result.raw_center ≈ expected atol=1e-12
            @test result.center ≈ expected atol=1e-12
            @test result.model_step_hours == step
            # SINDy dominance is a property of the fitted stack, not of this call; assert it holds so
            # a re-fit that violated the constraint could not be served silently.
            @test result.sindy_mass >= 0.60 - 1e-9
            @test sum(values(result.weights)) ≈ 1.0 atol=1e-9
        end
        # A quiet issue with a positive rate must not reach the deepening cell: the gate is zero, so
        # the coupling threshold cannot promote it.
        quiet = v22_serving_center(
            stack; model_steps = 3, latest_dst = -10.0, dst_delta_1h_nt = 2.0,
            vbsouth_mvm = 4.0, components...)
        @test quiet.regime === :quiet
        @test quiet.coupling_active_mvm == 0.0
        @test quiet.raw_center ≈ _hand_stack_sum(stack, 3, :quiet, components) atol=1e-12
        # Recovery: disturbed Dst with the ring current refilling.
        recovery = v22_serving_center(
            stack; model_steps = 2, latest_dst = -80.0, dst_delta_1h_nt = 4.0,
            vbsouth_mvm = 0.0, components...)
        @test recovery.regime === :recovery
        @test recovery.raw_center ≈ _hand_stack_sum(stack, 2, :recovery, components) atol=1e-12
    end

    @testset "served center fails closed on unusable inputs" begin
        stack = load_v22_serving_stack(DEPLOY_STACK)
        components = (served_v2_1 = -140.0, frozen_v2_1 = -130.0, persistence = -120.0,
                      burton = -150.0, burton_full = -145.0, obrien = -135.0)
        @test_throws ArgumentError v22_serving_center(
            stack; model_steps = 5, latest_dst = -120.0, dst_delta_1h_nt = -12.0,
            vbsouth_mvm = 3.2, components...)
        @test_throws ArgumentError v22_serving_center(
            stack; model_steps = 0, latest_dst = -120.0, dst_delta_1h_nt = -12.0,
            vbsouth_mvm = 3.2, components...)
        @test_throws ArgumentError v22_serving_center(
            stack; model_steps = 3, latest_dst = -120.0, dst_delta_1h_nt = -12.0,
            vbsouth_mvm = 3.2, served_v2_1 = NaN, frozen_v2_1 = -130.0, persistence = -120.0,
            burton = -150.0, burton_full = -145.0, obrien = -135.0)
        # A non-finite one-hour rate is neutral rather than fatal, matching the archived convention
        # for an interior Kyoto-Dst gap.
        neutral = v22_serving_center(
            stack; model_steps = 3, latest_dst = -80.0, dst_delta_1h_nt = NaN,
            vbsouth_mvm = 3.0, components...)
        @test neutral.coupling_active_mvm == 0.0
        @test neutral.regime === :recovery
        # The physical projection is the only difference between the archived sum and the served
        # center, and it engages only above the +50 nT ceiling.
        positive = (served_v2_1 = 50.0, frozen_v2_1 = 45.0, persistence = 77.0,
                    burton = 69.0, burton_full = 69.0, obrien = 69.0)
        high = v22_serving_center(
            stack; model_steps = 1, latest_dst = 77.0, dst_delta_1h_nt = 19.0,
            vbsouth_mvm = 0.0, positive...)
        @test high.raw_center > 50.0
        @test high.center == 50.0
    end

    @testset "archived static-stack column is reproduced" begin
        if !isfile(BASE_TABLE)
            @test_skip "V2.3 base table is absent; run validation/operational/v2_3_base_table.jl"
        else
            stack = load_v22_serving_stack(DEPLOY_STACK)
            # Stream the head of the archived table rather than materialising 400 MB: the identity is
            # a per-row property, so a contiguous prefix of scorable rows is a valid sample and the
            # standalone oracle (validation/operational/v2_2_served_identity.jl) covers all of them.
            wanted = 2_000
            worst = 0.0
            worst_coupling = 0.0
            checked = 0
            steps = Set(Int[])
            for row in CSV.Rows(BASE_TABLE; strict = true, reusebuffer = true)
                checked >= wanted && break
                String(row.partition) in ("DEV", "TEST") || continue
                step = parse(Int, String(row.model_step_hours))
                step in stack.supported_model_steps || continue
                f(name) = parse(Float64, String(getproperty(row, name)))
                result = v22_serving_center(
                    stack; model_steps = step,
                    latest_dst = f(:latest_dst_nt),
                    dst_delta_1h_nt = f(:dst_delta_1h_nt),
                    vbsouth_mvm = f(:VBsouth_mvm),
                    served_v2_1 = f(:served_v2_1_dst_nt),
                    frozen_v2_1 = f(:frozen_v2_1_dst_nt),
                    persistence = f(:persistence_dst_nt),
                    burton = f(:burton_dst_nt),
                    burton_full = f(:burton_full_dst_nt),
                    obrien = f(:obrien_dst_nt))
                worst = max(worst, abs(result.raw_center - f(:static_v2_2_dst_nt)))
                worst_coupling = max(worst_coupling,
                                     abs(result.coupling_active_mvm - f(:coupling_active_mvm)))
                push!(steps, step)
                checked += 1
            end
            @test checked >= 200
            @test length(steps) == length(stack.supported_model_steps)
            @test worst <= 1e-9
            @test worst_coupling <= 1e-9
        end
    end
end

end # module
