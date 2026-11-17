module IntervalMeanForecasterPackageTests

using Test, Dates, LinearAlgebra, SolarSINDy
const IMF = SolarSINDy.IntervalMeanForecaster
const T0 = DateTime(2018)
const SYNTHETIC_SOURCE = repeat("ab", 32)
const ROUNDING_RTOL = 64 * eps(Float64)

@testset "Interval moments and correlated observation update" begin
    # Brownian motion: Var(W_d)=d, Cov(W_d,mean(W))=d/2,
    # and Var(mean(W))=d/3. These values are exactly representable here.
    k = IMF.interval_kernel(0., 2., 3.)
    @test (k.phi, k.mean_coefficient, k.forcing_endpoint, k.forcing_mean) == (1., 1., 3., 1.5)
    @test (k.endpoint_process_variance, k.endpoint_mean_process_covariance,
           k.mean_process_variance) == (12., 6., 4.)
    decay = IMF.interval_kernel(log(2.), 0., 1.)
    @test isapprox(decay.phi, .5; rtol=ROUNDING_RTOL)
    @test isapprox(decay.mean_coefficient, .5 / log(2.); rtol=ROUNDING_RTOL)
    @test isapprox(decay.forcing_endpoint, .5 / log(2.); rtol=ROUNDING_RTOL)
    @test isapprox(decay.forcing_mean, (1 - .5 / log(2.)) / log(2.); rtol=ROUNDING_RTOL)
    noisy_decay = IMF.interval_kernel(1., 1., 1.)
    @test isapprox(noisy_decay.endpoint_process_variance, (1 - exp(-2.)) / 2; rtol=ROUNDING_RTOL)
    @test isapprox(noisy_decay.endpoint_mean_process_covariance, (1 - exp(-1.))^2 / 2; rtol=ROUNDING_RTOL)
    @test isapprox(noisy_decay.mean_process_variance, -.5 + 2 * exp(-1.) - exp(-2.) / 2; rtol=ROUNDING_RTOL)

    joint = IMF.interval_moments([2.], reshape([9.], 1, 1), [0.], [3.], [4.], 2.)
    @test joint.endpoint_mean == [10.]
    @test joint.interval_mean == 6.
    @test joint.endpoint_covariance == reshape([27.], 1, 1)
    @test joint.interval_variance == 15.
    @test joint.endpoint_interval_covariance == [18.]
    # Gaussian conditioning: gain=18/(15+3)=1 and posterior variance=27-18=9.
    posterior = IMF.condition_interval(joint, 9., 3.)
    @test posterior.mean == [13.]
    @test posterior.covariance == reshape([9.], 1, 1)
    @test posterior.gain == [1.]
    @test_throws ArgumentError IMF.interval_kernel(-1., 1., 1.)
    @test_throws ArgumentError IMF.interval_kernel(1., 1., 0.)
    @test_throws ArgumentError IMF.condition_interval(joint, 9., 0.)
    @test_throws ArgumentError IMF.interval_moments([0.], reshape([-1.], 1, 1), [1.], [1.], [0.], 1.)
end

@testset "Two-gain nonnegative least squares" begin
    identity_design = Matrix{Float64}(I, 2, 2)
    @test isapprox(collect(IMF.nnls2(identity_design, [2., 3.]).gains), [2., 3.]; rtol=ROUNDING_RTOL)
    @test isapprox(collect(IMF.nnls2(identity_design, [-2., 3.]).gains), [0., 3.]; rtol=ROUNDING_RTOL)
    # Equal collinear columns have minimum-norm nonnegative solution (1,1).
    @test isapprox(collect(IMF.nnls2([1. 1.; 2. 2.], [2., 4.]).gains), [1., 1.]; rtol=ROUNDING_RTOL)
    @test_throws DimensionMismatch IMF.nnls2(ones(2, 3), [1., 2.])
    @test_throws ArgumentError IMF.nnls2(identity_design, [1., 2.]; weights=[1., -1.])
end

@testset "Forecast clocks and label embargo" begin
    for delay in (0, 1), horizon in (4, 6)
        clock = IMF.forecast_clock(T0, horizon, delay)
        @test clock.anchor_tag == T0 - Hour(1 + delay)
        @test clock.anchor_receipt == T0
        @test clock.target_complete == T0 + Hour(horizon)
        @test clock.target_receipt == T0 + Hour(horizon + delay)
    end
    boundary = T0 + Hour(200)
    cutoff = boundary - Hour(168)
    issue = cutoff - Hour(7)
    four = IMF.forecast_clock(issue, 4, 1)
    six = IMF.forecast_clock(issue, 6, 1)
    @test IMF.fit_eligible(issue, six.target_complete, six.target_receipt, boundary)
    @test IMF.joint_fit_mask([issue], [four.target_complete], [four.target_receipt],
                            [six.target_complete], [six.target_receipt], boundary) == [true]
    @test IMF.joint_fit_mask([issue], [four.target_complete], [four.target_receipt],
                            [six.target_complete], [cutoff + Millisecond(1)], boundary) == [false]
    @test_throws ArgumentError IMF.forecast_clock(T0 + Minute(1), 4, 0)
    @test_throws ArgumentError IMF.forecast_clock(T0, 4, true)
end

function synthetic_input(values)
    IMF.HourlyInputs([T0 + Hour(i - 1) for i in axes(values, 2)], values;
                     source_sha256=SYNTHETIC_SOURCE)
end

function collect_stream(input; kwargs...)
    rows = NamedTuple[]
    result = IMF.stream_basis(input; delay_hours=1, taus=(4., 24.), sigma=3.,
                              sink=part -> append!(rows, part), kwargs...)
    rows, result
end

function checkpoint_values(cp)
    pending = cp.pending === nothing ? nothing :
        (cp.pending.index, cp.pending.start, cp.pending.stop, cp.pending.q, cp.pending.pressure)
    (cp.last_index, cp.last_issue, cp.source_sha256, cp.prefix_sha256,
     cp.delay_hours, cp.taus, cp.sigma, cp.sealed.boundary, cp.sealed.mean,
     cp.sealed.covariance, cp.basis, pending, cp.active, cp.consecutive_valid, cp.reset_tag)
end

@testset "Streaming, causal forecasts, fitting and model persistence" begin
    values = ones(17, 320)
    values[8, :] .= 400.
    values[6, :] .= -5.
    values[12, :] .= 2.
    values[15, :] .= [-30. + sin(i / 20) for i in axes(values, 2)]
    input = synthetic_input(values)
    rows, full = collect_stream(input; chunk_hours=7)
    @test length(rows) == full.row_count == 320
    @test !rows[167].model_available && rows[168].model_available
    @test full.max_buffered_rows <= 7
    immediate_rows = NamedTuple[]
    immediate = IMF.stream_basis(input; delay_hours=0, taus=(4., 24.), sigma=3.,
        stop_after_index=180, sink=part -> append!(immediate_rows, part))
    @test immediate_rows[168].model_available
    @test immediate.checkpoint.pending === nothing
    @test immediate.checkpoint.sealed.boundary == immediate.checkpoint.last_issue

    chunked, _ = collect_stream(input; chunk_hours=31)
    @test isequal(chunked, rows)
    first, partial = collect_stream(input; chunk_hours=13, stop_after_index=100)
    rest, resumed = collect_stream(input; chunk_hours=13, checkpoint=partial.checkpoint)
    @test isequal(vcat(first, rest), rows)
    @test isequal(checkpoint_values(resumed.checkpoint), checkpoint_values(full.checkpoint))

    # Change all unknown future speed/Dst values. The issued forecast stays
    # fixed even though its subsequently scored target labels change.
    future_values = copy(values)
    future_values[8, 191:end] .= 900.
    future_values[15, 191:end] .= -300.
    future, _ = collect_stream(synthetic_input(future_values); chunk_hours=19)
    @test future[190].offset == rows[190].offset
    @test future[190].coefficients == rows[190].coefficients
    @test future[190].variance == rows[190].variance
    @test future[190].target != rows[190].target

    # Independent manufactured affine responses give exact gains (2,3).
    setting = IMF.settings()[9]
    panel = [begin
        issue = T0 + Hour(i)
        clocks = (IMF.forecast_clock(issue, 4, 1), IMF.forecast_clock(issue, 6, 1))
        (issue=issue, driver_tag=clocks[1].driver_tag, driver_receipt=issue,
         anchor_tag=clocks[1].anchor_tag, anchor_complete=clocks[1].anchor_complete,
         anchor_receipt=clocks[1].anchor_receipt, anchor=-30.,
         target_tag=Tuple(c.target_tag for c in clocks),
         target_complete=Tuple(c.target_complete for c in clocks),
         target_receipt=Tuple(c.target_receipt for c in clocks),
         target=(2., 3.), scoreable=(true, true), model_available=true,
         offset=(0., 0.), coefficients=((1., 0.), (0., 1.)),
         dst_delay_hours=1, taus=setting.taus, sigma=setting.sigma,
         source_sha256=SYNTHETIC_SOURCE)
    end for i in vcat(collect(1:120), 200)]
    boundary = T0 + Hour(300)
    model = IMF.fit_gains(panel, boundary, setting; delay_hours=1, source_sha256=SYNTHETIC_SOURCE)
    @test model.fit_count == model.informative_count == 120
    @test isapprox(collect(model.gains), [2., 3.]; rtol=ROUNDING_RTOL)
    @test model.max_label_receipt <= boundary - Hour(168)
    predictions = IMF.predict_rows(model, panel)
    @test isapprox(collect(predictions[1].prediction), [2., 3.]; rtol=ROUNDING_RTOL)
    changed_labels = copy(panel)
    changed_labels[end] = merge(panel[end], (target=(-1.e6, 1.e6),))
    refit = IMF.fit_gains(changed_labels, boundary, setting; delay_hours=1, source_sha256=SYNTHETIC_SOURCE)
    @test refit.gains == model.gains
    @test refit.fit_data_sha256 == model.fit_data_sha256

    mktempdir() do directory
        saved = IMF.save_bundle(joinpath(directory, "model"), model, full.checkpoint;
                                provenance=(synthetic_fixture=true,))
        loaded = IMF.load_bundle(saved.directory; expected_manifest_sha256=saved.manifest_sha256,
                                 expected_sources=IMF.DstIntervalArtifacts.verify_source_identity())
        @test isequal(loaded.model, model)
        @test isequal(checkpoint_values(loaded.checkpoint), checkpoint_values(full.checkpoint))
        @test isequal(IMF.predict_rows(loaded.model, panel), predictions)
    end
end

end
