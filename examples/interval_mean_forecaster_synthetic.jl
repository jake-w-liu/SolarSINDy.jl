using Dates, SHA, SolarSINDy

"""
    run_interval_mean_forecaster_synthetic()

Exercise the maintained interval-mean namespace with generated hourly inputs.
The two latency assumptions are checked separately. Outputs demonstrate library
correctness on a synthetic fixture, not forecast skill or deployment readiness.
"""
function run_interval_mean_forecaster_synthetic()
    F = SolarSINDy.IntervalMeanForecaster
    sources = F.source_identity()
    mktempdir(; prefix="solarsindy-interval-example-") do directory
        origin = DateTime(2018)
        n = 900
        values = ones(17, n)
        values[8, :] .= 400.
        values[6, :] = [-6 + 2sin(i/31) for i in 1:n]
        values[12, :] = [2 + .2cos(i/41) for i in 1:n]
        values[15, :] = [-70 + 10sin(i/23) for i in 1:n]
        rawpath = joinpath(directory, "generated-hourly.lst")
        open(rawpath, "w") do io
            for i in 1:n
                tag = origin + Hour(i-1)
                fields = fill("0", 57)
                fields[1:3] .= string.((year(tag), dayofyear(tag), hour(tag)))
                for (channel, column) in enumerate(F.DstIntervalAdapter.Features.RAW_COLUMNS)
                    fields[column] = repr(values[channel, i])
                end
                println(io, join(fields, " "))
            end
        end
        digest = bytes2hex(sha256(read(rawpath)))
        input = F.read_author_hourly(rawpath; expected_sha256=digest)
        @assert input.values == values
        for delay in (0, 1)
            rows = NamedTuple[]
            streamed = F.stream_basis(input; delay_hours=delay, taus=(4., 24.), sigma=3.,
                chunk_hours=24, sink=chunk -> append!(rows, chunk))
            @assert length(rows) == streamed.row_count == n
            @assert streamed.max_buffered_rows <= 24
            # The recipe and fitting boundary are explicit; no method ranking
            # or real-data selection is performed by this example.
            model = F.fit_gains(rows, origin+Hour(700), F.settings()[9];
                delay_hours=delay, source_sha256=digest)
            predictions = F.predict_rows(model, rows)
            @assert length(predictions) == n
            saved = F.save_bundle(joinpath(directory, "bundle-L$delay"), model, streamed.checkpoint;
                provenance=(synthetic_fixture=true, input_sha256=digest,
                    scope="generated hourly library example"))
            loaded = F.load_bundle(saved.directory;
                expected_manifest_sha256=saved.manifest_sha256,
                expected_sources=F.DstIntervalArtifacts.verify_source_identity())
            @assert isequal(loaded.model, model)
            @assert isequal(loaded.checkpoint, streamed.checkpoint)
            @assert isequal(F.predict_rows(loaded.model, rows), predictions)
        end
    end
    @assert F.source_identity() == sources
    println("Interval-mean forecaster: synthetic parser/stream/fit/predict/bundle checks passed (L0 and L1).")
    nothing
end
