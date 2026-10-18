@testset "canonical fitting rejects fractional-hour boundaries before solving" begin
    for delay in (0,1)
        setting = DF.settings()[1]
        rows = DISK_METADATA.metadata_rows(setting,delay)
        boundary = DISK_METADATA.MBOUNDARY
        valid = DF.fit_gains(rows,boundary,setting;
            delay_hours=delay,source_sha256=DISK_METADATA.MSOURCE)
        @test DF.DstIntervalArtifacts._validate_model(valid) === valid
        for (slot,delta) in enumerate((-Millisecond(1),Millisecond(1),Minute(30)))
            bad = boundary+delta
            failure = try
                DF.fit_gains(rows,bad,setting;
                    delay_hours=delay,source_sha256=DISK_METADATA.MSOURCE)
                nothing
            catch error
                error
            end
            @test failure isa ArgumentError
            @test occursin("exact hour",sprint(showerror,failure))
            # Affine export retains its diagnostic-boundary semantics. Its
            # internal mathematical solve is not a canonical model constructor.
            input = DF.export_fit_input(joinpath(DISK_OUTPUT,"boundary-L$delay-$slot"),
                bad,setting;delay_hours=delay,source_sha256=DISK_METADATA.MSOURCE) do sink
                sink(rows)
            end
            diagnostic = DT._solve_fit_input(input.directory;
                expected_manifest_sha256=input.manifest_sha256)
            @test diagnostic.boundary == bad
            @test diagnostic.schema == "disk_affine_gain_research_result_v1"
            data = joinpath(input.directory,"joint-fit.f64")
            manifest = joinpath(input.directory,"manifest.toml")
            original_data = bytes2hex(sha256(read(data)))
            original_manifest = bytes2hex(sha256(read(manifest)))
            function reject_canonical_input()
                failure = try
                    DF.fit_gains_from_input(input.directory;
                        expected_manifest_sha256=input.manifest_sha256)
                    nothing
                catch error
                    error
                end
                @test failure isa ArgumentError
                @test occursin("exact hour",sprint(showerror,failure))
            end
            reject_canonical_input()
            @test bytes2hex(sha256(read(data))) == original_data
            @test bytes2hex(sha256(read(manifest))) == original_manifest
            # Make the data file unreadable as a record by replacing its owned
            # path with a directory. The same boundary error must occur first.
            preserved = data*".preserved"
            mv(data,preserved)
            mkdir(data)
            reject_canonical_input()
            @test bytes2hex(sha256(read(preserved))) == original_data
            @test bytes2hex(sha256(read(manifest))) == original_manifest
        end
    end
end
