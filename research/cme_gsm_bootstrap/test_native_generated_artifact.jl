using Test, Random, SHA, TOML

function test_native_generated_artifact(result; role, source_head)
    B = Main.CMEGSMBootstrap
    @testset "Real native matrix versus independent scalar RNG/byte oracle" begin
        population = B.load_population()
        rng = Random.MersenneTwister(170281)
        observed = 0
        open(joinpath(result.directory, "BOOTSTRAP_INDICES.csv"), "r") do stream
            for draw in 1:10000
                # This independent oracle serializes an entire tuple, rather
                # than calling the production writer or production CSV parser.
                row = ntuple(_ -> rand(rng, 1:31), 31)
                @test readline(stream; keep=true) == join(row, ',')*"\n"
                observed += 1
            end
            @test eof(stream)
        end
        @test observed == 10000 && population.K == 31
        @test B.validate_artifact(result.directory; expected_source_head=source_head,
            expected_matrix_sha256=result.matrix_sha256).matrix_sha256 == result.matrix_sha256
        @test_throws ArgumentError B.generate(result.directory; role, source_head, expected_runtime=VERSION)
        @test bytes2hex(sha256(read(joinpath(result.directory, "BOOTSTRAP_INDICES.csv")))) == result.matrix_sha256
        mktempdir() do temporary
            function changed_copy(mutate, name)
                destination = joinpath(temporary, name); mkdir(destination)
                for member in readdir(result.directory)
                    cp(joinpath(result.directory, member), joinpath(destination, member))
                end
                mutate(destination)
                @test_throws ArgumentError B.validate_artifact(destination; expected_source_head=source_head,
                    expected_matrix_sha256=result.matrix_sha256)
            end
            function change_manifest(destination, change)
                path = joinpath(destination, "INDEX_MANIFEST.toml")
                data = TOML.parsefile(path); change(data)
                open(path, "w") do io; TOML.print(io, data; sorted=true); end
                marker = TOML.parsefile(joinpath(destination, "COMPLETED.toml"))
                marker["manifest_sha256"] = bytes2hex(sha256(read(path)))
                open(joinpath(destination, "COMPLETED.toml"), "w") do io; TOML.print(io, marker; sorted=true); end
            end
            changed_copy("year-order") do destination
                change_manifest(destination, data -> reverse!(data["year_universe"]))
            end
            changed_copy("year-omission") do destination
                change_manifest(destination, data -> pop!(data["year_universe"]))
            end
            changed_copy("wrong-source") do destination
                change_manifest(destination, data -> (data["claimed_source_head"] = repeat("0", 40)))
            end
            changed_copy("matrix-tamper") do destination
                path = joinpath(destination, "BOOTSTRAP_INDICES.csv")
                bytes = read(path); bytes[1] = bytes[1] == 0x31 ? 0x32 : 0x31; write(path, bytes)
            end
            changed_copy("input-tamper") do destination
                open(joinpath(destination, "POPULATION_YEARS.toml"), "a") do io; write(io, '\n'); end
            end
        end
    end
    println("BOOTSTRAP_NATIVE_RNG_AND_ARTIFACT_TAMPER_ORACLES_PASSED")
end
