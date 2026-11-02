using Test, SHA, TOML
isdefined(Main, :CMEGSMBootstrap) || include(joinpath(@__DIR__, "bootstrap_indices.jl"))
const B = Main.CMEGSMBootstrap

@testset "Independent literal writer and validation oracles" begin
    # Literal fixture predates implementation. It is not a Julia RNG output.
    expected = UInt8[0x32,0x2c,0x31,0x0a,0x31,0x2c,0x32,0x0a,0x32,0x2c,0x32,0x0a]
    expected_sha = "4dc41bae5c2d906887e2b55d8586d810afa33f2f4ed7dea49f83dfa419321fb2"
    @test bytes2hex(sha256(expected)) == expected_sha
    picks = (2,1,1,2,2,2); seen = Ref(0)
    picker = function(range)
        @test range == 1:2
        seen[] += 1
        picks[seen[]]
    end
    io = IOBuffer(); B.write_index_csv(io, 2, 3, picker)
    @test seen[] == 6
    @test take!(io) == expected
    @test_throws ArgumentError B.write_index_csv(IOBuffer(), 0, 3, _ -> 1)
    @test_throws ArgumentError B.write_index_csv(IOBuffer(), 2, 0, _ -> 1)
    @test_throws ArgumentError B.write_index_csv(IOBuffer(), 2, 3, _ -> 0)
    @test_throws ArgumentError B.write_index_csv(IOBuffer(), 2, 3, _ -> 3)
    @test_throws ArgumentError B.write_index_csv(IOBuffer(), 2, 3, _ -> true)
    mktempdir() do temporary
        file = joinpath(temporary, "literal.csv")
        B.exclusive_bytes(file, expected)
        @test B.csv_values(file, 2, 3; expected_sha256=expected_sha) == [2 1; 1 2; 2 2]
        @test_throws Base.IOError B.exclusive_bytes(file, UInt8[0x31])
        @test read(file) == expected
        @test_throws ArgumentError B.csv_values(file, 2, 2; expected_sha256=expected_sha)
        @test_throws ArgumentError B.csv_values(file, 3, 3; expected_sha256=expected_sha)
        # Row/slot order changes are rejected against the independent byte pin.
        order = joinpath(temporary, "order.csv"); write(order, "1,2\n2,1\n2,2\n")
        @test_throws ArgumentError B.csv_values(order, 2, 3; expected_sha256=expected_sha)
        for (index, text) in enumerate(("2,0\n1,2\n2,2\n", "2,3\n1,2\n2,2\n", "02,1\n1,2\n2,2\n",
            "2,1\r\n1,2\r\n2,2\r\n", "2,1\n1,2\n2,2", "a,b\n2,1\n1,2\n2,2\n"))
            bad = joinpath(temporary, "bad-$index.csv"); write(bad, text)
            # A changed byte pin does not make bad domain/geometry/format valid.
            @test_throws ArgumentError B.csv_values(bad, 2, 3; expected_sha256=bytes2hex(sha256(codeunits(text))))
        end
    end
end

@testset "All551 IDs and pre-value year-universe lineage" begin
    population = B.load_population()
    @test length(population.ids) == 551
    @test population.years == Tuple(1995:2025)
    @test population.K == 31
    data = TOML.parsefile(B.DEFAULT_INPUT)
    reordered = deepcopy(data); reverse!(reordered["year_universe"])
    @test_throws ArgumentError B.population_values(reordered)
    omitted = deepcopy(data); pop!(omitted["year_universe"])
    @test_throws ArgumentError B.population_values(omitted)
    wrongyear = deepcopy(data); wrongyear["event"][1]["utc_year"] -= 1
    @test_throws ArgumentError B.population_values(wrongyear)
    duplicate = deepcopy(data); duplicate["event"][2]["id"] = duplicate["event"][1]["id"]
    @test_throws ArgumentError B.population_values(duplicate)
    wrongorder = deepcopy(data); reverse!(wrongorder["event"])
    @test_throws ArgumentError B.population_values(wrongorder)
    missing = deepcopy(data); pop!(missing["event"])
    @test_throws ArgumentError B.population_values(missing)
    mktempdir() do temporary
        changed = joinpath(temporary, "changed.toml")
        write(changed, read(B.DEFAULT_INPUT, String)*"\n")
        @test_throws ArgumentError B.load_population(changed)
    end
end

println("BOOTSTRAP_LITERAL_AND_POPULATION_ORACLES_PASSED;ACTUAL_RNG_HASH_NOT_PREDECLARED")
