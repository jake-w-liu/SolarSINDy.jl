module PaperSourceSchemasIndependentTests

using Test
using CSV
using DataFrames
using SHA

include(joinpath(@__DIR__, "..", "tools", "paper_source_schemas", "SuperlearnerAnnualSchema.jl"))
using .SuperlearnerAnnualSchema

const LEGACY = ["model_step_hours", "observation_dst_nt", "depth_bin", "v2_4e",
    "v2_4e_lo_nt", "v2_4e_hi_nt", "static_v2_2", "v2_3_shadow", "direct_gbm", "served_v2_1"]
const CANONICAL = ["model_step_hours", "observation_dst_nt", "depth_bin", "superlearner_e",
    "superlearner_e_lo_nt", "superlearner_e_hi_nt", "static_stack", "analog_shadow",
    "direct_gbm", "served_inertial_sindy"]

function literal_csv(path, headers=LEGACY)
    # Quoting is deliberate: duplicate raw names must survive the header probe.
    open(path, "w") do io
        println(io, join(["\"" * replace(h, "\"" => "\"\"") * "\"" for h in headers], ','))
        println(io, "6,-80,deep,-79,-95,-60,-76,-77,-75,-78")
        println(io, "1,-10,shallow,-9,-20,0,-8,-7,-6,-5")
    end
    return path
end

raw_pin(path) = (filesize(path), open(io -> bytes2hex(sha256(io)), path))

function annual_fixture(directory; headers=LEGACY)
    expected = Dict{String,Tuple{Int,String}}()
    for year in 2014:2025
        name = "learn_year_$(year).csv"
        expected[name] = raw_pin(literal_csv(joinpath(directory, name), headers))
    end
    return expected
end

@testset "Literal annual schema, raw-header ambiguity and rename identity" begin
    @test header_plan(LEGACY).raw_names == Tuple(LEGACY)
    @test header_plan(LEGACY).canonical_names == Tuple(CANONICAL)
    @test header_plan(CANONICAL).canonical_names == Tuple(CANONICAL)
    mixed = ["direct_gbm", "v2_4e_hi_nt", "depth_bin", "static_stack", "v2_4e",
        "observation_dst_nt", "analog_shadow", "v2_4e_lo_nt", "model_step_hours",
        "served_inertial_sindy", "v2_4e_half_width_nt", "V2_4E"]
    @test header_plan(mixed).canonical_names == ("direct_gbm", "superlearner_e_hi_nt",
        "depth_bin", "static_stack", "superlearner_e", "observation_dst_nt", "analog_shadow",
        "superlearner_e_lo_nt", "model_step_hours", "served_inertial_sindy",
        "v2_4e_half_width_nt", "V2_4E")
    @test header_plan(mixed).selected_source_names == Tuple(mixed[1:10])
    @test_throws Exception header_plan(vcat(LEGACY, ["v2_4e"]))
    @test_throws Exception header_plan(vcat(LEGACY, ["superlearner_e"]))
    @test_throws Exception header_plan(vcat(LEGACY, ["extra", "extra"]))
    @test_throws Exception header_plan(vcat(LEGACY, [""]))
    for index in (4, 5, 6, 7, 8, 10)
        missing_name = copy(LEGACY)
        deleteat!(missing_name, index)
        @test_throws Exception header_plan(missing_name)
        near_match = copy(LEGACY)
        near_match[index] *= " "
        @test_throws Exception header_plan(near_match)
    end

    bits = UInt64[0x7ff8000000000042, 0x8000000000000000]
    frame = DataFrame([
        "depth_bin" => ["deep", "shallow"], "v2_4e_hi_nt" => [Inf, -Inf],
        "model_step_hours" => [6, 1], "v2_4e" => copy(reinterpret(Float64, bits)),
        "observation_dst_nt" => Union{Missing,Float64}[missing, -55.0],
        "v2_4e_lo_nt" => [-0.0, 0.0], "static_v2_2" => [true, false],
        "v2_3_shadow" => ["quoted,value", "z"], "direct_gbm" => [4, -3],
        "served_v2_1" => Union{Missing,Int}[2, missing], "extra" => ['x', 'y']]; copycols=false)
    buffers = collect(eachcol(frame))
    types = eltype.(buffers)
    original_values = deepcopy(buffers)
    @test adapt_annual_columns!(frame) === frame
    @test names(frame) == ["depth_bin", "superlearner_e_hi_nt", "model_step_hours", "superlearner_e",
        "observation_dst_nt", "superlearner_e_lo_nt", "static_stack", "analog_shadow",
        "direct_gbm", "served_inertial_sindy", "extra"]
    @test eltype.(eachcol(frame)) == types
    for index in eachindex(buffers)
        @test frame[!, index] === buffers[index]
        @test isequal(frame[!, index], original_values[index])
    end
    @test collect(reinterpret(UInt64, frame.superlearner_e)) == UInt64[0x7ff8000000000042, 0x8000000000000000]
    @test collect(reinterpret(UInt64, frame.superlearner_e_lo_nt)) == UInt64[0x8000000000000000, 0x0000000000000000]
    @test adapt_annual_columns!(frame) === frame

    mktempdir() do directory
        valid = literal_csv(joinpath(directory, "valid.csv"))
        @test read_header_plan(valid).canonical_names == Tuple(CANONICAL)
        crlf = joinpath(directory, "crlf.csv")
        write(crlf, replace(read(valid, String), "\n" => "\r\n"))
        @test read_header_plan(crlf).raw_names == Tuple(LEGACY)
        duplicate = literal_csv(joinpath(directory, "duplicate.csv"), vcat(LEGACY, ["v2_4e"]))
        @test_throws Exception read_header_plan(duplicate)
        for values in ("-79,-79", "-79,300")
            ambiguous = joinpath(directory, "ambiguous.csv")
            write(ambiguous, join(vcat(LEGACY, ["superlearner_e"]), ',') * "\n" *
                "6,-80,deep," * first(split(values, ',')) * ",-95,-60,-76,-77,-75,-78," * last(split(values, ',')) * "\n")
            @test_throws Exception read_header_plan(ambiguous)
        end
        empty_name = literal_csv(joinpath(directory, "empty.csv"), vcat(LEGACY, [""]))
        @test_throws Exception read_header_plan(empty_name)
        malformed = joinpath(directory, "malformed.csv")
        # Keep every required name: this must reject malformed quoting itself.
        write(malformed, join(LEGACY, ',') * ",\"unterminated\n6,7\n")
        @test_throws Exception read_header_plan(malformed)
    end
end

@testset "All twelve raw annuals and schemas precede consumer admission" begin
    for headers in (LEGACY, CANONICAL)
        mktempdir() do directory
            expected = annual_fixture(directory; headers=headers)
            plans = prepare_annual_inputs(directory, expected)
            @test plans isa Tuple
            @test Tuple(p.year for p in plans) == Tuple(2014:2025)
            @test length(plans) == 12
            @test all(p.raw_bytes == expected[basename(p.path)][1] &&
                p.raw_sha256 == expected[basename(p.path)][2] for p in plans)
            frame = read_annual_frame(plans[1])
            @test names(frame) == CANONICAL
            @test frame.model_step_hours == [6, 1]
            @test frame.observation_dst_nt == [-80, -10]
            @test frame.depth_bin == ["deep", "shallow"]
            @test frame.superlearner_e == [-79, -9]
            @test frame.superlearner_e_lo_nt == [-95, -20]
            @test frame.superlearner_e_hi_nt == [-60, 0]
            @test frame.static_stack == [-76, -8]
            @test frame.analog_shadow == [-77, -7]
            @test frame.direct_gbm == [-75, -6]
            @test frame.served_inertial_sindy == [-78, -5]
            # A plan is invalid after even a same-size raw edit.
            original = read(plans[1].path, String)
            write(plans[1].path, replace(original, "-79" => "-70"))
            @test_throws Exception read_annual_frame(plans[1])
        end
    end
    for defect in (:size, :sha, :missing, :header, :duplicate, :ambiguous, :incomplete, :bad_pin)
        mktempdir() do directory
            expected = annual_fixture(directory)
            last_name = "learn_year_2025.csv"
            last_path = joinpath(directory, last_name)
            if defect == :size
                expected[last_name] = (expected[last_name][1] + 1, expected[last_name][2])
            elseif defect == :sha
                expected[last_name] = (expected[last_name][1], repeat("0", 64))
            elseif defect == :missing
                rm(last_path)
            elseif defect in (:header, :duplicate, :ambiguous)
                headers = defect == :header ? LEGACY[1:9] :
                    vcat(LEGACY, [defect == :duplicate ? "v2_4e" : "superlearner_e"])
                literal_csv(last_path, headers)
                expected[last_name] = raw_pin(last_path)
            elseif defect == :incomplete
                delete!(expected, last_name)
            else
                expected[last_name] = (expected[last_name][1], repeat("G", 64))
            end
            writer_called = Ref(false)
            @test_throws Exception begin
                prepare_annual_inputs(directory, expected)
                writer_called[] = true
            end
            @test !writer_called[]
        end
    end
end

end # module
