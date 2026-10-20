using Test, SHA, JSON3
const PERMIT_ORACLE_PATH=joinpath(@__DIR__,"frozen","permit_grammar_contract_v1.json")
@test bytes2hex(sha256(read(PERMIT_ORACLE_PATH)))=="099073f290056d2a00806bac6a580521cb013f00bebb19a5e65958415ed37480"
const PERMIT_ORACLE=JSON3.read(read(PERMIT_ORACLE_PATH),Dict{String,Any})
function grammar_fixture()
    plan=Dict{String,Any}("schema"=>"qualified_actual_initial_a_l0_permit_v1","complete"=>true,"stage"=>"initial_l0",
        "implementation_sources"=>Dict("grammar-only"=>repeat("a",64)),
        "reference_completion_sha256"=>repeat("b",64),"independent_audit_completion_sha256"=>repeat("c",64),
        "source_sha256"=>C.Inputs.ACTUAL_SOURCE,"admitted_input_seal"=>repeat("d",64),
        "initial_fit_count"=>18,"delay_hours"=>0,"final_fit_count"=>0,
        "test_scoring"=>false,"scientific_verdict"=>false,"serving_promotion"=>false,
        "qualification_file"=>abspath(joinpath(dirname(only(ARGS)),"grammar-only-never-written.json")),
        "qualification_sha256"=>repeat("e",64),"julia_version"=>string(VERSION),"julia_binary_sha256"=>repeat("f",64),
        "training_boundary"=>"2020-01-01T00:00:00","validation_start"=>"2020-01-01T00:00:00","validation_end"=>"2023-01-01T00:00:00")
    @test sort!(collect(keys(plan)))==PERMIT_ORACLE["fields"]
    plan
end
@testset "Pure stage grammar rejects scope ambiguity and grants no authority" begin
    plan=grammar_fixture()
    @test C.InitialFits.validate_permit_scope(plan)
    @test !ispath(plan["qualification_file"])
    for field in PERMIT_ORACLE["fields"]
        altered=deepcopy(plan);delete!(altered,field)
        @test_throws ArgumentError C.InitialFits.validate_permit_scope(altered)
    end
    altered=deepcopy(plan);altered["selected_final_refit"]=true
    @test_throws ArgumentError C.InitialFits.validate_permit_scope(altered)
    for (field,values) in (("initial_fit_count",(true,18.0,17,19)),("delay_hours",(false,0.0,1)),
            ("final_fit_count",(false,0.0,1)),("complete",(1,false)),("stage",("final_l0","initial_l1")),
            ("training_boundary",("2020-01-01T00:00:00.001","2021-01-01T00:00:00")),
            ("validation_start",("2020-01-01T01:00:00",)),("validation_end",("2024-01-01T00:00:00",)),
            ("qualification_file",("relative-proof.json",)),("source_sha256",(repeat("0",64),)))
        for value in values
            altered=deepcopy(plan);altered[field]=value
            @test_throws ArgumentError C.InitialFits.validate_permit_scope(altered)
        end
    end
    for field in PERMIT_ORACLE["forbidden_true"],value in (true,0)
        altered=deepcopy(plan);altered[field]=value
        @test_throws ArgumentError C.InitialFits.validate_permit_scope(altered)
    end
end
