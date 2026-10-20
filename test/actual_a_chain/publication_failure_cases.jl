using Test, Dates, SHA, TOML, JSON3, LinearAlgebra, SolarSINDy
BLAS.set_num_threads(1)
const PUBLICATION_ORACLE=joinpath(@__DIR__,"frozen","publication_failure_oracle_v1.json")
@test bytes2hex(sha256(read(PUBLICATION_ORACLE)))=="8a2ca3eecf23559b5c79338c1469c91b77ccbc79ccf925007e977b80d1cbb2fd"
length(ARGS)==5 || error("PROJECT EXPORT EXPORT_SHA256 AUDIT_SHA256 NEW_OWNED_STAGE")
include(joinpath(ARGS[1],"tools","actual_a_chain","ActualAChain.jl"))
const C=ActualAChain
const B=C.InitialFits.B
const EXPORT=realpath(ARGS[2])
const STAGE=abspath(ARGS[5])
!ispath(STAGE) && isdir(dirname(STAGE)) || error("exclusive stage output required")
reference=C.load_reference(EXPORT,ARGS[3];audit_completion_sha256=ARGS[4])
C.ReferenceImport.verified_metadata(reference).meta["evidence_mode"]=="synthetic_only" || error("generated control required")
before=C.ReferenceImport.file_inventory(EXPORT)
# This single-use process adds a narrower dispatch method; the original generic
# method performs every byte write. Only the new owned stage marker triggers a
# controlled exception after that write has completed. No tracked source edit.
@eval B begin
    const CONTROLLED_COMPLETION_PATH=$(joinpath(STAGE,"completion.json"))
    function _write_new(path::String,bytes::Vector{UInt8})
        result=invoke(_write_new,Tuple{Any,Any},path,bytes)
        path==CONTROLLED_COMPLETION_PATH && error("controlled completion publication failure")
        result
    end
end
failure=try
    Base.invokelatest(C.run_initial_l0,reference,STAGE;chunk_hours=24)
    nothing
catch error
    error
end
@testset "Complete generated output is quarantined when publication fails after writing" begin
    @test failure isa ErrorException
    @test sprint(showerror,failure)=="controlled completion publication failure"
    @test !ispath(joinpath(STAGE,"completion.json"))
    @test isfile(joinpath(STAGE,"completion.json.incomplete"))
    receipt=JSON3.read(read(joinpath(STAGE,"completion.json.incomplete")),Dict{String,Any})
    @test receipt["complete"]===true
    @test receipt["evidence_mode"]=="synthetic_only"
    @test receipt["initial_fit_count"]==18
    @test receipt["state_schedule_count"]==9
    failed=TOML.parsefile(joinpath(STAGE,"failed.toml"))
    @test failed["complete"]===false
    @test failed["completed_initial_fits"]==18
    @test failed["completed_state_schedules"]==9
    @test failed["error"]=="controlled completion publication failure"
    @test C.ReferenceImport.file_inventory(EXPORT)==before
    @test_throws ArgumentError C.verify_initial_l0(STAGE,bytes2hex(sha256(read(joinpath(STAGE,"completion.json.incomplete")))),reference)
end
println("CONTROLLED_PUBLICATION_FAILURE_REJECTED=true")
