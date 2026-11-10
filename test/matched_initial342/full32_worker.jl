module MatchedInitial342Full32Worker
using Test,Dates,SHA,JSON3,SolarSINDy
using LinearAlgebra: BLAS
include(joinpath(pkgdir(SolarSINDy),"tools","actual_b_chain","BMatchedReference342.jl"))
const Entry=BMatchedReference342
const FC=Entry.BMatched342Core
const I=Entry.BI
const P=Entry.P
const M=Entry.M
include("fixture.jl")
include("producer_audit_tests.jl")
function main()
    length(ARGS)==2||error("Full32 worker requires a pinned ordinary request")
    request_path,request_sha=ARGS
    isabspath(request_path)&&isfile(request_path)&&!islink(request_path)&&filesize(request_path)<=2*1024^2||error("Invalid full32 request path")
    body=P._bound(request_path,2*1024^2);bytes2hex(sha256(body))==request_sha||error("Full32 request pin differs")
    request=P._bounded_json_parse(body)
    @assert Set(String.(keys(request)))==Set(["schema","package_root","active_project","active_project_sha256","active_manifest_sha256",
        "depot_path","source_proof","output_root","result_path","original_head","projected"])
    @assert request.schema=="matched342_full32_owned_test_request_v1"
    @assert realpath(pkgdir(SolarSINDy))==request.package_root
    @assert realpath(Base.active_project())==request.active_project
    selected=Base.project_file_manifest_path(String(Base.active_project()))
    @assert selected!==nothing
    P._ordinary(selected)
    @assert realpath(selected)==realpath(joinpath(dirname(request.active_project),"Manifest.toml"))
    @assert bytes2hex(sha256(P._bound(request.active_project,2*1024^2)))==request.active_project_sha256
    @assert bytes2hex(sha256(P._bound(joinpath(dirname(request.active_project),"Manifest.toml"),2*1024^2)))==request.active_manifest_sha256
    @assert abspath.(DEPOT_PATH)==abspath.(String.(request.depot_path))
    @assert Threads.nthreads()==1
    BLAS.set_num_threads(1)
    @assert get(ENV,"OMP_NUM_THREADS","")=="1"
    proof_body=P._bound(request.source_proof.path,2*1024^2)
    @assert bytes2hex(sha256(proof_body))==request.source_proof.sha256
    proof=P._bounded_json_parse(proof_body)
    FC.recheck_source_proof(request.source_proof;expected_head=proof.head,expected_sources=Entry.source_identity())
    @testset "Full32 nonlinear and flat unchanged producer/auditor/B22 cases" begin
        for flat in (false,true)
            owned_case(request.output_root,flat ? "full32-flat" : "full32-nonlinear") do root
                full_producer_audit_checks(root;flat)
            end
        end
    end
    @assert bytes2hex(sha256(P._bound(request.active_project,2*1024^2)))==request.active_project_sha256
    @assert bytes2hex(sha256(P._bound(joinpath(dirname(request.active_project),"Manifest.toml"),2*1024^2)))==request.active_manifest_sha256
    FC.recheck_source_proof(request.source_proof;expected_head=proof.head,expected_sources=Entry.source_identity())
    result=Dict("schema"=>"matched342_full32_owned_test_worker_v1","complete"=>true,"evidence_mode"=>"synthetic_only",
        "julia_pid"=>getpid(),"cases"=>["nonlinear","flat"],"fit_count"=>64,"nonlinear_complete"=>true,"flat_complete"=>true,
        "package_root"=>request.package_root,"head"=>proof.head,"source_proof_sha256"=>request.source_proof.sha256,
        "active_project_sha256"=>request.active_project_sha256,"active_manifest_sha256"=>request.active_manifest_sha256)
    b2_write(request.result_path,result)
end
end
MatchedInitial342Full32Worker.main()
