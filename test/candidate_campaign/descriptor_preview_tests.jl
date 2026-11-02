"""Untrusted/generated metadata; no archival source, fit or native execution."""
module CandidateCampaignDescriptorTests
using SolarSINDy,Test,Dates,SHA,JSON3
include(joinpath(@__DIR__,"..","..","research","dbdt_completed_source","run_candidate_descriptor_preview.jl"))
const P=CandidateDescriptorNativePreview
const N=P.N
const R=N.R
include(joinpath(@__DIR__,"..","candidate_validation","generated_fixture.jl"))
literal(path)=JSON3.read(read(path,String),Dict{String,Any})
const L=literal(joinpath(@__DIR__,"descriptor_literals.json"))
const OLD=literal(joinpath(@__DIR__,"..","selected_point_bindings","full_codec_oracles.json"))["decisions"]["R2"]["request"]
function facts(station,row)
    Dict{String,Any}("operation"=>"fixed_candidate_2017_fit_input_only",
        "case_request_sha256"=>repeat("1",64),"case_manifest_sha256"=>repeat("2",64),"case_complete_sha256"=>repeat("3",64),
        "table_index"=>station=="CMO" ? 1 : 2,"station"=>station,"role"=>"primary","candidate_id"=>row["candidate_key"],
        "recipe_sha256"=>row["point_recipe_sha256"],"grid_sha256"=>L["R2_grid_sha256"],
        "split"=>Dict("start_inclusive"=>"2017-01-01T00:00:00.000Z","stop_exclusive"=>"2017-07-01T00:00:00.000Z",
            "target_asof"=>"2018-01-01T00:00:00.000Z","first_evaluation_issue"=>"2017-07-01T00:00:00.000Z"),
        "runtime"=>string(VERSION),"policy_sha256"=>repeat("4",64),"job_sha256"=>repeat("5",64),"point_source_sha256"=>repeat("6",64),
        "actual_source_access"=>false,"case_read_admission_sha256"=>nothing,"selection_sha256"=>nothing,"point"=>nothing)
end
@testset "All16 frozen recipe literals and both stations preserve descriptor fields" begin
    @test OLD["grid_sha256"]==L["R2_grid_sha256"]
    for row in OLD["producer_expectations"],station in ("CMO","FRD")
        known=facts(station,row);saved=deepcopy(known);out=N.descriptor_from_metadata(known)
        @test known==saved
        @test length(out)==23 && Set(keys(out))==Set(L["descriptor_keys"])
        @test all(out[k]==v for (k,v) in saved)
        @test out["schema"]==L["descriptor_schema"] && out["layout"]==L["layout"]
        @test out["phase_period_minutes"]==L["phase_period_minutes"] && out["work_budget"]==L["work_budget"]
        @test bytes2hex(sha256(N.canonical(out)))==L["canonical_descriptors"][station*"/"*row["candidate_key"]][string(VERSION)]
        @test N.recipe_for(row["candidate_key"]).sha256==row["point_recipe_sha256"]
        out["split"]["target_asof"]="2099-01-01T00:00:00.000Z"
        @test known==saved
    end
    known=facts("CMO",first(OLD["producer_expectations"]))
    for (key,bad) in (("station","OTHER"),("table_index",true),("role","holdout"),("runtime","1.14.0"),
                      ("grid_sha256",repeat("0",64)),("recipe_sha256",repeat("0",64)),("candidate_id","ridge-999"))
        changed=deepcopy(known);changed[key]=bad
        @test_throws ArgumentError N.descriptor_from_metadata(changed)
    end
end
@testset "Real generated Q.VerifiedCases no-fit production descriptor parity" begin
    mktempdir() do temporary
        root=realpath(temporary);cases=CandidateValidationGeneratedFixture.make_cases(root)
        @test cases isa N.Q.VerifiedCases && !cases.actual_source_authority_granted && !cases.actual_fit_performed
        split=N.CandidateSplit(DateTime(2020)+Minute(40),DateTime(2020)+Minute(310),DateTime(2020)+Minute(681))
        clock(t)=Dates.format(t,dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
        recipe_pins=Dict(row["candidate_key"]=>row["point_recipe_sha256"] for row in OLD["producer_expectations"])
        for (index,station) in enumerate(("CMO","FRD")),spec in N.S.POINT_GRID
            expected=Dict{String,Any}("schema"=>"candidate_known_input_descriptor_v1","operation"=>"fixed_candidate_2017_fit_input_only",
                "case_request_sha256"=>cases.request_sha256,"case_manifest_sha256"=>cases.manifest_sha256,"case_complete_sha256"=>cases.complete_sha256,
                "table_index"=>index,"station"=>station,"role"=>"primary","layout"=>"ground11","candidate_id"=>spec.id,
                "recipe_sha256"=>recipe_pins[spec.id],"grid_sha256"=>L["R2_grid_sha256"],"split"=>Dict(
                    "start_inclusive"=>clock(split.start),"stop_exclusive"=>clock(split.stop),"target_asof"=>clock(split.asof),"first_evaluation_issue"=>clock(split.first_issue)),
                "runtime"=>string(VERSION),"phase_period_minutes"=>31,"policy_sha256"=>N.policy_identity(),"job_sha256"=>N.job_identity(),
                "point_source_sha256"=>N.F.source_identity(),"actual_source_access"=>false,"case_read_admission_sha256"=>nothing,
                "selection_sha256"=>nothing,"point"=>nothing,"work_budget"=>L["work_budget"])
            actual=N.descriptor(cases,index,spec.id,split;operation="fixed_candidate_2017_fit_input_only")
            @test N.canonical(actual)==N.canonical(expected)
            known=Dict(k=>v for (k,v) in expected if !(k in ("schema","layout","phase_period_minutes","work_budget")))
            @test N.canonical(N.descriptor_from_metadata(known))==N.canonical(expected)
        end
        @test !cases.actual_source_authority_granted && !cases.actual_fit_performed
    end
end
function write_meta(root,name,value)
    path=joinpath(root,name);bytes=R.canonical_bytes(value;limit=1048576);R._exclusive_write(path,bytes)
    Dict("path"=>path,"sha256"=>N.digest(bytes))
end
function literal_bundle(root)
    table=Dict("id"=>"literal-CMO","station"=>"CMO","start_inclusive"=>"2017-01-01T00:00:00.000Z",
        "stop_exclusive"=>"2017-10-01T00:00:00.000Z","label_asof"=>"2018-01-01T00:00:00.000Z")
    request=Dict("schema"=>"archive_catalog_case_request_v1","evidence_mode"=>"actual_archival_cache","layout"=>"ground-only",
        "role"=>"primary","runtime"=>string(VERSION),"source"=>Dict("fixture_only"=>true),"tables"=>[table])
    req=write_meta(root,"request.json",request)
    manifest=write_meta(root,"manifest.json",Dict("schema"=>"archive_catalog_case_artifact_v1","request_sha256"=>req["sha256"],
        "layout"=>"ground-only","role"=>"primary","runtime"=>string(VERSION),"tables"=>[Dict("declaration"=>table)]))
    complete=write_meta(root,"COMPLETE.json",Dict("schema"=>"archive_catalog_case_complete_v1","manifest_sha256"=>manifest["sha256"]))
    admission=write_meta(root,"admission.json",Dict("schema"=>"archive_catalog_case_read_admission_v1","evidence_mode"=>"actual_archival_cache",
        "request_sha256"=>req["sha256"],"artifact_manifest_sha256"=>manifest["sha256"],"artifact_complete_sha256"=>complete["sha256"]))
    Dict{String,Any}("stage_key"=>"CMO/ridge-1/candidate-input","stage_operation"=>"candidate-input","candidate_id"=>"ridge-1",
        "table_index"=>1,"station"=>"CMO","case_request"=>req,"manifest"=>manifest,"complete"=>complete,"admission"=>admission,
        "point"=>nothing,"selection_sha256"=>nothing,"selection"=>nothing)
end
function add_point_metadata(root,bundle)
    value=deepcopy(bundle);recipe=N.recipe_for("ridge-1");source=N.F.source_identity()
    context=Dict("case_request_sha256"=>value["case_request"]["sha256"],"table_index"=>1,"table"=>Dict("station"=>"CMO"),
        "recipe_sha256"=>recipe.sha256,"source_sha256"=>source,"start_inclusive"=>"2017-01-01T00:00:00.000Z",
        "stop_exclusive"=>"2017-07-01T00:00:00.000Z","rows"=>1000)
    c=write_meta(root,"point-context.json",context);model=repeat("7",64)
    receipt=write_meta(root,"point-receipt.json",Dict("context_sha256"=>c["sha256"],"model_sha256"=>model,"recipe_sha256"=>recipe.sha256,
        "source_sha256"=>source,"actual_source_access"=>true,"actual_fit_performed"=>true,"fixture_only"=>true))
    descriptor=Dict("context_sha256"=>c["sha256"],"recipe_sha256"=>recipe.sha256,"model_sha256"=>model,
        "receipt_sha256"=>receipt["sha256"],"source_sha256"=>source)
    value["point"]=Dict("descriptor"=>descriptor,"context"=>c,"receipt"=>receipt)
    value["stage_key"]="CMO/ridge-1/candidate-validation";value["stage_operation"]="candidate-validation";value
end
function add_selection_metadata(root,bundle)
    value=deepcopy(bundle);evidence=repeat("8",64)
    choice=Dict("station"=>"CMO","selection_evidence_sha256"=>evidence,"family_winners"=>[
        Dict("family"=>"ridge","candidate_id"=>"ridge-1"),Dict("family"=>"boost","candidate_id"=>"boost-3-100"),
        Dict("family"=>"forest","candidate_id"=>"forest-9-64")])
    result=Dict("operation"=>"select-R2","complete"=>true,"selection"=>choice,"fixture_only"=>true,"native_execution_evidence"=>false)
    r=write_meta(root,"selection-result.json",result)
    owner=write_meta(root,"selection-owner.json",Dict("schema"=>"candidate_native_stage_evidence_v1","complete"=>true,"operation"=>"select-R2",
        "internal_result_sha256"=>r["sha256"],"outputs"=>result,"fixture_only"=>true,"actual_authority"=>false,"native_execution_evidence"=>false))
    value["selection"]=Dict("family"=>"ridge","result"=>r,"receipt"=>owner);value["selection_sha256"]=evidence
    value["stage_key"]="CMO/ridge/selected-refit-input";value["stage_operation"]="selected-refit-input";value
end
function run_literal_fixture()
    mktempdir() do temporary
        root=realpath(temporary);bundle=literal_bundle(root);result=P.preview_known_metadata(bundle)
        @test result["descriptor_only"]===true && result["metadata_unverified"]===true
        @test result["authority_granted"]===false && result["actual_source_bodies_read"]===false && result["actual_fits"]==0
        @test result["descriptor_sha256"]==N.digest(N.canonical(result["descriptor"]))
        point=add_point_metadata(root,bundle);checked=P.preview_known_metadata(point)
        @test checked["descriptor"]["point"]==point["point"]["descriptor"]
        wrong=deepcopy(point);wrong["point"]["descriptor"]["source_sha256"]=repeat("0",64)
        @test_throws ArgumentError P.preview_known_metadata(wrong)
        selected=add_selection_metadata(root,bundle);value=P.preview_known_metadata(selected)
        @test value["descriptor"]["selection_sha256"]==repeat("8",64)
        wrong=deepcopy(selected);wrong["selection_sha256"]=repeat("9",64)
        @test_throws ArgumentError P.preview_known_metadata(wrong)
        for (key,bad) in (("stage_key","FRD/ridge-1/candidate-input"),("station","FRD"),("table_index",2),
            ("stage_operation","candidate-fit"),("candidate_id","ridge-10"))
            wrong=deepcopy(bundle);wrong[key]=bad
            @test_throws ArgumentError P.preview_known_metadata(wrong)
        end
        (;descriptor_only=true,metadata_unverified=true,actual_source_bodies_read=false,actual_fits=0,authority_granted=false)
    end
end
@testset "Pinned untrusted header/point/selection metadata, no numerical body" begin
    run_literal_fixture()
end
end
