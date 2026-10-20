using Test, Dates, SHA, TOML, JSON3, LinearAlgebra, SolarSINDy
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"..","..","tools","actual_a_chain","ActualAChain.jl"))
const C=ActualAChain
const X=C.InitialFits
const F=SolarSINDy.IntervalMeanForecaster
const A=F.DstIntervalAdapter
const S=F.DstIntervalStream
const T=F.DstIntervalTraining
const B=F.DstIntervalArtifacts
length(ARGS)==4 || error("EXPORT_ROOT EXPORT_SHA256 AUDIT_SHA256 OUTPUT")
oracle_path=joinpath(@__DIR__,"frozen","initial_stage_cohorts_v1.json")
@test bytes2hex(sha256(read(oracle_path)))=="fa0c46a40e8d1dd56178cdb068aed9db390a74588cefd7c410fcf92a76ac7b34"
const O=JSON3.read(read(oracle_path),Dict{String,Any})
reference=C.load_reference(ARGS[1],ARGS[2];audit_completion_sha256=ARGS[3])
frame=C.load_prefix(reference)
before=C.Inputs.input_digest(frame.input)
@testset "Stage boundaries reject invalid or existing destinations before fitting" begin
    rejected=ARGS[4]*"-rejected"
    for chunk in (0,4097,true,24.0)
        @test_throws ArgumentError C.run_initial_l0(reference,rejected;chunk_hours=chunk)
        @test !ispath(rejected)
    end
    permit=X.InitialExecutionPermit(abspath(joinpath(dirname(ARGS[4]),"not-an-actual-permit.json")),repeat("0",64))
    @test_throws ArgumentError C.run_initial_l0(reference,rejected;execution_permit=permit)
    @test !ispath(rejected)
    inside=joinpath(reference.directory,"forbidden-output")
    @test !ispath(inside)
    @test_throws ArgumentError C.run_initial_l0(reference,inside)
    @test !ispath(inside)
    existing=ARGS[4]*"-existing";mkdir(existing)
    write(joinpath(existing,"sentinel"),"preserve-existing-output")
    @test_throws ArgumentError C.run_initial_l0(reference,existing)
    @test read(joinpath(existing,"sentinel"),String)=="preserve-existing-output"
    @test readdir(existing)==["sentinel"]
    @test_throws ArgumentError C.verify_initial_l0(existing,repeat("0",64),reference)
end
measured=@timed C.run_initial_l0(reference,ARGS[4];chunk_hours=24)
result=measured.value
completion=JSON3.read(read(joinpath(result.directory,"completion.json")),Dict{String,Any})

@testset "Generated initial execution closes all eighteen models and nine schedules" begin
    @test result.initial_fit_count==18
    @test result.state_schedule_count==9
    @test completion["complete"]===true
    @test completion["evidence_mode"]=="synthetic_only"
    @test completion["initial_fit_count"]==18
    @test completion["state_schedule_count"]==9
    @test completion["final_fit_count"]==0
    @test completion["delay_hours"]==0
    @test completion["actual_execution_permit_sha256"]===nothing
    @test all(completion[key]===false for key in ("test_scoring","scientific_verdict","serving_promotion"))
    @test length(completion["models"])==18
    @test [m["index"] for m in completion["models"]]==collect(1:18)
    @test allunique(m["model_id"] for m in completion["models"])
    @test all(s["row_count"]==900 && s["reference_union_rows"]==226 && s["max_buffered_rows"]<=24 for s in completion["streams"])
    @test length(result.selection.panels)==18
    @test result.selection.status in (:selected,:none)
    @test C.Inputs.input_digest(frame.input)==before
end

base=C.ReferenceImport.reference_rows(reference)
initial=DateTime(2019)+Hour(500)
panels=Dict{Int,Any}()
@testset "Independent raw cohort and direct fit agree with persisted disk models" begin
    for setting in A.settings()[1:2:end]
        # Direct small generated replay is an independent test route. The
        # executor retains only its 226 union rows, never these full panels.
        rows=NamedTuple[]
        S.stream_basis(frame.input;delay_hours=0,taus=setting.taus,sigma=setting.sigma,
            chunk_hours=7,sink=chunk->append!(rows,chunk))
        joint=[i for i in eachindex(rows) if all(rows[i].scoreable) &&
            all(A.fit_eligible(rows[i].issue,rows[i].target_complete[j],rows[i].target_receipt[j],initial) for j in 1:2)]
        @test joint==O["joint_fit_origin_rows"]
        @test length(joint)==326
        @test [i for i in joint if rows[i].model_available]==O["model_available_joint_origin_rows"]
        @test count(i->!rows[i].model_available,joint)==167
        for index in (setting.index,setting.index+1)
            record=completion["models"][index]
            loaded=B.load_bundle(joinpath(result.directory,record["bundle"]);
                expected_manifest_sha256=record["bundle_manifest_sha256"],expected_sources=B.verify_source_identity())
            model=loaded.model
            direct=T.fit_gains(rows,initial,A.settings()[index];delay_hours=0,source_sha256=frame.input.source_sha256)
            @test isequal(model,direct)
            @test model.fit_count==326
            @test 100<=model.informative_count<=159
            @test model.storm_count==count(j->rows[j[1]].target[j[2]]<=-50,Iterators.product(joint,1:2))
            @test model.max_fit_issue==DateTime(2019)+Hour(326)
            @test model.max_target_complete==initial-Hour(168)
            @test model.max_label_receipt==initial-Hour(168)
            basis=X.read_basis(joinpath(result.directory,record["basis"]),record["basis_sha256"],base,A.settings()[index],model.source_sha256)
            @test length(basis)==226
            expected=[row for row in rows if row.issue in getproperty.(base,:issue)]
            @test isequal(basis,expected)
            predicted=T.predict_rows(loaded.model,basis)
            points=read(joinpath(result.directory,record["points"]))
            expected_bytes=IOBuffer()
            for row in predicted, point in row.prediction
                write(expected_bytes,htol(reinterpret(UInt64,point)))
            end
            @test points==take!(expected_bytes)
            panels[index]=predicted
        end
    end
    independent=T.select_setting(panels,reference;validation_start=DateTime(2019)+Hour(501),
        validation_end=DateTime(2019)+Hour(900),delay_hours=0)
    @test isequal(result.selection,independent)
    @test [m.n for m in first(independent.panels).pooled]==[224,222]
    @test [m.n for m in first(independent.panels).storm]==[93,91]
end

@testset "Pinned readback closes raw design, certificates, checkpoints and points without fitting" begin
    inventory=X.stage_inventory(result.directory).sha256
    closed=X.verify_initial_l0(result.directory,result.completion_sha256,reference;chunk_hours=7)
    @test closed.complete
    @test closed.models_checked==18
    @test closed.state_replays==9
    @test closed.new_gain_fits==0
    @test closed.final_refits==0
    @test closed.scientific_verdict===false
    @test isequal(closed.selection,result.selection)
    @test X.stage_inventory(result.directory).sha256==inventory
    @test_throws ArgumentError X.verify_initial_l0(result.directory,repeat("0",64),reference)
end

function refresh_stage(directory,metadata)
    marker=joinpath(directory,"completion.json")
    inventory=X.stage_inventory(directory)
    delete!(inventory.sha256,"completion.json")
    metadata["output_sha256"]=inventory.sha256
    metadata["output_bytes"]=inventory.bytes-filesize(marker)
    write(marker,JSON3.write(metadata))
    bytes2hex(sha256(read(marker)))
end

function owned_copy_case(change::Function,label;positive=false,expected_error=nothing)
    directory=abspath(joinpath(dirname(result.directory),"readback-"*label))
    @test dirname(directory)==abspath(dirname(result.directory)) && startswith(basename(directory),"readback-")
    @test !ispath(directory)
    cp(result.directory,directory;follow_symlinks=false)
    metadata=JSON3.read(read(joinpath(directory,"completion.json")),Dict{String,Any})
    change(directory,metadata)
    digest=refresh_stage(directory,metadata)
    before=X.stage_inventory(directory).sha256
    if positive
        checked=X.verify_initial_l0(directory,digest,reference)
        @test checked.complete && checked.new_gain_fits==0
        @test isequal(checked.selection,result.selection)
    else
        failure=try
            X.verify_initial_l0(directory,digest,reference)
            nothing
        catch error
            error
        end
        @test failure isa ArgumentError
        expected_error===nothing || @test occursin(expected_error,sprint(showerror,failure))
    end
    @test X.stage_inventory(directory).sha256==before
    # The path is a newly created, ordinary sibling of this generated fixture's
    # stage, and its complete bounded inventory was just checked above.
    @test realpath(dirname(directory))==realpath(dirname(result.directory))
    @test !islink(directory) && !C.ReferenceImport.B.reparse(directory)
    rm(directory;recursive=true)
end

function rewrite_bundle(directory,metadata,name,change)
    record=metadata["models"][1];bundle=joinpath(directory,record["bundle"])
    path=joinpath(bundle,name)
    payload=change(B._decode(TOML.parsefile(path)["value"]))
    write(path,B._typed_payload_bytes(payload))
    manifest_path=joinpath(bundle,"manifest.toml");manifest=TOML.parsefile(manifest_path)
    for member in manifest["members"]
        file=joinpath(bundle,member["name"])
        member["sha256"]=bytes2hex(sha256(read(file)));member["bytes"]=filesize(file)
    end
    if name=="model.toml"
        manifest["model_id"]=payload.model_id;manifest["setting_index"]=payload.setting_index
        record["model_id"]=payload.model_id;record["gains"]=collect(payload.gains)
    end
    write(manifest_path,B._toml_bytes(manifest))
    record["bundle_manifest_sha256"]=bytes2hex(sha256(read(manifest_path)))
end

function replace_model(model;solver=model.solver,setting_index=model.setting_index,eta=model.eta)
    names=keys(model)[1:end-1]
    base=NamedTuple{names}(Tuple(model)[1:end-1])
    base=merge(base,(solver=solver,setting_index=setting_index,eta=eta))
    merge(base,(model_id=T._hash(base),))
end

@testset "Output relocation and rehashed semantic corruption controls" begin
    owned_copy_case((directory,metadata)->nothing,"relocated";positive=true)
    for (label,key,value,message) in (("fit-count-bool","initial_fit_count",true,"fit/stage count"),
            ("score-claim","test_scoring",true,"exceeds its scope"),
            ("source","source_sha256",repeat("b",64),"raw prefix provenance"),
            ("selection","selected_index",18,"saved selection differs"))
        owned_copy_case((directory,metadata)->(metadata[key]=value),label;expected_error=message)
    end
    owned_copy_case("extra-member";expected_error="member set") do directory,metadata
        write(joinpath(directory,"unreported.txt"),"extra")
    end
    owned_copy_case("points";expected_error="saved points differ") do directory,metadata
        record=metadata["models"][1];path=joinpath(directory,record["points"])
        bytes=read(path);words=reinterpret(UInt64,bytes)
        words[1]=htol(reinterpret(UInt64,reinterpret(Float64,ltoh(words[1]))+1))
        write(path,bytes);record["points_sha256"]=bytes2hex(sha256(bytes))
    end
    owned_copy_case("fit-record";expected_error="records/masks differ") do directory,metadata
        record=metadata["models"][1];root=joinpath(directory,record["fit_input"])
        path=joinpath(root,"joint-fit.f64");bytes=read(path);bytes[1]⊻=0x01;write(path,bytes)
        manifest_path=joinpath(root,"manifest.toml");manifest=TOML.parsefile(manifest_path)
        manifest["data_sha256"]=bytes2hex(sha256(bytes));write(manifest_path,B._toml_bytes(manifest))
        record["fit_input_manifest_sha256"]=bytes2hex(sha256(read(manifest_path)))
    end
    owned_copy_case("checkpoint";expected_error="saved checkpoint differs") do directory,metadata
        rewrite_bundle(directory,metadata,"checkpoint.toml",cp->merge(cp,(sealed=merge(cp.sealed,
            (mean=(cp.sealed.mean[1]+1,cp.sealed.mean[2]),)),)))
    end
    owned_copy_case("provenance";expected_error="provenance differs") do directory,metadata
        rewrite_bundle(directory,metadata,"provenance.toml",p->merge(p,(scientific_verdict=true,)))
    end
    owned_copy_case("solver-loss";expected_error="saved loss/winner policy differs") do directory,metadata
        rewrite_bundle(directory,metadata,"model.toml",m->replace_model(m;
            solver=merge(m.solver,(weighted_sse=m.solver.weighted_sse+1,))))
    end
    owned_copy_case("model-recipe";expected_error="saved model fit/source/clocks differ") do directory,metadata
        rewrite_bundle(directory,metadata,"model.toml",m->replace_model(m;setting_index=2,eta=A.settings()[2].eta))
    end
end

println(JSON3.write((generated_only=true,executor_initial_fits=18,independent_direct_oracle_fits=18,
    state_schedules=9,joint_origins=326,available_origins=159,selection_status=result.selection.status,
    stage_bytes=completion["output_bytes"],elapsed_seconds=measured.time,allocated_bytes=measured.bytes,
    actual_data=false,test_scoring=false,scientific_verdict=false)))
