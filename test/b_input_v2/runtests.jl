module BInputV2Tests
using Test,SolarSINDy,Dates,SHA,JSON3
using LinearAlgebra: BLAS
include(joinpath(pkgdir(SolarSINDy),"tools","actual_b_chain","run_initial_l0_native.jl"))
const Entry=BInitialNativeStage
const B=Entry.B
const T=SolarSINDy.DurationPointTraining
const P=SolarSINDy.DurationPointArtifacts
const S=B.BSelection
const I=B.BInputs
const LITERAL_PATH=joinpath(@__DIR__,"source_dag_literals_v1.json")
# Reuse the existing bounded reference FILE writer, never its array-only view.
# Including this file defines helpers; it does not execute reference_checks().
include(joinpath(@__DIR__,"..","b_point_training","reference_checks.jl"))
include("fixture.jl")
const ORACLE=b2_json(LITERAL_PATH)

function source_and_clock_checks(root)
    views=String[]
    for delay in (0,1)
        f=b2_source_fixture(joinpath(root,"positive-L$delay");delay)
        @test !ispath(joinpath(f.base,"reference-matched"))
        @test !ispath(joinpath(f.base,"joined-matched"))
        @test length(readdir(f.base))==1
        source=b2_admit_source(f);expected=b2_expected_view(f)
        push!(views,source.view.view_id)
        @test source.view.view_id==expected.view_id
        @test source.view.input_prefix_sha256==expected.input_prefix_sha256
        @test source.view.wind_sequence_sha256==expected.wind_sequence_sha256
        @test source.view.input_admission_sha256==f.proof_sha
        @test source.view.source_hours==1024&&filesize(joinpath(f.directory,"raw.bin"))==139264
        @test source.metadata.delay_hours==delay&&I.recheck_source_v2(source)
        @test Set(keys(f.common))==Set(String.(ORACLE["common_fields"]))
        @test Set(keys(f.proof))==Set(vcat(String.(ORACLE["common_fields"]),String.(ORACLE["source_extra_fields"])))
        @test Set(getproperty.(source.members,:role))==Set(String.(ORACLE["source_roles"]))
        @test all(field->!haskey(f.proof,field),ORACLE["forbidden_source_fields"])
        observed=ORACLE["cohorts"][delay+1]
        # One-based raw index i has issue hour i; independent integer receipts.
        training=[[i for i in 1:1024 if i-delay>=1&&i+h+delay<=344] for h in (4,6)]
        validation=[[i for i in 512:1024 if i+h+delay<=856] for h in (4,6)]
        @test length.(training)==Int.(observed["training_count"])
        @test first.(training)==Int.(observed["training_first"])
        @test last.(training)==Int.(observed["training_last"])
        @test length.(validation)==Int.(observed["validation_count"])
        @test first.(validation)==Int.(observed["validation_first"])
        @test last.(validation)==Int.(observed["validation_last"])
        @test setdiff(validation[1],validation[2])==Int.(observed["h4_only"])
        fcopy=joinpath(root,"relocated-source-L$delay");cp(f.directory,fcopy)
        relocated=I.admit_source_prefix_v2(joinpath(fcopy,"source-proof.json");expected_source_proof_sha256=f.proof_sha,
            roots=Dict("source"=>fcopy),expected_sources=P.source_identity())
        @test relocated.view.view_id==source.view.view_id
        @test relocated.source_admission_id==source.source_admission_id
        joined=b2_join_fixture(f,source)
        checked=Entry.reconstruct_inputs_v2(joined.input_spec,joined.reference_spec;
            expected_evidence_mode=:synthetic_only,expected_sources=P.source_identity())
        @test checked.admission isa I.BPrefixAdmission
        @test checked.reference isa S.BReferencePanel&&checked.reference.audit isa S.BReferenceAudit
        @test checked.admission.view.view_id==source.view.view_id==b2_expected_view(f).view_id
        @test checked.admission.view.input_admission_sha256==f.proof_sha
        @test checked.admission.admission_id==b2_expected_admission(joined)
        @test checked.admission.metadata.input_contract_version===2
        @test I.recheck(checked.admission)&&I.recheck_source_v2(source)
        @test length(checked.reference.rows)==observed["union_count"]
        @test filesize(joinpath(joined.reference_directory,"predictions.bin"))==observed["selected_bytes"]
        @test filesize(joinpath(joined.reference_directory,"candidates","h4","setting-1.bin"))==observed["candidate_bytes"]
        @test S._check_reference(checked.reference;matched=true)
        same_bytes_other_root=b2_join_fixture(f,source;tag="separate-valid-root")
        @test S._check_reference(same_bytes_other_root.reference;matched=true)
        @test_throws ArgumentError Entry.reconstruct_inputs_v2(joined.input_spec,same_bytes_other_root.reference_spec;
            expected_evidence_mode=:synthetic_only)
        @test_throws ArgumentError I.admit_prefix(joined.envelope_path;expected_envelope_sha256=joined.envelope_sha,
            roots=joined.roots,expected_sources=P.source_identity(),independent_admission=(;path=f.path,sha256=f.proof_sha))
        if delay==0
            original=b2_join_fixture(f,source;tag="original",kind=:original_reference_diagnostic_only)
            @test original.admission.view.view_id==source.view.view_id
            @test original.admission.admission_id!=joined.admission.admission_id
            @test b2_file(original.join_path)!=b2_file(joined.join_path)
            @test b2_file(original.envelope_path)!=b2_file(joined.envelope_path)
            @test b2_admit_source(f).source_admission_id==source.source_admission_id
            # Both references separately pass all bounded readers. A checked
            # helper must still bind the exact pair joined in this admission.
            @test_throws ArgumentError Entry.reconstruct_inputs_v2(original.input_spec,joined.reference_spec;expected_evidence_mode=:synthetic_only)
            swapped_path=merge(original.reference_spec,(;completion_path=joined.reference_spec.completion_path,roots=joined.reference_spec.roots))
            swapped_pins=merge(original.reference_spec,(;pins=joined.reference_spec.pins))
            @test_throws ArgumentError Entry.reconstruct_inputs_v2(original.input_spec,swapped_path;expected_evidence_mode=:synthetic_only)
            @test_throws ArgumentError Entry.reconstruct_inputs_v2(original.input_spec,swapped_pins;expected_evidence_mode=:synthetic_only)
        end
    end
    @test length(unique(views))==2
    unchanged=b2_source_fixture(joinpath(root,"same-raw-upstream-provenance-A"))
    changed=b2_source_fixture(joinpath(root,"same-raw-upstream-provenance-B");
        provenance_changes=Dict("source_format_sha256"=>b2_sha(codeunits("different valid synthetic provenance witness"))))
    first_source=b2_admit_source(unchanged);second_source=b2_admit_source(changed)
    @test unchanged.common["raw_prefix_sha256"]==changed.common["raw_prefix_sha256"]
    @test first_source.view.input_prefix_sha256==second_source.view.input_prefix_sha256
    @test first_source.view.view_id!=second_source.view.view_id
    @test first_source.view.view_id==b2_expected_view(unchanged).view_id&&second_source.view.view_id==b2_expected_view(changed).view_id
end

function source_rejection_checks(root)
    number=0
    for field in String.(ORACLE["forbidden_source_fields"]), role in (:provenance,:audit,:source)
        number+=1;changes=Dict{String,Any}(field=>repeat("0",64))
        options=role==:provenance ? (;provenance_changes=changes) : role==:audit ? (;audit_changes=changes) : (;source_changes=changes)
        f=b2_source_fixture(joinpath(root,"future-$number");options...)
        @test_throws ArgumentError b2_admit_source(f)
    end
    for (label,options) in (
        ("Boolean-delay",(;common_changes=Dict("receipt_delay_hours"=>true))),
        ("float-channels",(;common_changes=Dict("channel_count"=>17.0))),
        ("too-many-rows",(;common_changes=Dict("prefix_rows"=>500001))),
        ("non-hour-split",(;common_changes=Dict("validation_start_ms"=>string(Dates.value(DateTime(1900)+Hour(512))+1)))),
        ("actual-relabel",(;common_changes=Dict("evidence_mode"=>"archival_assumed_completion_research"))),
        ("layout",(;provenance_changes=Dict("raw_layout"=>"row_major"))),
        ("parser",(;provenance_changes=Dict("parser_source_sha256"=>repeat("0",64)))),
        ("format-grammar",(;provenance_changes=Dict("source_format_sha256"=>"not-a-hash"))),
        ("wrong-source",(;provenance_changes=Dict("source_sha256"=>repeat("0",64)))),
        ("wrong-audit",(;audit_changes=Dict("raw_provenance_sha256"=>repeat("0",64)))),
        ("old-schema",(;source_changes=Dict("schema"=>"independently_audited_b_prefix_v1"))),
        ("reduced-limit",(;source_changes=Dict("limits"=>merge(deepcopy(ORACLE["limits"]),Dict("chunk_rows"=>4095))))))
        f=b2_source_fixture(joinpath(root,label);options...)
        @test_throws ArgumentError b2_admit_source(f)
    end
    for (label,edit) in (
        ("duplicate-role",d->(d["members"][2]["role"]="raw_prefix")),
        ("traversal",d->(d["members"][1]["path"]="../raw.bin")),
        ("case-alias",d->(d["members"][1]["path"]="RAW.BIN")),
        ("row-size",d->(d["members"][1]["bytes"]-=8)),
        ("source-closure",d->pop!(d["implementation_sources"])))
        f=b2_source_fixture(joinpath(root,label);source_edit=edit)
        @test_throws ArgumentError b2_admit_source(f)
    end
    missing=b2_source_fixture(joinpath(root,"canonical-missing");raw_change=v->(v[1,1]=reinterpret(Float64,UInt64(0x7ff8000000000000))))
    @test isnan(b2_admit_source(missing).view.input.values[1,1])
    for word in (UInt64(0x7ff8000000000001),UInt64(0x7ff0000000000000),UInt64(0xfff0000000000000))
        f=b2_source_fixture(joinpath(root,"unsupported-word-"*string(word));raw_change=v->(v[1,1]=reinterpret(Float64,word)))
        @test_throws ArgumentError b2_admit_source(f)
    end
    for name in ("raw.bin","provenance.json","source-audit.json","source-proof.json")
        f=b2_source_fixture(joinpath(root,"changed-"*replace(name,'.'=>'-')));source=b2_admit_source(f)
        path=joinpath(f.directory,name);body=read(path);body[1]=xor(body[1],UInt8(1));write(path,body)
        @test_throws ArgumentError I.recheck_source_v2(source)
    end
    oversized=b2_source_fixture(joinpath(root,"oversized-source-proof"))
    open(joinpath(oversized.directory,"source-proof.json"),"w") do io
        block=fill(UInt8(' '),65536);for _ in 1:32;write(io,block);end;write(io,UInt8(' '))
    end
    @test_throws ArgumentError I.admit_source_prefix_v2(oversized.path;expected_source_proof_sha256=b2_file(oversized.path),roots=oversized.roots,expected_sources=P.source_identity())
    if !Sys.iswindows()
        f=b2_source_fixture(joinpath(root,"real-symbolic-link"));target=joinpath(f.directory,"raw.bin")
        outside=joinpath(root,"linked-raw-target.bin");mv(target,outside);symlink(outside,target)
        @test_throws ArgumentError b2_admit_source(f)
    end
end

function join_rejection_checks(root)
    mutations=[
        ("future-envelope",j->(j["envelope_sha256"]=repeat("0",64)),identity),
        ("own-hash",j->(j["reference_join_audit_sha256"]=repeat("0",64)),identity),
        ("wrong-view",j->(j["input_view_id"]=repeat("0",64)),identity),
        ("wrong-prefix",j->(j["input_prefix_sha256"]=repeat("0",64)),identity),
        ("wrong-delay",j->(j["receipt_delay_hours"]=1),identity),
        ("wrong-source",j->(j["source_sha256"]=repeat("0",64)),identity),
        ("wrong-source-proof",j->(j["source_admission_sha256"]=repeat("0",64)),identity),
        ("wrong-provenance",j->(j["raw_provenance_sha256"]=repeat("0",64)),identity),
        ("wrong-reference",j->(j["reference_completion_sha256"]=repeat("0",64)),identity),
        ("wrong-reference-audit",j->(j["reference_audit_sha256"]=repeat("0",64)),identity),
        ("extra-envelope",identity,e->(e["unchecked_ready"]=true)),
        ("extra-role",identity,e->push!(e["members"],deepcopy(e["members"][1]))),
        ("duplicate-role",identity,e->(e["members"][end]["role"]="reference_audit")),
        ("source-path-alias",identity,e->(e["members"][1]["path"]="RAW.BIN")),
        ("stale-source-size",identity,e->(e["members"][1]["bytes"]-=8))]
    for (label,join_change,envelope_change) in mutations
        f=b2_source_fixture(joinpath(root,label));source=b2_admit_source(f);joined=b2_join_fixture(f,source)
        b2_reseal_join!(joined;join_change,envelope_change)
        @test_throws ArgumentError b2_admit_join(joined)
        @test I.recheck_source_v2(source)
    end
    f=b2_source_fixture(joinpath(root,"shared-entry-expected-identities"));source=b2_admit_source(f);joined=b2_join_fixture(f,source)
    for field in (:expected_view_id,:expected_admitted_seal)
        wrong=merge(joined.input_spec,NamedTuple{(field,)}((repeat("0",64),)))
        @test_throws ArgumentError Entry.reconstruct_inputs_v2(wrong,joined.reference_spec;expected_evidence_mode=:synthetic_only)
    end
    @test_throws ArgumentError Entry.reconstruct_inputs_v2(joined.input_spec,joined.reference_spec;expected_evidence_mode=:archival_assumed_completion_research)
    @test_throws ArgumentError Entry.reconstruct_inputs_v2(joined.input_spec,joined.reference_spec;
        expected_evidence_mode=:synthetic_only,expected_input_prefix_sha256=repeat("0",64))
    @test_throws ArgumentError Entry.reconstruct_inputs_v2(merge(joined.input_spec,(;ready=true)),joined.reference_spec;expected_evidence_mode=:synthetic_only)
    # J's coherent outer pins cannot replace the complete reference reader.
    path=joinpath(joined.reference_directory,"predictions.bin");body=read(path);body[end]=xor(body[end],UInt8(1));write(path,body)
    @test_throws ArgumentError Entry.reconstruct_inputs_v2(joined.input_spec,joined.reference_spec;expected_evidence_mode=:synthetic_only)
    @test I.recheck_source_v2(source)
    # Rehash both R/RA IDs, their member ledger, outside pins, J and E. Known
    # invalid headers must now fail the metadata-only reader before decoding
    # any raw prefix, as well as failing the identical shared entry.
    for (label,change_completion,change_audit) in (
        ("coherently-rehashed-reference-view",d->(d["input_view_id"]=repeat("0",64)),d->(d["input_view_id"]=repeat("0",64))),
        ("coherently-rehashed-reference-prefix",d->(d["input_prefix_sha256"]=repeat("0",64)),d->(d["input_prefix_sha256"]=repeat("0",64))),
        ("unknown-reference-header",d->(d["unchecked_ready"]=true),identity),
        ("unknown-reference-audit",identity,d->(d["future_envelope_sha256"]=repeat("0",64))),
        ("reference-audit-flag",identity,d->(d["initial_only"]=false)),
        ("reference-history",d->(d["training_history_start"]=string(Dates.value(DateTime(1900)+Hour(1)))),d->(d["training_history_start"]=string(Dates.value(DateTime(1900)+Hour(1))))))
        f2=b2_source_fixture(joinpath(root,label));source2=b2_admit_source(f2);second=b2_join_fixture(f2,source2)
        supplied=b2_reseal_reference_headers!(second;completion_change=change_completion,audit_change=change_audit)
        @test_throws ArgumentError b2_admit_join(second;metadata_only=true,reference_pins=supplied.reference_spec.pins)
        @test_throws ArgumentError b2_admit_join(second;reference_pins=supplied.reference_spec.pins)
        @test_throws ArgumentError Entry.reconstruct_inputs_v2(supplied.input_spec,supplied.reference_spec;expected_evidence_mode=:synthetic_only)
        @test I.recheck_source_v2(source2)
    end
end

function complete_file_handoff(root)
    f=b2_source_fixture(joinpath(root,"full36-ordinary-files"));source=b2_admit_source(f)
    @test !ispath(joinpath(f.base,"reference-matched"))&&!ispath(joinpath(f.base,"joined-matched"))
    joined=b2_join_fixture(f,source)
    checked=Entry.reconstruct_inputs_v2(joined.input_spec,joined.reference_spec;expected_evidence_mode=:synthetic_only)
    @test checked.reference.audit isa S.BReferenceAudit
    worker=time_ns();started=time_ns()
    stage=B.BInitialPointFits.run_initial_l0(checked.admission,checked.reference,joinpath(root,"full36-stage");chunk_hours=7)
    fit_elapsed_ns=Int(time_ns()-started);started=time_ns()
    verified=B.BInitialPointFits.verify_initial_l0(stage.directory,checked.admission,checked.reference;
        expected_completion_sha256=stage.completion_sha256,expected_sources=P.source_identity(),chunk_hours=2)
    readback_elapsed_ns=Int(time_ns()-started)
    result=Entry._assembled_result(stage,verified;actual=false,fit_elapsed_ns,readback_elapsed_ns,worker_elapsed_ns=Int(time_ns()-worker))
    @test result["complete"]===true&&result["actual_source_access"]===false&&result["evidence_mode"]=="synthetic_only"
    for (key,value) in (("initial_hazard_fit_count",2),("initial_gain_fit_count",36),("state_schedule_count",9),
            ("beta_design_count",18),("verified_model_count",36),("verified_certificate_count",36),
            ("verified_hazard_fit_calls",0),("verified_gain_fit_calls",0),("final_fit_count",0),("calibration_fit_count",0))
        @test result[key] isa Integer&&!(result[key] isa Bool)&&result[key]==value
    end
    @test result["selected_index"]===nothing&&result["scientific_verdict"]===false&&result["serving_promotion"]===false
    completion=b2_json(joinpath(stage.directory,"completion.json"))
    @test completion["admitted_input_seal"]==checked.admission.admission_id
    @test completion["input_view_id"]==b2_expected_view(f).view_id
    @test completion["actual_execution_permit_sha256"]===nothing&&completion["qualified_head"]===nothing
    @test completion["input_envelope_sha256"]==joined.envelope_sha
    @test completion["reference_completion_sha256"]==joined.witness.pins.completion_sha256
    @test result["output_file_count"]==423&&result["output_directory_count"]==116
    @test I.recheck_source_v2(source)&&I.recheck(checked.admission)
end

function run_checks()
    base=haskey(ENV,"RUNNER_TEMP") ? realpath(ENV["RUNNER_TEMP"]) : realpath(tempdir())
    root=realpath(mktempdir(base;prefix="solarsindy-b-source-v2-",cleanup=false));success=false
    caller_threads=BLAS.get_num_threads()
    try
        BLAS.set_num_threads(1)
        @testset "Acyclic bounded B source and reference composition" begin
            @test b2_file(LITERAL_PATH)=="94e65b88b4b0a293e0ee111bf8e3a1aebf91f78470aa383e1b307f4e7311c464"
            source_and_clock_checks(root)
            source_rejection_checks(root)
            join_rejection_checks(root)
            complete_file_handoff(root)
        end
        success=true
    finally
        BLAS.set_num_threads(caller_threads)
        if success
            @assert dirname(realpath(root))==base&&startswith(basename(root),"solarsindy-b-source-v2-")
            rm(root;recursive=true)
        else;println("B_SOURCE_V2_FAILURE_FIXTURE_ROOT=",root);end
    end
    nothing
end
run_checks()
end
