module BPointSyntheticFixture
using SolarSINDy,Dates,SHA,JSON3
const T=SolarSINDy.DurationPointTraining
const P=SolarSINDy.DurationPointArtifacts
const REPO=normpath(joinpath(@__DIR__,"..",".."))
include(joinpath(REPO,"tools","actual_b_chain","ActualBChain.jl"))
const B=ActualBChain
const S=B.BSelection
function fixture(root,n=1024)
    mkdir(root);t0=DateTime(1900);boundary=t0+Hour(n÷2);stop=t0+Hour(n)
    raw=ones(17,n)
    for i in 1:n
        tag=i-1;raw[6,i]=tag%10<5 ? 1. : (-1.,-2.5,-5.,-7.5)[(tag÷10)%4+1]
        raw[8,i]=400.;raw[12,i]=2.;raw[15,i]=tag%192<96 ? -60. : -30.
    end
    io=IOBuffer();for x in raw;write(io,htol(reinterpret(UInt64,x)));end;data=take!(io);source=T._sha(data)
    P._exclusive(joinpath(root,"raw.bin"),data)
    rows=T.BPointPredictionRow[]
    # Independent direct per-lead receipt-clock union, no candidate/model mask.
    cutoff=stop-Day(7)
    for i in 1:n
        issue=t0+Hour(i)
        keep=boundary<=issue<stop&&any(issue+Hour(h)<=cutoff for h in (4,6))
        keep||continue
        anchor=raw[15,i];target=(raw[15,i+4],raw[15,i+6])
        push!(rows,T.BPointPredictionRow(issue,(anchor,anchor),anchor,target,false,:synthetic_persistence_reference,source,0))
    end
    T._need(length(rows)==n÷2-171,"generated independent union geometry differs")
    rawid=S._row_identity(rows);predid=T._id([(;issue=x.issue,means=x.means) for x in rows])
    completion=(;schema="synthetic_reference_completion_v1",complete=true,evidence_mode=:synthetic_only,source_sha256=source,raw_row_identity_sha256=rawid,prediction_panel_sha256=predid)
    audit=(;schema="synthetic_reference_audit_v1",complete=true,evidence_mode=:synthetic_only,source_sha256=source,raw_row_identity_sha256=rawid)
    inputaudit=(;schema="synthetic_input_audit_v1",complete=true,evidence_mode=:synthetic_only,source_sha256=source,source_rows=n)
    provenance=(;schema="generated_raw_provenance_v1",actual_data=false,source_sha256=source,start=t0,source_rows=n)
    for (name,value) in (("reference.json",completion),("reference-audit.json",audit),("input-audit.json",inputaudit),("provenance.json",provenance));P._exclusive(joinpath(root,name),T._bytes(value));end
    hash(name)=P._file_sha(joinpath(root,name))[1]
    refpin=hash("reference.json");refaudit=hash("reference-audit.json");inputpin=hash("input-audit.json")
    proof=(;schema="independently_audited_b_prefix_v1",complete=true,evidence_mode=:synthetic_only,source_sha256=source,raw_prefix_sha256=source,prefix_start_ms=t0,prefix_stop_ms=stop,prefix_rows=n,training_boundary_ms=boundary,validation_start_ms=boundary,validation_end_ms=stop,receipt_delay_hours=0,reference_completion_sha256=refpin,reference_audit_sha256=refaudit,audit_completion_sha256=inputpin,implementation_sources=T._hashdata(T._hashmap(P.source_identity())))
    P._exclusive(joinpath(root,"admission.json"),T._bytes(proof));proofpin=hash("admission.json")
    members=[begin bytes=filesize(joinpath(root,name));(;root="input",path=name,bytes,sha256=hash(name),role) end for (name,role) in (("raw.bin","raw_prefix"),("reference.json","reference_completion"),("reference-audit.json","reference_audit"),("admission.json","source_admission"),("input-audit.json","independent_audit"),("provenance.json","raw_provenance"))]
    envelope=(;schema="b_point_input_envelope_v1",complete=true,evidence_mode=:synthetic_only,availability_kind="assumed_completed_hour",source_sha256=source,prefix_start_ms=t0,prefix_stop_ms=stop,prefix_rows=n,channel_count=17,raw_prefix_member="input/raw.bin",raw_prefix_bytes=length(data),raw_prefix_sha256=source,training_boundary_ms=boundary,validation_start_ms=boundary,validation_end_ms=stop,receipt_delay_hours=0,independent_admission_completion_sha256=proofpin,audit_completion_sha256=inputpin,reference_kind=:original_reference_diagnostic_only,reference_completion_sha256=refpin,reference_audit_sha256=refaudit,implementation_sources=T._hashdata(T._hashmap(P.source_identity())),limits=P.LIMITS,members)
    path=joinpath(root,"envelope.json");P._exclusive(path,T._bytes(envelope))
    admission=B.BInputs.admit_prefix(path;expected_envelope_sha256=hash("envelope.json"),roots=Dict("input"=>root),expected_sources=P.source_identity(),independent_admission=(;path=joinpath(root,"admission.json"),sha256=proofpin))
    meta=(;schema="b_point_reference_panel_v1",evidence_mode=:synthetic_only,reference_kind=:original_reference_diagnostic_only,source_sha256=source,delay_hours=0,training_boundary=boundary,validation_start=boundary,validation_end=stop,training_history_start=t0,model_ids=[T._sha("synthetic_reference_h4"),T._sha("synthetic_reference_h6")],max_fit_target_complete=boundary-Day(7),max_fit_label_receipt=boundary-Day(7),feature_contract_sha256=T._sha("synthetic_reference_features"),fixed_grid_manifest_sha256=T._sha("synthetic_original_reference_grid"),selected_setting_ids=[29,29],tuning_loss_contract_sha256=T._sha("synthetic_reference_loss"),tuning_proof_completion_sha256=T._sha("synthetic_reference_tuning"),prediction_panel_sha256=predid,raw_row_identity_sha256=rawid,reference_completion_sha256=refpin,audit_completion_sha256=refaudit)
    admission,S._reference(meta,rows)
end
function run_stage(root;chunk_hours=7)
    root=realpath(root)
    admission,reference=fixture(joinpath(root,"input"))
    stage=B.BInitialPointFits.run_initial_l0(admission,reference,joinpath(root,"stage");chunk_hours)
    verified=B.BInitialPointFits.verify_initial_l0(stage.directory,admission,reference;expected_completion_sha256=stage.completion_sha256,expected_sources=P.source_identity(),chunk_hours=2)
    T._need(stage.gain_fit_count==36&&verified.complete&&verified.model_count==36&&verified.certificate_count==36&&verified.gain_fit_calls==0&&verified.hazard_fit_calls==0,"generated B stage/readback differs")
    (;admission,reference,stage,verified)
end
end
