"""Fit, independently audit and read back the four actual matched342 cells."""
module Matched342InitialActual
using SolarSINDy, JSON3, LinearAlgebra
include("BMatchedReference342.jl")
const Entry=BMatchedReference342
const FC=Entry.BMatched342Core
const T=Entry.T
const P=Entry.P
const BI=Entry.BI
const Producer=Entry.BMatchedInitialProducer
const Audit=Entry.MatchedReferenceAudit342
const REQUEST_FIELDS=(:schema,:evidence_mode,:sources,:output_directory,:audit_directory,:audit_contract_sha256)

function write_json(path,data)
    body=T._bytes(data)
    T._need(length(body)<=T.MAX_META,"matched342 output metadata exceeds its byte bound")
    P._exclusive(path,body)
    (;path,sha256=T._sha(body))
end
identified(data,key)=merge(data,NamedTuple{(key,)}((T._id(data),)))

function main(request_path,request_sha)
    BLAS.set_num_threads(1)
    controls=Dict{String,String}()
    ref=FC._pinned((;path=request_path,sha256=request_sha),controls)
    request=FC._json_ref(ref,controls;fields=REQUEST_FIELDS)
    T._need(request.schema=="matched342_initial_actual_request_v2"&&
        request.evidence_mode=="archival_assumed_completion_research"&&T._hex(request.audit_contract_sha256),
        "actual matched342 request schema/mode differs")
    T._need((request.sources isa AbstractVector||request.sources isa Tuple)&&length(request.sources)==2,
        "actual matched342 requires separate L0 and L1 source specifications")
    admissions=map(enumerate(request.sources)) do (index,spec)
        T._fields(spec,(:delay_hours,:source_proof,:roots))
        T._int(spec.delay_hours,index-1,index-1)
        source=FC._pinned(spec.source_proof,controls)
        roots=Dict(String(key)=>String(value) for (key,value) in pairs(spec.roots))
        admission=BI.admit_source_prefix_v2(source.path;expected_source_proof_sha256=source.sha256,
            roots,expected_sources=P.source_identity())
        T._need(admission.metadata.evidence_mode===:archival_assumed_completion_research,
            "actual matched342 rejects generated source admissions")
        admission
    end
    pair=FC.bind_source_pair(admissions...;loss_contract_sha256=Entry.LOSS_CONTRACT_SHA256,
        expected_sources=Entry.source_identity())
    audit_directory=Audit.audit_root(String(request.audit_directory);fresh=true)
    T._need(!Producer._overlap(audit_directory,String(request.output_directory))&&
        !Producer._overlap(audit_directory,Entry.PACKAGE_ROOT)&&!Producer._overlap(audit_directory,ref.path),
        "actual matched342 audit output overlaps code, input or producer output")
    for admission in admissions, source_root in values(admission.roots)
        T._need(!Producer._overlap(audit_directory,source_root),"actual matched342 audit output overlaps source data")
    end
    feature=(;member=Entry.FEATURE_CONTRACT_MEMBER,path=joinpath(Entry.PACKAGE_ROOT,Entry.FEATURE_CONTRACT_MEMBER),
        sha256=Entry.FEATURE_CONTRACT_SHA256)
    loss=(;member=Entry.LOSS_CONTRACT_MEMBER,path=joinpath(Entry.PACKAGE_ROOT,Entry.LOSS_CONTRACT_MEMBER),
        sha256=Entry.LOSS_CONTRACT_SHA256)
    started=time_ns();gc_before=Base.gc_num()
    draft=Producer.produce_initial_actual(admissions...,String(request.output_directory);
        feature_contract=feature,loss_contract=loss,expected_sources=Entry.source_identity())
    metadata=Producer.draft_metadata(draft)
    component,_=P._json(metadata.component_completion_path;fields=Producer.COMPONENT_FIELDS)
    contracts=merge(FC._contracts(),(;audit_contract_sha256=String(request.audit_contract_sha256)))
    audit_request=identified((;schema="matched342_initial_component_audit_request_v1",
        evidence_mode="archival_assumed_completion_research",operation="independent_saved_state_audit_initial32",
        component=(;path=metadata.component_completion_path,sha256=metadata.component_completion_sha256),
        source_bindings=component.source_bindings,contracts,runtime=component.runtime,
        drafts=Tuple((;delay_hours=index-1,root=metadata.reference_roots[index],members=component.drafts[index].members)
            for index in 1:2),output_directory=String(request.audit_directory)),:request_id)
    audit_input=write_json(String(request.audit_directory)*".request.json",audit_request)
    checked=Audit.audit_component(pair,audit_input.path;request_sha256=audit_input.sha256,
        expected_sources=Entry.source_identity(),expected_audit_contract_sha256=String(request.audit_contract_sha256))
    audit=Audit.component_audit_metadata(checked)
    publication=Producer.publish_initial(draft,checked;expected_audit_completion_sha256=audit.audit_completion_sha256)
    audit_path=joinpath(audit.audit_root,"audit-completion.json")
    reread=Audit.read_component_audit(pair,audit_path,audit.audit_completion_sha256;
        expected_component_sha256=metadata.component_completion_sha256,expected_sources=Entry.source_identity(),
        expected_audit_contract_sha256=String(request.audit_contract_sha256))
    Audit.recheck_component_audit(reread)
    FC.recheck_pair(pair);FC._check_controls(controls)
    result=(;complete=true,evidence_mode="archival_assumed_completion_research",request_sha256=ref.sha256,
        publication,component_audit=(;path=audit_path,sha256=audit.audit_completion_sha256),
        observations=(;elapsed_ns=Int(time_ns()-started),allocated_bytes=Base.GC_Diff(Base.gc_num(),gc_before).allocd,
            peak_rss_bytes=Sys.maxrss(),threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads()),
        scientific_qualified=false,serving_promoted=false)
    println("MATCHED342_ACTUAL_COMPLETE ",JSON3.write(result));flush(stdout)
    result
end
end
if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    length(ARGS)==2||error("usage: run_matched342_initial_actual.jl REQUEST_JSON REQUEST_SHA256")
    Matched342InitialActual.main(ARGS...)
end
