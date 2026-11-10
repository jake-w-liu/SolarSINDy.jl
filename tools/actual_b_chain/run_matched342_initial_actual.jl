"""Actual initial32 entry: supplied authority precedes all archive source decoding."""
module Matched342InitialActual
include("run_matched342_full_shape_profile.jl")
const Profile=Matched342FullShapeProfile
using JSON3,LinearAlgebra
const FC=Profile.FC;const T=Profile.T;const P=Profile.P;const Entry=Profile.Entry
const Producer=Profile.Producer;const Audit=Profile.Audit
const need=Profile.need;const fields=Profile.fields;const same=Profile.same
const ACTUAL_FIELDS=(:schema,:evidence_mode,:operation,:execution_authority,:sources,:source_proof,:runtime,:output_directory,:owner_directory,:request_id)
function validate_actual_metadata(d)
    fields(d,ACTUAL_FIELDS);FC._check_id(d,:request_id)
    need(d.schema=="matched342_initial_actual_request_v1"&&d.evidence_mode=="archival_assumed_completion_research"&&
        d.operation=="initial_matched342_32fits","actual request mode/operation differs")
    Profile.ref_grammar(d.execution_authority);Profile.ref_grammar(d.source_proof);Audit.check_runtime(d.runtime)
    Profile.absolute_grammar(d.output_directory);Profile.absolute_grammar(d.owner_directory)
    need((d.sources isa Tuple||d.sources isa AbstractVector)&&length(d.sources)==2,"two outside source specifications required")
    for (k,s) in enumerate(d.sources)
        fields(s,(:delay_hours,:source_proof,:roots));T._int(s.delay_hours,k-1,k-1);Profile.ref_grammar(s.source_proof)
        need(1<=length(s.roots)<=32,"ordinary source root map required")
        for (label,path) in pairs(s.roots)
            need((label isa AbstractString||label isa Symbol)&&occursin(r"\A[A-Za-z0-9_-]+\z",String(label)),"source root label differs")
            Profile.absolute_grammar(path)
        end
    end
    d
end
function attempt_marker_path(directory,sha)
    Profile.absolute_grammar(directory);need(T._hex(sha),"permit SHA grammar differs")
    joinpath(dirname(directory),basename(directory)*".matched342-"*sha*".attempt.json")
end
function main(request_path,request_sha,bridge_path)
    BLAS.set_num_threads(1);controls=Dict{String,String}()
    ref=FC._pinned((;path=request_path,sha256=request_sha),controls);d=FC._json_ref(ref,controls;fields=ACTUAL_FIELDS)
    validate_actual_metadata(d)
    # Only bounded control metadata is read here. The outside permit's complete
    # qualifications/resource proof are admitted before BI decodes either archive.
    permit=FC._json_ref(d.execution_authority,controls;fields=FC.PERMIT_FIELDS);FC._check_id(permit,:permit_id)
    need(permit.schema=="qualified_actual_initial_matched342_permit_v1"&&permit.complete===true&&
        permit.evidence_mode==d.evidence_mode&&permit.operation==d.operation&&same(d.source_proof,permit.source_proof)&&
        same(d.runtime,permit.runtime)&&d.output_directory==permit.execution.directory,"outside actual request/permit differs")
    Profile.validate_limits(permit.limits)
    resource=FC._json_ref(permit.resource_qualification,controls;fields=FC.RESOURCE_FIELDS)
    bridge_data=(;owner_directory=d.owner_directory,qualified_head=permit.qualified_head,source_proof=d.source_proof,
        fixture_spec=resource.fixture_spec,runtime=d.runtime,limits=permit.limits)
    prelaunch=Profile.read_bridge(bridge_path,ref,bridge_data;mode=:archival_assumed_completion_research,permit_sha=d.execution_authority.sha256)
    witness=FC._json_ref(prelaunch,controls;fields=FC.PRELAUNCH_FIELDS)
    marker=attempt_marker_path(String(permit.execution.directory),String(d.execution_authority.sha256))
    m,_=P._json(marker;fields=(:schema,:execution_directory,:permit_sha256,:request_sha256,:owner_pid,:created_monotonic_ns,:marker_id))
    FC._check_id(m,:marker_id)
    need(m.schema=="matched342_permanent_actual_attempt_v1"&&m.execution_directory==d.output_directory&&
        m.permit_sha256==d.execution_authority.sha256&&m.request_sha256==ref.sha256&&m.owner_pid==witness.owner_pid,
        "permanent outside actual attempt marker differs")
    T._int(m.created_monotonic_ns,1,typemax(Int64));controls[P._ordinary(marker)]=P._file_sha(marker)[1]
    authority=FC.admit_execution_authority(d.execution_authority;expected_sources=Entry.source_identity(),prelaunch_admission=prelaunch)
    meta=FC.execution_authority_metadata(authority)
    need(meta.execution_directory==d.output_directory,"actual authority output differs")
    input=Profile.admit_sources(d.sources;mode=:archival_assumed_completion_research)
    FC.recheck_execution_authority(authority,input.pair);FC._check_controls(controls)
    feature=(;member=Entry.FEATURE_CONTRACT_MEMBER,path=joinpath(Profile.REPO,Entry.FEATURE_CONTRACT_MEMBER),sha256=Entry.FEATURE_CONTRACT_SHA256)
    loss=(;member=Entry.LOSS_CONTRACT_MEMBER,path=joinpath(Profile.REPO,Entry.LOSS_CONTRACT_MEMBER),sha256=Entry.LOSS_CONTRACT_SHA256)
    draft=Producer.produce_initial_actual(input.admissions...,String(d.output_directory);qualification=permit.component_qualification,
        feature_contract=feature,loss_contract=loss,qualified_head=String(permit.qualified_head),execution_authority=authority,expected_sources=Entry.source_identity())
    # Historical permit links are checked by the independent reader after fitting;
    # an elapsed active fitting clock is not reinterpreted as audit authority.
    contracts=merge(FC._freeze(permit.contracts),(;audit_contract_sha256=Profile.AUDIT_SHA))
    published=Profile.audit_and_publish(input.pair,draft,String(d.owner_directory),
        (;component_qualification=permit.component_qualification,source_proof=d.source_proof,contracts,runtime=d.runtime);authority_ref=d.execution_authority)
    # This is a postcondition on the same bounded owner stage, not a new fit:
    # include the audit's observed RSS/allocation/descriptor use in its budget.
    FC.recheck_execution_authority(authority,input.pair);FC._check_controls(controls)
    result=Profile.identified((;schema="matched342_initial_actual_result_v1",complete=true,evidence_mode=d.evidence_mode,
        request_sha256=ref.sha256,execution_authority_sha256=d.execution_authority.sha256,pair_id=FC.source_pair_metadata(input.pair).pair_id,
        component=published.component,component_audit=published.component_audit,publication=published.publication,
        scientific_qualified=false,serving_promoted=false),:result_id)
    saved=Profile.write_json(joinpath(String(d.owner_directory),"ACTUAL_RESULT.json"),result)
    println("MATCHED342_ACTUAL_COMPLETE ",JSON3.write((;request=ref,julia_pid=getpid(),result=saved,execution_authority_sha256=d.execution_authority.sha256)));flush(stdout)
    nothing
end
end
if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    length(ARGS)==3||error("actual initial32 requires request path, SHA and on-launch bridge path")
    Matched342InitialActual.main(ARGS...)
end
