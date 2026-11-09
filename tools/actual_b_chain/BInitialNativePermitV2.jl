# Included in BInitialPointFits. Legacy permit/schema/owner and numerical code stay intact.
export OwnedInitialPointExecutionPermitV2
const OWNED_V2_OWNER="ccbf1cbd0709e50d0b5080bdbe4977ae80ebd1649237e016588d24800ddf2e21"
const OWNED_V2_LEGACY="15d2644ae5eb2e87ce31c9913f103f9fa7e940b67403b94f29ef2ed5a4d175b8"
const OWNED_V2_FIELDS=(P.PERMIT_FIELDS...,:execution_directory,:execution_intent_sha256,:native_driver_sha256,:native_entry_sha256,:preserved_native_owner_sha256)
const OWNED_V2_REQUEST=(:schema,:evidence_mode,:operation,:execution_directory,:intent,:permit,:source_proof,:root_qualification,:b_execution_qualification,:julia,:runtime)
const OWNED_V2_INTENT=(:schema,:evidence_mode,:stage,:execution_directory,:contract_sha256,:implementation_sources,:input,:reference,:runtime,:programs,:limits,:scope)
const OWNED_V2_SCOPE=(:training_semantics,:training_boundary,:validation_start,:validation_end,:embargo_hours,:delay_hours,:beta_order,:b_setting_indices,:initial_hazard_fit_count,:initial_gain_fit_count,:state_schedule_count,:beta_design_count,:final_fit_count,:calibration_fit_count,:exposed_development_scoring,:joint_primary_selection,:scientific_verdict,:serving_promotion)
struct OwnedInitialPointExecutionPermitV2
    path::String
    sha256::String
    intent_path::String
    intent_sha256::String
end
struct _OwnedInitialAdmissionV2
    path::String
    sha256::String
    qualification_path::String
    qualification_sha256::String
    qualified_head::String
    controls::Tuple
end
_v2_ref(ref)=begin T._fields(ref,(:path,:sha256));T._need(ref.path isa AbstractString&&isabspath(ref.path)&&T._hex(ref.sha256),"B v2 outside pin grammar");body=P._bound(ref.path);T._need(T._sha(body)==ref.sha256,"B v2 outside pin differs");P._bounded_json_parse(body) end
_v2_canonical_ref(ref)=begin value=_v2_ref(ref);T._need(T._bytes(value)==P._bound(ref.path),"B v2 canonical metadata required");value end
function _v2_scope(d)
    T._fields(d,OWNED_V2_SCOPE)
    T._need(d.training_semantics=="option_T_in_sample_two_stage_batch_v1","B v2 training semantics differs")
    for (k,v) in ((:training_boundary,DateTime(2020)),(:validation_start,DateTime(2020)),(:validation_end,DateTime(2023)))
        T._need(T._clock(getproperty(d,k))==v,"B v2 fixed split differs")
    end
    for (k,v) in ((:initial_hazard_fit_count,2),(:initial_gain_fit_count,36),(:state_schedule_count,9),(:beta_design_count,18),(:final_fit_count,0),(:calibration_fit_count,0),(:delay_hours,0),(:embargo_hours,168))
        T._need(T._int(getproperty(d,k),0,500000)==v,"B v2 count differs")
    end
    for k in (:exposed_development_scoring,:joint_primary_selection,:scientific_verdict,:serving_promotion)
        T._need(getproperty(d,k)===false,"B v2 exceeds initial scope")
    end
    T._need(collect(d.beta_order)==[5,20]&&all(x->x isa Integer&&!(x isa Bool),d.beta_order)&&collect(d.b_setting_indices)==collect(1:36)&&all(x->x isa Integer&&!(x isa Bool),d.b_setting_indices),"B v2 fixed order differs")
    nothing
end
function _v2_context(request,request_sha,argv_path)
    root=P._ordinary(request.execution_directory);T._need(isabspath(request.execution_directory)&&root==request.execution_directory&&isdir(root),"B v2 execution directory differs")
    for name in ("B1_NATIVE_ATTEMPT.json","B1_OWNER_LAUNCH.json","B1_TIMING.json","b1-argv.json","CONTROLS_BEFORE.json")
        T._need(isfile(P._member(root,name)),"B v2 owned parent marker absent")
    end
    timing,body=P._json(P._member(root,"B1_TIMING.json"));launch,_=P._json(P._member(root,"B1_OWNER_LAUNCH.json"));attempt,_=P._json(P._member(root,"B1_NATIVE_ATTEMPT.json"))
    T._need(get(ENV,"SOLARSINDY_B_V2_NATIVE_OWNER_SHA256","")==OWNED_V2_OWNER&&get(ENV,"SOLARSINDY_B_V2_REQUEST_SHA256","")==request_sha&&get(ENV,"SOLARSINDY_B_V2_TIMING_SHA256","")==T._sha(body),"B v2 owned environment/context differs")
    T._need(timing.schema=="b_initial_l0_timing_v2"&&timing.request_sha256==request_sha&&timing.intent_sha256==request.intent.sha256&&timing.execution_directory==root&&timing.owner_sha256==OWNED_V2_OWNER&&timing.legacy_owner_sha256==OWNED_V2_LEGACY,"B v2 timing binding differs")
    start=T._int(timing.start_ns,0,typemax(Int64));deadline=T._int(timing.deadline_ns,0,typemax(Int64));T._need(deadline-start==7200*10^9,"B v2 fixed Python deadline differs")
    # Python monotonic_ns is NOT compared to Julia time_ns. Parent/pump enforce it.
    T._need(launch.schema=="b_initial_l0_owner_launch_v2"&&launch.request_sha256==request_sha&&launch.timing_sha256==T._sha(body)&&launch.owner_pid==timing.parent_pid&&string(launch.pid)==get(ENV,"SOLARSINDY_B_V2_SUPERVISOR_PID",""),"B v2 supervisor identity differs")
    T._need(attempt.schema=="b_initial_l0_native_attempt_v2"&&attempt.request_sha256==request_sha&&attempt.intent_sha256==request.intent.sha256&&attempt.permit_sha256==request.permit.sha256&&attempt.timing_sha256==T._sha(body)&&attempt.parent_pid==timing.parent_pid,"B v2 spent attempt differs")
    T._need(abspath(argv_path)==P._member(root,"b1-argv.json"),"B v2 fixed argv path differs")
    argv,_=P._json(argv_path)
    expected=String[request.julia.path,"--startup-file=no","--compiled-modules=no","--pkgimages=no","--threads=1","--project="*REPO,joinpath(REPO,"tools","actual_b_chain","run_initial_l0_native.jl"),"--request",get(ENV,"SOLARSINDY_B_V2_REQUEST_PATH",""),"--request-sha256",request_sha,"--argv",argv_path]
    T._need(collect(argv)==expected&&abspath(PROGRAM_FILE)==joinpath(REPO,"tools","actual_b_chain","run_initial_l0_native.jl"),"B v2 fixed entry/argv differs")
    root
end
function _v2_precheck(permit::OwnedInitialPointExecutionPermitV2;argv_path=nothing)
    d,body=P._json(permit.path;fields=OWNED_V2_FIELDS)
    T._need(T._sha(body)==permit.sha256&&d.schema=="qualified_actual_initial_b_l0_owned_permit_v2"&&d.complete===true&&d.stage=="initial_l0"&&d.contract_sha256==T.CONTRACT,"B v2 raw permit differs")
    intent,ib=P._json(permit.intent_path;fields=OWNED_V2_INTENT)
    T._need(T._sha(ib)==permit.intent_sha256==d.execution_intent_sha256&&intent.schema=="b_initial_l0_execution_intent_v2"&&intent.stage=="initial_l0"&&intent.evidence_mode=="archival_assumed_completion_research"&&intent.contract_sha256==T.CONTRACT&&intent.execution_directory==d.execution_directory,"B v2 intent differs")
    _v2_scope(intent.scope)
    for name in OWNED_V2_SCOPE;T._need(T._bytes(getproperty(d,name))==T._bytes(getproperty(intent.scope,name)),"B v2 permit/intent scope differs");end
    # Bind all knowable input/reference metadata BEFORE raw prefix decoding.
    input=intent.input;T._fields(input,(:envelope,:roots,:independent_admission,:expected_view_id,:expected_admitted_seal))
    T._fields(input.envelope,(:path,:sha256));T._fields(input.independent_admission,(:path,:sha256))
    ref=intent.reference;T._fields(ref,(:completion_path,:roots,:pins));pins=_v2_canonical_ref(ref.pins)
    info=BInputs._joined_metadata_v2(input.envelope.path;expected_envelope_sha256=input.envelope.sha256,roots=Dict(String(k)=>String(v) for (k,v) in pairs(input.roots)),expected_sources=P.source_identity(),independent_admission=(;path=String(input.independent_admission.path),sha256=String(input.independent_admission.sha256)),expected_view_id=input.expected_view_id,expected_input_prefix_sha256=d.input_prefix_sha256,reference_pins=ref.pins)
    env=info.data
    T._need(env.evidence_mode=="archival_assumed_completion_research"&&env.source_sha256==BInputs.ACTUAL_SOURCE&&T._int(env.receipt_delay_hours,0,0)==0,"B v2 actual joined source header differs")
    T._need(d.source_sha256==env.source_sha256&&d.input_envelope_sha256==input.envelope.sha256&&d.input_view_id==input.expected_view_id==info.join.input_view_id&&d.admitted_input_seal==input.expected_admitted_seal&&d.input_prefix_sha256==info.join.input_prefix_sha256,"B v2 pre-body joined intent differs")
    T._need(P._sources(env.implementation_sources)==P._sources(intent.implementation_sources),"B v2 joined implementation map differs")
    for (k,e) in ((:training_boundary,:training_boundary_ms),(:validation_start,:validation_start_ms),(:validation_end,:validation_end_ms))
        T._need(getproperty(d,k)==getproperty(env,e),"B v2 pre-body split differs")
    end
    T._need(pins.completion_sha256==env.reference_completion_sha256==d.reference_completion_sha256&&pins.audit_sha256==env.reference_audit_sha256==d.reference_audit_sha256,"B v2 pre-body reference identity differs")
    T._need(BInputs._role_v2(info.members,"reference_completion").absolute==ref.completion_path,"B v2 joined reference path differs")

    own=T._hashmap(P.source_identity());T._need(P._sources(d.implementation_sources)==P._sources(intent.implementation_sources)==own,"B v2 current implementation map differs")
    for limits in (d.limits,intent.limits)
        T._fields(limits,keys(P.LIMITS));T._need(all(T._int(getproperty(limits,k),1,getproperty(P.LIMITS,k))==getproperty(P.LIMITS,k) for k in keys(P.LIMITS)),"B v2 limits differ")
    end
    request_path=get(ENV,"SOLARSINDY_B_V2_REQUEST_PATH","");request_sha=get(ENV,"SOLARSINDY_B_V2_REQUEST_SHA256","")
    T._need(isabspath(request_path)&&T._hex(request_sha),"B v2 request outside binding absent")
    r,rb=P._json(request_path;fields=OWNED_V2_REQUEST)
    T._need(T._sha(rb)==request_sha&&r.schema=="b_initial_l0_native_request_v2"&&r.operation=="initial-l0"&&r.evidence_mode==intent.evidence_mode&&r.execution_directory==d.execution_directory&&r.permit.path==permit.path&&r.permit.sha256==permit.sha256&&r.intent.path==permit.intent_path&&r.intent.sha256==permit.intent_sha256,"B v2 request/raw permit differs")
    root=_v2_context(r,request_sha,argv_path===nothing ? joinpath(r.execution_directory,"b1-argv.json") : argv_path)
    programs=intent.programs;T._fields(programs,(:driver,:entry,:versioned_owner,:preserved_owner,:utility_driver))
    for (name,path) in ((:driver,"tools/actual_b_chain/run_initial_l0_native.py"),(:entry,"tools/actual_b_chain/run_initial_l0_native.jl"),(:versioned_owner,"research/dbdt_completed_source/candidate_native_owner.py"),(:preserved_owner,"research/dbdt_completed_source/native_stage.py"),(:utility_driver,"research/dbdt_completed_source/run_candidate_stage.py"))
        ref=getproperty(programs,name);T._fields(ref,(:path,:sha256));T._need(abspath(ref.path)==joinpath(REPO,split(path,'/')...)&&P._file_sha(ref.path)[1]==ref.sha256,"B v2 fixed program differs")
    end
    T._need(d.native_owner_helper_sha256==OWNED_V2_OWNER&&d.preserved_native_owner_sha256==OWNED_V2_LEGACY&&d.native_driver_sha256==programs.driver.sha256&&d.native_entry_sha256==programs.entry.sha256,"B v2 owner/driver differs")
    T._need(Threads.nthreads()==1&&BLAS.get_num_threads()==1&&get(ENV,"OMP_NUM_THREADS","")=="1"&&d.julia_version==string(VERSION)&&d.julia_binary_sha256==P._file_sha(joinpath(Sys.BINDIR,Base.julia_exename()))[1]&&r.julia.sha256==d.julia_binary_sha256,"B v2 native runtime differs")
    T._need(Sys.free_memory()>=8589934592&&Base.Filesystem.diskstat(root).available>=17179869184,"B v2 native memory/disk admission")
    head=readchomp(Cmd(["git","-C",REPO,"rev-parse","HEAD"]))
    T._need(d.qualified_head==head&&isempty(readchomp(Cmd(["git","-C",REPO,"status","--porcelain"]))),"B v2 clean current head differs")
    proof=_v2_ref(r.source_proof);T._fields(proof,(:complete,:head,:tree,:source_files,:source_sha256,:commit_object_ids,:all_clean_and_checkout_filters_match))
    T._need(proof.complete===true&&proof.all_clean_and_checkout_filters_match===true&&proof.head==head&&proof.tree==readchomp(Cmd(["git","-C",REPO,"rev-parse","HEAD^{tree}"]))&&T._int(proof.source_files,1,8192)==length(proof.source_sha256),"B v2 whole source proof differs")
    for (name,hash) in pairs(proof.source_sha256)
        path=String(name);T._need(!isabspath(path)&&all(x->x!=".."&&x!="."&&!isempty(x),split(path,'/'))&&P._file_sha(joinpath(REPO,split(path,'/')...))[1]==hash,"B v2 whole source byte differs")
    end
    qualification=P._bound(d.qualification_file);T._need(T._sha(qualification)==d.qualification_sha256==r.b_execution_qualification.sha256&&d.qualification_file==r.b_execution_qualification.path,"B v2 qualification pin differs")
    q=P._software_qualification(qualification;schema="b_point_execution_software_qualification_v1",evidence_mode="software_qualified",head,sources=own)
    T._need(q.root_qualification_sha256==r.root_qualification.sha256&&q.source_proof_sha256==r.source_proof.sha256,"B v2 qualification root links differ")
    qualified=_v2_ref(r.root_qualification);e=qualified.candidate_pipeline_full18
    T._need(q.canonical_harness_sha256==e.harness_raw_sha256,"B v2 canonical harness raw receipt linkage differs")
    T._need(qualified.complete===true&&qualified.head==head&&qualified.source_proof.sha256==r.source_proof.sha256&&qualified.strict_docs_both_complete===true&&qualified.unfiltered_package_and_full_experiments_executed===true&&qualified.software_qualification_only===true&&e.complete===true&&e.head==head&&length(e.jobs)==18&&e.both_actual_strict_docs_passed===true&&e.unfiltered_Pkg_and_complete_examples_passed===true&&e.eight_full_resources_verified===true&&e.agreement48_verified===true,"B v2 current full18 qualification absent")
    expected=Set(vcat(["Julia $v / $os-latest" for v in ("1.12.6","1.13.1") for os in ("ubuntu","windows","macos")],["Documentation build","Julia 1.12 deployment images","Canonical development harness","Full-scale cross-runtime agreement"],["Full-scale $v / $role / $layout" for v in ("1.12.6","1.13.1") for role in ("primary","independent") for layout in ("ground-only","mixed")]))
    T._need(Set(String(j.name) for j in e.jobs)==expected&&allunique([j.job_id for j in e.jobs]),"B v2 full18 job identity differs")
    for j in e.jobs
        T._need(T._int(j.native_exit,0,0)==0&&j.timed_out===false&&j.head==head&&j.conclusion=="success"&&P._file_sha(j.raw_path)[1]==j.raw_sha256,"B v2 genuine native qualification differs")
    end
    # The parent's complete, pinned control manifest includes H names/raw and all inputs.
    before,_=P._json(joinpath(root,"CONTROLS_BEFORE.json"));controls=Pair{String,String}[]
    for (path,hash) in pairs(before)
        T._need(T._hex(hash)&&P._file_sha(String(path))[1]==hash,"B v2 supplied control changed before body admission");push!(controls,String(path)=>String(hash))
    end
    for path in (request_path,permit.path,permit.intent_path,joinpath(root,"B1_TIMING.json"),joinpath(root,"B1_OWNER_LAUNCH.json"),joinpath(root,"b1-argv.json"))
        push!(controls,path=>P._file_sha(path)[1])
    end
    (;request=r,intent,expected_input_prefix_sha256=String(d.input_prefix_sha256),permission=_OwnedInitialAdmissionV2(permit.path,permit.sha256,String(d.qualification_file),String(d.qualification_sha256),head,Tuple(controls)))
end
function _permission(a,permit::OwnedInitialPointExecutionPermitV2)
    T._need(a.view.evidence_mode==:archival_assumed_completion_research,"B v2 actual permit rejects synthetic admission")
    T._need(hasproperty(a.metadata,:input_contract_version)&&a.metadata.input_contract_version===2,"B v2 permit requires acyclic joined admission")
    checked=_v2_precheck(permit);d,_=P._json(permit.path;fields=OWNED_V2_FIELDS)
    T._need(d.source_sha256==a.view.source_sha256&&d.input_prefix_sha256==a.view.input_prefix_sha256&&d.input_view_id==a.view.view_id&&d.input_envelope_sha256==a.envelope_sha256&&d.admitted_input_seal==a.admission_id&&d.reference_completion_sha256==a.metadata.reference_completion_sha256&&d.reference_audit_sha256==a.metadata.reference_audit_sha256,"B v2 reconstructed admission differs")
    input=checked.intent.input
    T._need(input.expected_view_id==a.view.view_id&&input.expected_admitted_seal==a.admission_id&&input.envelope.sha256==a.envelope_sha256,"B v2 expected reconstructed input differs")
    checked.permission
end
function _permission_recheck(permission::_OwnedInitialAdmissionV2)
    for (path,hash) in permission.controls;T._need(P._file_sha(path)[1]==hash,"B v2 external control changed during stage");end
    T._need(P._file_sha(permission.path)[1]==permission.sha256&&P._file_sha(permission.qualification_path)[1]==permission.qualification_sha256,"B v2 authority changed")
    P.source_identity();nothing
end