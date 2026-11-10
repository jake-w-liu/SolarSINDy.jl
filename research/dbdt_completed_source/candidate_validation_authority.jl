"""Separate input-only permits. An original F fit grant is never constructed here."""
for name in (:InputPreparationAuthority,:CandidateValidationAuthority,:IndependentReadAuthority,:SelectedRefitAuthority)
    @eval struct $name
        grant_path::String
        grant_sha256::String
        source_proof_path::String
        source_proof_sha256::String
        qualification_path::String
        qualification_sha256::String
    end
end
function _spend_generated(directory,d,kind)
    path=directory*(kind===:input ? ".CANDIDATE_FIT_INPUT_ATTEMPT.json" : ".CANDIDATE_VALIDATION_ATTEMPT.json")
    bytes=canonical(Dict("schema"=>"generated_candidate_operation_attempt_v1","descriptor_sha256"=>digest(canonical(d)),
        "operation"=>d["operation"],"started_utc"=>F.clock(now(UTC)),"actual_source_authority"=>false,"native_fits"=>0);limit=16384)
    R._exclusive_write(path,bytes);(;path,bytes,started=time_ns())
end
function _close_generated(attempt)
    need(R.bounded_read(attempt.path;limit=16384)==attempt.bytes && (time_ns()-attempt.started)/1e9<=7200,"generated candidate spent attempt changed")
    job_identity();nothing
end
"""Outside pins for a finished native stage and its retained raw transcript."""
struct NativeEvidence
    receipt_path::String
    receipt_sha256::String
    raw_path::String
    raw_sha256::String
end
const NATIVE_DRIVER_FILES=("research/dbdt_completed_source/run_candidate_stage.py","research/dbdt_completed_source/run_candidate_stage.jl",
    "research/dbdt_completed_source/candidate_native_owner.py","research/dbdt_completed_source/native_stage.py")
const NATIVE_DRIVER_PINS=Tuple(digest(read(joinpath(pkgdir(SolarSINDy),split(name,'/')...))) for name in NATIVE_DRIVER_FILES)
function _checked_native_stage(evidence,operation,authority)
    need(evidence isa NativeEvidence,"native stage outside pins required")
    owner=A._pinned(evidence.receipt_path,evidence.receipt_sha256;limit=262144)
    root=realpath(pkgdir(SolarSINDy));paths=Tuple(joinpath(root,split(name,'/')...) for name in NATIVE_DRIVER_FILES)
    need(all(digest(read(path))==pin for (path,pin) in zip(paths,NATIVE_DRIVER_PINS)) &&
        NATIVE_DRIVER_PINS[4]=="8799458af4fe532c62726df503bb88a18385fe670485363290e45447737cc836","native loaded driver changed")
    need(owner["schema"]=="candidate_native_stage_evidence_v1" && owner["complete"]===true && owner["actual_authority"]===true &&
        owner["operation"]==operation && owner["driver_sha256"]==NATIVE_DRIVER_PINS[1] && owner["entry_sha256"]==NATIVE_DRIVER_PINS[2] &&
        owner["runner_sha256"]==NATIVE_DRIVER_PINS[3] && owner["legacy_runner_sha256"]==NATIVE_DRIVER_PINS[4] && owner["source_proof_sha256"]==authority.source_proof_sha256 &&
        owner["qualification_sha256"]==authority.qualification_sha256 && owner["grant_sha256"]==authority.grant_sha256 &&
        owner["job_sha256"]==job_identity() && owner["stage_seconds_limit"]==7200 && owner["raw_bytes_limit"]==64*1024^2,
        "native fixed operation/driver/source binding differs")
    request=A._pinned(owner["request_path"],owner["request_sha256"];canonical=true,limit=1048576)
    need(request["schema"]=="candidate_native_stage_request_v1" && request["operation"]==operation &&
        request["evidence_mode"]=="actual_archival_cache" && request["source_proof"]["sha256"]==authority.source_proof_sha256 &&
        request["qualification"]["sha256"]==authority.qualification_sha256,"native request/source binding differs")
    q=request["qualification_authority"]
    need(all(q[String(name)]==getproperty(authority,name) for name in fieldnames(typeof(authority))),"native request permit differs")
    grant=A._pinned(authority.grant_path,authority.grant_sha256;canonical=true,limit=131072)
    _whole_qualification(authority,grant)
    directory=request["execution_directory"];J._ordinary(directory;directory=true)
    need(J._same(abspath(evidence.receipt_path),joinpath(directory,"CANDIDATE_NATIVE_RECEIPT.json")) &&
        J._same(abspath(evidence.raw_path),joinpath(directory,"julia.log")),"native output paths differ")
    runtime=owner["runtime"];argv=owner["argv"]
    expected=String[request["julia"]["path"],"--startup-file=no","--compiled-modules=no","--pkgimages=no","--threads=1","--project="*root,
        paths[2],"--request",owner["request_path"],"--request-sha256",owner["request_sha256"],"--argv",joinpath(directory,"stage-argv.json")]
    need(argv==expected && runtime["argv"]==expected && owner["argv_sha256"]==digest(canonical(expected;limit=65536)) &&
        runtime["executable_sha256"]==request["julia"]["sha256"]==F.digest(read(first(expected))) &&
        runtime["compiled_modules"]===false && runtime["pkgimages"]===false && F.isint(runtime["blas_threads"]) &&
        F.isint(runtime["omp_threads"]) && runtime["blas_threads"]==runtime["omp_threads"]==1,"native argv/runtime differs")
    R._ownership_check(runtime,owner["ownership"],owner["descendant_cleanup"])
    inner=owner["julia_native"];outer=owner["supervisor_native"]
    need(F.isint(inner["native_exit"]) && inner["native_exit"]==0 && F.isint(outer["native_exit"]) && outer["native_exit"]==0 &&
        inner["timed_out"]===false && outer["timed_out"]===false && inner["complete"]===true && inner["raw_byte_limit_exceeded"]===false &&
        inner["reader_joined"]===true && inner["reader_pipe_close_pending"]===false &&
        inner["exact_child_reaped"]===true && outer["leader_reaped"]===true && owner["ownership_boundary_closed"]===true &&
        owner["all_descendants_os_reaped"]===false && owner["julia_leader_reaped"]===true && owner["supervisor_leader_reaped"]===true &&
        owner["owned_launch"]["pid"]==outer["pid"] && inner["pid"]!=outer["pid"] &&
        owner["native_owner_wall_seconds"]<=7200,"native inner/outer exit ownership differs")
    log,_=FS._read(evidence.raw_path,64*1024^2)
    need(digest(codeunits(log))==evidence.raw_sha256==inner["raw_sha256"]==owner["raw_sha256"] &&
        F.isint(inner["raw_bytes"]) && inner["raw_bytes"]==ncodeunits(log),"native actual raw transcript differs")
    before=owner["controls_before"];after=owner["controls_after"]
    b=A._pinned(before["path"],before["sha256"];canonical=true,limit=8*1024^2)
    a=A._pinned(after["path"],after["sha256"];canonical=true,limit=8*1024^2)
    need(b==a && before["sha256"]==after["sha256"] && owner["changed_control_count"]===0 && owner["control_read_error_count"]===0,
        "native guarded bytes changed")
    for (path,pin) in zip(paths,NATIVE_DRIVER_PINS);need(get(b,path,nothing)==pin,"native driver missing from guarded closure");end
    need(get(b,owner["request_path"],nothing)==owner["request_sha256"] && get(b,first(expected),nothing)==runtime["executable_sha256"],"native request/executable outside guard")
    internal=A._pinned(joinpath(directory,"JULIA_STAGE_RESULT.json"),owner["internal_result_sha256"];canonical=true,limit=1048576)
    need(internal["complete"]===true && internal["operation"]==operation && internal["request_sha256"]==owner["request_sha256"] &&
        internal["runtime"]==runtime && internal["job_sha256"]==job_identity(),"native internal result differs")
    required=operation=="candidate-fit" ? ("point",) : operation=="candidate-validation" ? ("output_context_sha256","output_complete_sha256") :
        ("output_witness_sha256","output_manifest_sha256","output_complete_sha256","input_descriptor_sha256")
    need(all(haskey(internal,key) && owner[key]==internal[key] for key in required),"native completed output pins differ")
    owner
end
const REQUIRED_JOBS=Set(vcat(["Julia $v / $s-latest" for v in ("1.12.6","1.13.1") for s in ("ubuntu","windows","macos")],
    ["Documentation build","Julia 1.12 deployment images","Canonical development harness","Full-scale cross-runtime agreement"],
    ["Full-scale $v / $r / $l" for v in ("1.12.6","1.13.1") for r in ("primary","independent") for l in ("ground-only","mixed")]))
function _expected_harness_checks(proof)
    prefix=["Julia executable found","Project.toml is present and nonempty","src/ contains Julia source","test/runtests.jl exists",
        "CODE_NOTES.md records implementation correctness context","TEST_REPORT.md records test coverage, tolerances, oracles, or anti-false-test checks"]
    parses=["parse ok: "*name for name in sort!(collect(keys(proof["source_sha256"]))) if endswith(name,".jl")]
    suffix=["tests contain @testset and test assertions","no obvious tautological tests found","randomness appears to use explicit seed/RNG",
        "no TODO/FIXME/HACK markers in source/tests","Pkg.test() passed","examples/experiments.jl executed"]
    vcat(prefix,parses,suffix)
end
function _locked_split(split,kind)
    expected=kind===:input ? (DateTime(2017),DateTime(2017,7,1)) : kind===:refit ?
        (DateTime(2017),DateTime(2017,10,1)) : (DateTime(2017,7,1),DateTime(2017,10,1))
    need((split.start,split.stop)==expected && split.stop<=split.asof<=DateTime(2018),"candidate locked2017 split differs")
    kind===:validation && need(split.first_issue==DateTime(2017,10,1),"candidate validation first issue differs")
end
function _whole_qualification(authority,grant)
    proof=A._pinned(authority.source_proof_path,authority.source_proof_sha256;limit=1048576)
    qual=A._pinned(authority.qualification_path,authority.qualification_sha256;limit=1048576)
    A._qualification(proof,qual,authority,grant)
    # Legacy eight-job checks remain mandatory; this extension cannot replace
    # them, and is additionally required for this new numerical pipeline.
    ext=get(qual,"candidate_pipeline_full18",nothing)
    J._object(ext,Set(("complete","head","jobs","harness_counts","harness_raw_path","harness_raw_sha256","harness_named_checks_sha256","both_actual_strict_docs_passed",
        "unfiltered_Pkg_and_complete_examples_passed","eight_full_resources_verified","agreement48_verified")))
    expected_checks=_expected_harness_checks(proof)
    need(get(proof["source_sha256"],"tools/canonical_dev_harness.sh",nothing)=="c7cb2ad1d05a77cf0462c9af6f82f17394d2fc37476fe8c68e57bd4a625c2b28","candidate canonical H producer changed")
    hlog,_=FS._read(ext["harness_raw_path"],64*1024^2)
    named=[String(m.captures[1]) for m in eachmatch(r"(?m)^[ \t]*\[PASS\] ([^\r\n]+)\r?$",hlog)]
    warnings=[String(m.captures[1]) for m in eachmatch(r"(?m)^[ \t]*\[WARN\] ([^\r\n]+)\r?$",hlog)]
    summaries=Dict(name=>[String(m.captures[1]) for m in eachmatch(Regex("(?m)^"*name*": ([0-9]+)\\r?\$"),hlog)] for name in ("PASS","WARN","FAIL"))
    need(digest(codeunits(hlog))==J._sha(ext["harness_raw_sha256"]) && named==expected_checks &&
        !occursin("[FAIL] ",hlog) && summaries==Dict("PASS"=>[string(length(expected_checks))],"WARN"=>["3"],"FAIL"=>["0"]) &&
        ext["harness_named_checks_sha256"]==digest(canonical(expected_checks;limit=262144)) && warnings==[
        "loose tolerances found; confirm each is justified","paper_dst_superlearner/data directory not found","paper_dst_superlearner/figs directory not found"],
        "candidate canonical H named checks differ")
    need(ext["complete"]===true && ext["head"]==proof["head"] && ext["harness_counts"]==Dict("PASS"=>length(expected_checks),"WARN"=>3,"FAIL"=>0) &&
        all(x->F.isint(x),values(ext["harness_counts"])) && ext["both_actual_strict_docs_passed"]===true &&
        ext["unfiltered_Pkg_and_complete_examples_passed"]===true && ext["eight_full_resources_verified"]===true &&
        ext["agreement48_verified"]===true,"candidate requires current complete18 qualification")
    jobs=ext["jobs"];need(jobs isa AbstractVector && length(jobs)==18 && Set(row["name"] for row in jobs)==REQUIRED_JOBS &&
        length(unique(row["job_id"] for row in jobs))==18,"candidate full18 job inventory")
    for row in jobs
        J._object(row,Set(("name","job_id","head","conclusion","raw_path","raw_sha256","native_exit","timed_out")))
        need(F.isint(row["job_id"]) && row["job_id"]>0 && row["head"]==proof["head"] && row["conclusion"]=="success" &&
            F.isint(row["native_exit"]) && row["native_exit"]==0 && row["timed_out"]===false,"candidate native CI stage failed")
        text,_=FS._read(row["raw_path"],64*1024^2);need(digest(codeunits(text))==J._sha(row["raw_sha256"]),"candidate CI raw changed")
    end
    for (name,pin) in zip(WORKER_NAMES,WORKER_PINS)
        need(get(proof["source_sha256"],"research/dbdt_completed_source/"*name,nothing)==pin,"candidate worker absent from qualification")
    end
    for (name,pin) in zip(NATIVE_DRIVER_FILES,NATIVE_DRIVER_PINS)
        need(get(proof["source_sha256"],name,nothing)==pin,"native driver absent from full qualification")
    end
    need(get(proof["source_sha256"],K.JOB_NAME,nothing)==K.JOB_PIN &&
        get(proof["source_sha256"],K.CODEC_NAME,nothing)==K.CODEC_PIN,"candidate K closure absent from qualification")
    proof,qual
end
function _native_evidence(evidence,point,producer)
    need(evidence isa NativeEvidence,"candidate requires original native owner/raw evidence")
    owner=_checked_native_stage(evidence,"candidate-fit",producer)
    text,_=FS._read(evidence.raw_path,64*1024^2)
    need(digest(codeunits(text))==J._sha(evidence.raw_sha256),"candidate producer raw SHA differs")
    auth=K._producer_evidence(point,producer)
    original_grant=A._pinned(producer.grant_path,producer.grant_sha256;canonical=true,limit=131072)
    _whole_qualification(producer,original_grant)
    need(owner["schema"]=="candidate_native_stage_evidence_v1" && owner["complete"]===true &&
        owner["head"]==auth["head"] && owner["grant_sha256"]==producer.grant_sha256 &&
        owner["source_proof_sha256"]==producer.source_proof_sha256 && owner["qualification_sha256"]==producer.qualification_sha256 &&
        owner["raw_sha256"]==evidence.raw_sha256 && F.isint(owner["native_exit"]) && owner["native_exit"]==0 &&
        owner["timed_out"]===false && owner["ownership_boundary_closed"]===true && owner["stage_seconds_limit"]==7200 &&
        owner["available_physical_bytes"]>=8589934592 && owner["free_disk_bytes"]>=17179869184 &&
        owner["point"]==point_descriptor(point),"candidate original native producer evidence differs")
    path=joinpath(realpath(dirname(producer.grant_path)),"POINT_FIT_ATTEMPT.json")
    bytes=R.bounded_read(path;limit=16384);attempt=decode(bytes;limit=16384)
    need(digest(bytes)==auth["fit_attempt_sha256"] && attempt["grant_sha256"]==producer.grant_sha256 &&
        attempt["context_sha256"]==point.expected.context_sha256 && attempt["recipe_sha256"]==point.expected.recipe_sha256 &&
        attempt["source_sha256"]==point.expected.source_sha256 && attempt["expected_model_fits"]==1,"candidate original fit attempt differs")
    outer_path=joinpath(realpath(dirname(producer.grant_path)),"CANDIDATE_POINT_FIT_ATTEMPT.json")
    outer=decode(R.bounded_read(outer_path;limit=16384);limit=16384)
    need(outer["schema"]=="candidate_point_fit_attempt_v1" && outer["grant_sha256"]==producer.grant_sha256 &&
        outer["context_sha256"]==point.expected.context_sha256 && recipe_for(outer["candidate_id"]).sha256==point.expected.recipe_sha256,
        "candidate fixed-grid outer fit attempt differs")
    J._sha(outer["input_descriptor_sha256"])
    owner
end
function _admit(value,index,id,split,directory,authority,kind;point=nothing,producer=nothing,native=nothing,selection=nothing)
    required=kind in (:input,:refitinput) ? InputPreparationAuthority : kind===:read ? IndependentReadAuthority : CandidateValidationAuthority
    need(authority isa required,"candidate requires distinct input-only operation authority")
    _assert_fresh_destination(directory)
    _locked_split(split,kind===:read ? :validation : kind===:refitinput ? :refit : kind)
    admission=F._archival_request(value,index,split.start,split.stop,split.asof)
    operation=kind===:refitinput ? "selected_R2_2017_refit_input_only" : kind===:input ? "fixed_candidate_2017_fit_input_only" : kind===:read ?
        "fixed_candidate_2017_independent_read_only" : "fixed_candidate_2017_validation_only"
    if kind===:refitinput
        need(selection isa CheckedR2Selection && selection.actual &&
            id==_selected_id(selection,recipe_for(id).kind,split;actual=true),"refit input requires producer-checked selected recipe")
        _selection_target(selection,value.cases,index;actual=true)
    else
        need(selection===nothing,"candidate operation cannot consume selected-refit evidence")
    end
    d=descriptor(value.cases,index,id,split;operation=kind===:read ? "fixed_candidate_2017_validation_only" : operation,
        point,actual=true,admission=digest(B.admission_bytes(value)),selection_sha256=selection===nothing ? nothing : selection.evidence_sha256)
    grant=A._pinned(authority.grant_path,authority.grant_sha256;canonical=true,limit=131072)
    fields=Set(("schema","operation","input_descriptor_sha256","source_proof_sha256","qualification_sha256","head",
        "case_read_admission_sha256","runtime","execution_directory","attempt_path","work_budget",
        "actual_source_access_authorized","actual_model_fit_authorized","new_case_generation_authorized",
        "dataset_discovery","provider_receipts_available"))
    J._object(grant,fields)
    need(grant["schema"]=="candidate_input_only_authority_v1" && grant["operation"]==operation &&
        grant["input_descriptor_sha256"]==digest(canonical(d)) && grant["source_proof_sha256"]==J._sha(authority.source_proof_sha256)==admission["reader_source_proof_sha256"] &&
        grant["qualification_sha256"]==J._sha(authority.qualification_sha256)==admission["reader_qualification_sha256"] &&
        grant["case_read_admission_sha256"]==d["case_read_admission_sha256"] && grant["runtime"]==string(VERSION) &&
        canonical(grant["work_budget"])==canonical(work_budget()) && grant["actual_source_access_authorized"]===true &&
        grant["actual_model_fit_authorized"]===false && grant["new_case_generation_authorized"]===false &&
        grant["dataset_discovery"]===false && grant["provider_receipts_available"]===false,"candidate input authority differs")
    execution=grant["execution_directory"];J._ordinary(execution;directory=true)
    need(isabspath(execution) && J._same(realpath(execution),realpath(dirname(authority.grant_path))) &&
        J._same(abspath(directory),joinpath(realpath(execution),kind===:read ? "independent-verification" : "output")),"candidate execution path differs")
    attempt=joinpath(realpath(execution),kind in (:input,:refitinput) ? "CANDIDATE_FIT_INPUT_ATTEMPT.json" :
        kind===:read ? "CANDIDATE_INDEPENDENT_READ_ATTEMPT.json" : "CANDIDATE_VALIDATION_ATTEMPT.json")
    need(grant["attempt_path"]==attempt && !ispath(attempt) && !islink(attempt),"candidate attempt must be fresh")
    proof,_=_whole_qualification(authority,grant)
    need(proof["head"]==admission["reader_head"] && BLAS.get_num_threads()==1,"candidate qualification/BLAS differs")
    need(Sys.free_memory()>=8589934592 && Base.Filesystem.diskstat(execution).available>=17179869184,"candidate native memory/disk admission")
    if !(kind in (:input,:refitinput));_native_evidence(native,point,producer);end
    auth=Dict("schema"=>"candidate_operation_admission_v1","operation"=>operation,"head"=>proof["head"],
        "grant_sha256"=>authority.grant_sha256,"source_proof_sha256"=>authority.source_proof_sha256,
        "qualification_sha256"=>authority.qualification_sha256,"input_descriptor_sha256"=>digest(canonical(d)),
        "case_read_admission_sha256"=>d["case_read_admission_sha256"],"native_fits"=>0)
    (;auth,d,attempt,grant)
end
function _admit_and_spend(value,index,id,split,directory,authority,kind;kwargs...)
    a=_admit(value,index,id,split,directory,authority,kind;kwargs...)
    bytes=canonical(Dict("schema"=>"candidate_operation_attempt_v1","grant_sha256"=>authority.grant_sha256,
        "input_descriptor_sha256"=>digest(canonical(a.d)),"operation"=>a.auth["operation"],"started_utc"=>F.clock(now(UTC)));limit=16384)
    R._exclusive_write(a.attempt,bytes)
    auth=copy(a.auth);auth["attempt_sha256"]=digest(bytes)
    auth,a.d,(path=a.attempt,bytes=bytes,started=time_ns())
end
function _close_admission(value,index,id,split,directory,authority,kind,d,attempt;kwargs...)
    # Re-admit every immutable input without allowing a second attempt. The
    # marker is independently checked and never removed, including on failure.
    need(R.bounded_read(attempt.path;limit=16384)==attempt.bytes && (time_ns()-attempt.started)/1e9<=7200,"candidate attempt/time changed")
    grant=A._pinned(authority.grant_path,authority.grant_sha256;canonical=true,limit=131072)
    _whole_qualification(authority,grant)
    point=get(kwargs,:point,nothing)
    current=descriptor(value.cases,index,id,split;operation=d["operation"],point,actual=true,admission=digest(B.admission_bytes(value)),selection_sha256=d["selection_sha256"])
    need(canonical(current)==canonical(d),"candidate admission changed during work")
    point===nothing || _native_evidence(get(kwargs,:native,nothing),point,get(kwargs,:producer,nothing))
    nothing
end
