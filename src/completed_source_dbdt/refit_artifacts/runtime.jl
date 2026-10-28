const _PACKAGE_ROOT = normpath(joinpath(@__DIR__,"..","..",".."))
const _OWNED_RUNNER_SHA = "0b0f2327841230e5ac87d03bd67d1cbaf10717e0a52ce0134d71169a36146b89"
_stream_sha(path) = open(io->bytes2hex(SHA.sha256(io)),path)

function _git_read(arguments...)
    git=Sys.which("git");require(git!==nothing,"Git is required for native research execution")
    strip(read(Cmd(vcat([git,"-C",_PACKAGE_ROOT],String[arguments...])),String))
end
function _code_files(stage::String)
    common=["src/completed_minute_replay.jl","src/completed_source_dbdt/CompletedSourceDBDT.jl",
        "src/completed_source_dbdt/adapter.jl","src/completed_source_dbdt/reference/dbdt_causal_forecaster.jl"]
    for name in ("CompletedRefitArtifacts.jl","fields.jl","codec.jl","protocol.jl","types.jl","store.jl","runtime.jl","source_input.jl","jobs.jl","publication.jl")
        push!(common,"src/completed_source_dbdt/refit_artifacts/"*name)
    end
    append!(common,["research/dbdt_completed_source/run_fit_job.jl","research/dbdt_completed_source/finalize_artifact.jl"])
    append!(common,["research/dbdt_completed_source/native_stage.py","research/dbdt_completed_source/run_synthetic_chain.py",
        "research/dbdt_completed_source/synthetic_sources.py","research/dbdt_completed_source/synthetic_checkpoint.py"])
    if stage in ("fit_ridge","fit_independent_check","refit_point","calibrate","cal_independent_check","load_cal_check","calibration")
        append!(common,["src/completed_source_dbdt/refit_artifacts/ridge.jl","src/completed_source_dbdt/refit_artifacts/numerical_audit.jl","src/completed_source_dbdt/refit_artifacts/calibration.jl"])
    end
    sort!(unique(common))
end
function _code_hashes(stage::String)
    files=_code_files(stage)
    require(length(files)<=MAX_FIELDS,"relevant code map exceeds declared object bound")
    result=Dict{String,Any}(name=>_stream_sha(joinpath(_PACKAGE_ROOT,name)) for name in files)
    result["_source_inventory"]=digest(_source_inventory_bytes())
    require(length(result)<=MAX_FIELDS,"relevant code/inventory map exceeds declared bound")
    result
end
function _context_codes(context,stage::String)
    codes=context["code_hashes"];sha_string(codes["_source_inventory"])
    expected=Dict{String,Any}(name=>_stream_sha(joinpath(_PACKAGE_ROOT,name)) for name in _code_files(stage))
    require(Dict(k=>v for (k,v) in codes if k!="_source_inventory")==expected,"relevant loaded source identities differ")
    require(context["julia_project_sha256"]==_stream_sha(joinpath(_PACKAGE_ROOT,"Project.toml")) &&
        context["julia_manifest_sha256"]==_stream_sha(joinpath(_PACKAGE_ROOT,"Manifest.toml")),"project/manifest identity differs")
    nothing
end

function _source_inventory_bytes()
    require(isempty(_git_read("status","--porcelain","--untracked-files=normal")),"native inventory requires a clean source checkpoint")
    names=split(_git_read("ls-files"),'\n')
    require(0<length(names)<=MAX_ARRAY && issorted(names) && length(unique(names))==length(names),"tracked source inventory bound/order differs")
    rows=Any[]
    for name in names
        require(!isabspath(name) && !isempty(name) && !(".." in split(replace(name,'\\'=>'/'),'/')),"unsafe inventory source path")
        path=joinpath(_PACKAGE_ROOT,name)
        require(isfile(path) && !islink(path),"source inventory requires ordinary files")
        push!(rows,Any[name,_stream_sha(path)])
    end
    canonical_bytes(Any[_git_read("rev-parse","HEAD"),_git_read("rev-parse","HEAD^{tree}"),rows])
end
function _inventory_check(bytes,context;check_current=false)
    value=_array(decode_canonical(bytes));require(length(value)==3,"invalid native source inventory shape")
    require(_gitsha(value[1])==context["package_commit"] && _gitsha(value[2])==context["package_tree"],"native source inventory commit/tree differs")
    names=String[];mapping=Dict{String,String}()
    for entry in _array(value[3];nonempty=true)
        pair=_array(entry);require(length(pair)==2,"invalid inventory entry")
        name=bounded_string(pair[1];nonempty=true)
        require(!isabspath(name) && !(".." in split(replace(name,'\\'=>'/'),'/')),"unsafe source inventory path")
        require(!haskey(mapping,name),"duplicate source inventory path")
        mapping[name]=sha_string(pair[2]);push!(names,name)
        check_current && require(isfile(joinpath(_PACKAGE_ROOT,name)) && !islink(joinpath(_PACKAGE_ROOT,name)) &&
            _stream_sha(joinpath(_PACKAGE_ROOT,name))==mapping[name],"tracked source bytes differ from native inventory")
    end
    require(issorted(names),"native source inventory is not ordered")
    for (name,pin) in context["code_hashes"]
        name=="_source_inventory" && continue
        require(get(mapping,name,nothing)==pin,"loaded code is absent/different in native inventory")
    end
    nothing
end

function _runtime_id(runtime)
    _schema(runtime,"runtime")
    # Invocation paths bind the native receipt independently through argv_sha256.
    # They are not a numerical-runtime identity: fit/cal/load jobs have different
    # input/output paths even on the exact same executable/library/thread setup.
    value=Dict{String,Any}(k=>v for (k,v) in runtime if k!="argv")
    digest(canonical_bytes(value;limit=MAX_METADATA_BYTES))
end
function _library_text(handle,names)
    for name in names
        pointer=LinearAlgebra.Libdl.dlsym(handle,name;throw_error=false)
        if pointer!==nothing && pointer!=C_NULL
            result=ccall(pointer,Cstring,())
            require(result!=C_NULL,"BLAS identity query returned null")
            return unsafe_string(result)
        end
    end
    fail("BLAS does not expose the required verified identity")
end
function _inspect_runtime(argv;native=true)
    if native
        require(!Base.get_zero_subnormals(),"native job requires gradual underflow")
        require(Threads.nthreads()==1 && BLAS.get_num_threads()==1,"native fitting/checking requires one Julia/BLAS thread")
        require(get(ENV,"OMP_NUM_THREADS","")=="1","OMP_NUM_THREADS must be explicitly1")
        require(VERSION in (v"1.12.6",v"1.13.1"),"unsupported synthetic native Julia version")
        require(Base.JLOptions().use_compiled_modules==0 && Base.JLOptions().use_pkgimages==0,"native job requires compiled-modules=no/pkgimages=no")
    end
    libraries=BLAS.get_config().loaded_libs
    active=filter(lib->lib.interface==(Base.USE_BLAS64 ? :ilp64 : :lp64),libraries)
    require(length(active)==1,"native BLAS must have one unambiguous active backend")
    library=only(active)
    path=LinearAlgebra.Libdl.dlpath(library.handle)
    require(isabspath(path) && isfile(path) && occursin("openblas",lowercase(basename(path))),"unqualified native BLAS backend")
    version=_library_text(library.handle,("openblas_get_config64_","openblas_get_config"))
    core=_library_text(library.handle,("openblas_get_corename64_","openblas_get_corename"))
    executable=joinpath(Sys.BINDIR,Sys.iswindows() ? "julia.exe" : "julia")
    arguments=String[bounded_string(v;nonempty=true) for v in _array(argv;nonempty=true)]
    require(isabspath(first(arguments)) && _stream_sha(first(arguments))==_stream_sha(executable),"native executable argument differs")
    Dict{String,Any}("julia_version"=>string(VERSION),"executable_sha256"=>_stream_sha(executable),
        "architecture"=>string(Sys.ARCH)*"/"*string(Sys.KERNEL),"argv"=>arguments,
        "julia_project_sha256"=>_stream_sha(joinpath(_PACKAGE_ROOT,"Project.toml")),
        "julia_manifest_sha256"=>_stream_sha(joinpath(_PACKAGE_ROOT,"Manifest.toml")),
        "blas_vendor"=>"OpenBLAS/"*String(library.interface),"blas_version"=>version*";core="*core,
        "blas_library_sha256"=>_stream_sha(path),"blas_threads"=>BLAS.get_num_threads(),
        "omp_threads"=>something(tryparse(Int,get(ENV,"OMP_NUM_THREADS","")),0),
        "rng_kind"=>"none","rng_seed"=>0,"compiled_modules"=>Base.JLOptions().use_compiled_modules!=0,
        "pkgimages"=>Base.JLOptions().use_pkgimages!=0)
end

function _runtime_os(runtime)
    tag=bounded_string(runtime["architecture"];nonempty=true)
    require(tag in ("x86_64/NT","x86_64/Linux","aarch64/Linux","x86_64/Darwin","aarch64/Darwin"),"unsupported/ambiguous recorded architecture and OS")
    last(split(tag,'/'))
end
function _ownership_check(runtime,ownership,cleanup)
    os=_runtime_os(runtime)
    expected=os=="NT" ? ("CREATE_SUSPENDED -> AssignProcessToJobObject -> ResumeThread","private Windows Job Object closed with KILL_ON_JOB_CLOSE") :
        ("start_new_session=True","owned POSIX session killed after stage")
    require((ownership,cleanup)==expected,"unsupported, mixed, unowned or wrong-OS native ownership tuple")
    nothing
end
function _same_numeric_runtime(context)
    # This is a diagnostic choice, not a creation credential. A normal loader
    # can run without Git or native launch flags. Unsupported current identity
    # selects foreign arithmetic checks, never a fabricated matching runtime.
    argv=[joinpath(Sys.BINDIR,Sys.iswindows() ? "julia.exe" : "julia")]
    try
        runtime=_inspect_runtime(argv;native=false)
        _runtime_id(runtime)==context["runtime_id"]
    catch error
        error isa InterruptException && rethrow()
        false
    end
end

function _validate_runtime(runtime)
    _schema(runtime,"runtime")
    require(runtime["julia_version"] in ("1.12.6","1.13.1"),"unsupported native runtime version")
    for name in ("executable_sha256","julia_project_sha256","julia_manifest_sha256","blas_library_sha256");sha_string(runtime[name]);end
    os=_runtime_os(runtime);bounded_string(runtime["blas_vendor"];nonempty=true);bounded_string(runtime["blas_version"];nonempty=true)
    argv=_array(runtime["argv"];nonempty=true)
    require(all(v->v isa String && ncodeunits(v)<=MAX_STRING_BYTES,argv) && _absolute_for_os(first(argv),os),"invalid recorded absolute native argv for its OS")
    require(os=="NT" ? endswith(lowercase(first(argv)),"julia.exe") : endswith(first(argv),"/julia"),"recorded native executable basename differs")
    for flag in ("--startup-file=no","--compiled-modules=no","--pkgimages=no","--threads=1")
        require(count(==(flag),argv)==1,"native required option is absent/duplicated")
    end
    require(checked_integer(runtime["blas_threads"],1,1)==1 && checked_integer(runtime["omp_threads"],1,1)==1,"native thread identity differs")
    require(runtime["rng_kind"]=="none" && checked_integer(runtime["rng_seed"],0,0)==0,"unexpected ridge RNG")
    require(runtime["compiled_modules"]===false && runtime["pkgimages"]===false,"native image flags differ")
    runtime
end

function _common_context(protocol::RefitProtocol,stage::String,runtime;source=nothing,schedule=nothing)
    _validate_runtime(runtime)
    require(isempty(_git_read("status","--porcelain","--untracked-files=normal")),"native research jobs require an exact clean source checkpoint")
    spec=_spec(protocol)
    result=Dict{String,Any}(k=>deepcopy(spec[k]) for k in ("protocol_id","station","ground_product","wind_product_or_null","units","layout","feature_names","cohort_kind","policy","availability_kind","measured_receipts_available"))
    merge!(result,Dict{String,Any}("protocol_sha256"=>protocol.sha256,"source_manifest_sha256_or_null"=>_nullable_sha(source),
        "schedule_manifest_sha256_or_null"=>_nullable_sha(schedule),"package_commit"=>_git_read("rev-parse","HEAD"),
        "package_tree"=>_git_read("rev-parse","HEAD^{tree}"),"code_hashes"=>_code_hashes(stage),
        "julia_project_sha256"=>runtime["julia_project_sha256"],"julia_manifest_sha256"=>runtime["julia_manifest_sha256"],"runtime_id"=>_runtime_id(runtime)))
    result
end
function _draft_common(protocol::RefitProtocol,stage::String;source=nothing,schedule=nothing)
    argv=[joinpath(Sys.BINDIR,Sys.iswindows() ? "julia.exe" : "julia"),"--startup-file=no","--compiled-modules=no","--pkgimages=no","--threads=1"]
    _common_context(protocol,stage,_inspect_runtime(argv);source,schedule)
end
function _with_kind(common,kind_context)
    value=deepcopy(common)
    require(!haskey(value,"kind_context"),"scientific common context must not contain future kind context")
    value["kind_context"]=kind_context
    value
end

function _controls(value)
    rows=_array(value;nonempty=true)
    names=String[];result=Dict{String,String}()
    for row in rows
        _schema(row,"native_control_entry");name=bounded_string(row["name"];nonempty=true)
        require(!haskey(result,name),"duplicate native control")
        result[name]=sha_string(row["sha256"]);push!(names,name)
    end
    require(issorted(names),"native controls must be sorted")
    result
end
function _deps(request)
    _schema(request,"job_request");result=Dict{String,String}();names=String[]
    for row in _array(request["dependency_pins"])
        _schema(row,"dependency_pin_entry");role=identifier(row["role"])
        require(!haskey(result,role),"duplicate dependency role")
        result[role]=sha_string(row["manifest_sha256"]);push!(names,role)
    end
    require(issorted(names),"dependency role map is not canonical")
    result
end
function _validate_job_request(request,protocol::RefitProtocol;runtime=nothing)
    _schema(request,"job_request")
    require(request["schema"]=="dbdt_completed_job_request_v1","native request schema differs")
    identifier(request["job_id"])
    op=bounded_string(request["operation"];nonempty=true)
    require(op in OPERATIONS && request["evidence_mode"]=="synthetic_only" &&
        sha_string(request["protocol_pin"])==protocol.sha256,"unsupported native request")
    input=_nullable_sha(request["input_manifest_pin_or_null"])
    sha_string(request["input_data_pin"])
    recipe=_nullable_sha(request["recipe_pin_or_null"])
    permission=_nullable_sha(request["source_permission_pin_or_null"])
    require(op=="source_audit" ? input===nothing : input!==nothing,"native input manifest null applicability differs")
    require(op in ("fit_ridge","fit_independent_check") ? recipe!==nothing : recipe===nothing,
        "native recipe pin null applicability differs")
    if op in ("source_audit","source_independent_audit")
        expected=digest(canonical_bytes(_spec(protocol)["source_permissions"];limit=MAX_METADATA_BYTES))
        require(permission==expected,"native source permission pin differs")
    else
        require(permission===nothing,"native source permission pin must be null for this operation")
    end
    required=request["required_runtime"]
    _validate_runtime(required)
    if runtime!==nothing
        _validate_runtime(runtime)
        require(required==runtime,"native inspected runtime differs from request")
    end
    work=bounded_string(request["work_directory"];nonempty=true)
    require(_absolute_for_os(work,_runtime_os(required)),"invalid recorded native work directory for its OS")
    deps=_deps(request)
    expected=op=="fit_independent_check" ? Set(["producer"]) : op=="calibrate" ? Set(["point"]) :
        op=="cal_independent_check" ? Set(["point","producer"]) : Set{String}()
    require(Set(keys(deps))==expected,"native dependency roles differ")
    op
end

function _guard_file(path,pin;limit=MAX_MEMBER_BYTES)
    bytes=bounded_read(path;limit)
    require(digest(bytes)==sha_string(pin),"caller-pinned file bytes differ")
    bytes
end

function _native_receipt_check(bundle::_Bundle,pins)
    require(bundle.kind=="native_job","expected committed native job")
    request=_member_json(bundle,"job.json");_validate_job_request(request,bundle.protocol)
    receiptbytes=_member(bundle,"native-receipt.json");receipt=decode_canonical(receiptbytes);_schema(receipt,"native_receipt")
    internal=_member_json(bundle,"internal-receipt.json");_schema(internal,"job_internal_receipt")
    context=decode_canonical(bundle.context);k=context["kind_context"];op=k["operation"]
    require(receipt["schema"]=="dbdt_owned_native_receipt_v1" && receipt["operation"]==op && request["operation"]==op && internal["operation"]==op,"native operation differs")
    require(request["schema"]=="dbdt_completed_job_request_v1" && internal["schema"]=="dbdt_completed_job_internal_receipt_v1","native request/receipt schema differs")
    require(request["evidence_mode"]=="synthetic_only" && request["protocol_pin"]==bundle.protocol.sha256,"native request protocol differs")
    require(receipt["protocol_sha256"]==bundle.protocol.sha256 && internal["protocol_sha256"]==bundle.protocol.sha256,"native receipt protocol differs")
    require(receipt["job_id"]==identifier(request["job_id"]),"native job id differs")
    _validate_runtime(receipt["runtime"])
    require(receipt["runtime"]==request["required_runtime"],"native runtime/argv is not the caller-requested identity")
    require(receipt["argv_sha256"]==digest(canonical_bytes(receipt["runtime"]["argv"])) && receipt["runner_sha256"]==_OWNED_RUNNER_SHA,"native invocation/runner pin differs")
    require(checked_integer(receipt["native_exit"],0,0)==0 && receipt["timed_out"]===false && internal["completed"]===true,"native job did not succeed")
    _ownership_check(receipt["runtime"],receipt["ownership"],receipt["descendant_cleanup"])
    require(read_time_text(receipt["started_utc"])<=read_time_text(receipt["finished_utc"]),"native receipt clock runs backwards")
    controls=_controls(receipt["controls_before_sha256"])
    require(controls==_controls(receipt["controls_after_sha256"]),"native guarded inputs/source changed")
    for (name,pin) in (("protocol",bundle.protocol.sha256),("request",digest(_member(bundle,"job.json"))),
                      ("source_inventory",context["code_hashes"]["_source_inventory"]),
                      ("runtime_executable",receipt["runtime"]["executable_sha256"]),("runtime_blas",receipt["runtime"]["blas_library_sha256"]))
        require(get(controls,name,nothing)==pin,"required native control is absent/different: "*name)
    end
    inputpin=request["input_manifest_pin_or_null"]
    inputpin!==nothing && require(get(controls,"input",nothing)==inputpin,"required native input control differs")
    inventorypin=context["code_hashes"]["_source_inventory"]
    inventory=_guard_file(_resolve(pins,inventorypin;directory=false),inventorypin)
    _inventory_check(inventory,context)
    require(receipt["input_manifest_sha256_or_null"]==request["input_manifest_pin_or_null"]==internal["input_manifest_sha256_or_null"]==k["input_manifest_sha256_or_null"],"native input manifest differs")
    require(receipt["input_data_sha256"]==request["input_data_pin"]==internal["input_data_sha256"]==k["input_data_sha256"],"native input data differs")
    payloadnames=_chunk_catalog([first(p) for p in bundle.members],"payload";single="payload.json")
    chunks=[_member(bundle,n) for n in payloadnames]
    payloadpin=length(chunks)==1 ? digest(only(chunks)) : _frame_digest("DBDTPAYLOAD1",chunks)
    require(receipt["payload_sha256"]==internal["payload_sha256"]==k["payload_sha256"]==payloadpin,"native payload bytes differ")
    require(receipt["internal_receipt_sha256"]==k["internal_receipt_sha256"]==digest(_member(bundle,"internal-receipt.json")),"native internal receipt differs")
    require(receipt["log_sha256"]==digest(_member(bundle,"native.log")) && k["native_receipt_sha256"]==digest(receiptbytes),"native log/external receipt differs")
    require(internal["loaded_code_hashes"]==context["code_hashes"] && internal["runtime_id"]==context["runtime_id"]==_runtime_id(receipt["runtime"]),"native internal code/runtime differs")
    _context_codes(context,op)
    # A receipt copied inside its own untrusted artifact is not a caller trust
    # anchor. The reviewed external receipt must be separately pinned to an
    # ordinary file outside the committed native-job directory.
    external=_resolve(pins,digest(receiptbytes);directory=false)
    relative=relpath(realpath(external),bundle.directory)
    require(relative==".." || startswith(relative,".."*string(Base.Filesystem.path_separator)),"native receipt lacks an external caller trust anchor")
    require(_guard_file(external,digest(receiptbytes);limit=MAX_METADATA_BYTES)==receiptbytes,"external native receipt differs")
    receipt
end
