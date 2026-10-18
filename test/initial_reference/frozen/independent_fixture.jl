# Independent, visibly synthetic filesystem fixture. No production builder is used.
# All model/data names contain inert bytes. Nothing here is an actual forecast run.
using SHA, JSON3
fixture_sha(path)=bytes2hex(sha256(read(path)))
function new_file(path,bytes)
    mkpath(dirname(path))
    io=Base.Filesystem.open(path,Base.Filesystem.JL_O_WRONLY|Base.Filesystem.JL_O_CREAT|Base.Filesystem.JL_O_EXCL,0o600)
    try;write(io,bytes);finally;close(io);end
    fixture_sha(path)
end
function new_json(path,object)
    new_file(path,JSON3.write(object))
end
function record(root,path,digest)
    Dict{String,Any}("root"=>root,"path"=>path,"sha256"=>digest)
end
function fixture(base)
    mkdir(base)
    roots=Dict(name=>joinpath(base,name) for name in ("source","package","artifacts","reader","closure","support"))
    foreach(mkdir,values(roots))
    source_names=["tools/dst_forecaster/audit_sandu_matched_outputs.py";["s$i.txt" for i in 2:13]]
    package_names=vcat(["Project.toml","Manifest.toml","src/SolarSINDy.jl","src/analog_features.jl","src/analog_gbm.jl"],
        ["p$i.txt" for i in 6:388])
    reader_names=["independent_completed_gbm.py","bounded_bson.py","independent_gbm_math.jl","FROZEN_ORACLES.json","ARITHMETIC_ADDENDUM.md"]
    files=Dict{String,Any}()
    for (group,names) in (("source",source_names),("package",package_names),("reader",reader_names))
        files[group]=Dict(name=>new_file(joinpath(roots[group],name),"INERT SYNTHETIC $group $name\n") for name in names)
    end
    support=Dict{String,Any}()
    roles=("driver","julia","native","raw","format","synthetic_gate","closure_driver","closure_log",
        "original_log_1","original_log_2","runtime_log_1","runtime_log_2","runtime_log_3","runtime_log_4",
        "helper_log_1","helper_log_2","initial_imputed","initial_features","initial_availability",
        "final_imputed","final_features","final_availability","harness_log")
    for role in roles
        name=role*".txt";digest=new_file(joinpath(roots["support"],name),"INERT SYNTHETIC SUPPORT $role\n")
        support[role]=record("support",name,digest)
    end
    receipts=Dict{String,Any}()
    function receipt(name,object;root="closure",filename=name*".json")
        digest=new_json(joinpath(roots[root],filename),object)
        receipts[name]=record(root,filename,digest)
        digest
    end
    runtime=Dict{String,Any}("success"=>true,"source_unchanged"=>true,"files"=>files["package"],"git_commit"=>repeat("b",40),
        "runs"=>[Dict("name"=>name,"native_exit"=>0,"log_sha256"=>support["runtime_log_$i"]["sha256"])
            for (i,name) in enumerate(("serving","maturity","package","experiments"))])
    runtime_sha=receipt("runtime",runtime)
    config=Dict{String,Any}("mode"=>"synthetic_fixture","horizons"=>[4,6],"embargo_hours"=>168,"extra_dst_delay_hours"=>0,
        "extra_driver_delay_hours"=>0,"target_imputation"=>false,"novel_or_operational_superiority_claim"=>false)
    complete_sources=merge(files["source"],Dict("../runtime/"*name=>files["package"][name] for name in package_names[1:5]))
    outputs=Dict(name=>new_file(joinpath(roots["artifacts"],name),"INERT SYNTHETIC OUTPUT $name\n") for name in
        ("started.json","initial_imputation.json","initial_imputed_counts.json","gbm29_selection_h4.json",
         "gbm29_selection_h6.json","input_hourly_channels.f64","input_hourly_channels.json"))
    e=Dict{String,Any}("source_sha256"=>files["source"],"package_files"=>files["package"],"package_commit"=>repeat("b",40),
        "configuration"=>config,"completion_source_sha256"=>complete_sources,"runtime_gate_sha256"=>runtime_sha,
        "raw_hourly_matrix_sha256"=>outputs["input_hourly_channels.f64"],
        "checker_sha256"=>files["source"][source_names[1]])
    for (role,field) in (("driver","driver_sha256"),("julia","julia_binary_sha256"),("native","native_library_sha256"),
        ("raw","input_sha256"),("format","format_sha256"),("synthetic_gate","synthetic_gate_sha256"))
        e[field]=support[role]["sha256"]
    end
    c=Dict{String,Any}("complete"=>true,"input_sha256"=>e["input_sha256"],"raw_hourly_matrix_sha256"=>e["raw_hourly_matrix_sha256"],
        "source_sha256"=>complete_sources,"configuration"=>config,"no_original_model_parity"=>true,"no_live_serving_changes"=>true,
        "native"=>Dict("sha256"=>e["native_library_sha256"],"version"=>[3,2,0],"live_objects"=>0),"output_sha256"=>outputs)
    completion_sha=receipt("completion",c;root="artifacts",filename="completion.json")
    e["completion_sha256"]=completion_sha
    files["artifacts"]=merge(outputs,Dict("completion.json"=>completion_sha))
    t=Dict{String,Any}(key=>deepcopy(e[key]) for key in ("driver_sha256","runtime_gate_sha256","synthetic_gate_sha256",
        "source_sha256","package_files","package_commit","input_sha256","format_sha256","native_library_sha256","julia_binary_sha256"))
    merge!(t,Dict("success"=>true,"all13_runner_files_unchanged"=>true,"all388_package_files_unchanged"=>true,
        "independent_confirmation"=>false,"novelty_or_operational_superiority"=>false,
        "runs"=>[Dict("name"=>name,"native_exit"=>0,"timed_out"=>false,"log_sha256"=>support["original_log_$i"]["sha256"])
            for (i,name) in enumerate(("real_fit","independent_reader"))]))
    terminal_sha=receipt("terminal",t)
    a=Dict{String,Any}("complete"=>true,"completion_sha256"=>completion_sha,"checker_sha256"=>e["checker_sha256"],
        "source_mode"=>"synthetic_fixture","horizons"=>[4,6],"checks"=>100,"new_model_fitting"=>false,
        "author_or_ml_library_imported"=>false,"scientific_verdict"=>false)
    receipt("legacy",a)
    q=Dict{String,Any}("complete"=>true,"real_output_or_raw_read"=>false,"original_fixture_unchanged"=>true,
        "all13_runner_sources_unchanged"=>true,"all388_runtime_files_unchanged"=>true,"source_sha256"=>files["reader"],
        "cases"=>[Dict("expectation_met"=>true,"expected_rejection"=>isodd(i),"native_exit"=>isodd(i) ? 1 : 0) for i in 1:25],
        "unit_runs"=>[Dict("native_exit"=>0) for _ in 1:3])
    qualification_sha=receipt("qualification",q)
    h=Dict{String,Any}("success"=>true,"native_exit"=>0,"changed_package_files"=>[],"changed_reader_files"=>[],
        "changed_support_files"=>[],"package_files_before"=>Dict("h$i"=>repeat("c",64) for i in 1:395),
        "reader_source_sha256"=>files["reader"],"qualification_sha256"=>qualification_sha,"log_sha256"=>support["harness_log"]["sha256"])
    receipt("harness",h)
    p=Dict{String,Any}("schema"=>"sandu_additive_real_closure_inputs_v1","reader_sources"=>files["reader"])
    for (field,key) in (("terminal","terminal"),("completion","completion"),("legacy_oracle","legacy"),("runtime_gate","runtime"),
        ("qualification","qualification"),("full_harness","harness"))
        p[field]=Dict("path"=>"ORIGINAL SYNTHETIC PROVENANCE "*field,"sha256"=>receipts[key]["sha256"])
    end
    plan_sha=receipt("closure_plan",p)
    stages=Dict(stage=>Dict{String,Any}("fit_count"=>Dict("4"=>24,"6"=>22),
        "input_sha256"=>support[stage*"_imputed"]["sha256"],"features_sha256"=>support[stage*"_features"]["sha256"],
        "availability_sha256"=>support[stage*"_availability"]["sha256"]) for stage in ("initial","final"))
    s=Dict{String,Any}("schema"=>"sandu_completed_gbm_independent_semantics_v1","complete"=>true,"mode"=>"synthetic_fixture",
        "completion_sha256"=>completion_sha,"artifact_sha256"=>files["artifacts"],"original_execution_source_sha256"=>complete_sources,
        "source_sha256"=>files["reader"],"julia_binary_sha256"=>e["julia_binary_sha256"],
        "helper_runs"=>[Dict("native_exit"=>0,"binary_sha256"=>e["julia_binary_sha256"],"log_sha256"=>support["helper_log_$i"]["sha256"]) for i in 1:2],
        "stages"=>stages,"checks"=>29,"gbm_predictions"=>28,"gbm_fallbacks"=>2,"new_model_fitting"=>false,
        "scientific_verdict"=>false,"author_or_ml_library_imported"=>false,"native_training_availability_independently_reconstructed"=>false)
    semantic_sha=receipt("semantics",s)
    z=Dict{String,Any}("schema"=>"sandu_completed_gbm_independent_closure_v1","complete"=>true,"native_exit"=>0,
        "new_model_fitting"=>false,"source_run_modified"=>false,"independent_confirmation"=>false,"scientific_verdict"=>false,
        "original_terminal_sha256"=>terminal_sha,"original_completion_sha256"=>completion_sha,"additive_receipt_sha256"=>semantic_sha,
        "original_source_sha256"=>files["source"],"original_runtime_sha256"=>files["package"],"original_execution_source_sha256"=>complete_sources,
        "reader_source_sha256"=>files["reader"],"log_sha256"=>support["closure_log"]["sha256"],"driver_sha256"=>support["closure_driver"]["sha256"],
        "checks"=>29,"gbm_predictions"=>28,"gbm_fallbacks"=>2,"stages"=>stages,"original_artifact_sha256"=>files["artifacts"],
        "plan_sha256"=>plan_sha,"verified_input_plan"=>p)
    receipt("closure",z)
    manifest=Dict{String,Any}("schema"=>"bound_initial_reference_inputs_v1","mode"=>"synthetic_fixture","expected"=>e,
        "files"=>files,"receipts"=>receipts,"support"=>support,"root_roles"=>sort!(collect(keys(roots))))
    path=joinpath(base,"manifest.json");digest=new_json(path,manifest)
    (base=base,roots=roots,path=path,sha256=digest,plan=manifest,terminal=t,completion=c,legacy=a,runtime=runtime,
        semantics=s,closure=z,closure_plan=p,qualification=q,harness=h)
end
