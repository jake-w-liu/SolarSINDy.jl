function structural_job_request(protocol,operation;tag="x86_64/Linux")
    runtime=structural_runtime(tag,endswith(tag,"/NT") ? "Q:/unopened/julia.exe" : "/unopened/julia")
    permission=R.digest(R.canonical_bytes(R._spec(protocol)["source_permissions"];limit=R.MAX_METADATA_BYTES))
    roles=operation=="fit_independent_check" ? ["producer"] : operation=="calibrate" ? ["point"] :
        operation=="cal_independent_check" ? ["point","producer"] : String[]
    Dict{String,Any}("schema"=>"dbdt_completed_job_request_v1","job_id"=>"structural-shape-only",
        "operation"=>operation,"evidence_mode"=>"synthetic_only","protocol_pin"=>protocol.sha256,
        "input_manifest_pin_or_null"=>operation=="source_audit" ? nothing : repeat("1",64),
        "input_data_pin"=>repeat("2",64),
        "recipe_pin_or_null"=>operation in ("fit_ridge","fit_independent_check") ? repeat("3",64) : nothing,
        "source_permission_pin_or_null"=>operation in ("source_audit","source_independent_audit") ? permission : nothing,
        "required_runtime"=>runtime,"dependency_pins"=>Any[Dict{String,Any}("role"=>role,"manifest_sha256"=>repeat("4",64)) for role in roles],
        "work_directory"=>endswith(tag,"/NT") ? "Q:/unopened/work" : "/unopened/work")
end

@testset "closed12-field request grammar is structural and grants no handles" begin
    protocol=make_protocol(protocol_fixture())
    # These absent foreign paths are metadata; this validator performs no file,
    # active-project or Git checks and returns only an operation string.
    for tag in ("x86_64/NT","x86_64/Linux","aarch64/Linux","x86_64/Darwin","aarch64/Darwin"), operation in R.OPERATIONS
        request=structural_job_request(protocol,operation;tag)
        @test R._validate_job_request(request,protocol)==operation
    end
    invalid=Tuple{String,String,Any}[]
    for (field,values) in (
        "schema"=>Any[nothing,true,"unknown"],"job_id"=>Any[nothing,true,"",repeat("x",513),"bad\nidentifier"],
        "operation"=>Any[nothing,true,"unknown"],"evidence_mode"=>Any[nothing,true,"actual"],
        "protocol_pin"=>Any[nothing,true,Dict("untyped"=>true),"not-a-sha",repeat("f",64)],
        "input_manifest_pin_or_null"=>Any[nothing,true,Dict("untyped"=>true),"not-a-sha"],
        "input_data_pin"=>Any[nothing,true,Dict("untyped"=>true),"not-a-sha"],
        "recipe_pin_or_null"=>Any[true,Dict("untyped"=>true),Any[],"not-a-sha",repeat("f",64)],
        "source_permission_pin_or_null"=>Any[true,Dict("untyped"=>true),Any[],"not-a-sha",repeat("f",64)],
        "required_runtime"=>Any[true,Dict{String,Any}()],"dependency_pins"=>Any[nothing,true,Any[true]],
        "work_directory"=>Any[nothing,true,"relative/work"])
        for value in values;push!(invalid,("load_cal_check",field,value));end
    end
    for operation in ("fit_ridge","fit_independent_check")
        push!(invalid,(operation,"recipe_pin_or_null",nothing))
    end
    for operation in ("source_audit","source_independent_audit")
        push!(invalid,(operation,"source_permission_pin_or_null",nothing))
        push!(invalid,(operation,"source_permission_pin_or_null",repeat("f",64)))
    end
    push!(invalid,("source_audit","input_manifest_pin_or_null",repeat("f",64)))
    for (operation,field,value) in invalid
        request=structural_job_request(protocol,operation);request[field]=value
        @test_throws ArgumentError R._validate_job_request(request,protocol)
    end
    for (field,value) in (("blas_threads",true),("omp_threads",true),("rng_seed",false),("compiled_modules",0),("pkgimages",0))
        request=structural_job_request(protocol,"load_cal_check");actual=deepcopy(request["required_runtime"])
        request["required_runtime"][field]=value
        @test request["required_runtime"]==actual # Julia equality alone cannot establish wire types.
        @test_throws ArgumentError R._validate_job_request(request,protocol;runtime=actual)
    end
    for dependency in (
        Any[Dict("role"=>true,"manifest_sha256"=>repeat("1",64))],
        Any[Dict("role"=>"unknown","manifest_sha256"=>repeat("1",64))],
        Any[Dict("role"=>"producer","manifest_sha256"=>true)],
        Any[Dict("role"=>"producer","manifest_sha256"=>"not-a-sha")],
        Any[Dict("role"=>"producer","manifest_sha256"=>repeat("1",64)),Dict("role"=>"producer","manifest_sha256"=>repeat("1",64))])
        request=structural_job_request(protocol,"fit_independent_check");request["dependency_pins"]=dependency
        @test_throws ArgumentError R._validate_job_request(request,protocol)
    end
    request=structural_job_request(protocol,"cal_independent_check");reverse!(request["dependency_pins"])
    @test_throws ArgumentError R._validate_job_request(request,protocol)
    request=structural_job_request(protocol,"load_cal_check");request["extra"]=nothing
    @test_throws ArgumentError R._validate_job_request(request,protocol)
    request=structural_job_request(protocol,"load_cal_check");delete!(request["required_runtime"],"rng_kind")
    @test_throws ArgumentError R._validate_job_request(request,protocol)
    request=structural_job_request(protocol,"load_cal_check");request["required_runtime"]["extra"]=nothing
    @test_throws ArgumentError R._validate_job_request(request,protocol)
    request=structural_job_request(protocol,"load_cal_check");push!(request["required_runtime"]["argv"],"--threads=1")
    @test_throws ArgumentError R._validate_job_request(request,protocol)
end

@testset "native and data-only entry paths both enforce request shape first" begin
    protocol=make_protocol(protocol_fixture())
    request=structural_job_request(protocol,"source_audit");request["recipe_pin_or_null"]=Dict("untyped"=>true)
    failure=try R._job_common(request,protocol,Dict{String,String}(),request["required_runtime"]);nothing catch error;error end
    @test failure isa ArgumentError
    @test occursin("expected a string",sprint(showerror,failure))
    members=structural_native_members(protocol)
    members=[first(pair)=="job.json" ? first(pair)=>R._jsontext(request) : pair for pair in members]
    destination=joinpath(CODEC_TEST_DIRECTORY,"invalid-request-shape-only")
    bundle=R._publish_bundle(destination,protocol,"native_job",structural_native_context(protocol),members,0;pins=Dict{String,String}())
    # This untrusted structural bundle has no execution evidence. The request
    # type error must occur before its intentionally absent receipt fields.
    failure=try R._native_receipt_check(bundle,Dict{String,String}());nothing catch error;error end
    @test failure isa ArgumentError
    @test occursin("expected a string",sprint(showerror,failure))
end
