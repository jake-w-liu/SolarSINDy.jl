function structural_runtime(tag,path)
    Dict{String,Any}("julia_version"=>"1.12.6","executable_sha256"=>repeat("1",64),"architecture"=>tag,
        "argv"=>[path,"--startup-file=no","--compiled-modules=no","--pkgimages=no","--threads=1"],
        "julia_project_sha256"=>repeat("2",64),"julia_manifest_sha256"=>repeat("3",64),
        "blas_vendor"=>"OpenBLAS/ilp64","blas_version"=>"structural-only","blas_library_sha256"=>repeat("4",64),
        "blas_threads"=>1,"omp_threads"=>1,"rng_kind"=>"none","rng_seed"=>0,"compiled_modules"=>false,"pkgimages"=>false)
end
@testset "recorded runtime OS and complete ownership pairs" begin
    windows=("CREATE_SUSPENDED -> AssignProcessToJobObject -> ResumeThread","private Windows Job Object closed with KILL_ON_JOB_CLOSE")
    posix=("start_new_session=True","owned POSIX session killed after stage")
    for tag in ("x86_64/NT","x86_64/Linux","aarch64/Linux","x86_64/Darwin","aarch64/Darwin")
        nt=endswith(tag,"/NT");runtime=structural_runtime(tag,nt ? "Q:/unopened/julia.exe" : "/unopened/julia")
        good=nt ? windows : posix;bad=nt ? posix : windows
        # Metadata validation must not follow these deliberately absent paths.
        @test R._validate_runtime(runtime)===runtime
        @test R._ownership_check(runtime,good...)===nothing
        @test_throws ArgumentError R._ownership_check(runtime,bad...)
        @test_throws ArgumentError R._ownership_check(runtime,good[1],bad[2])
        @test_throws ArgumentError R._ownership_check(runtime,"unowned",good[2])
    end
    @test_throws ArgumentError R._validate_runtime(structural_runtime("x86_64","Q:/unopened/julia.exe"))
    @test_throws ArgumentError R._validate_runtime(structural_runtime("x86_64/Unknown","/unopened/julia"))
    @test_throws ArgumentError R._validate_runtime(structural_runtime("x86_64/Linux","Q:/unopened/julia.exe"))
    @test_throws ArgumentError R._validate_runtime(structural_runtime("x86_64/NT","/unopened/julia"))
    @test_throws ArgumentError R._validate_runtime(structural_runtime("x86_64/Linux","relative/julia"))
    runtime=structural_runtime("x86_64/Linux","/unopened/julia");runtime["compiled_modules"]=true
    @test_throws ArgumentError R._validate_runtime(runtime)
    runtime=structural_runtime("x86_64/Linux","/unopened/julia");runtime["blas_threads"]=true
    @test_throws ArgumentError R._validate_runtime(runtime)
end
@testset "portable source metadata versus active source opens" begin
    for path in ("Q:/unopened/approved-source.json","/unopened/approved-source.json")
        value=protocol_fixture();value["source_permissions"][1]["path_or_null"]=path
        protocol=make_protocol(value)
        @test R._spec(protocol)["source_permissions"][1]["path_or_null"]==path
        @test_throws ArgumentError R._audit_source_bodies(protocol)
    end
end
@testset "strict residual feature count and gradual underflow" begin
    Z=zeros(10,9);Z[:,1].=1.0;y=ones(10);weights=vcat(0.0,ones(8))
    beta=vcat(1.0+2.0^-44,zeros(8));qr=vcat(1.0,zeros(8))
    eta=(big(1)//big(2)^44)/(2+big(1)//big(2)^44);u=big(1)//big(2)^53
    @test 8*30*u/(1-30*u)<eta<8*32*u/(1-32*u)
    @test_throws ArgumentError R._ridge_enclosure(Z,y,weights,beta,qr,256)
    # Warm the exact same specialization under normal arithmetic before the
    # mode switch. A cold-only probe missed this confirmed enforcement failure.
    @test R._check_prediction_bound([0.0],[0.0],[1.0],[0.0,0.0],[0.0],0.0).reference.hi==0
    old=Base.get_zero_subnormals()
    try
        supported=Base.set_zero_subnormals(true)
        @test supported isa Bool
        if supported
            @test Base.get_zero_subnormals()
            @test_throws ArgumentError R._check_prediction_bound([0.0],[0.0],[1.0],[0.0,0.0],[0.0],0.0)
            @test_throws ArgumentError R._ridge_enclosure(Z,y,weights,qr,qr,256)
            native_argv=[joinpath(Sys.BINDIR,Sys.iswindows() ? "julia.exe" : "julia"),"--startup-file=no","--compiled-modules=no","--pkgimages=no","--threads=1"]
            @test_throws ArgumentError R._inspect_runtime(native_argv)
        else
            # Julia exposes this as an optional hardware capability. There is
            # no skipped proof: verify actual gradual mode and positive cases.
            @test !Base.get_zero_subnormals()
            @test R._check_prediction_bound([0.0],[0.0],[1.0],[0.0,0.0],[0.0],0.0).reference.hi==0
            @test R._ridge_enclosure(Z,y,weights,qr,qr,256).eta.hi==0
            @test Base.set_zero_subnormals(false)
            @test !Base.get_zero_subnormals()
        end
    finally
        @test Base.set_zero_subnormals(old)
        @test Base.get_zero_subnormals()==old
    end
end
