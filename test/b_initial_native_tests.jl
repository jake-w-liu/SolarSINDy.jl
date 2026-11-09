# Independent generated full36 handoff and fixed owned Python controls.
module BInitialNativePackageTests
using Test,SHA,Dates,JSON3,SolarSINDy
include(joinpath(pkgdir(SolarSINDy),"tools","actual_b_chain","run_initial_l0_native.jl"))
include(joinpath(@__DIR__,"completed_refit_artifacts","run_ci_native_chain.jl"))
module OriginalBRoute
    using SolarSINDy
    include(joinpath(pkgdir(SolarSINDy),"examples","support","b_point_generated_fixture.jl"))
end
const ORIGINAL=OriginalBRoute.BPointSyntheticFixture
const SUPPORT=CompletedRefitNativeDriver
const CONTROL=joinpath(@__DIR__,"b_initial_native","run_controls.py")
const CONTROL_SHA="177bda7d43e455ac6907cc7c5cfb8fe94ea0dacb7c767f6f5b7351c873414f3e"
const TREE_ORACLE=joinpath(@__DIR__,"b_point_training","tree_inventory_oracles_v1.json")
const INDEPENDENT=joinpath(@__DIR__,"b_point_training","independent_literals_v1.json")

file_sha(path)=open(io->bytes2hex(sha256(io)),path)
json(path)=JSON3.read(read(path),Dict{String,Any})
float_bits(x)=reinterpret(Float64,parse(UInt64,String(x);base=16))

function run_python_controls()
    @test file_sha(CONTROL)==CONTROL_SHA
    python=SUPPORT.choose_python()
    base=haskey(ENV,"RUNNER_TEMP") ? realpath(ENV["RUNNER_TEMP"]) : realpath(tempdir())
    parent=realpath(mktempdir(base;prefix="solarsindy-b1-native-controls-",cleanup=false))
    output=joinpath(parent,"proof")
    command=addenv(Cmd([python,"-B","-X","utf8",CONTROL,"--output",output]),
        "PYTHONDONTWRITEBYTECODE"=>"1","TMP"=>parent,"TEMP"=>parent,"TMPDIR"=>parent)
    SUPPORT._run_support_command(command,parent)
    path=joinpath(output,"TERMINAL.json")
    isfile(path)&&!islink(path)||error("ordinary B1 control receipt required")
    before=stat(path);0<before.size<=131072||error("B1 receipt byte bound")
    body=open(io->read(io,131073),path);after=stat(path)
    length(body)==before.size==after.size&&before.inode==after.inode&&
        before.mtime==after.mtime&&before.ctime==after.ctime||error("B1 receipt changed")
    receipt=JSON3.read(body,Dict{String,Any})
    @test receipt["schema"]=="b1_generated_native_controls_v1"
    @test receipt["complete"]===true&&receipt["methods"]==30
    @test receipt["native"]["native_exit"] isa Integer&&!(receipt["native"]["native_exit"] isa Bool)
    @test receipt["native"]["native_exit"]==0&&receipt["native"]["timed_out"]===false
    @test receipt["native"]["leader_reaped"]===true&&receipt["native"]["ownership_boundary_closed"]===true
    @test isempty(receipt["changed_controls"])
    @test receipt["controls_before_sha256"]==receipt["controls_after_sha256"]
    @test file_sha(joinpath(output,"native.log"))==receipt["raw_sha256"]
    @test receipt["Julia_executed"]===false&&receipt["actual_observations"]==0
    @test receipt["source_authority_granted"]===false&&receipt["scientific_qualification"]===false
    println("B1_GENERATED_NATIVE_CONTROL_RECEIPTS=",output)
    nothing
end

function check_generated(generated,oracle)
    stage=generated.stage.directory;result=generated.result
    completion=json(joinpath(stage,"completion.json"))
    @test result["schema"]=="b_initial_l0_internal_result_v2"&&result["complete"]===true
    @test result["evidence_mode"]=="synthetic_only"&&result["actual_source_access"]===false
    for (key,value) in (("initial_hazard_fit_count",2),("initial_gain_fit_count",36),
            ("state_schedule_count",9),("beta_design_count",18),("final_fit_count",0),
            ("calibration_fit_count",0),("verified_model_count",36),("verified_certificate_count",36),
            ("verified_hazard_fit_calls",0),("verified_gain_fit_calls",0),("verified_raw_replay_A_streams",9))
        @test result[key] isa Integer&&!(result[key] isa Bool)&&result[key]==value
    end
    @test result["selected_index"]===nothing&&result["scientific_verdict"]===false&&result["serving_promotion"]===false
    @test completion["actual_execution_permit_sha256"]===nothing&&completion["qualified_head"]===nothing
    @test completion["final_fit_count"]==completion["calibration_fit_count"]==0
    for key in ("exposed_development_scoring","joint_primary_selection","scientific_verdict","serving_promotion")
        @test completion[key]===false
    end
    @test completion["provisional_selection"]["selected_index"]===nothing
    @test [law["beta"] for law in completion["hazards"]]==[5,20]
    @test [model["setting_index"] for model in completion["models"]]==collect(1:36)
    @test generated.admission.view.source_hours==1024
    # Direct receipt-clock union: i in512:852 gives852-512+1=341.
    @test length(generated.reference.rows)==341
    @test first(generated.reference.rows).issue==DateTime(1900)+Hour(512)
    @test last(generated.reference.rows).issue==DateTime(1900)+Hour(852)
    files=String[];directories=String[]
    for (parent,dirs,names) in walkdir(stage;follow_symlinks=false)
        append!(files,[replace(relpath(joinpath(parent,n),stage),'\\'=>'/') for n in names])
        append!(directories,[replace(relpath(joinpath(parent,n),stage),'\\'=>'/') for n in dirs])
    end
    @test sort(files)==sort(vcat(String.(oracle["files"]),["completion.json"]))
    @test sort(directories)==sort(String.(oracle["directories"]))
    @test length(files)==424&&length(directories)==116
    @test result["output_file_count"]==completion["output_file_count"]==423
    @test result["output_directory_count"]==completion["output_directory_count"]==116
    @test result["completion_sha256"]==generated.stage.completion_sha256==file_sha(joinpath(stage,"completion.json"))
    @test result["output_sha256"]==completion["output_sha256"]
    support=json(joinpath(stage,"support.json"))
    @test support["all36_checked"]===true&&support["accepted"]===true&&support["gain_fit_count"]==0
    @test [row["setting_index"] for row in support["settings"]]==collect(1:36)
    @test all(row->row["informative_origins"]>=100&&row["supported"]===true,support["settings"])
    for key in ("fit_elapsed_ns","readback_elapsed_ns","worker_elapsed_ns","julia_peak_rss_bytes")
        @test result[key] isa Integer&&!(result[key] isa Bool)&&result[key]>=0
    end
    @test result["julia_peak_rss_measured"]===true
    @test result["worker_elapsed_ns"]>=result["fit_elapsed_ns"]+result["readback_elapsed_ns"]
    # Measurements are evidence, not invented performance/resource thresholds.
    for index in 1:36
        @test filesize(joinpath(stage,"points","setting-$index.bin"))==16+341*16
    end
    nothing
end

function compare_old_route(generated,old,literals)
    for index in 1:36
        first=json(joinpath(generated.stage.directory,"models","setting-$index","model.json"))
        second=json(joinpath(old.stage.directory,"models","setting-$index","model.json"))
        @test first["setting_index"]==second["setting_index"]==index
        @test first["beta"]==second["beta"]==(isodd(index) ? 5 : 20)
        @test first["a_setting_index"]==second["a_setting_index"]==(index+1)÷2
        @test float_bits(first["eta"])==float_bits(second["eta"])==((index-1)%4<2 ? 0. : .25)
        @test first["fit_count"]==second["fit_count"]
        @test first["informative_count"]==second["informative_count"]>=100
        # Retain the old independently derived NNLS gain tolerances. This
        # compares numerical gains, not context/chunk-dependent artifact IDs.
        @test all(isapprox(float_bits(a),float_bits(b);rtol=literals["nnls"]["gain_rtol"],
            atol=literals["nnls"]["gain_atol"]) for (a,b) in zip(first["gains"],second["gains"]))
    end
    nothing
end

@testset "B1 native generated controls and real full36 handoff" begin
    run_python_controls()
    @test file_sha(TREE_ORACLE)=="e829beee61d48333689fea77527c5f6593e86412a1a21d66ab84ab917a4ebc96"
    @test file_sha(INDEPENDENT)=="736e8e052c513647dd1c712e447fb801fb4d452381e1dfa7d9ec3ebbabe74b58"
    oracle=json(TREE_ORACLE);literals=json(INDEPENDENT)
    mktempdir() do temporary
        root=realpath(temporary);newroot=joinpath(root,"new-entry");oldroot=joinpath(root,"old-route")
        mkdir(newroot);mkdir(oldroot)
        generated=BInitialNativeStage.run_generated(newroot)
        Base.invokelatest(check_generated,generated,oracle)
        old=ORIGINAL.run_stage(oldroot;chunk_hours=2)
        compare_old_route(generated,old,literals)
        # Derive the admission's own nested module to preserve nominal types.
        point=getfield(parentmodule(parentmodule(typeof(generated.admission))),:BInitialPointFits)
        raw=Base.invokelatest(getfield(point,:OwnedInitialPointExecutionPermitV2),
            joinpath(root,"missing-permit.json"),repeat("0",64),
            joinpath(root,"missing-intent.json"),repeat("0",64))
        @test_throws ArgumentError Base.invokelatest(getfield(point,:_permission),generated.admission,raw)
        # Coherent completion rehash must not hide a false point/model link.
        clone=joinpath(root,"rehash-wrong-model");cp(generated.stage.directory,clone)
        completion=json(joinpath(clone,"completion.json"))
        completion["models"][1]["points"]["model_id"]=repeat("0",64)
        body=SolarSINDy.DurationPointTraining._bytes(completion)
        write(joinpath(clone,"completion.json"),body)
        @test_throws ArgumentError Base.invokelatest(getfield(point,:verify_initial_l0),clone,
            generated.admission,generated.reference;expected_completion_sha256=bytes2hex(sha256(body)),
            expected_sources=SolarSINDy.DurationPointArtifacts.source_identity(),chunk_hours=2)
    end
end
end
