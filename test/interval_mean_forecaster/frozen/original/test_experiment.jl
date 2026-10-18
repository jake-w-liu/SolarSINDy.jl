using Dates, Test, SHA, TOML
include("DstIntervalAdapter.jl")
include("DstIntervalStream.jl")
include("DstIntervalTraining.jl")
include("DstIntervalArtifacts.jl")
include("DstIntervalExperiment.jl")
using .DstIntervalAdapter, .DstIntervalStream, .DstIntervalTraining, .DstIntervalArtifacts, .DstIntervalExperiment
const A=DstIntervalAdapter
const B=DstIntervalArtifacts
const E=DstIntervalExperiment
root=mktempdir(;cleanup=false)
println("RETAINED_EXPERIMENT_FIXTURES=",root)

# This source is generated here. It has the literal57-column author format,
# but is not an observation from the real archive or an external model.
n=1600;t=DateTime(2018)
sourcepath=joinpath(root,"synthetic57.lst")
rawvalues=ones(17,n)
rawvalues[8,:].=400.;rawvalues[6,:]=[-6+2sin(i/31) for i in 1:n]
rawvalues[12,:]=[2+.2cos(i/41) for i in 1:n]
rawvalues[15,:]=[-70+10sin(i/23) for i in 1:n]
open(sourcepath,"w") do io
    for i in 1:n
        tag=t+Hour(i-1);fields=fill("0",57)
        fields[1]=string(year(tag));fields[2]=string(dayofyear(tag));fields[3]=string(hour(tag))
        for (j,col) in enumerate(A.Features.RAW_COLUMNS);fields[col]=repr(rawvalues[j,i]);end
        println(io,join(fields," "))
    end
end
source=bytes2hex(sha256(read(sourcepath)))
input=A.Features.read_author_hourly(sourcepath;expected_sha256=source)
initial=t+Hour(800);ending=t+Hour(1400)
provenance=(synthetic_fixture=true,input_sha256=source,generator="test_experiment.jl")

function references(error)
    result=Dict{Int,Any}()
    for delay in (0,1)
        rows=NamedTuple[]
        stream_basis(input;delay_hours=delay,taus=(4.,24.),sigma=3.,sink=x->append!(rows,x))
        preds=[merge(r,(prediction=Tuple(y+error for y in r.target),)) for r in rows if initial<=r.issue<ending]
        result[delay]=(rows=preds,training_boundary=initial,
            max_fit_target_complete=(initial-Hour(168+delay),initial-Hour(168+delay)),
            max_fit_label_receipt=(initial-Hour(168),initial-Hour(168)),
            source_sha256=source,delay_hours=delay,
            model_ids=(bytes2hex(sha256("synthetic-reference4-L$delay")),bytes2hex(sha256("synthetic-reference6-L$delay"))),
            evidence_kind=:synthetic_initial_reference)
    end
    result
end

function spy_run(name,reference;fail_stage=nothing)
    calls=NamedTuple[];published=String[]
    function fit_spy(rows,boundary,setting;kwargs...)
        push!(calls,(boundary=boundary,index=setting.index,delay=kwargs[:delay_hours]))
        fail_stage==:initial && length(calls)==10 && error("independent initialfailure injection")
        fail_stage==:final && boundary==ending && error("independent finalfailure injection")
        fit_gains(rows,boundary,setting;kwargs...)
    end
    function publish_spy(directory,model,checkpoint;kwargs...)
        push!(published,String(directory))
        fail_stage==:publish && error("independent publishfailure injection")
        save_bundle(directory,model,checkpoint;kwargs...)
    end
    directory=joinpath(root,name)
    result=try
        run_synthetic_experiment(input,directory;initial_boundary=initial,validation_end=ending,final_boundary=ending,
            references_by_delay=reference,provenance=provenance,fit_impl=fit_spy,publish_impl=publish_spy,chunk_hours=511)
    catch exception
        exception
    end
    (;result,calls,published,directory)
end

@testset "syntheticparserfitselectrefitsavereload" begin
    @test input.values==rawvalues
    @test input.time==[t+Hour(i-1) for i in 1:n]
    @test_throws ArgumentError A.Features.read_author_hourly(sourcepath;expected_sha256=repeat("0",64))
    weak=references(1000.)
    selected=spy_run("selected",weak)
    @test !(selected.result isa Exception)
    if !(selected.result isa Exception)
        result=selected.result
        @test result.initial_fit_calls==36
        @test result.final_fit_calls==result.publish_calls==2
        @test count(x->x.boundary==initial,selected.calls)==36
        @test count(x->x.boundary==ending,selected.calls)==2
        @test length(selected.published)==2
        @test all(result.by_delay[d].selection.status==:selected for d in (0,1))
        @test all(length(result.by_delay[d].initial_model_ids)==18 for d in (0,1))
        @test all(length(result.by_delay[d].selection.panels)==18 for d in (0,1))
        completion=TOML.parsefile(joinpath(result.directory,"completion.toml"))
        @test completion["evidence_mode"]=="synthetic_only_no_skill_verdict"
        @test completion["initial_fit_calls"]==36 && completion["final_fit_calls"]==2
        @test completion["source_hashes"]==Dict(E.verify_source_identity())
        @test bytes2hex(sha256(read(joinpath(result.directory,"completion.toml"))))==result.completion_sha256
        for delay in (0,1)
            branch=result.by_delay[delay]
            loaded=load_bundle(branch.bundle.directory;expected_manifest_sha256=branch.bundle.manifest_sha256,
                expected_sources=B.verify_source_identity())
            @test isequal(loaded.model,branch.final_model)
            @test loaded.model.max_label_receipt==ending-Hour(168)
            @test loaded.provenance.selected_index==branch.selection.selected_index
        end
    else
        showerror(stdout,selected.result);println()
    end
    none=spy_run("none",references(0.))
    @test !(none.result isa Exception)
    if !(none.result isa Exception)
        @test none.result.initial_fit_calls==36 && none.result.final_fit_calls==none.result.publish_calls==0
        @test length(none.calls)==36 && isempty(none.published)
        @test all(none.result.by_delay[d].selection.status==:none for d in (0,1))
        @test all(none.result.by_delay[d].final_model===nothing for d in (0,1))
        @test !ispath(joinpath(none.directory,"final-L0")) && !ispath(joinpath(none.directory,"final-L1"))
    end
    blocked=spy_run("blocked",Dict(0=>nothing,1=>nothing))
    @test !(blocked.result isa Exception)
    if !(blocked.result isa Exception)
        @test length(blocked.calls)==36 && isempty(blocked.published)
        @test blocked.result.final_fit_calls==0
        @test all(blocked.result.by_delay[d].selection.status==:blocked_reference_absent for d in (0,1))
    end
    for stage in (:initial,:final,:publish)
        failed=spy_run("failure-$stage",weak;fail_stage=stage)
        @test failed.result isa ErrorException
        @test occursin("failure injection",sprint(showerror,failed.result))
        @test !ispath(joinpath(failed.directory,"completion.toml"))
        @test isfile(joinpath(failed.directory,"failure.txt"))
    end
    @test_throws Base.IOError run_synthetic_experiment(input,selected.directory;initial_boundary=initial,
        validation_end=ending,final_boundary=ending,references_by_delay=weak,provenance=provenance)
    @test_throws ArgumentError run_synthetic_experiment(input,joinpath(root,"not-synthetic");initial_boundary=initial,
        validation_end=ending,final_boundary=ending,references_by_delay=weak,
        provenance=(synthetic_fixture=false,input_sha256=source))
end
