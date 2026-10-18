using Test,Dates,SHA,TOML
const SOURCEDIR=ARGS[1]
const OUTDIR=ARGS[2]
const FIXTURE=ARGS[3]
for name in ("DstIntervalAdapter.jl","DstIntervalStream.jl","DstIntervalTraining.jl","DstIntervalArtifacts.jl","DstIntervalExperiment.jl")
    include(joinpath(SOURCEDIR,name))
end
const A=DstIntervalAdapter;const S=DstIntervalStream;const T=DstIntervalTraining
const F=DstIntervalArtifacts;const X=DstIntervalExperiment
const B0=DateTime(2019,1,25);const B1=DateTime(2019,2,15)
hashfile(path)=bytes2hex(sha256(read(path)))
@assert hashfile(FIXTURE)=="061bbabbcef461d56d4b1dd9f4ffd1e0de607efda70fabfefe4b25d4199ea31b"
raw=[split(line,'\t') for line in readlines(FIXTURE)[2:end]]
const TIMES=[DateTime(row[1]) for row in raw]
const VALUES=ones(17,length(raw))
for (j,row) in enumerate(raw)
    for (column,field) in ((8,2),(6,3),(12,4),(15,5));VALUES[column,j]=parse(Float64,row[field]);end
end
const ID=hashfile(FIXTURE)
const INPUT=S.HourlyInputs(TIMES,VALUES;source_sha256=ID,availability_kind=:assumed_completed_hour)
const PROVENANCE=(synthetic_fixture=true,input_sha256=ID,scope="Independent fallback/control-flow oracle; no forecast-skill claim")

function reference(L,error)
    rows=NamedTuple[]
    for j in eachindex(TIMES)
        issue=TIMES[j]+Hour(1);c4=A.forecast_clock(issue,4,L);c6=A.forecast_clock(issue,6,L)
        anchor=j-L>=1 ? VALUES[15,j-L] : NaN
        target=ntuple(slot->j+(4,6)[slot]<=length(TIMES) ? VALUES[15,j+(4,6)[slot]] : NaN,2)
        push!(rows,(;issue,driver_tag=c4.driver_tag,driver_receipt=issue,anchor_tag=c4.anchor_tag,
            anchor_complete=c4.anchor_complete,anchor_receipt=c4.anchor_receipt,anchor,
            target_tag=(c4.target_tag,c6.target_tag),target_complete=(c4.target_complete,c6.target_complete),
            target_receipt=(c4.target_receipt,c6.target_receipt),target,
            scoreable=ntuple(h->isfinite(anchor)&&isfinite(target[h]),2),
            prediction=ntuple(h->isfinite(target[h]) ? target[h]+error : anchor,2),
            source_sha256=ID,dst_delay_hours=L))
    end
    (;rows,training_boundary=B0,max_fit_target_complete=(B0-Hour(168+L),B0-Hour(168+L)),
        max_fit_label_receipt=(B0-Hour(168),B0-Hour(168)),source_sha256=ID,delay_hours=L,
        model_ids=("a"^64,"b"^64),evidence_kind=:synthetic_initial_reference)
end

function run_case(name,refs;fail_fit=0,fail_publish=false)
    fits=NamedTuple[];publications=NamedTuple[]
    function fitspy(rows,boundary,setting;kwargs...)
        push!(fits,(;boundary,index=setting.index,L=kwargs[:delay_hours]))
        length(fits)==fail_fit && error("owned fifth initial fit failure")
        T.fit_gains(rows,boundary,setting;kwargs...)
    end
    function publishspy(directory,model,checkpoint;kwargs...)
        push!(publications,(;L=model.delay_hours,index=model.setting_index,directory))
        fail_publish && error("owned publication failure")
        F.save_bundle(directory,model,checkpoint;kwargs...)
    end
    directory=joinpath(OUTDIR,name)
    result=X.run_synthetic_experiment(INPUT,directory;initial_boundary=B0,validation_end=B1,final_boundary=B1,
        references_by_delay=refs,provenance=PROVENANCE,fit_impl=fitspy,publish_impl=publishspy,chunk_hours=24)
    (;result,fits,publications,directory)
end

@testset "Independent actual36-fit selected/none/blocked orchestration" begin
    for (name,refs,status) in (("selected",Dict(L=>reference(L,10.) for L in (0,1)),:selected),
        ("none",Dict(L=>reference(L,0.) for L in (0,1)),:none),
        ("missing",Dict(0=>nothing,1=>nothing),:blocked_reference_absent))
        case=run_case(name,refs)
        initial=[r for r in case.fits if r.boundary==B0];final=[r for r in case.fits if r.boundary==B1]
        @test length(initial)==case.result.initial_fit_calls==36
        for L in (0,1)
            @test sort([r.index for r in initial if r.L==L])==collect(1:18)
            detail=case.result.by_delay[L]
            @test detail.selection.status==status
            @test length(detail.initial_model_ids)==18
            @test length(unique(detail.initial_model_ids))==18
            if status==:selected
                @test detail.selection.selected_index==1
                @test detail.final_model.setting_index==1
                @test detail.final_model.fit_count==(L==0 ? 906 : 904)
                @test detail.final_model.informative_count==408
                @test detail.selection.panels[1].pooled[1].n==(L==0 ? 333 : 332)
                @test detail.selection.panels[1].pooled[2].n==(L==0 ? 331 : 330)
                @test detail.selection.panels[1].pooled[1].rmse≈(4+L)/10 atol=1e-10
                @test detail.selection.panels[1].pooled[2].rmse≈(6+L)/10 atol=1e-10
                @test detail.selection.panels[1].normalized_storm_rmse≈(L==0 ? .05 : .06) atol=1e-10
                bundle=detail.bundle
                @test Set(readdir(bundle.directory))==Set(["model.toml","checkpoint.toml","provenance.toml","manifest.toml"])
                loaded=F.load_bundle(bundle.directory;expected_manifest_sha256=bundle.manifest_sha256,expected_sources=F.verify_source_identity())
                @test loaded.model.gains==detail.final_model.gains
                @test loaded.model.model_id==detail.final_model.model_id==bundle.model_id
                @test loaded.model.source_sha256==ID
            else
                @test detail.selection.selected_index===nothing
                @test detail.final_model===nothing
                @test detail.bundle===nothing
                @test !ispath(joinpath(case.directory,"final-L$L"))
            end
        end
        wanted=status==:selected ? 2 : 0
        @test length(final)==case.result.final_fit_calls==wanted
        @test length(case.publications)==case.result.publish_calls==wanted
        @test isfile(joinpath(case.directory,"completion.toml"))
        @test !isfile(joinpath(case.directory,"failure.txt"))
    end
end

@testset "Independent failure markers, resource cap and existing output" begin
    refs=Dict(L=>reference(L,10.) for L in (0,1))
    @test_throws ErrorException run_case("injected-fit-failure",refs;fail_fit=5)
    @test isfile(joinpath(OUTDIR,"injected-fit-failure","failure.txt"))
    @test !isfile(joinpath(OUTDIR,"injected-fit-failure","completion.toml"))
    @test_throws ErrorException run_case("injected-publish-failure",refs;fail_publish=true)
    @test isfile(joinpath(OUTDIR,"injected-publish-failure","failure.txt"))
    @test !isfile(joinpath(OUTDIR,"injected-publish-failure","completion.toml"))
    completed_before=read(joinpath(OUTDIR,"selected","completion.toml"))
    @test_throws Base.IOError run_case("selected",refs)
    @test read(joinpath(OUTDIR,"selected","completion.toml"))==completed_before
    @test isfile(joinpath(OUTDIR,"selected","completion.toml"))
    large=S.HourlyInputs([DateTime(2019)+Hour(j) for j in 0:20000],ones(17,20001);source_sha256="c"^64)
    calls=Ref(0)
    reject_fit=(args...;kwargs...)->begin calls[]+=1;error("large input reached a fit") end
    @test_throws ArgumentError X.run_synthetic_experiment(large,joinpath(OUTDIR,"over-limit");initial_boundary=B0,
        validation_end=B1,final_boundary=B1,references_by_delay=refs,
        provenance=(synthetic_fixture=true,input_sha256="c"^64),fit_impl=reject_fit)
    @test calls[]==0
    @test !isfile(joinpath(OUTDIR,"over-limit","completion.toml"))
end

function named_field(node,name)
    @assert node["type"]=="namedtuple"
    node["items"][only(findall(==(name),node["names"]))]
end
function rewrite(path,document)
    open(path,"w") do stream;TOML.print(stream,document;sorted=true);end
end
function rehash_member(directory,name)
    path=joinpath(directory,"manifest.toml");manifest=TOML.parsefile(path)
    member=only(filter(m->m["name"]==name,manifest["members"]))
    member["sha256"]=hashfile(joinpath(directory,name));member["bytes"]=filesize(joinpath(directory,name))
    rewrite(path,manifest);hashfile(path)
end

@testset "Independent exact bundle members and hash-rebound corruption" begin
    bundle=joinpath(OUTDIR,"selected","final-L0")
    pin=hashfile(joinpath(bundle,"manifest.toml"))
    @test_throws ArgumentError F.load_bundle(bundle;expected_manifest_sha256="0"^64,expected_sources=F.verify_source_identity())
    mutations=("old-member-hash","missing-checkpoint","extra-member","negative-gain","wrong-source","checkpoint-source")
    for name in mutations
        copydir=joinpath(OUTDIR,"corrupt-"*name);cp(bundle,copydir)
        newpin=pin
        if name=="old-member-hash"
            open(joinpath(copydir,"model.toml"),"a") do stream;write(stream,"\n# changed\n");end
        elseif name=="missing-checkpoint"
            rm(joinpath(copydir,"checkpoint.toml"))
        elseif name=="extra-member"
            write(joinpath(copydir,"unexpected.txt"),"owned extra")
        elseif name in ("negative-gain","wrong-source")
            path=joinpath(copydir,"model.toml");document=TOML.parsefile(path)
            field=named_field(document["value"],name=="negative-gain" ? "gains" : "source_sha256")
            name=="negative-gain" ? (field["items"][1]["bits"]="bff0000000000000") : (field["value"]="f"^64)
            rewrite(path,document);newpin=rehash_member(copydir,"model.toml")
        else
            path=joinpath(copydir,"checkpoint.toml");document=TOML.parsefile(path)
            named_field(document["value"],"source_sha256")["value"]="f"^64
            rewrite(path,document);newpin=rehash_member(copydir,"checkpoint.toml")
        end
        @test_throws ArgumentError F.load_bundle(copydir;expected_manifest_sha256=newpin,expected_sources=F.verify_source_identity())
    end
end
