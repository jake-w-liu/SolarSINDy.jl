using Dates, Test, SHA, TOML
include("DstIntervalAdapter.jl")
include("DstIntervalStream.jl")
include("DstIntervalTraining.jl")
include("DstIntervalArtifacts.jl")
using .DstIntervalAdapter, .DstIntervalStream, .DstIntervalTraining, .DstIntervalArtifacts
const A=DstIntervalArtifacts
source=bytes2hex(sha256("synthetic-bundle-input"))
n=900;origin=DateTime(2018)
times=[origin+Hour(i-1) for i in 1:n]
values=ones(17,n);values[8,:].=400.;values[6,:]=[-6+sin(i/20) for i in 1:n]
values[12,:].=2.;values[15,:]=[-60+30sin(i/45) for i in 1:n]
input=HourlyInputs(times,values;source_sha256=source)
setting=settings()[9]
root=mktempdir(;cleanup=false)
println("RETAINED_BUNDLE_FIXTURES=",root)

function rewrite(path,document)
    open(path,"w") do io;TOML.print(io,document;sorted=true);end
end
function rebound_copy(bundle;member=nothing,transform=identity,manifest_transform=identity)
    target=joinpath(root,"negative-$(time_ns())");cp(bundle.directory,target)
    manifestpath=joinpath(target,"manifest.toml");manifest=TOML.parsefile(manifestpath)
    if member!==nothing
        path=joinpath(target,member);doc=TOML.parsefile(path);transform(doc);rewrite(path,doc)
        item=only(filter(x->x["name"]==member,manifest["members"]))
        item["bytes"]=filesize(path);item["sha256"]=bytes2hex(sha256(read(path)))
    end
    manifest_transform(manifest);rewrite(manifestpath,manifest)
    target,bytes2hex(sha256(read(manifestpath)))
end
function change_field(doc,name,new)
    value=doc["value"];index=findfirst(==(name),value["names"]);value["items"][index]=A._encode(new)
end

@testset "hashboundsyntheticbundleroundtrip" begin
    for delay in (0,1)
        rows=NamedTuple[]
        streamed=stream_basis(input;delay_hours=delay,taus=setting.taus,sigma=setting.sigma,
            stop_after_index=600,sink=x->append!(rows,x))
        model=fit_gains(rows,origin+Hour(550),setting;delay_hours=delay,source_sha256=source)
        dest=joinpath(root,"bundle-L$delay")
        saved=save_bundle(dest,model,streamed.checkpoint;provenance=(scope="synthetic only",input_sha256=source,stage=:synthetic))
        loaded=load_bundle(dest;expected_manifest_sha256=saved.manifest_sha256,expected_sources=A.verify_source_identity())
        @test isequal(loaded.model,model)
        @test loaded.checkpoint==streamed.checkpoint
        @test loaded.provenance.stage==:synthetic
        @test isequal(predict_rows(loaded.model,rows),predict_rows(model,rows))
        resumed=NamedTuple[]
        result=stream_basis(input;delay_hours=delay,taus=setting.taus,sigma=setting.sigma,
            checkpoint=loaded.checkpoint,sink=x->append!(resumed,x))
        @test result.checkpoint.last_index==n
        @test first(resumed).issue==origin+Hour(601)
        @test_throws Base.IOError save_bundle(dest,model,streamed.checkpoint;provenance=(scope="synthetic only",))
        @test_throws ArgumentError load_bundle(dest;expected_manifest_sha256=repeat("0",64),expected_sources=A.verify_source_identity())
        @test_throws ArgumentError load_bundle(dest;expected_manifest_sha256=saved.manifest_sha256,expected_sources=())
        # Byte corruption remains rejected without updating trusted manifest.
        bad=joinpath(root,"corrupt-L$delay");cp(dest,bad)
        open(joinpath(bad,"model.toml"),"a") do io;write(io,"\n# tampered\n");end
        @test_throws ArgumentError load_bundle(bad;expected_manifest_sha256=saved.manifest_sha256,expected_sources=A.verify_source_identity())
        # Semantic controls intentionally recompute member and manifest digests:
        # their failure therefore proves schema/identity checks beyond hashing.
        for (field,value) in (("schema","invented"),("model_id",repeat("1",64)),("source_sha256",repeat("2",64)),
                ("contract_sha256",repeat("3",64)),("gains",(-1.,1.)),("delay_hours",1-delay),
                ("max_label_receipt",origin+Hour(550)),("fit_count",1))
            path,pin=rebound_copy(saved;member="model.toml",transform=d->change_field(d,field,value))
            @test_throws ArgumentError load_bundle(path;expected_manifest_sha256=pin,expected_sources=A.verify_source_identity())
        end
        for (field,value) in (("schema","other"),("clock_contract","target_start"),("delay_hours",1-delay),
                ("model_id",repeat("4",64)),("source_sha256",repeat("5",64)))
            path,pin=rebound_copy(saved;manifest_transform=d->(d[field]=value))
            @test_throws ArgumentError load_bundle(path;expected_manifest_sha256=pin,expected_sources=A.verify_source_identity())
        end
        for (field,value) in (("last_issue",origin),("source_sha256",repeat("6",64)),("consecutive_valid",-1),
                ("basis",((Inf,0.),(0.,0.))))
            if field=="basis"
                # The writer itself fails closed before malformed numeric TOML.
                @test_throws ArgumentError A._encode(value)
            else
                path,pin=rebound_copy(saved;member="checkpoint.toml",transform=d->change_field(d,field,value))
                @test_throws ArgumentError load_bundle(path;expected_manifest_sha256=pin,expected_sources=A.verify_source_identity())
            end
        end
        extra=joinpath(root,"extra-L$delay");cp(dest,extra);write(joinpath(extra,"unknown"),"no")
        @test_throws ArgumentError load_bundle(extra;expected_manifest_sha256=saved.manifest_sha256,expected_sources=A.verify_source_identity())
        changed=copy(values);changed[6,599]-=1
        altered=HourlyInputs(times,changed;source_sha256=source)
        @test_throws ArgumentError stream_basis(altered;delay_hours=delay,taus=setting.taus,sigma=setting.sigma,checkpoint=loaded.checkpoint)
    end
end
