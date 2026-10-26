module ArchiveSnapshotShardGeometryTests
using SolarSINDy,Test,JSON3,SHA,Dates
const API=SolarSINDy.ArchiveCatalogSnapshotValues
const J=SolarSINDy.ArchiveCatalogSourceJob
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const ORACLE=JSON3.read(read(joinpath(@__DIR__,"snapshot_fixtures","FROZEN_MULTISHARD_ORACLES_V1.json")),Any)
const ROLES=("primary","independent")
@assert all(role->role in ("primary","independent"),ROLES)
digest(x)=bytes2hex(sha256(IOBuffer(x)))
function rebuild(text,marker,members,role)
    API.reconstruct(text,marker,members;manifest_sha256=digest(text),complete_sha256=digest(marker),
                    expected_layout="ground-only",expected_role=role,expected_runtime=string(VERSION),expected_evidence_mode="generated")
end
function run(OUTPUT)
    @test ORACLE["rows"]==8300 && ORACLE["actual_data"]===false
    for role in ROLES
        mktempdir(OUTPUT) do directory
            directory=realpath(directory);inputs_dir=joinpath(directory,"inputs");mkdir(inputs_dir)
            inputs=Any[]
            for fixture in ORACLE["fixtures"]
                bytes=read(joinpath(@__DIR__,"snapshot_fixtures",fixture["path"]))
                @test length(bytes)==fixture["bytes"] && digest(bytes)==fixture["sha256"]
                name="generated-ground-"*fixture["station"]*".txt";path=joinpath(inputs_dir,name)
                R._exclusive_write(path,bytes)
                push!(inputs,Dict("logical_id"=>name,"path"=>path,"kind"=>"ground","station"=>fixture["station"],
                    "stream_labels"=>[fixture["station"]],"product"=>"quasi-definitive","start_inclusive"=>ORACLE["start"],
                    "stop"=>ORACLE["stop"],"stop_inclusive"=>true,"bytes"=>length(bytes),"sha256"=>digest(bytes),
                    "role"=>"generated_fixture","provenance"=>"generated_fixture"))
            end
            request=Dict("schema"=>"archive_catalog_audit_request_v1","evidence_mode"=>"generated","layout"=>"ground-only",
                         "role"=>role,"runtime"=>string(VERSION),"data_root"=>inputs_dir,"inputs"=>inputs)
            request_path=joinpath(directory,"request.json");R._exclusive_write(request_path,R.canonical_bytes(request;limit=131072))
            source=joinpath(directory,"snapshot");J.reconstruct(request_path,source)
            text=read(joinpath(source,"manifest.json"),String);marker_text=read(joinpath(source,"COMPLETE.json"),String)
            manifest=R.decode_canonical(collect(codeunits(text));limit=1_048_576)
            marker=R.decode_canonical(collect(codeunits(marker_text));limit=4096)
            members=Dict{String,String}(m["name"]=>read(joinpath(source,m["name"]),String) for m in manifest["members"])
            value=rebuild(text,marker_text,members,role)
            @testset "Frozen independent8300-row binary oracles $role" begin
                for fixture in ORACLE["fixtures"]
                    stream="ground-"*fixture["station"];catalog=getproperty(value.catalogs,Symbol(stream))
                    @test catalog.rows==8300 && ncodeunits(catalog.selected_storage)==fixture["selected_bytes"]
                    @test catalog.selected_sha256==fixture["selected_sha256"]
                    @test collect(catalog.body_ids)==[fixture["body_id"]]
                    body=only(filter(b->haskey(b["body_ids"],stream),manifest["bodies"]))
                    raw=members[body["raw"]]
                    @test ncodeunits(raw)==fixture["raw_bytes"] && digest(raw)==fixture["raw_sha256"]
                    @test bytes2hex(codeunits(raw)[1:48])==fixture["first_raw_hex"]
                    @test bytes2hex(codeunits(raw)[end-47:end])==fixture["last_raw_hex"]
                end
            end
            changed=deepcopy(manifest);changed_members=copy(members)
            shards=changed["catalogs"]["ground-CMO"]["shards"]
            @test length(shards)==2
            left,right=shards
            first_bytes=collect(codeunits(members[left]));second_bytes=collect(codeunits(members[right]))
            @test length(first_bytes)==8192*80 && length(second_bytes)==108*80
            changed_members[left]=String([first_bytes;second_bytes[1:80]])
            changed_members[right]=String(second_bytes[81:end])
            @test changed_members[left]*changed_members[right]==members[left]*members[right]
            for name in (left,right)
                descriptor=only(filter(m->m["name"]==name,changed["members"]))
                descriptor["bytes"]=ncodeunits(changed_members[name]);descriptor["sha256"]=digest(changed_members[name])
            end
            changed_text=String(R.canonical_bytes(changed;limit=1_048_576))
            changed_marker=deepcopy(marker);changed_marker["manifest_sha256"]=digest(changed_text)
            changed_marker["snapshot_bytes_without_marker"]=ncodeunits(changed_text)+sum(m["bytes"] for m in changed["members"])
            error_message=nothing
            try
                rebuild(changed_text,String(R.canonical_bytes(changed_marker;limit=4096)),changed_members,role)
            catch error
                error isa ArgumentError || rethrow()
                error_message=error.msg
            end
            @testset "Rehashed unchanged-whole-payload shard shift rejected $role" begin
                @test error_message=="snapshot selected shard width"
            end
        end
    end
    open(joinpath(OUTPUT,"result.json"),"w") do io
        write(io,JSON3.write(Dict("complete"=>true,"runtime"=>string(VERSION),"generated_roles"=>2,
                                "actual_source_access"=>false,"actual_features"=>false,"actual_fits"=>0)))
    end
    println("snapshot-shard-geometry-controls-complete")
end
mktempdir() do output
    run(realpath(output))
end

end
