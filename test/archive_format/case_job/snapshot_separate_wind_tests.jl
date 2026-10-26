module ArchiveSnapshotSeparateWindTests
using SolarSINDy,Test,JSON3,SHA
const CatalogSnapshotValues=SolarSINDy.ArchiveCatalogSnapshotValues
const J=SolarSINDy.ArchiveCatalogSourceJob
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const PACKAGE=pkgdir(SolarSINDy)
digest(bytes)=bytes2hex(sha256(IOBuffer(bytes)))
function run(ROLE,OUTPUT)
    fixture=JSON3.read(read(joinpath(PACKAGE,"test","archive_format","source_job","fixtures","FROZEN_GENERATED_FIXTURES_AND_ORACLES_V1.json")),Any)
    mktempdir(OUTPUT) do root
        root=realpath(root);inputs_dir=joinpath(root,"inputs");mkdir(inputs_dir)
        inputs=Any[]
        for original in fixture["mixed_inputs"]
            descriptor=fixture["fixtures"][original]
            labels=descriptor["kind"]=="wind" ? (["CMO"],["FRD"]) : (descriptor["stream_labels"],)
            for stream_labels in labels
                name=descriptor["kind"]=="wind" ? "generated-wind-"*only(stream_labels)*".csv" : original
                bytes=Vector{UInt8}(codeunits(descriptor["text"]));path=joinpath(inputs_dir,name)
                R._exclusive_write(path,bytes)
                push!(inputs,Dict("logical_id"=>name,"path"=>path,"kind"=>descriptor["kind"],"station"=>descriptor["station"],
                    "stream_labels"=>stream_labels,"product"=>descriptor["product"],"start_inclusive"=>fixture["start"],
                    "stop"=>fixture["stop"],"stop_inclusive"=>descriptor["kind"]=="ground","bytes"=>length(bytes),
                    "sha256"=>digest(bytes),"role"=>"generated_fixture","provenance"=>"generated_fixture"))
            end
        end
        request=Dict("schema"=>"archive_catalog_audit_request_v1","evidence_mode"=>"generated","layout"=>"mixed",
            "role"=>ROLE,"runtime"=>string(VERSION),"data_root"=>inputs_dir,"inputs"=>inputs)
        path=joinpath(root,"request.json");R._exclusive_write(path,R.canonical_bytes(request;limit=131072))
        source=joinpath(root,"snapshot");manifest=J.reconstruct(path,source)
        @testset "Qualified source job accepts separate station wind declarations $ROLE" begin
            @test length(inputs)==4 && length(manifest["read_trace"])==4
            @test Set(keys(manifest["catalogs"]))==Set(fixture["expected_mixed_catalogs"])
            @test [body["declaration"]["stream_labels"] for body in manifest["bodies"] if body["declaration"]["kind"]=="wind"]==[["CMO"],["FRD"]]
            for stream in fixture["expected_mixed_catalogs"]
                oracle=fixture["catalog_oracles"][stream];catalog=manifest["catalogs"][stream]
                for key in ("selected_sha256","raw_catalog_sha256","raw_rows","rows","declared_minutes","duplicates","missing_minutes","uncovered_minutes")
                    @test catalog[key]==oracle[key]
                end
                @test catalog["body_ids"]==[oracle["body_id"]]
                selected=join(read(joinpath(source,name),String) for name in catalog["shards"])
                @test ncodeunits(selected)==oracle["selected_bytes"] && digest(selected)==oracle["selected_sha256"]
                body=only(filter(body->haskey(body["body_ids"],stream),manifest["bodies"]))
                raw=read(joinpath(source,body["raw"]),String)
                @test ncodeunits(raw)==oracle["raw_bytes"] && digest(raw)==oracle["raw_sha256"]
            end
        end
        text=read(joinpath(source,"manifest.json"),String);marker=read(joinpath(source,"COMPLETE.json"),String)
        members=Dict{String,String}(entry["name"]=>read(joinpath(source,entry["name"]),String) for entry in manifest["members"])
        value=nothing;failure=nothing
        try
            value=CatalogSnapshotValues.reconstruct(text,marker,members;manifest_sha256=digest(text),complete_sha256=digest(marker),
                expected_layout="mixed",expected_role=ROLE,expected_runtime=string(VERSION),expected_evidence_mode="generated")
        catch error
            error isa ArgumentError || rethrow()
            failure=error.msg
        end
        @testset "Valid separately declared wind snapshot must reconstruct $ROLE" begin
            @test failure===nothing
            if failure===nothing
                for stream in fixture["expected_mixed_catalogs"]
                    catalog=getproperty(value.catalogs,Symbol(stream));oracle=fixture["catalog_oracles"][stream]
                    @test catalog.selected_sha256==oracle["selected_sha256"]
                    @test catalog.raw_catalog_sha256==oracle["raw_catalog_sha256"]
                end
                @test value.actual_source_authority_granted===false && value.actual_fits==0
            else
                println("verified-valid-source-rejected: ",failure)
            end
        end
    end
    open(joinpath(OUTPUT,"result.json"),"w") do stream
        write(stream,JSON3.write(Dict("complete"=>true,"runtime"=>string(VERSION),"role"=>ROLE,"actual_source_access"=>false,"actual_features"=>false,"actual_fits"=>0)))
    end
end
for role in ("primary","independent")
    mktempdir() do output
        run(role,realpath(output))
    end
end

end
