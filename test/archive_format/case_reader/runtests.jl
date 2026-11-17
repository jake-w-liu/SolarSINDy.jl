module ArchiveCatalogCaseReaderTests
using Dates,SHA,SolarSINDy,Test,JSON3
const B=SolarSINDy.ArchiveCatalogCaseReader
const Q=SolarSINDy.ArchiveCatalogCaseJob
const A=SolarSINDy.ArchiveCatalogCaseAuthority
const J=SolarSINDy.ArchiveCatalogSourceJob
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const C=SolarSINDy.ArchiveCatalogCases
const START=DateTime(2020)
const FIXTURE=JSON3.read(read(joinpath(pkgdir(SolarSINDy),"test","archive_format","windows","fixtures","FROZEN_INPUTS_AND_ORACLES_V1.json")),Any)
digest(bytes)=bytes2hex(sha256(bytes))
clock(time)=Dates.format(time,dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
function failure(action,expected)
    message=nothing
    try;action();catch error;error isa ArgumentError || rethrow();message=error.msg;end
    @test message==expected
end
function write_json(path,value;limit=131072)
    bytes=R.canonical_bytes(value;limit);R._exclusive_write(path,bytes);digest(bytes)
end

@testset "Existing-case reader generated recomputation and immutable receipt" begin
    for role in ("primary","independent"),layout in ("ground-only","mixed")
        mktempdir() do temporary
            root=realpath(temporary);input=joinpath(root,"inputs");mkdir(input);descriptors=Any[]
            for (kind,station,text) in [("ground","CMO",FIXTURE["ground"]),
                    ("ground","FRD",replace(FIXTURE["ground"],"CMO"=>"FRD")),("wind",nothing,FIXTURE["wind"])]
                layout=="ground-only" && kind=="wind" && continue
                name=kind=="wind" ? "generated-wind.csv" : "generated-"*station*".txt"
                path=joinpath(input,name);bytes=Vector{UInt8}(codeunits(text));R._exclusive_write(path,bytes)
                push!(descriptors,Dict("logical_id"=>name,"path"=>path,"kind"=>kind,"station"=>station,
                    "stream_labels"=>kind=="wind" ? ["CMO","FRD"] : [station],
                    "product"=>kind=="wind" ? "OMNI_HRO_1MIN" : "quasi-definitive",
                    "start_inclusive"=>clock(START),"stop"=>clock(START+Minute(90)),"stop_inclusive"=>kind=="ground",
                    "bytes"=>length(bytes),"sha256"=>digest(bytes),"role"=>"generated_fixture","provenance"=>"generated_fixture"))
            end
            source_request=Dict("schema"=>"archive_catalog_audit_request_v1","evidence_mode"=>"generated","layout"=>layout,
                "role"=>role,"runtime"=>string(VERSION),"data_root"=>input,"inputs"=>descriptors)
            source_path=joinpath(root,"source-request.json");write_json(source_path,source_request)
            source=joinpath(root,"source");J.reconstruct(source_path,source)
            source_pins=Dict("directory"=>source,"manifest_sha256"=>digest(read(joinpath(source,"manifest.json"))),
                "complete_sha256"=>digest(read(joinpath(source,"COMPLETE.json"))))
            request=Dict("schema"=>"generated_catalog_case_request_v1","evidence_mode"=>"generated","layout"=>layout,
                "role"=>role,"runtime"=>string(VERSION),"source"=>source_pins,
                "tables"=>[Dict("id"=>station*"-generated","station"=>station,"start_inclusive"=>clock(START),
                    "stop_exclusive"=>clock(START+Minute(91)),"label_asof"=>clock(START+Minute(121))) for station in ("CMO","FRD")])
            request_path=joinpath(root,"request.json");rp=write_json(request_path,request)
            directory=joinpath(root,"cases");Q.publish(request_path,directory;request_sha256=rp)
            mp=digest(read(joinpath(directory,"manifest.json")));cp=digest(read(joinpath(directory,"COMPLETE.json")))
            original=Q.read_verified(request_path,directory;request_sha256=rp,manifest_sha256=mp,complete_sha256=cp)
            reread(dir=directory;m=mp,c=cp,producer=nothing,reader=nothing)=B.read_reverified(request_path,dir;
                request_sha256=rp,manifest_sha256=m,complete_sha256=c,producer,reader)
            value=reread();receipt=R.decode_canonical(B.admission_bytes(value);limit=131072)
            @test value isa B.ReverifiedCases && isimmutable(value)
            @test value.cases.actual_source_authority_granted===false && value.cases.actual_fit_performed===false
            @test receipt==Dict("schema"=>"archive_catalog_case_read_admission_v1","evidence_mode"=>"generated",
                "request_sha256"=>rp,"artifact_manifest_sha256"=>mp,"artifact_complete_sha256"=>cp,
                "all_artifact_bytes_recomputed"=>true,"actual_model_fit_authorized"=>false,
                "new_case_generation_authorized"=>false,"provider_receipts_available"=>false,"actual_fits"=>0)
            @test_throws MethodError B.ReverifiedCases(original,String(B.admission_bytes(value)))
            owned=B.admission_bytes(value);owned[1]=0x00
            @test B.admission_bytes(value)[1]==UInt8('{')
            for index in 1:2
                @test value.cases.features[index].payload==original.features[index].payload
                @test value.cases.labels[index].payload==original.labels[index].payload
                counts=C.case_counts(value.cases.features[index],value.cases.labels[index])
                @test (counts.scheduled,counts.features_accepted,counts.labels_accepted,counts.supervised)==(91,60,61,30)
                arrays=Q.split_arrays(value.cases,index;start=START+Minute(40),stop=START+Minute(80),asof=START+Minute(121))
                @test arrays.issues==collect(START+Minute(40):Minute(1):START+Minute(49)) && length(arrays.targets)==10
                @test arrays.targets==vcat(fill(100.0,5),fill(5.0,5)) && !arrays.actual_fit_performed
                equal=START+Minute(62)+Second(45)
                @test isempty(Q.split_arrays(value.cases,index;start=START+Minute(32),stop=equal,asof=START+Minute(121)).targets)
                @test length(Q.split_arrays(value.cases,index;start=START+Minute(32),stop=equal+Millisecond(1),asof=START+Minute(121)).targets)==1
            end
            absent=joinpath(root,"never-created");pin="0"^64
            for (p,r) in ((true,nothing),(nothing,true),(true,true))
                @test_throws ArgumentError reread(;producer=p,reader=r)
            end
            failure(()->reread(;m=pin),"case reader metadata pin")
            changed=joinpath(root,"changed");Base.cp(directory,changed)
            metadata=R.decode_canonical(read(joinpath(changed,"manifest.json"));limit=1_048_576)
            marker=R.decode_canonical(read(joinpath(changed,"COMPLETE.json"));limit=4096)
            name=metadata["tables"][1]["feature_shards"][1];path=joinpath(changed,name);bytes=read(path)
            bytes[19]=0x01;write(path,bytes)
            only(filter(entry->entry["name"]==name,metadata["members"]))["sha256"]=digest(bytes)
            text=R.canonical_bytes(metadata;limit=1_048_576);write(joinpath(changed,"manifest.json"),text)
            marker["manifest_sha256"]=digest(text)
            marker["snapshot_bytes_without_marker"]=length(text)+sum(entry["bytes"] for entry in metadata["members"])
            write(joinpath(changed,"COMPLETE.json"),R.canonical_bytes(marker;limit=4096))
            failure(()->reread(changed;m=digest(text),c=digest(read(joinpath(changed,"COMPLETE.json")))),
                "case reader recomputed payload differs")
            @test digest(read(joinpath(directory,"manifest.json")))==mp && digest(read(joinpath(directory,"COMPLETE.json")))==cp
            @test !ispath(absent)
        end
    end
end
end
