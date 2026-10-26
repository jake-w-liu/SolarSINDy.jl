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
function producer(absent,pin)
    A.Authority(absent,pin,absent,pin,absent,pin,absent,pin,absent,pin,absent,pin)
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
            for (p,r) in ((producer(absent,pin),nothing),(nothing,B.ReaderAuthority(absent,pin,absent,pin,absent,pin)),(true,true))
                failure(()->reread(;producer=p,reader=r),"generated case rereads cannot consume actual authorities")
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
@testset "Existing-case read permit negatives fail before producer or source bodies" begin
    mktempdir() do temporary
        root=realpath(temporary);absent=joinpath(root,"never-created");pin="0"^64;different="1"^64
        p=producer(absent,pin);reader=B.ReaderAuthority(absent,pin,absent,pin,absent,pin)
        request=Dict("schema"=>"archive_catalog_case_request_v1","evidence_mode"=>"actual_archival_cache",
            "layout"=>"ground-only","role"=>"primary","runtime"=>string(VERSION),
            "source"=>Dict("directory"=>absent,"manifest_sha256"=>pin,"complete_sha256"=>pin),
            "tables"=>[Dict("id"=>"CMO-declaration-only","station"=>"CMO","start_inclusive"=>"2017-01-01T00:00:00.000Z",
                "stop_exclusive"=>"2017-01-01T00:01:00.000Z","label_asof"=>"2017-01-01T00:32:00.000Z")])
        request_path=joinpath(root,"request.json");rp=write_json(request_path,request)
        for (old,new) in ((nothing,nothing),(p,nothing),(nothing,reader),(reader,p),(Dict(),Dict()))
            failure(()->B.read_reverified(request_path,absent;request_sha256=rp,manifest_sha256=pin,complete_sha256=pin,
                producer=old,reader=new),"existing archival cases require distinct producer and reader authorities")
        end
        # These deliberately invalid permits are rejection fixtures. No positive
        # archival qualification, feature grant or source artifact is fabricated.
        base=Dict("schema"=>"archive_catalog_existing_case_read_grant_v1","operation"=>"verify_existing_cases_only",
            "request_sha256"=>rp,"artifact_manifest_sha256"=>pin,"artifact_complete_sha256"=>pin,
            "producer_grant_sha256"=>pin,"producer_source_proof_sha256"=>pin,"producer_qualification_sha256"=>pin,
            "reader_source_proof_sha256"=>pin,"reader_qualification_sha256"=>pin,"head"=>"0"^40,
            "actual_source_access_authorized"=>true,"actual_model_fit_authorized"=>false,
            "new_case_generation_authorized"=>false,"dataset_discovery"=>false,"provider_receipts_available"=>false)
        changes=[("schema","wrong"),("operation","fit"),("request_sha256",different),
            ("artifact_manifest_sha256",different),("artifact_complete_sha256",different),
            ("producer_grant_sha256",different),("producer_source_proof_sha256",different),
            ("producer_qualification_sha256",different),("reader_source_proof_sha256",different),
            ("reader_qualification_sha256",different),("actual_source_access_authorized",false),
            ("actual_model_fit_authorized",true),("new_case_generation_authorized",true),
            ("dataset_discovery",true),("provider_receipts_available",true)]
        for (index,(field,bad)) in enumerate(changes)
            grant=copy(base);grant[field]=bad;path=joinpath(root,"invalid-permit-$index.json");gp=write_json(path,grant)
            supplied=B.ReaderAuthority(path,gp,absent,pin,absent,pin)
            failure(()->B.read_reverified(request_path,absent;request_sha256=rp,manifest_sha256=pin,complete_sha256=pin,
                producer=p,reader=supplied),"existing-case read grant differs")
        end
        @test !ispath(absent)
        @test length(readdir(root))==length(changes)+1
    end
end
@testset "Loaded reader inventory remains strict" begin
    inventory=Dict(name=>"0"^64 for name in ("src/archive_format/ArchiveCatalogCaseAuthority.jl",
        "src/archive_format/ArchiveCatalogCaseJob.jl","src/archive_format/ArchiveCatalogCaseComparison.jl",
        "src/archive_format/ArchiveCatalogSnapshotFilesystem.jl","src/archive_format/ArchiveCatalogSnapshotValues.jl",
        "src/archive_format/ArchiveCatalogSourceJob.jl"))
    proof=Dict("complete"=>true,"all_commit_filters_match"=>true,"source_sha256"=>inventory)
    failure(()->A._qualification_loaded(proof),"loaded package differs from qualified source proof")
end
println("GENERATED_CASE_READER_TRANSITION_PASSED;ACTUAL_SOURCE_ACCESS=false;ACTUAL_FITS=0")
end
