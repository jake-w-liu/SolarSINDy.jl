module ArchiveCatalogCaseJobGeneratedTests
using Dates,SolarSINDy,Test,JSON3,SHA
const Q=SolarSINDy.ArchiveCatalogCaseJob
const F=SolarSINDy.ArchiveCatalogSnapshotFilesystem
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const J=SolarSINDy.ArchiveCatalogSourceJob
const C=SolarSINDy.ArchiveCatalogCases
const PACKAGE=pkgdir(SolarSINDy)
const START=DateTime(2020)
digest(bytes)=bytes2hex(sha256(IOBuffer(bytes)))
clock(time)=Dates.format(time,dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
const FIXTURE=JSON3.read(read(joinpath(PACKAGE,"test","archive_format","windows","fixtures","FROZEN_INPUTS_AND_ORACLES_V1.json")),Any)
const WHOLE=JSON3.read(read(joinpath(PACKAGE,"test","archive_format","cases","INDEPENDENT_WHOLE_NUMERIC_ORACLE_V1.json")),Any)
const BINARY=JSON3.read(read(joinpath(PACKAGE,"test","archive_format","cases","FROZEN_BINARY_CONTRACT_AND_ORACLES_V1.json")),Any)
function repack(directory,mutate)
    metadata=R.decode_canonical(read(joinpath(directory,"manifest.json"));limit=1_048_576)
    marker=R.decode_canonical(read(joinpath(directory,"COMPLETE.json"));limit=4096)
    mutate(metadata,marker)
    text=R.canonical_bytes(metadata;limit=1_048_576)
    marker["manifest_sha256"]=digest(text)
    marker["snapshot_bytes_without_marker"]=length(text)+sum(entry["bytes"] for entry in metadata["members"])
    write(joinpath(directory,"manifest.json"),text)
    write(joinpath(directory,"COMPLETE.json"),R.canonical_bytes(marker;limit=4096))
    digest(text),digest(read(joinpath(directory,"COMPLETE.json")))
end
repack(mutate::Function,directory::AbstractString)=repack(directory,mutate)
function check_failure(action,expected)
    message=nothing
    try;action();catch error;error isa ArgumentError || rethrow();message=error.msg;end
    @test message==expected
end
function run_generated_case_job_tests(output)
    results=Any[]
    for role in ("primary","independent"),layout in ("ground-only","mixed")
        mktempdir(output) do root
            root=realpath(root);input=joinpath(root,"inputs");mkdir(input);descriptors=Any[]
            for (kind,station,text) in [("ground","CMO",FIXTURE["ground"]),("ground","FRD",replace(FIXTURE["ground"],"CMO"=>"FRD")),
                                        ("wind",nothing,FIXTURE["wind"])]
                layout=="ground-only" && kind=="wind" && continue
                name=kind=="wind" ? "generated-wind.csv" : "generated-ground-"*station*".txt"
                path=joinpath(input,name);R._exclusive_write(path,Vector{UInt8}(codeunits(text)))
                push!(descriptors,Dict("logical_id"=>name,"path"=>path,"kind"=>kind,"station"=>station,
                    "stream_labels"=>kind=="wind" ? ["CMO","FRD"] : [station],"product"=>kind=="wind" ? "OMNI_HRO_1MIN" : "quasi-definitive",
                    "start_inclusive"=>clock(START),"stop"=>clock(START+Minute(90)),"stop_inclusive"=>kind=="ground",
                    "bytes"=>ncodeunits(text),"sha256"=>digest(text),"role"=>"generated_fixture","provenance"=>"generated_fixture"))
            end
            source_request=Dict("schema"=>"archive_catalog_audit_request_v1","evidence_mode"=>"generated","layout"=>layout,
                "role"=>role,"runtime"=>string(VERSION),"data_root"=>input,"inputs"=>descriptors)
            path=joinpath(root,"source-request.json");R._exclusive_write(path,R.canonical_bytes(source_request;limit=131072))
            source=joinpath(root,"source");J.reconstruct(path,source)
            pins=Dict("directory"=>source,"manifest_sha256"=>digest(read(joinpath(source,"manifest.json"))),"complete_sha256"=>digest(read(joinpath(source,"COMPLETE.json"))))
            request=Dict("schema"=>"generated_catalog_case_request_v1","evidence_mode"=>"generated","layout"=>layout,"role"=>role,
                "runtime"=>string(VERSION),"source"=>pins,"tables"=>[
                    Dict("id"=>"CMO-generated","station"=>"CMO","start_inclusive"=>clock(START),"stop_exclusive"=>clock(START+Minute(91)),"label_asof"=>clock(START+Minute(121))),
                    Dict("id"=>"FRD-generated","station"=>"FRD","start_inclusive"=>clock(START),"stop_exclusive"=>clock(START+Minute(91)),"label_asof"=>clock(START+Minute(121)))])
            request_path=joinpath(root,"cases.json");R._exclusive_write(request_path,R.canonical_bytes(request;limit=131072));request_pin=digest(read(request_path))
            artifact=joinpath(root,"cases");Q.publish(request_path,artifact;request_sha256=request_pin)
            mp=digest(read(joinpath(artifact,"manifest.json")));cpin=digest(read(joinpath(artifact,"COMPLETE.json")))
            read_case(directory=artifact;m=mp,c=cpin)=Q.read_verified(request_path,directory;request_sha256=request_pin,manifest_sha256=m,complete_sha256=c)
            value=read_case()
            @testset "Generated filesystem artifacts and independent numerical oracles $role/$layout" begin
                @test value isa Q.VerifiedCases && value.actual_source_authority_granted===false && value.actual_fit_performed===false
                @test length(value.features)==length(value.labels)==2
                for index in 1:2
                    f=value.features[index];l=value.labels[index];counts=C.case_counts(f,l)
                    agreement=SolarSINDy.ArchiveCatalogCaseComparison.compare_case_bytes(f.payload,l.payload,f.payload,l.payload;
                        mode=f.mode,start=START,stop=START+Minute(91))
                    @test agreement.scheduled==91 && agreement.features_accepted==60 && agreement.labels_accepted==61 && agreement.supervised==30
                    @test agreement.changed_feature_numbers==agreement.changed_label_numbers==0
                    @test !agreement.source_provenance_qualified && !agreement.actual_fit_performed
                    @test counts.scheduled==91 && counts.features_accepted==60 && counts.labels_accepted==61 && counts.supervised==30
                    arrays=C.case_arrays(f,l)
                    @test arrays.issues==collect(START+Minute(31):Minute(1):START+Minute(60))
                    @test size(arrays.features)==(30,layout=="mixed" ? 8 : 11)
                    @test C.feature_row(f,33).raw_now==5.0
                    @test string(reinterpret(UInt64,C.label_row(l,33).target);base=16,pad=16)==FIXTURE["target_expected_bits"]
                    if layout=="mixed"
                        for expected in WHOLE["rows"]
                            row=expected["issue_minute"]+1;actual=C.feature_row(f,row);label=C.label_row(l,row)
                            @test actual.accepted==expected["feature_accepted"] && label.accepted==expected["label_accepted"]
                            if actual.accepted
                                gold=expected["feature_numeric13"]
                                @test isapprox(actual.raw_now,gold[1];atol=1e-13,rtol=0)
                                @test isapprox(actual.scale,gold[2];atol=1e-13,rtol=0)
                                for column in 1:8;@test isapprox(actual.features[column],gold[column+2];atol=1e-13,rtol=0);end
                            end
                            @test expected["target"]===nothing ? label.target===nothing : reinterpret(UInt64,label.target)==reinterpret(UInt64,Float64(expected["target"]))
                        end
                        index==1 && @test bytes2hex(codeunits(f.payload)[32*160+1:33*160])==BINARY["mixed8_issue32_feature_hex"]
                    else
                        for (actual,gold) in zip(C.feature_row(f,33).features,FIXTURE["ground11_expected"])
                            @test isapprox(actual,gold;atol=1e-13,rtol=0)
                        end
                    end
                    domain=Q.split_arrays(value,index;start=START+Minute(40),stop=START+Minute(80),asof=START+Minute(121))
                    @test domain.issues==collect(START+Minute(40):Minute(1):START+Minute(49))
                    @test length(domain.targets)==10 && domain.actual_fit_performed===false
                    # Literal boundary: issue32 target becomes available at62:45.
                    equal=START+Minute(62)+Second(45);calls=Ref(0)
                    spy=function (audit,row)
                        calls[]+=1
                        @test row==33
                        C.label_row(audit,row)
                    end
                    empty=Q._split_arrays(value,index,START+Minute(32),equal,START+Minute(121),spy)
                    @test isempty(empty.targets) && calls[]==0
                    admitted=Q._split_arrays(value,index,START+Minute(32),equal+Millisecond(1),START+Minute(121),spy)
                    @test admitted.issues==[START+Minute(32)] && calls[]==1
                    @test reinterpret(UInt64,only(admitted.targets))==parse(UInt64,FIXTURE["target_expected_bits"];base=16)
                    domain.features[1,1]=-999
                    @test C.feature_row(f,41).features[1]!=-999
                end
                @test_throws ArgumentError Q.publish(request_path,artifact;request_sha256=request_pin)
                @test digest(read(joinpath(artifact,"manifest.json")))==mp
                @test_throws MethodError Q.split_arrays((;features=value.features),1;start=START,stop=START+Minute(1),asof=START+Minute(2))
                @test_throws MethodError Q.VerifiedCases()
                @test_throws ArgumentError Q.split_arrays(value,true;start=START,stop=START+Minute(1),asof=START+Minute(2))
                altered=deepcopy(request);altered["evidence_mode"]="actual_archival_cache";altered["source"]["directory"]=joinpath(root,"must-not-open")
                blocked=joinpath(root,"actual-request.json");R._exclusive_write(blocked,R.canonical_bytes(altered;limit=131072))
                check_failure(()->Q.publish(blocked,joinpath(root,"must-not-publish");request_sha256=digest(read(blocked))),"generated case job authority required")
                @test !ispath(joinpath(root,"must-not-open")) && !ispath(joinpath(root,"must-not-publish"))
                extra=joinpath(source,"foreign.txt");write(extra,"unrecognized")
                check_failure(()->F.load_generated(source;manifest_sha256=pins["manifest_sha256"],complete_sha256=pins["complete_sha256"],role,layout),"snapshot filesystem fileset differs")
                rm(extra) # Exact generated file created immediately above.
                for (name,change,expected) in [
                    ("bool-count",(m,c)->(m["tables"][1]["counts"]["features_rejected"]=true),"snapshot integer bound"),
                    ("bool-scheduled",(m,c)->(m["tables"][1]["scheduled"]=true),"snapshot integer bound"),
                    ("wrong-context",(m,c)->(m["role"]=role=="primary" ? "independent" : "primary"),"case reader request binding")]
                    changed=joinpath(root,name);cp(artifact,changed)
                    mh,ch=repack(changed,change)
                    check_failure(()->read_case(changed;m=mh,c=ch),expected)
                end
                changed=joinpath(root,"rehashed-padding");cp(artifact,changed)
                mh,ch=repack(changed) do metadata,marker
                    name=metadata["tables"][1]["feature_shards"][1];file=joinpath(changed,name)
                    bytes=read(file);bytes[19]=0x01;write(file,bytes)
                    descriptor=only(filter(entry->entry["name"]==name,metadata["members"]));descriptor["sha256"]=digest(bytes)
                end
                check_failure(()->read_case(changed;m=mh,c=ch),"case reader recomputed payload differs")
            end
            push!(results,Dict("role"=>role,"layout"=>layout,"feature_sha256"=>[f.payload_sha256 for f in value.features],"label_sha256"=>[l.payload_sha256 for l in value.labels]))
        end
    end
    write(joinpath(output,"result.json"),JSON3.write(Dict("complete"=>true,"runtime"=>string(VERSION),"generated_cases"=>results,"actual_source_access"=>false,"actual_features"=>false,"actual_fits"=>0,"production_qualified"=>false)))
end
mktempdir() do directory
    run_generated_case_job_tests(realpath(directory))
    result=JSON3.read(read(joinpath(directory,"result.json")),Any)
    @testset "Within-runtime role payload equality" begin
        for layout in ("ground-only","mixed")
            pair=filter(case->case["layout"]==layout,result["generated_cases"])
            @test length(pair)==2
            @test pair[1]["feature_sha256"]==pair[2]["feature_sha256"]
            @test pair[1]["label_sha256"]==pair[2]["label_sha256"]
        end
    end
end
end

include(joinpath(@__DIR__,"authority_tests.jl"))
include(joinpath(@__DIR__,"comparison_tests.jl"))
include(joinpath(@__DIR__,"snapshot_value_tests.jl"))
include(joinpath(@__DIR__,"snapshot_separate_wind_tests.jl"))
include(joinpath(@__DIR__,"snapshot_shard_geometry_tests.jl"))
include(joinpath(@__DIR__,"rounding","runtests.jl"))
