module ArchiveSnapshotValueTests
using SolarSINDy,Test,JSON3,SHA
const CatalogSnapshotValues=SolarSINDy.ArchiveCatalogSnapshotValues
const PACKAGE=pkgdir(SolarSINDy)
include(joinpath(PACKAGE,"test","archive_format","source_job","contract_body.jl"))
const V=CatalogSnapshotValues
hashtext(x)=bytes2hex(sha256(codeunits(x)))
function repack(manifest,marker)
    text=String(RJ.canonical_bytes(manifest;limit=1_048_576))
    copy_marker=deepcopy(marker)
    copy_marker["manifest_sha256"]=hashtext(text)
    copy_marker["snapshot_bytes_without_marker"]=ncodeunits(text)+sum(m["bytes"] for m in manifest["members"])
    text,String(RJ.canonical_bytes(copy_marker;limit=4096))
end
function rebuild(text,marker,members,role,layout)
    V.reconstruct(text,marker,members;manifest_sha256=hashtext(text),complete_sha256=hashtext(marker),
                  expected_layout=layout,expected_role=role,expected_runtime=string(VERSION),expected_evidence_mode="generated")
end
function reject_metadata(f,m,c,members,role,layout)
    changed=deepcopy(m);f(changed)
    text,marker=repack(changed,c)
    @test_throws ArgumentError rebuild(text,marker,members,role,layout)
end
function reject_large_metadata(large,members,role,layout)
    try
        V.reconstruct(large,"",members;manifest_sha256=repeat("0",64),complete_sha256=repeat("0",64),
                      expected_layout=layout,expected_role=role,expected_runtime=string(VERSION),expected_evidence_mode="generated")
    catch error
        error isa ArgumentError && error.msg=="snapshot buffer size" || rethrow()
        return true
    end
    false
end
function run(OUTPUT)
    fixture_path=joinpath(PACKAGE,"test","archive_format","source_job","fixtures","FROZEN_GENERATED_FIXTURES_AND_ORACLES_V1.json")
    fixture=JSON3.read(read(fixture_path),Any)
    results=Any[]
    for role in ("primary","independent"),layout in ("ground-only","mixed")
        mktempdir(OUTPUT) do root
            root=realpath(root)
            @testset "Fresh generated source snapshot $role/$layout" begin
                source_job_contract(root,role,layout,fixture_path)
                source=joinpath(root,"snapshot")
                text=read(joinpath(source,"manifest.json"),String);marker_text=read(joinpath(source,"COMPLETE.json"),String)
                manifest=RJ.decode_canonical(Vector{UInt8}(codeunits(text));limit=1_048_576)
                marker=RJ.decode_canonical(Vector{UInt8}(codeunits(marker_text));limit=4096)
                members=Dict{String,String}(m["name"]=>read(joinpath(source,m["name"]),String) for m in manifest["members"])
                value=rebuild(text,marker_text,members,role,layout)
                @test !value.actual_source_authority_granted && !value.provider_receipts_available && !value.independent_source_qualification && value.actual_fits==0
                expected=fixture[layout=="ground-only" ? "expected_ground_only_catalogs" : "expected_mixed_catalogs"]
                @test Set(string.(keys(value.catalogs)))==Set(expected)
                for stream in expected
                    catalog=getproperty(value.catalogs,Symbol(stream));oracle=fixture["catalog_oracles"][stream]
                    @test catalog.selected_sha256==oracle["selected_sha256"]
                    @test catalog.raw_catalog_sha256==oracle["raw_catalog_sha256"]
                    @test catalog.rows==oracle["rows"] && catalog.raw_rows==oracle["raw_rows"]
                end
                @test Set(value.validated_member_names)==Set(keys(members))
                reject_metadata(manifest,marker,members,role,layout) do m;m["catalogs"][first(expected)]["rows"]=true;end
                reject_metadata(manifest,marker,members,role,layout) do m;m["read_trace"][1]["bytes"]=true;end
                reject_metadata(manifest,marker,members,role,layout) do m;m["provider_receipts_available"]=true;end
                reject_metadata(manifest,marker,members,role,layout) do m;m["independent_source_qualification"]=true;end
                reject_metadata(manifest,marker,members,role,layout) do m;m["actual_fits"]=true;end
                reject_metadata(manifest,marker,members,role,layout) do m;m["evidence_mode"]="actual_archival_cache";end
                reject_metadata(manifest,marker,members,role,layout) do m;m["unknown"]=0;end
                reject_metadata(manifest,marker,members,role,layout) do m;m["catalogs"][first(expected)]["selected_sha256"]=repeat("0",64);end
                reject_metadata(manifest,marker,members,role,layout) do m;m["bodies"][1]["declaration"]["provenance"]="bad\nvalue";end
                changed=deepcopy(manifest);entry=changed["bodies"][1]
                entry["body_ids"][first(keys(entry["body_ids"]))]=repeat("0",64)
                bad_text,bad_marker=repack(changed,marker)
                @test_throws ArgumentError rebuild(bad_text,bad_marker,members,role,layout)
                missing=copy(members);delete!(missing,first(keys(missing)))
                @test_throws ArgumentError rebuild(text,marker_text,missing,role,layout)
                extra=copy(members);extra["unrecognized.bin"]=""
                @test_throws ArgumentError rebuild(text,marker_text,extra,role,layout)
                raw_name=manifest["bodies"][1]["raw"]
                corrupted=copy(members);bytes=collect(codeunits(corrupted[raw_name]));@test !isempty(bytes);bytes[end]⊻=0x01;corrupted[raw_name]=String(bytes)
                @test_throws ArgumentError rebuild(text,marker_text,corrupted,role,layout)
                changed=deepcopy(manifest)
                descriptor=only(filter(m->m["name"]==raw_name,changed["members"]))
                descriptor["sha256"]=hashtext(corrupted[raw_name])
                bad_text,bad_marker=repack(changed,marker)
                @test_throws ArgumentError rebuild(bad_text,bad_marker,corrupted,role,layout)
                bad=deepcopy(marker);bad["members"]=true
                @test_throws ArgumentError rebuild(text,String(RJ.canonical_bytes(bad)),members,role,layout)
                bad=deepcopy(marker);bad["manifest_sha256"]=repeat("0",64)
                @test_throws ArgumentError rebuild(text,String(RJ.canonical_bytes(bad)),members,role,layout)
                @test_throws ArgumentError V.reconstruct(text,marker_text,members;manifest_sha256=repeat("0",64),complete_sha256=hashtext(marker_text),expected_layout=layout,expected_role=role,expected_runtime=string(VERSION),expected_evidence_mode="generated")
                large=repeat("x",4*1024^2)
                for _ in 1:3;@test reject_large_metadata(large,members,role,layout);end
                allocated=@allocated @test reject_large_metadata(large,members,role,layout)
                @test allocated<1024^2
                vector=collect(codeunits(large))
                for _ in 1:3;@test reject_large_metadata(vector,members,role,layout);end
                allocated_vector=@allocated @test reject_large_metadata(vector,members,role,layout)
                @test allocated_vector<1024^2
                saved=getproperty(value.catalogs,Symbol(first(expected))).selected_sha256
                members[first(keys(members))]="changed after reconstruction"
                @test getproperty(value.catalogs,Symbol(first(expected))).selected_sha256==saved
                push!(results,Dict("role"=>role,"layout"=>layout,"catalogs"=>length(expected),"oversize_string_allocated"=>allocated,"oversize_vector_allocated"=>allocated_vector))
            end
        end
    end
    result=Dict("complete"=>true,"runtime"=>string(VERSION),"generated_cases"=>results,"actual_source_access"=>false,"actual_features"=>false,"actual_fits"=>0,"production_qualified"=>false)
    open(joinpath(OUTPUT,"result.json"),"w") do io;write(io,JSON3.write(result));end
    println("snapshot-values-generated-controls-complete")
end
mktempdir() do output
    run(realpath(output))
end
end
