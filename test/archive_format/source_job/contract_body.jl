using SolarSINDy, Test, JSON3, SHA
const J = SolarSINDy.ArchiveCatalogSourceJob
const RJ = SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
source_job_hash(bytes) = bytes2hex(sha256(bytes))
function source_job_contract(directory, role, layout, fixture_path)
    fixture = JSON3.read(read(fixture_path),Any)
    inputs_dir = joinpath(directory,"inputs"); mkdir(inputs_dir)
    names = fixture[layout=="ground-only" ? "ground_only_inputs" : "mixed_inputs"]
    inputs = Any[]
    for name in names
        f = fixture["fixtures"][name]
        bytes = Vector{UInt8}(codeunits(f["text"]))
        path = joinpath(inputs_dir,name); RJ._exclusive_write(path,bytes)
        push!(inputs,Dict("logical_id"=>name,"path"=>path,"kind"=>f["kind"],"station"=>f["station"],
            "stream_labels"=>f["stream_labels"],"product"=>f["product"],"start_inclusive"=>fixture["start"],
            "stop"=>fixture["stop"],"stop_inclusive"=>f["kind"]=="ground","bytes"=>length(bytes),
            "sha256"=>source_job_hash(bytes),"role"=>"generated_fixture","provenance"=>"generated_fixture"))
    end
    request = Dict("schema"=>"archive_catalog_audit_request_v1","evidence_mode"=>"generated",
        "layout"=>layout,"role"=>role,"runtime"=>string(VERSION),"data_root"=>inputs_dir,"inputs"=>inputs)
    request_path = joinpath(directory,"request.json")
    RJ._exclusive_write(request_path,RJ.canonical_bytes(request;limit=131072))
    output = joinpath(directory,"snapshot")
    manifest = J.reconstruct(request_path,output)
    marker = RJ.decode_canonical(read(joinpath(output,"COMPLETE.json")))
    @test marker["manifest_sha256"]==source_job_hash(read(joinpath(output,"manifest.json")))
    @test manifest["role"]==role
    @test manifest["layout"]==layout
    @test manifest["runtime"]==string(VERSION)
    @test manifest["provider_receipts_available"]===false
    @test manifest["independent_source_qualification"]===false
    @test manifest["actual_fits"]==0
    @test length(manifest["read_trace"])==length(inputs)
    @test marker["members"]==length(readdir(output))
    @test marker["snapshot_bytes_without_marker"]==sum(filesize(joinpath(output,n)) for n in readdir(output) if n!="COMPLETE.json")
    expected = fixture[layout=="ground-only" ? "expected_ground_only_catalogs" : "expected_mixed_catalogs"]
    @test Set(keys(manifest["catalogs"]))==Set(expected)
    for stream in expected
        oracle = fixture["catalog_oracles"][stream]
        catalog = manifest["catalogs"][stream]
        for key in ("selected_sha256","raw_catalog_sha256","raw_rows","rows","declared_minutes",
                    "duplicates","missing_minutes","uncovered_minutes")
            @test catalog[key]==oracle[key]
        end
        @test catalog["body_ids"]==[oracle["body_id"]]
        @test catalog["complete_vectors"]==oracle["rows"]
        selected = reduce(vcat,(read(joinpath(output,n)) for n in catalog["shards"]))
        @test length(selected)==oracle["selected_bytes"]
        @test source_job_hash(selected)==oracle["selected_sha256"]
        body = only(filter(b->haskey(b["body_ids"],stream),manifest["bodies"]))
        raw = read(joinpath(output,body["raw"]))
        @test length(raw)==oracle["raw_bytes"]
        @test source_job_hash(raw)==oracle["raw_sha256"]
        @test bytes2hex(raw[1:48])==oracle["first_raw_hex"]
        @test bytes2hex(raw[end-47:end])==oracle["last_raw_hex"]
    end
    payloads = Dict{String,String}()
    for member in manifest["members"]
        bytes = read(joinpath(output,member["name"]))
        @test length(bytes)==member["bytes"]<=8_388_608
        @test source_job_hash(bytes)==member["sha256"]
        payloads[member["name"]]=member["sha256"]
    end
    for body in manifest["bodies"]
        @test String(read(joinpath(output,body["original"])))==fixture["fixtures"][body["logical_id"]]["text"]
    end
    original_marker = read(joinpath(output,"COMPLETE.json"))
    @test_throws ArgumentError J.reconstruct(request_path,output)
    @test read(joinpath(output,"COMPLETE.json"))==original_marker
    mkdir(joinpath(directory,"incomplete"))
    @test_throws ArgumentError J.reconstruct(request_path,joinpath(directory,"incomplete"))

    # Every rejection has its own immutable request and destination. Changed
    # declarations cannot contaminate subsequent tests or reuse a completion.
    counter = Ref(0)
    function reject(value; bytes=nothing, authority=nothing, absent=true)
        counter[]+=1
        name = "negative-"*lpad(string(counter[]),3,'0')
        path = joinpath(directory,name*".json")
        RJ._exclusive_write(path,bytes===nothing ? RJ.canonical_bytes(value;limit=131072) : bytes)
        destination = joinpath(directory,name)
        @test_throws ArgumentError J.reconstruct(path,destination;authority)
        @test !isfile(joinpath(destination,"COMPLETE.json"))
        absent && @test !ispath(destination)
    end
    altered = deepcopy(request); altered["extra"]=true; reject(altered)
    altered = deepcopy(request); altered["inputs"][1]["bytes"]=true; reject(altered)
    altered = deepcopy(request); push!(altered["inputs"],deepcopy(altered["inputs"][1])); reject(altered)
    altered = deepcopy(request); altered["inputs"][1]["start_inclusive"]="2020-01-01T00:00:01.000Z"; reject(altered)
    altered = deepcopy(request); altered["inputs"][1]["stop"]="2020-02-03T00:00:00.000Z"; reject(altered)
    altered = deepcopy(request); altered["inputs"][1]["path"]="relative.txt"; reject(altered)
    altered = deepcopy(request); altered["data_root"]=directory; reject(altered)
    altered = deepcopy(request); altered["runtime"]="1.0.0"; reject(altered)
    altered = deepcopy(request); altered["inputs"][1]["sha256"]="0"^64; reject(altered;absent=false)
    altered = deepcopy(request); altered["inputs"][1]["bytes"]+=1; reject(altered)
    altered = deepcopy(request); altered["evidence_mode"]="actual_archival_cache"; reject(altered)
    unused_authority = J.Authority(request_path,"0"^64,request_path,"0"^64,request_path,"0"^64,request_path,"0"^64)
    reject(request;authority=unused_authority)
    incomplete_proof = Dict("complete"=>true,"all_commit_filters_match"=>true,
        "source_sha256"=>Dict("Project.toml"=>source_job_hash(read(joinpath(pkgdir(SolarSINDy),"Project.toml")))))
    @test_throws ArgumentError J._source_inventory(incomplete_proof)
    if layout=="ground-only"
        altered = deepcopy(request)
        wind = deepcopy(altered["inputs"][1]); wind["kind"]="wind"; wind["logical_id"]="generated-forbidden-wind.csv"
        wind["path"]=joinpath(inputs_dir,"generated-must-not-open.csv")
        push!(altered["inputs"],wind); reject(altered)
        @test !ispath(wind["path"])
        @test all(t->!occursin("wind",t["logical_id"]),manifest["read_trace"])
    end
    canonical = RJ.canonical_bytes(request;limit=131072)
    reject(request;bytes=vcat(UInt8[0x20],canonical))
    reject(request;bytes=vcat(Vector{UInt8}(codeunits("{\"schema\":\"duplicate\",")),canonical[2:end]))

    # Overlapping valid values disagree. Both independent raw-ledger algorithms
    # must reject the source rather than silently choose a complete witness.
    altered = deepcopy(request)
    conflict = deepcopy(only(filter(i->i["kind"]=="ground" && i["station"]=="CMO",altered["inputs"])))
    conflict["logical_id"]="generated-conflict.txt"; conflict["path"]=joinpath(inputs_dir,"generated-conflict.txt")
    conflict_bytes = Vector{UInt8}(codeunits(replace(fixture["fixtures"]["generated-cmo.txt"]["text"],
        "00:01:00.000 001 3 4"=>"00:01:00.000 001 4 4")))
    RJ._exclusive_write(conflict["path"],conflict_bytes)
    conflict["bytes"]=length(conflict_bytes); conflict["sha256"]=source_job_hash(conflict_bytes)
    push!(altered["inputs"],conflict); reject(altered;absent=false)
    if !Sys.iswindows()
        target = inputs[1]["path"]; link = joinpath(inputs_dir,"generated-link.txt"); symlink(target,link)
        altered = deepcopy(request); altered["inputs"][1]["path"]=link; reject(altered)
        linked_root = joinpath(directory,"linked-inputs"); symlink(inputs_dir,linked_root)
        altered = deepcopy(request); altered["data_root"]=linked_root; reject(altered)
    end
    Dict("complete"=>true,"role"=>role,"layout"=>layout,"runtime"=>string(VERSION),
        "payload_sha256"=>payloads,"negative_cases"=>counter[],"actual_data"=>false,"actual_fits"=>0)
end
