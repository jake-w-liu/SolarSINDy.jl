"""Optional native agreement for eight separately retained generated full-size jobs."""

using SolarSINDy,Dates,SHA,JSON3
length(ARGS)==2 || error("usage: archive_case_full_scale_agreement.jl INPUT_DIRECTORY FRESH_REPORT")
const C=SolarSINDy.ArchiveCatalogCaseComparison
digest(path)=open(io->bytes2hex(sha256(io)),path)
function read_table(root,table,field,members)
    io=IOBuffer()
    for name in table[field]
        entry=only(filter(x->x["name"]==name,members))
        @assert name isa String && occursin(r"\A[a-zA-Z0-9_.-]+\z",name) && name!="." && name!=".."
        path=joinpath(root,name);@assert isfile(path) && !islink(path)
        @assert filesize(path)==entry["bytes"] && digest(path)==entry["sha256"]
        write(io,read(path))
    end
    value=String(take!(io))
    @assert bytes2hex(sha256(value))==table[field=="feature_shards" ? "feature_payload_sha256" : "label_payload_sha256"]
    value
end
reports=Any[]
for layout in ("ground-only","mixed")
    jobs=[joinpath(ARGS[1],"full-scale-"*v*"-"*role*"-"*layout,"output","cases") for (v,role) in
        (("1.12.6","primary"),("1.12.6","independent"),("1.13.1","primary"),("1.13.1","independent"))]
    for other in 2:4
        left=JSON3.read(read(joinpath(jobs[1],"manifest.json"),String),Dict{String,Any})
        right=JSON3.read(read(joinpath(jobs[other],"manifest.json"),String),Dict{String,Any})
        @assert length(left["tables"])==length(right["tables"])==8
        for index in 1:8
            a=left["tables"][index];b=right["tables"][index];@assert a["declaration"]==b["declaration"]
            declaration=a["declaration"]
            result=C.compare_case_bytes(read_table(jobs[1],a,"feature_shards",left["members"]),
                read_table(jobs[1],a,"label_shards",left["members"]),
                read_table(jobs[other],b,"feature_shards",right["members"]),
                read_table(jobs[other],b,"label_shards",right["members"]);
                mode=layout=="ground-only" ? :ground11 : :mixed8,
                start=DateTime(declaration["start_inclusive"][1:end-1]),stop=DateTime(declaration["stop_exclusive"][1:end-1]))
            @assert result.complete && !result.actual_fit_performed && !result.actual_source_authority_granted
            @assert result.scheduled==a["counts"]["scheduled"] && result.supervised==a["counts"]["supervised"]
            push!(reports,(;layout,other,index,result));GC.gc()
        end
    end
end
@assert C.policy_digest()=="666f9e8112c4ee7747e2121e4cb3b5ed38cf081484468e4cc2399bea271b8c94"
@assert length(reports)==48
@assert sum(x.result.scheduled for x in reports)==7914240
@assert Sys.maxrss()<=8*1024^3
report=(;complete=true,table_comparisons=48,rows_compared=7914240,policy_sha256=C.policy_digest(),
    peak_native_process_RSS_bytes=Sys.maxrss(),actual_fits=0,actual_source_access=false,reports)
SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts._exclusive_write(
    ARGS[2],Vector{UInt8}(codeunits(JSON3.write(report)*"\n")))
println("FULL_SCALE_NATIVE_AGREEMENT_PASSED")
