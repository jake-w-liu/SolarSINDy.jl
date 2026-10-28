# Optional full-size generated numeric/input-export workload; no actual data or fitting.
using Dates,SHA,Test,SolarSINDy
const F=SolarSINDy.ArchiveSourceBoundPointFit
const Q=SolarSINDy.ArchiveCatalogCaseJob
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
length(ARGS)==5 || error("expected module/moduleSHA/request/requestSHA/output")
module_path,module_pin,request_path,request_pin,output=ARGS
@assert bytes2hex(sha256(read(module_path)))==module_pin
include(module_path)
const P=PointInputExport
request_bytes=R.bounded_read(request_path;limit=131072)
@assert bytes2hex(sha256(request_bytes))==request_pin
request=R.decode_canonical(request_bytes;limit=131072)
@assert request["actual_source_access"]===false && request["runtime"]==string(VERSION)
@assert request["schema"]=="generated_input_export_resource_request_v1" && request["layout"] in ("ground-only","mixed")
@assert isabspath(output) && !ispath(output);mkdir(output)
p=request["layout"]=="ground-only" ? 11 : 8;n=600000
X=Matrix{Float64}(undef,n,p)
for j in 1:p,i in 1:n;X[i,j]=mod(i*(2j+1),257)/256;end
targets=Float64[(mod(i,19)+1)/4 for i in 1:n]
raw=Float64[mod(i,7)/8 for i in 1:n];scales=Float64[1+mod(i,13)/16 for i in 1:n]
arrays=(;features=X,rows=collect(1:n),issues=[DateTime(2020)+Minute(i-1) for i in 1:n],targets,raw_now=raw,scales)
before=bytes2hex(sha256(reinterpret(UInt8,vec(X))));before_target=bytes2hex(sha256(reinterpret(UInt8,targets)))
started=time_ns();body,input_sha=P._stream(arrays);elapsed=Int(time_ns()-started)
@assert before==bytes2hex(sha256(reinterpret(UInt8,vec(X)))) && before_target==bytes2hex(sha256(reinterpret(UInt8,targets)))
primitive=joinpath(output,"primitive");mkdir(primitive);members=Any[]
for (index,at) in enumerate(1:P.CHUNK_BYTES:length(body))
    bytes=body[at:min(at+P.CHUNK_BYTES-1,length(body))];name="input-"*lpad(index,4,'0')*".bin"
    R._exclusive_write(joinpath(primitive,name),bytes)
    push!(members,Dict("name"=>name,"bytes"=>length(bytes),"sha256"=>bytes2hex(sha256(bytes))))
end
primitive_report=Dict("complete"=>true,"numeric_primitive_only"=>true,"rows"=>n,"columns"=>p,"input_sha256"=>input_sha,
    "input_bytes"=>length(body),"members"=>members,"elapsed_nanoseconds"=>elapsed,"actual_source_access"=>false,"actual_fits"=>0)
R._exclusive_write(joinpath(primitive,"REPORT.json"),R.canonical_bytes(primitive_report;limit=65536))
println("MAXIMUM_NUMERIC_INPUT_EXPORTED;ROWS=",n,";BYTES=",length(body),";PEAK_RSS_BYTES=",Sys.maxrss());flush(stdout)
body=nothing;arrays=nothing;X=nothing;targets=nothing;raw=nothing;scales=nothing;GC.gc()
cases=Q.read_verified(request["case_request_path"],request["case_directory"];request_sha256=request["case_request_sha256"],
    manifest_sha256=request["case_manifest_sha256"],complete_sha256=request["case_complete_sha256"])
case_request=R.decode_canonical(Vector{UInt8}(codeunits(cases.request_json));limit=131072)
@assert cases.actual_source_authority_granted===false && cases.actual_fit_performed===false
@assert case_request["layout"]==request["layout"] && case_request["runtime"]==string(VERSION) && case_request["role"]=="primary"
exports=Any[]
for index in (1,5)
    table=case_request["tables"][index]
    @assert table["start_inclusive"]=="2017-01-01T00:00:00.000Z" && table["stop_exclusive"]=="2018-01-01T00:00:00.000Z"
    destination=joinpath(output,table["station"]*"-full-year")
    started=time_ns();pins=P.export_generated(cases,index,destination;start=DateTime(2017),stop=DateTime(2018),asof=DateTime(2018)+Minute(31))
    @assert pins.rows==525539 && pins.chunks>1 && pins.actual_source_access===false && pins.actual_fits==0
    planning=P.planning_request(cases,index;start=DateTime(2017),stop=DateTime(2018),asof=DateTime(2018)+Minute(31))
    push!(exports,Dict("directory"=>destination,"table_index"=>index,"station"=>table["station"],"rows"=>pins.rows,"columns"=>pins.columns,
        "context_sha256"=>pins.context_sha256,"manifest_sha256"=>pins.manifest_sha256,"complete_sha256"=>pins.complete_sha256,
        "receipt_sha256"=>pins.receipt_sha256,"input_sha256"=>pins.input_sha256,"chunks"=>pins.chunks,
        "entry_sha256"=>module_pin,"numeric_source_sha256"=>planning["numeric_source_sha256"],"library_sha256"=>planning["library_sha256"],
        "elapsed_nanoseconds"=>Int(time_ns()-started),"peak_RSS_bytes"=>Sys.maxrss()))
    @assert Sys.maxrss()<=8*1024^3
    println("FULL_YEAR_SOURCE_INPUT_EXPORTED;STATION=",table["station"],";ROWS=",pins.rows,";CHUNKS=",pins.chunks,";PEAK_RSS_BYTES=",Sys.maxrss());flush(stdout)
end
@assert length(exports)==2 && Sys.maxrss()<=8*1024^3
report=Dict("complete"=>true,"runtime"=>string(VERSION),"layout"=>request["layout"],"primitive"=>primitive_report,"exports"=>exports,
    "peak_RSS_bytes"=>Sys.maxrss(),"actual_source_access"=>false,"actual_fits"=>0,"scientific_performance_claim"=>false)
R._exclusive_write(joinpath(output,"REPORT.json"),R.canonical_bytes(report;limit=262144))
println("INPUT_EXPORT_RESOURCE_JOB_PASSED;ACTUAL_FITS=0;PEAK_RSS_BYTES=",Sys.maxrss())
