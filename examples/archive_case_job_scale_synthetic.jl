"""Optional full-size generated artifact validation; no actual data or fitting."""
module ArchiveCaseScaleSynthetic
using SolarSINDy,Dates,SHA,JSON3
const J=SolarSINDy.ArchiveCatalogSourceJob
const Q=SolarSINDy.ArchiveCatalogCaseJob
const C=SolarSINDy.ArchiveCatalogCases
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
clock(time)=Dates.format(time,dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
digest(path)=open(io->bytes2hex(sha256(io)),path)
const EVALUATION_STARTS=(DateTime(2023,3),DateTime(2024,5),DateTime(2024,10))
function periods()
    vcat([(DateTime(2017,month),DateTime(2017,month)+Month(1),DateTime(2017)) for month in 1:12],
        [(start,start+Month(1),start) for start in EVALUATION_STARTS])
end
function descriptor(path,bytes,kind,station,start,stop)
    R._exclusive_write(path,bytes)
    Dict("logical_id"=>basename(path),"path"=>path,"kind"=>kind,"station"=>station,
        "stream_labels"=>(kind=="wind" ? ["CMO","FRD"] : [station]),
        "product"=>(kind=="wind" ? "OMNI_HRO_1MIN" : "quasi-definitive"),
        "start_inclusive"=>clock(start),"stop"=>clock(kind=="ground" ? stop-Minute(1) : stop),
        "stop_inclusive"=>kind=="ground","bytes"=>length(bytes),"sha256"=>bytes2hex(sha256(bytes)),
        "role"=>"generated_fixture","provenance"=>"generated_fixture")
end
function generate_inputs(root,layout)
    mkdir(root);inputs=Any[]
    for (start,stop,origin) in periods()
        tag=Dates.format(start,dateformat"yyyy-mm")
        for (station,factor) in (("CMO",1),("FRD",2))
            io=IOBuffer()
            for line in (" Format                 IAGA-2002 |"," Source of Data         GENERATED |",
                " IAGA CODE              $station |"," Reported               XYZF |",
                " Data Interval Type     Filtered 1-minute |"," Data Type              quasi-definitive |",
                "DATE TIME DOY $(station)X $(station)Y $(station)Z $(station)F |")
                println(io,line)
            end
            for time in start:Minute(1):stop-Minute(1)
                k=div(Dates.value(time-origin),60000);value=mod(k,1000)
                println(io,Dates.format(time,dateformat"yyyy-mm-dd HH:MM:SS.sss"),' ',
                    lpad(string(dayofyear(time)),3,'0'),' ',3factor*value,' ',4factor*value," 0 0")
            end
            path=joinpath(root,"generated-"*station*"-"*tag*".txt")
            push!(inputs,descriptor(path,take!(io),"ground",station,start,stop))
        end
        if layout=="mixed"
            io=IOBuffer()
            for time in start:Minute(1):stop-Minute(1);println(io,clock(time),",-2,-2,400,5");end
            path=joinpath(root,"generated-wind-"*tag*".csv")
            push!(inputs,descriptor(path,take!(io),"wind",nothing,start,stop))
        end
    end
    @assert length(inputs)==(layout=="mixed" ? 45 : 30)
    inputs
end
function declarations()
    output=Any[]
    for station in ("CMO","FRD"),start in (DateTime(2017),EVALUATION_STARTS...)
        stop=start==DateTime(2017) ? DateTime(2018) : start+Month(1)
        push!(output,Dict("id"=>station*"-generated-"*Dates.format(start,dateformat"yyyy-mm"),
            "station"=>station,"start_inclusive"=>clock(start),"stop_exclusive"=>clock(stop),"label_asof"=>clock(stop+Minute(31))))
    end
    output
end
function expected_values(factor,regime)
    setprecision(256) do
        low=BigFloat(5factor);high=BigFloat(4995factor)
        now=regime==2 ? high : low
        average=regime==0 ? low : (29low+high)/30
        peak=regime==0 ? low : high
        deviation=regime==0 ? BigFloat(0) : (high-low)/sqrt(BigFloat(30))
        ground=[log1p(now),log1p(average),log1p(peak),log1p(deviation),now/(1+peak),average/(1+peak),deviation/(1+average)]
        (;raw=Float64.([now,average,peak,deviation]),ground=Float64.(ground),scale=Float64(1+ground[3]),peak=Float64(peak))
    end
end
function verify(value,tables)
    summaries=Any[]
    for (index,table) in enumerate(tables)
        start=DateTime(table["start_inclusive"][1:end-1]);stop=DateTime(table["stop_exclusive"][1:end-1])
        n=div(Dates.value(stop-start),60000);factor=table["station"]=="CMO" ? 1 : 2
        f=value.features[index];l=value.labels[index];counts=C.case_counts(f,l)
        @assert counts.scheduled==n && counts.features_accepted==n-31 && counts.labels_accepted==n-30 && counts.supervised==n-61
        references=[expected_values(factor,regime) for regime in 0:2]
        samples=Set(filter(i->0<=i<n,(0,1,30,31,32,970,999,1000,1001,1030,8191,8192,n-31,n-30,n-1)))
        for row in 1:n
            i=row-1;issue=start+Minute(i);feature=C.feature_row(f,row);label=C.label_row(l,row)
            @assert feature.issue==label.issue==issue && feature.label_ready==label.label_ready==issue+Minute(30)+Second(45)
            @assert feature.accepted==(i>=31) && label.accepted==(i+30<n)
            if label.accepted
                target=(1000-mod(i,1000)<=30 ? 4995.0 : 5.0)*factor
                @assert reinterpret(UInt64,label.target)==reinterpret(UInt64,target)
            end
            feature.accepted || continue
            residue=mod(i,1000);regime=1<=residue<=30 ? (residue==1 ? 2 : 1) : 0
            reference=references[regime+1]
            @assert reinterpret(UInt64,feature.raw_now)==reinterpret(UInt64,reference.raw[1])
            @assert isapprox(feature.scale,reference.scale;atol=1e-13,rtol=0)
            if f.mode===:ground11
                for column in 1:7;@assert isapprox(feature.features[column],reference.ground[column];atol=1e-13,rtol=0);end
                if i in samples
                    setprecision(256) do
                        daily=2*BigFloat(pi)*(60hour(issue)+minute(issue))/1440
                        seasonal=2*BigFloat(pi)*(dayofyear(issue)-1)/daysinyear(issue)
                        expected=Float64.([sin(daily),cos(daily),sin(seasonal),cos(seasonal)])
                        for column in 1:4;@assert isapprox(feature.features[column+7],expected[column];atol=1e-13,rtol=0);end
                    end
                end
            else
                bound=128*(2.0^-53)*reference.peak
                for column in 1:4;@assert abs(feature.features[column]-reference.raw[column])<=bound;end
                @assert feature.features[5:8]==(400.0,-2.0,2.0,800.0)
            end
        end
        push!(summaries,Dict("id"=>table["id"],"scheduled"=>n,"features_accepted"=>counts.features_accepted,
            "labels_accepted"=>counts.labels_accepted,"supervised"=>counts.supervised,
            "feature_payload_sha256"=>f.payload_sha256,"label_payload_sha256"=>l.payload_sha256))
    end
    @assert sum(row["scheduled"] for row in summaries)==1319040
    @assert sum(row["features_accepted"] for row in summaries)==1318792
    @assert sum(row["labels_accepted"] for row in summaries)==1318800
    @assert sum(row["supervised"] for row in summaries)==1318552
    summaries
end
"""Run the full eight-table generated workload in a fresh ordinary directory.

All output and partial failures remain available for inspection. The workload
validates formats, whole-schedule oracles and native process RSS, not forecast
skill. Invoke explicitly; it is separate from the short generated example.
"""
function run_scale_example(directory;role,layout)
    @assert role in ("primary","independent") && layout in ("ground-only","mixed")
    directory=normpath(abspath(directory));J._ordinary(dirname(directory);directory=true)
    @assert !ispath(directory) && !islink(directory)
    mkdir(directory);started=time();phases=Dict{String,Float64}()
    mark=time();inputs=generate_inputs(joinpath(directory,"inputs"),layout);phases["generate_inputs"]=time()-mark
    request=Dict("schema"=>"archive_catalog_audit_request_v1","evidence_mode"=>"generated","layout"=>layout,
        "role"=>role,"runtime"=>string(VERSION),"data_root"=>joinpath(directory,"inputs"),"inputs"=>inputs)
    path=joinpath(directory,"source-request.json");R._exclusive_write(path,R.canonical_bytes(request;limit=131072))
    source=joinpath(directory,"source");mark=time();J.reconstruct(path,source);phases["source_reconstruction"]=time()-mark
    tables=declarations();request=Dict("schema"=>"generated_catalog_case_request_v1","evidence_mode"=>"generated",
        "layout"=>layout,"role"=>role,"runtime"=>string(VERSION),
        "source"=>Dict("directory"=>source,"manifest_sha256"=>digest(joinpath(source,"manifest.json")),"complete_sha256"=>digest(joinpath(source,"COMPLETE.json"))),"tables"=>tables)
    path=joinpath(directory,"case-request.json");R._exclusive_write(path,R.canonical_bytes(request;limit=131072));request_pin=digest(path)
    destination=joinpath(directory,"cases");mark=time();metadata=Q.publish(path,destination;request_sha256=request_pin);phases["publish"]=time()-mark
    @assert length(metadata["members"])+2==334
    mark=time();value=Q.read_verified(path,destination;request_sha256=request_pin,
        manifest_sha256=digest(joinpath(destination,"manifest.json")),complete_sha256=digest(joinpath(destination,"COMPLETE.json")))
    phases["read_verified"]=time()-mark
    @assert !value.actual_source_authority_granted && !value.actual_fit_performed
    mark=time();summaries=verify(value,tables);phases["independent_whole_schedule_oracles"]=time()-mark
    peak=Sys.maxrss();@assert peak<=8*1024^3
    report=Dict("schema"=>"generated_case_artifact_scale_report_v1","complete"=>true,"runtime"=>string(VERSION),
        "role"=>role,"layout"=>layout,"tables"=>summaries,"artifact_members"=>334,"phase_seconds"=>phases,
        "total_seconds"=>time()-started,"peak_native_process_RSS_bytes"=>peak,"actual_source_access"=>false,
        "actual_features"=>false,"actual_fits"=>0,"forecast_skill_qualified"=>false,"novelty_qualified"=>false)
    R._exclusive_write(joinpath(directory,"SCALE_RESULT.json"),Vector{UInt8}(codeunits(JSON3.write(report)*"\n")))
    println(JSON3.write(report));report
end
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==3 || error("usage: archive_case_job_scale_synthetic.jl FRESH_OUTPUT ROLE LAYOUT")
    ArchiveCaseScaleSynthetic.run_scale_example(ARGS[1];role=ARGS[2],layout=ARGS[3])
end
