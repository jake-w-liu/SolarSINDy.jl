module DailyOperationalReview

using Dates, HTTP, JSON3, MbedTLS, SHA
include(joinpath(@__DIR__, "live_log_lock.jl"))

utc(t) = string(t)*"Z"
digest(bytes) = bytes2hex(sha256(bytes))
getpath(value, keys...) = isempty(keys) ? value :
    value isa AbstractDict ? getpath(get(value,first(keys),nothing),Base.tail(keys)...) : nothing

function capture_file(path; reference=now(UTC), check_age=false)
    body = nothing; sha = nothing
    try
        filesize(path) <= 16*1024*1024 || error("review input exceeds 16 MiB")
        bytes = read(path); body = String(copy(bytes)); sha = digest(bytes)
        payload = endswith(path,".json") ? JSON3.read(body,Dict{String,Any}) : nothing
        age_minutes = nothing
        if check_age
            stamp = String(payload["generated_utc"])
            endswith(stamp,"Z") || error("claim timestamp is not UTC")
            age_minutes = Dates.value(reference-DateTime(chop(stamp;tail=1)))/60000
        end
        fresh = !check_age || 0<=age_minutes<=130
        return (;available=true,fresh,age_minutes,sha256=sha,body,payload,error=nothing)
    catch error
        error isa InterruptException && rethrow()
        return (;available=false,fresh=false,age_minutes=nothing,sha256=sha,
            body,payload=nothing,error=sprint(showerror,error))
    end
end

function capture_endpoint(url; http_get=HTTP.get, clock=()->now(UTC))
    started = clock()
    body = nothing; sha = nothing; completed = nothing; http_status = nothing
    try
        response = http_get(url;connect_timeout=3,readtimeout=10,retries=0,
            socket_type_tls=MbedTLS.SSLContext)
        completed = clock()
        http_status = response.status
        length(response.body) <= 1024*1024 || error("dashboard response exceeds 1 MiB")
        body = String(copy(response.body)); sha = digest(response.body)
        completed >= started || error("clock moved backward during dashboard receipt")
        http_status == 200 || error("dashboard returned HTTP $http_status")
        payload = JSON3.read(body,Dict{String,Any})
        return (;available=true,url,started_utc=utc(started),completed_utc=utc(completed),
            http_status,sha256=sha,body,payload,error=nothing)
    catch error
        error isa InterruptException && rethrow()
        completed === nothing && (completed = clock())
        return (;available=false,url,started_utc=utc(started),completed_utc=utc(completed),
            http_status,sha256=sha,body,payload=nothing,error=sprint(showerror,error))
    end
end

display_value(x) = x === nothing ? "unavailable" : string(x)
function report(record)
    a3 = record.claim.payload
    violations = getpath(a3,"integrity","violations")
    rows = getpath(a3,"marginal","n_rows")
    coverage = getpath(a3,"marginal","coverage_90")
    health = getpath(record.dashboard.payload,"status")
    lines = ["# Daily operational review", "", "Captured: $(record.generated_utc)", "",
        "Dashboard status: $(display_value(health)).",
        "A3 audit available/fresh: $(record.claim.available)/$(record.claim.fresh).",
        "A3 matured rows: $(display_value(rows)). Coverage: $(display_value(coverage)).",
        "A3 historical integrity findings: $(violations isa AbstractVector ? length(violations) : "unavailable").",
        "A3 marginal claim: $(display_value(getpath(a3,"marginal","claim_ready"))).",
        "A3 storm claim: $(display_value(getpath(a3,"storm","claim_ready"))).", "",
        "## Ground observations", ""]
    for (station,capture) in pairs(record.ground)
        push!(lines,"$station: response available=$(capture.available); " *
            "measurement available=$(display_value(getpath(capture.payload,"available"))); " *
            "product=$(display_value(getpath(capture.payload,"data_type"))); " *
            "age minutes=$(display_value(getpath(capture.payload,"age_minutes"))); " *
            "forecast status=$(display_value(getpath(capture.payload,"forecast_status"))).")
    end
    push!(lines,"", "CMO receipt-time screen: $(display_value(getpath(record.cmo_screen.payload,"receipt_time_screen_pass"))).",
        "Observed derivatives and future forecasts are separate products.",
        "", "## Matched point comparisons", "")
    push!(lines,record.comparison.available ? record.comparison.body :
        "Comparison unavailable: $(record.comparison.error)")
    push!(lines,"", "The JSON receipt retains source bytes, hashes, timestamps, and errors.",
        "Process health and elapsed collection time do not establish forecast accuracy.")
    return join(lines,'\n')*"\n"
end

function verify_review(path)
    isdir(path) && !islink(path) || error("daily review is not a regular directory: $path")
    manifest = JSON3.read(read(joinpath(path,"manifest.json")))
    for name in ("review.json","review.md")
        file = joinpath(path,name)
        isfile(file) && !islink(file) || error("daily review file is missing or linked: $file")
        digest(read(file)) == manifest[name] || error("daily review checksum mismatch: $file")
    end
    return path
end

function write_daily_review(monitor_dir; reference=now(UTC), http_get=HTTP.get,
        clock=()->now(UTC), dashboard_health_url=get(ENV,"SOLARSINDY_REVIEW_DASH_URL",
            "http://127.0.0.1:$(get(ENV,"SWM_PORT","8723"))/api/health"),
        screen_path=joinpath(@__DIR__,"../deploy/cmo_ground_shadow/decision.json"))
    endswith(dashboard_health_url,"/api/health") ||
        throw(ArgumentError("daily-review dashboard URL must end in /api/health"))
    root = joinpath(abspath(monitor_dir),"reviews"); mkpath(root)
    path = joinpath(root,string(Date(reference)))
    return _with_forecast_log_lock(joinpath(root,"daily-review")) do
        islink(path) && error("daily review path is a symlink: $path")
        ispath(path) && return verify_review(path)
        claim = capture_file(joinpath(monitor_dir,"v2_4_live_claim_status.json");reference,check_age=true)
        comparison = capture_file(joinpath(monitor_dir,"live_comparison_report.md");reference)
        dashboard = capture_endpoint(dashboard_health_url;http_get,clock)
        base = chop(dashboard_health_url;tail=length("/api/health"))
        ground = (;CMO=capture_endpoint(base*"/api/dbdt?station=CMO";http_get,clock),
                    FRD=capture_endpoint(base*"/api/dbdt?station=FRD";http_get,clock))
        record = (;schema_version=1,generated_utc=utc(clock()),review_date=string(Date(reference)),
            claim,comparison,dashboard,ground,cmo_screen=capture_file(screen_path;reference))
        temporary = mktempdir(root;prefix=".pending-")
        try
            json = JSON3.write(record); markdown = report(record)
            write(joinpath(temporary,"review.json"),json)
            write(joinpath(temporary,"review.md"),markdown)
            write(joinpath(temporary,"manifest.json"),JSON3.write(Dict(
                "review.json"=>digest(json),"review.md"=>digest(markdown))))
            mv(temporary,path)
        finally
            isdir(temporary) && rm(temporary;recursive=true)
        end
        return verify_review(path)
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==1 || error("Usage: daily_review.jl MONITOR_DIR")
    println(write_daily_review(only(ARGS)))
end

end # module
