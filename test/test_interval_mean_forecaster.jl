module IntervalMeanForecasterPackageTests

using Test, Dates, SHA, TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DRIVER = joinpath(@__DIR__, "interval_mean_forecaster", "worker.jl")
const JOBS = ("smoke", "scanner", "independent2272", "public-baseline", "public-candidate",
              "cache514", "original", "stage4", "stream-baseline", "stream-candidate",
              "training362", "warm44", "tenk-baseline", "tenk-candidate", "grammar-producer", "large79")
const EVIDENCE = mktempdir(; prefix="solarsindy-interval-functional-", cleanup=false)
println("INTERVAL_FORECASTER_RETAINED_EVIDENCE=", EVIDENCE)

@testset "Mandatory package-loaded interval-mean forecaster" begin
    records = Dict{String,Any}[]
    for job in JOBS
        out = joinpath(EVIDENCE, job)
        mkdir(out)
        scratch = joinpath(out, "tmp")
        mkdir(scratch)
        logpath = joinpath(EVIDENCE, job * ".log")
        cmd = `$(Base.julia_cmd()) --startup-file=no --pkgimages=no --threads=1 --project=$ROOT $DRIVER $job $out`
        # Per-child temporary roots contain frozen fixtures that use mktempdir.
        # Parent environment and the original full test suite remain unchanged.
        command = addenv(cmd, "TMPDIR"=>scratch, "TMP"=>scratch, "TEMP"=>scratch)
        started = string(now(UTC))
        proc = open(logpath, "w") do log
            run(pipeline(command; stdout=log, stderr=log); wait=false)
        end
        pid = getpid(proc)
        # This is an explicit audit timeout, not a claim of a hard RSS bound.
        timeout = timedwait(() -> process_exited(proc), 1800.; pollint=0.1) == :timed_out
        if timeout
            kill(proc)
        end
        wait(proc)
        native_good = !timeout && success(proc)
        receipt_path = joinpath(out, "worker.toml")
        receipt_valid = false
        if native_good && isfile(receipt_path)
            try
                receipt = TOML.parsefile(receipt_path)
                receipt_valid = get(receipt, "complete", false) === true && get(receipt, "job", "") == job
            catch error
                println("INVALID_INTERVAL_WORKER_RECEIPT=", sprint(showerror, error))
            end
        end
        good = native_good && receipt_valid
        push!(records, Dict("job"=>job, "argv"=>String.(cmd.exec), "pid"=>pid,
            "started_utc"=>started, "completed_utc"=>string(now(UTC)),
            "native_exit"=>proc.exitcode, "timed_out"=>timeout,
            "receipt_valid"=>receipt_valid,
            "environment_controls"=>Dict("TMPDIR"=>scratch, "TMP"=>scratch, "TEMP"=>scratch),
            "log_sha256"=>bytes2hex(sha256(read(logpath)))))
        open(joinpath(EVIDENCE, "terminal.toml"), "w") do io
            complete = length(records)==length(JOBS) && all(r -> r["native_exit"]==0 &&
                !r["timed_out"] && r["receipt_valid"], records)
            TOML.print(io, Dict("complete"=>complete, "runs"=>records,
                "worker_sha256"=>bytes2hex(sha256(read(DRIVER))), "required_jobs"=>collect(JOBS),
                "scope"=>"synthetic correctness and same-platform before/after bits; no benchmark or empirical skill"), sorted=true)
        end
        if !good
            println("FAILED_INTERVAL_WORKER=", job, " LOG=", logpath)
            print(read(logpath, String))
        end
        @test good
        @test receipt_valid
        if good
            receipt = TOML.parsefile(joinpath(out, "worker.toml"))
            @test receipt["complete"] === true
            @test receipt["job"] == job
        end
        good || break
    end
    @test length(records) == length(JOBS)
end

end
