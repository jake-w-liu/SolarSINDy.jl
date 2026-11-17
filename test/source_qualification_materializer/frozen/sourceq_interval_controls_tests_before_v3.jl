module SourceQIntervalPackageControls
using Test, SHA, JSON3, SolarSINDy
include(joinpath(@__DIR__, "completed_refit_artifacts", "run_ci_native_chain.jl"))
const SUPPORT = CompletedRefitNativeDriver
const CONTROL = joinpath(@__DIR__, "run_sourceq_interval_controls.py")
const CONTROL_SHA = "8571b5c0321f10deda20f20fa1ccbbdb25e3d48402c2df8a9eedad5e8c7fd430"
const CASES = (
    ("materializer24", 24, "matched342_sourceq_independent_tests_v1"),
    ("workers10", 10, "interval_workers_qualification_independent_controls_v1"),
    ("canonical3", 3, "sourceq_canonical_native_boundary_independent_tests_v1"),
    ("copy2", 2, "sourceq_copy_boundary_independent_tests_v1"),
    ("publication5", 5, "sourceq_and_interval_cli_publication_independent_tests_v1"),
    ("evidence5", 5, "sourceq_finite_evidence_gaps_independent_tests_v1"),
    ("lifecycle2", 2, "sourceq_lifecycle_event_independent_tests_v1"),
)
file_sha(path) = open(io -> bytes2hex(sha256(io)), path)
integer(value) = value isa Integer && !(value isa Bool)

function bounded(path, limit; empty=false)
    isfile(path) && !islink(path) || error("ordinary engineering evidence required")
    size = filesize(path)
    (empty ? 0 <= size <= limit : 0 < size <= limit) || error("engineering evidence exceeds bound")
    bytes = open(io -> read(io, limit + 1), path)
    length(bytes) == size && length(bytes) <= limit || error("engineering evidence grew during read")
    bytes
end
read_record(path) = JSON3.read(bounded(path, 2 * 1024^2), Dict{String,Any})

function run_tests()
    @testset "Mandatory offline SourceQ and interval engineering controls" begin
        isfile(CONTROL) && !islink(CONTROL) && file_sha(CONTROL) == CONTROL_SHA ||
            error("reviewed SourceQ/interval controller changed")
        parent = realpath(get(ENV, "RUNNER_TEMP", tempdir()))
        root = realpath(mktempdir(parent; prefix="matched342-sourceq-interval-", cleanup=false))
        output = joinpath(root, "evidence")
        logpath = joinpath(root, "controller.log")
        python = SUPPORT.choose_python()
        command = addenv(Cmd([python, "-I", "-B", "-X", "utf8", CONTROL, "--output-directory", output]),
                         "PYTHONDONTWRITEBYTECODE" => "1", "PYTHONUTF8" => "1")
        process = nothing
        started = time_ns()
        within_watchdog() = time_ns() - started < UInt64(930 * 10^9)
        try
            open(logpath, "w") do log
                process = run(pipeline(command; stdout=log, stderr=log); wait=false)
                while process_running(process)
                    within_watchdog() || error("SourceQ/interval controller watchdog expired")
                    sleep(.05)
                end
                wait(process)
                within_watchdog() || error("SourceQ/interval controller completed after watchdog")
                success(process) || error("SourceQ/interval controls failed; see " * logpath)
            end
            bytes = bounded(joinpath(output, "TERMINAL.json"), 2 * 1024^2)
            within_watchdog() || error("SourceQ/interval terminal read exceeded watchdog")
            result = JSON3.read(bytes, Dict{String,Any})
            @test result["schema"] == "sourceq_interval_registered_controls_v1"
            @test result["complete"] === true && result["source_unchanged"] === true
            @test integer(result["cases"]) && result["cases"] == result["expected_cases"] == 7
            @test integer(result["unittest_methods"]) && result["unittest_methods"] == result["expected_methods"] == 51
            @test result["stage_seconds"] == 120 && result["owner_cleanup_seconds"] == 20
            @test result["postcheck_reserve_seconds"] == 5
            @test integer(result["actual_Python_stage_launches"]) && result["actual_Python_stage_launches"] == 7
            @test all(integer(result[key]) for key in ("started_ns", "finished_ns", "deadline_ns"))
            @test result["started_ns"] <= result["finished_ns"] < result["deadline_ns"]
            @test result["deadline_ns"] == result["started_ns"] + 900 * 10^9
            @test result["source_before"] == result["source_after"]
            for (path, pin) in result["source_after"]
                @test isabspath(path) && isfile(path) && !islink(path)
                @test file_sha(path) == pin
            end
            for key in ("Julia_launched", "actual_data_exposed", "native_Darwin_qualified", "qualification_issued",
                        "source_qualified", "software_qualified", "scientific_qualified")
                @test result[key] === false
            end
            @test length(result["case_receipts"]) == 7
            for (index, (name, methods, schema)) in enumerate(CASES)
                row = result["case_receipts"][index]
                @test row["name"] == name && row["expected_methods"] == methods
                @test row["expected_schema"] == schema && row["verified"] === true
                @test row["timeout_seconds"] == 120
                @test isfinite(row["observed_stage_wall_seconds"]) && 0 <= row["observed_stage_wall_seconds"] < 120
                @test row["admission"]["available_physical_bytes"] >= 536870912
                @test row["admission"]["free_disk_bytes"] >= 8589934592
                native = row["native"]
                launch = read_record(joinpath(output, name * "-launch.json"))
                @test integer(native["pid"]) && native["pid"] > 0 && native["pid"] == launch["pid"]
                @test native["native_exit"] == 0 && native["timed_out"] === false
                @test native["leader_reaped"] === true && native["ownership_boundary_closed"] === true
                @test native["all_descendants_os_reaped"] === false
                @test isfinite(native["wall_seconds"]) && 0 <= native["wall_seconds"] < 120
                rawpath = joinpath(output, name * ".log")
                raw = bounded(rawpath, 4 * 1024^2; empty=true)
                @test row["raw"]["path"] == rawpath && row["raw"]["bytes"] == length(raw)
                @test row["raw"]["sha256"] == bytes2hex(sha256(raw))
                casepath = joinpath(output, name, "RESULT.json")
                casebytes = bounded(casepath, 2 * 1024^2)
                @test row["result"]["path"] == casepath && row["result"]["bytes"] == length(casebytes)
                @test row["result"]["sha256"] == bytes2hex(sha256(casebytes))
                value = JSON3.read(casebytes, Dict{String,Any})
                @test value["schema"] == schema && value["complete"] === true
                @test value["unittest_methods"] == value["expected_methods"] == methods
                @test all(integer(value[key]) && value[key] == 0 for key in ("failures", "errors", "skipped"))
                @test value["source_before"] == value["source_after"] && value["source_unchanged"] === true
                @test all(value[key] === false for key in ("source_qualified", "software_qualified", "scientific_qualified"))
            end
            @test result["admission"]["available_physical_bytes"] >= 536870912
            @test result["admission"]["free_disk_bytes"] >= 8589934592
            @test file_sha(CONTROL) == CONTROL_SHA
            within_watchdog() || error("SourceQ/interval verification exceeded watchdog")
            println("SOURCEQ_INTERVAL_51_EVIDENCE=", output, " terminal_sha256=", bytes2hex(sha256(bytes)))
            within_watchdog() || error("SourceQ/interval output completed after watchdog")
        catch
            println(stderr, "SOURCEQ_INTERVAL_51_FAILURE_RAW=", logpath)
            if isfile(logpath)
                open(logpath, "r") do io
                    seekend(io)
                    seek(io, max(0, position(io) - 65536))
                    write(stderr, read(io, 65536))
                    println(stderr)
                end
            end
            rethrow()
        finally
            if process !== nothing && process_running(process)
                kill(process)
                cleanup = time_ns()
                escalated = false
                while process_running(process)
                    if !escalated && time_ns() - cleanup >= UInt64(35 * 10^9)
                        kill(process, Base.SIGKILL)
                        escalated = true
                    end
                    time_ns() - cleanup < UInt64(40 * 10^9) || error("SourceQ/interval controller cleanup not observed within40s")
                    sleep(.05)
                end
                wait(process)
            end
        end
    end
end
end
SourceQIntervalPackageControls.run_tests()
