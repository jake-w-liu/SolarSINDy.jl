#!/usr/bin/env julia
# Long-run allocation/resource soak for the package-native live monitor engine.
#
# Run from the package root:
#   julia --project=. validation/monitor_soak.jl
#
# Environment:
#   SOLARSINDY_SOAK_BATCHES       measured 1/2/3/6 h batches (default 1000)
#   SOLARSINDY_SOAK_WARMUP        unmeasured warm-up batches (default 5)
#   SOLARSINDY_SOAK_SAMPLE_EVERY  resource sample interval (default 10 for
#                                 <=200 batches, otherwise 50)
#   SOLARSINDY_SOAK_INPUT         auto, real, or synthetic (default auto)
#   SOLARSINDY_SOAK_CSV           optional sample CSV output path
#
# A resource-growth failure requires five consecutive post-warm samples to be
# nondecreasing and to grow by both 25% and 64 MiB (Julia live heap) or 128 MiB
# (RSS). These deliberately conservative thresholds avoid treating allocator
# retention as a leak while still detecting sustained, operationally material
# growth. File descriptors may not grow at all after warm-up, and the fixed
# input snapshot may create at most one pending row per requested horizon.

isdefined(@__MODULE__, :LiveVerifyConfig) || include(joinpath(@__DIR__, "..", "examples", "live_forecast_verify.jl"))

using Printf

const SOAK_HORIZONS = (1, 2, 3, 6)
const MIB = 1024^2

_env_int(name, default) = parse(Int, get(ENV, name, string(default)))

function _synthetic_inputs(cfg::LiveVerifyConfig)
    issue_time = DateTime(2026, 7, 15, 12, 30)
    sw_times = collect((issue_time - Hour(12)):Minute(1):issue_time)
    plasma = DataFrame(
        time_tag=sw_times,
        speed=fill(500.0, length(sw_times)),
        density=fill(6.0, length(sw_times)),
    )
    mag = DataFrame(
        time_tag=sw_times,
        bz_gsm=fill(-4.0, length(sw_times)),
        by_gsm=fill(1.0, length(sw_times)),
    )
    latest_dst_time = _floor_hour(issue_time) - Hour(1)
    dst_times = collect((latest_dst_time - Hour(12)):Hour(1):latest_dst_time)
    dst_values = collect(range(-20.0, -44.0; length=length(dst_times)))
    return prepare_issue_inputs(
        cfg;
        issue_time,
        plasma_fn=() -> plasma,
        mag_fn=() -> mag,
        dst_fn=() -> (dst_times, dst_values),
    )
end

function _configs(monitor_dir::AbstractString, calibration_path::AbstractString)
    log_path = joinpath(monitor_dir, "live_forecast_log.csv")
    report_path = joinpath(monitor_dir, "live_comparison_report.md")
    return [LiveVerifyConfig(;
        model=:v2,
        horizon_hours=h,
        log_path,
        report_path,
        v2_calibration_path=calibration_path,
    ) for h in SOAK_HORIZONS]
end

function _issue_batch!(configs, inputs)
    for cfg in configs
        issue_forecast(
            cfg;
            inputs,
            write_trajectory=false,
            verbose=false,
        )
    end
    return nothing
end

function _quiet(f)
    redirect_stdout(devnull) do
        f()
    end
end

function _log_rows(path::AbstractString)
    isfile(path) || return 0
    return max(countlines(path) - 1, 0)
end

function _live_heap_bytes()
    isdefined(Base, :gc_live_bytes) || return missing
    return Int(getfield(Base, :gc_live_bytes)())
end

function _rss_bytes()
    try
        return 1024 * parse(Int, strip(read(`ps -o rss= -p $(getpid())`, String)))
    catch
        return missing
    end
end

function _fd_count()
    path = isdir("/proc/$(getpid())/fd") ? "/proc/$(getpid())/fd" : "/dev/fd"
    return length(readdir(path))
end

function _resource_sample(batch, log_path, elapsed, allocated, n_batches)
    GC.gc(true)
    return (
        batch=batch,
        live_heap_bytes=_live_heap_bytes(),
        rss_bytes=_rss_bytes(),
        fd_count=_fd_count(),
        log_rows=_log_rows(log_path),
        mean_batch_seconds=n_batches == 0 ? 0.0 : elapsed / n_batches,
        mean_batch_alloc_bytes=n_batches == 0 ? 0.0 : allocated / n_batches,
    )
end

function _sustained_growth(samples, field; absolute_bytes, relative=0.25, run=5)
    values = [getproperty(sample, field) for sample in samples]
    any(ismissing, values) && return false
    length(values) < run && return false
    for start in 1:(length(values) - run + 1)
        segment = Int.(values[start:(start + run - 1)])
        growth = segment[end] - segment[1]
        if all(diff(segment) .>= 0) && growth > absolute_bytes &&
           growth > relative * max(segment[1], 1)
            return true
        end
    end
    return false
end

function _fmt_mib(value)
    ismissing(value) && return "NA"
    return @sprintf("%.1f", value / MIB)
end

function _run_warmup!(configs, inputs, warmup)
    _quiet() do
        for _ in 1:warmup
            _issue_batch!(configs, inputs)
        end
    end
    return nothing
end

function _prepare_and_warm!(mode, monitor_dir, warmup, calibration_path)
    configs = _configs(monitor_dir, calibration_path)
    if mode != "synthetic"
        try
            inputs = _quiet(() -> prepare_issue_inputs(first(configs)))
            _run_warmup!(configs, inputs, warmup)
            return (; source="real", inputs, configs, fallback=nothing)
        catch error
            mode == "real" && rethrow()
            rm(monitor_dir; recursive=true, force=true)
            mkpath(monitor_dir)
            inputs = _quiet(() -> _synthetic_inputs(first(configs)))
            _run_warmup!(configs, inputs, warmup)
            return (; source="synthetic", inputs, configs,
                    fallback=sprint(showerror, error))
        end
    end
    inputs = _quiet(() -> _synthetic_inputs(first(configs)))
    _run_warmup!(configs, inputs, warmup)
    return (; source="synthetic", inputs, configs, fallback=nothing)
end

function monitor_soak_main()
    batches = _env_int("SOLARSINDY_SOAK_BATCHES", 1000)
    warmup = _env_int("SOLARSINDY_SOAK_WARMUP", 5)
    sample_every = _env_int(
        "SOLARSINDY_SOAK_SAMPLE_EVERY", batches <= 200 ? 10 : 50,
    )
    mode = lowercase(get(ENV, "SOLARSINDY_SOAK_INPUT", "auto"))
    batches >= 1 || error("SOLARSINDY_SOAK_BATCHES must be positive")
    warmup >= 1 || error("SOLARSINDY_SOAK_WARMUP must be positive")
    sample_every >= 1 || error("SOLARSINDY_SOAK_SAMPLE_EVERY must be positive")
    mode in ("auto", "real", "synthetic") ||
        error("SOLARSINDY_SOAK_INPUT must be auto, real, or synthetic")

    mktempdir(; prefix="solarsindy-monitor-soak-") do monitor_dir
        calibration_path = DEFAULT_V2_CALIBRATION_PATH
        setup = _prepare_and_warm!(mode, monitor_dir, warmup, calibration_path)
        log_path = first(setup.configs).log_path
        expected_rows = _log_rows(log_path)
        1 <= expected_rows <= length(SOAK_HORIZONS) || error(
            "warm-up produced $expected_rows rows; expected 1-$(length(SOAK_HORIZONS))",
        )

        samples = NamedTuple[]
        push!(samples, _resource_sample(0, log_path, 0.0, 0, 0))
        baseline_fd = samples[1].fd_count
        elapsed = 0.0
        allocated = 0
        measured = 0
        _quiet() do
            for batch in 1:batches
                timing = @timed _issue_batch!(setup.configs, setup.inputs)
                elapsed += timing.time
                allocated += timing.bytes
                measured += 1
                if batch % sample_every == 0 || batch == batches
                    sample = _resource_sample(batch, log_path, elapsed, allocated, measured)
                    sample.fd_count <= baseline_fd || error(
                        "file descriptors grew from $baseline_fd to $(sample.fd_count)",
                    )
                    sample.log_rows <= expected_rows || error(
                        "fixed input snapshot grew the log from $expected_rows to $(sample.log_rows) rows",
                    )
                    push!(samples, sample)
                    elapsed = 0.0
                    allocated = 0
                    measured = 0
                end
            end
        end

        _sustained_growth(samples, :live_heap_bytes; absolute_bytes=64MIB) &&
            error("live heap has a sustained post-warm growth trend above 64 MiB and 25%")
        _sustained_growth(samples, :rss_bytes; absolute_bytes=128MIB) &&
            error("RSS has a sustained post-warm growth trend above 128 MiB and 25%")

        csv_path = strip(get(ENV, "SOLARSINDY_SOAK_CSV", ""))
        if !isempty(csv_path)
            mkpath(dirname(abspath(csv_path)))
            sample_table = DataFrame(samples)
            insertcols!(sample_table, 1, :input_source => fill(setup.source, nrow(sample_table)))
            CSV.write(csv_path, sample_table)
        end

        println("monitor soak: PASS input=$(setup.source) batches=$batches warmup=$warmup sample_every=$sample_every")
        setup.fallback === nothing || println("real-input fallback: $(setup.fallback)")
        println("batch  heap_MiB  rss_MiB  fd  rows  seconds/batch  alloc_MiB/batch")
        for sample in samples
            @printf("%5d  %8s  %7s  %2d  %4d  %13.4f  %15.2f\n",
                    sample.batch, _fmt_mib(sample.live_heap_bytes),
                    _fmt_mib(sample.rss_bytes), sample.fd_count, sample.log_rows,
                    sample.mean_batch_seconds,
                    sample.mean_batch_alloc_bytes / MIB)
        end
        isempty(csv_path) || println("samples: $(abspath(csv_path))")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    monitor_soak_main()
end
