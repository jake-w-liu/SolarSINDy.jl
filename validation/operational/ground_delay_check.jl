module GroundDelayCheck

using CSV, DataFrames, Dates, JSON3, SHA, Statistics

const MODEL_PATH = normpath(joinpath(@__DIR__, "../../deploy/cmo_ground_shadow/model.json"))
const SOURCE_HASH = "a7386f50ea3e5fe920e224fb56b5040b9a4e7c188b9e54ec68559d0cabf2c47d"
const DELAYS = (0, 5, 10)
good(x) = x isa Real && !(x isa Bool) && isfinite(x)

function model_point(model, window)
    length(model.mu)==length(model.sigma)==4 && length(model.beta)==5 &&
        all(good,model.mu) && all(good,model.beta) &&
        all(x->good(x) && x>0,model.sigma) || throw(ArgumentError("invalid frozen ground coefficients"))
    length(window) == 30 && all(x -> good(x) && x >= 0, window) ||
        throw(ArgumentError("features require 30 finite nonnegative one-minute derivatives"))
    features = (last(window), mean(window), maximum(window), std(window))
    value = Float64(model.beta[1]) + sum(Float64(model.beta[k+1]) *
        (log1p(features[k]) - Float64(model.mu[k])) / Float64(model.sigma[k]) for k in 1:4)
    point = max(0, expm1(value))
    isfinite(point) || throw(ArgumentError("nonfinite ground prediction"))
    return point
end

function evaluate_delay(times, derivative, model, delay::Integer)
    delay in DELAYS || throw(ArgumentError("delay must be one of the frozen 0, 5, 10 minute checks"))
    length(times) == length(derivative) || throw(ArgumentError("unequal observation lengths"))
    all(diff(times) .> Millisecond(0)) || throw(ArgumentError("observation times must be unique and increasing"))
    irregular = cumsum(vcat(0, diff(times) .!= Minute(1)))
    indices = [i for i in 31:length(times)-30-delay
        if irregular[i+30+delay] == irregular[i-30] &&
           all(x -> good(x) && x >= 0, @view derivative[i-29:i+30+delay])]
    isempty(indices) && throw(ArgumentError("no complete observation windows"))
    dates = times[indices]
    prediction = [model_point(model, @view derivative[i-29:i]) for i in indices]
    target = [maximum(@view derivative[i+delay+1:i+delay+30]) for i in indices]
    past = [maximum(@view derivative[i-29:i]) for i in indices]
    residual = (target .- prediction) ./ (1 .+ past)
    seed = Float64.(model.calibration_seed)
    length(seed) == 1440 && all(isfinite, seed) || throw(ArgumentError("invalid calibration seed"))
    upper = similar(prediction)
    epoch = 1
    for k in eachindex(indices)
        k > 1 && dates[k] - dates[k-1] > Day(1) && (epoch = k)
        # Outcome completion is target-end + publication delay. Require receipt strictly
        # before decision time; equality cannot establish processing order.
        stop = searchsortedlast(dates, dates[k] - Minute(30+delay) - Millisecond(1))
        history = stop >= epoch ? vcat(seed, @view residual[epoch:stop]) : seed
        retained = @view history[max(1,end-1439):end]
        rank = min(length(retained), ceil(Int, .9*(length(retained)+1)))
        upper[k] = max(0, prediction[k] + sort(retained)[rank]*(1+past[k]))
    end
    return DataFrame(; anchor=dates, target_start=dates .+ Minute(delay),
        target_end=dates .+ Minute(delay+30), prediction, target, past, upper)
end

function metrics(frame, model)
    nrow(frame) > 0 || throw(ArgumentError("cannot score an empty forecast panel"))
    controls = [frame.past, fill(Float64(model.climatology_native), nrow(frame)),
                fill(Float64(model.climatology_log), nrow(frame))]
    rmse(p) = sqrt(mean(abs2, p .- frame.target))
    logrmse(p) = sqrt(mean(abs2, log1p.(p) .- log1p.(frame.target)))
    point_rmse = rmse(frame.prediction); log_rmse = logrmse(frame.prediction)
    baseline_rmse = minimum(rmse, controls); baseline_log_rmse = minimum(logrmse, controls)
    coverage = count(frame.target .<= frame.upper)/nrow(frame)
    return (; n=nrow(frame), point_rmse, log_rmse, baseline_rmse, baseline_log_rmse,
        coverage, native_pass=point_rmse<=baseline_rmse, log_pass=log_rmse<=baseline_log_rmse,
        coverage_pass=.88<=coverage<=.92)
end

function read_period(raw_root, period)
    rawdir = joinpath(raw_root, period in (2018, 2021) ? "additional_storm_raw" : "transfer_raw")
    receipts = JSON3.read(read(joinpath(rawdir, "receipt.json")))
    source = filter(q -> q.station == "CMO" && q.product == "adjusted" &&
        year(DateTime(String(q.start))) == period, receipts)
    isempty(source) && error("No archived CMO adjusted receipts for $period")
    observations = Dict{DateTime,Tuple{Float64,Float64}}()
    hashes = String[]
    for q in source
        successful = filter(a -> get(a.result,:status,0)==200 && get(a.result,:times,0)>0, q.attempts)
        isempty(successful) && error("No successful response for $(q.name)")
        receipt = last(successful).result
        bytes = read(joinpath(rawdir, String(receipt.filename)))
        bytes2hex(sha256(bytes)) == receipt.sha256 || error("Raw receipt hash mismatch")
        push!(hashes, String(receipt.sha256))
        raw = JSON3.read(bytes); meta = raw.metadata.intermagnet
        meta.imo.iaga_code == "CMO" && meta.data_type == "adjusted" &&
            meta.reported_orientation == "XY" && meta.sampling_period == 60 || error("Wrong USGS metadata")
        x = only(c.values for c in raw.values if c.metadata.element == "X" && c.metadata.station == "CMO")
        y = only(c.values for c in raw.values if c.metadata.element == "Y" && c.metadata.station == "CMO")
        length(raw.times) == length(x) == length(y) || error("Unequal USGS channels")
        for i in eachindex(raw.times)
            time = DateTime(chop(String(raw.times[i]); tail=1))
            pair = (good(x[i]) ? Float64(x[i]) : NaN, good(y[i]) ? Float64(y[i]) : NaN)
            haskey(observations,time) && !isequal(observations[time],pair) && error("Conflicting raw observation")
            observations[time] = pair
        end
    end
    times = sort!(collect(keys(observations))); derivative = fill(NaN,length(times))
    for i in 2:length(times)
        times[i]-times[i-1] == Minute(1) || continue
        previous, current = observations[times[i-1]], observations[times[i]]
        derivative[i] = hypot(current[1]-previous[1], current[2]-previous[2])
    end
    return (; times, derivative, hashes)
end

function run_check(raw_root, output)
    ispath(output) && error("Refusing to overwrite delay-check evidence: $output")
    source_bytes = read(joinpath(raw_root,"stable_candidates_verified/artifacts.json"))
    bytes2hex(sha256(source_bytes)) == SOURCE_HASH || error("Original ground model artifact changed")
    original = only(m for m in JSON3.read(source_bytes) if m.station=="CMO" && m.method=="log_features_ridge")
    model = JSON3.read(read(MODEL_PATH))
    for key in (:mu, :sigma, :beta, :calibration_seed, :dataset_sha256)
        model[key] == original[key] || error("Frozen ground model changed: $key")
    end
    mkpath(output); results = []; availability = []
    for period in (2018,2021,2024,2026)
        series = read_period(raw_root, period)
        push!(availability,(;period,source_points=length(series.times),
            finite_derivatives=count(isfinite,series.derivative),raw_hashes=series.hashes))
        for delay in DELAYS
            frame = evaluate_delay(series.times,series.derivative,model,delay)
            CSV.write(joinpath(output,"CMO_adjusted_$(period)_delay$(delay).csv"),frame)
            push!(results,merge((;period,delay_minutes=delay),metrics(frame,model)))
        end
    end
    CSV.write(joinpath(output,"metrics.csv"),DataFrame(results))
    decision = (; model_sha256=bytes2hex(sha256(read(MODEL_PATH))),
        protocol_sha256=bytes2hex(sha256(read(joinpath(@__DIR__,"ground_shadow_protocol.md")))),
        receipt_time_screen_pass=all(r.native_pass && r.log_pass && r.coverage_pass for r in results),
        retrospective_only=true,prospective_qualification=false,serving_enabled=false)
    write(joinpath(output,"decision.json"),JSON3.write(decision))
    write(joinpath(output,"availability.json"),JSON3.write(availability))
    println(JSON3.write(results)); println(JSON3.write(decision))
    return decision
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("Usage: ground_delay_check.jl REMEDIATION_ARCHIVE NEW_OUTPUT_DIR")
    run_check(abspath(ARGS[1]),abspath(ARGS[2]))
end

end # module
